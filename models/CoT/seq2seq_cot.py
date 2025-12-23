"""
Chain-of-Thought extension pour ALFRED
"""

import sys
import os
import collections
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


# ✓✓✓ FIX cuDNN ✓✓✓
torch.backends.cudnn.enabled = False
print("⚠️  cuDNN désactivé pour compatibilité GPU")


# Ajouter ALFRED au path pour trouver models.model
ALFRED_ROOT = Path.home() / "Bureau" / "Alfred" / "alfred"
if str(ALFRED_ROOT) not in sys.path:
    sys.path.insert(0, str(ALFRED_ROOT))
    
# Import direct du baseline
from models.model.seq2seq_im_mask import Module as BaseModule


class CoTModule(BaseModule):
    """
    Seq2Seq + Chain-of-Thought
    """
    
    def __init__(self, args, vocab):
    
        print("\n" + "="*70)
        print("INITIALIZING CHAIN-OF-THOUGHT MODULE")
        print("="*70)
        
        # Initialiser le parent
        super().__init__(args, vocab)
        
        # Hyperparamètres CoT
        self.use_cot = getattr(args, 'use_cot', True)
        self.max_subgoals = getattr(args, 'max_subgoals', 10)
        self.cot_loss_weight = getattr(args, 'cot_loss_weight', 0.5)
        
        if self.use_cot:
            print(f"\n✓ Chain-of-Thought ENABLED")
            print(f"  Max subgoals: {self.max_subgoals}")
            print(f"  CoT loss weight: {self.cot_loss_weight}")
            print(f"  High-level vocab size: {len(vocab['action_high'])}")
            
            # ✓ Projection pour passer de 2*dhid à dhid
            self.project_cont = nn.Linear(2*args.dhid, args.dhid)
            self.project_subgoals = nn.Linear(args.dhid, 2*args.dhid)
            # Modules CoT
            self.subgoal_decoder = nn.LSTM(
                args.demb + args.dhid,  # ← Changé de 2*dhid à dhid
                args.dhid,
                batch_first=True
            )
            
            self.subgoal_classifier = nn.Linear(
                args.dhid, 
                len(vocab['action_high'])
            )
            
            self.emb_subgoal = nn.Embedding(
                len(vocab['action_high']), 
                args.demb
            )
            
            # Tokens
            self.subgoal_start = vocab['action_high'].word2index('<<seg>>')
            self.subgoal_stop = vocab['action_high'].word2index('<<stop>>')
            
            print(f"  Subgoal tokens: start=<<seg>> ({self.subgoal_start}), stop=<<stop>> ({self.subgoal_stop})")
            print(f"  Context projection: 2*{args.dhid} → {args.dhid}")
            print(f"  Subgoals projection: {args.dhid} → 2*{args.dhid}") 
        else:
            print("\n✗ Chain-of-Thought DISABLED")
        
        print("="*70 + "\n")


    def forward(self, feat, max_decode=300):
        """Forward avec CoT optionnel"""
        
        # ✓✓✓ FIX GPU : Déplacer features vers le device du modèle ✓✓✓
        device = next(self.parameters()).device
        
        # Déplacer tous les tenseurs vers GPU/CPU
        for key in feat:
            if isinstance(feat[key], torch.Tensor):
                feat[key] = feat[key].to(device)
        
        # Encoder instruction
        cont_lang, enc_lang = self.encode_lang(feat)
        
        # Générer subgoals (CoT)
        if self.use_cot and not self.test_mode:
            subgoals_logits, subgoals_hidden = self._generate_subgoals(
                cont_lang, enc_lang, feat
            )
            feat['subgoals_logits'] = subgoals_logits
            feat['subgoals_hidden'] = subgoals_hidden
            
            # Enrichir le contexte
            enc_lang = self._enhance_context(enc_lang, subgoals_hidden)
        
        # Décoder actions
        state_0 = cont_lang, torch.zeros_like(cont_lang)
        frames = self.vis_dropout(feat['frames'])
        
        res = self.dec(
            enc_lang, 
            frames, 
            max_decode=max_decode,
            gold=feat['action_low'], 
            state_0=state_0
        )
        
        feat.update(res)
        return feat
    
    
    def _generate_subgoals(self, cont_lang, enc_lang, feat):
        """Génère les sous-buts"""
        batch_size = cont_lang.size(0)
        device = cont_lang.device
        
        # ✓ Projeter cont_lang de (batch, 2*dhid) à (batch, dhid)
        cont_lang_proj = self.project_cont(cont_lang)
        
        # État initial avec la bonne dimension
        h_0 = cont_lang_proj.unsqueeze(0)  # (1, batch, dhid)
        c_0 = torch.zeros_like(h_0)
        state = (h_0, c_0)
        
        # Token de départ
        input_tok = torch.full(
            (batch_size,), 
            self.subgoal_start, 
            dtype=torch.long, 
            device=device
        )
        
        subgoals_logits = []
        subgoals_hidden = []
        
        for t in range(self.max_subgoals):
            emb = self.emb_subgoal(input_tok)
            # ✓ Utiliser cont_lang_proj au lieu de cont_lang
            lstm_input = torch.cat([emb, cont_lang_proj], dim=-1).unsqueeze(1)
            
            output, state = self.subgoal_decoder(lstm_input, state)
            output = output.squeeze(1)
            
            logits = self.subgoal_classifier(output)
            
            subgoals_logits.append(logits)
            subgoals_hidden.append(output)
            
            if not self.test_mode and 'action_high' in feat:
                if t < feat['action_high'].size(1):
                    input_tok = feat['action_high'][:, t]
                else:
                    input_tok = torch.full(
                        (batch_size,), 
                        self.subgoal_stop,
                        dtype=torch.long, 
                        device=device
                    )
            else:
                input_tok = logits.argmax(dim=-1)
                if (input_tok == self.subgoal_stop).all():
                    break
        
        subgoals_logits = torch.stack(subgoals_logits, dim=1)
        subgoals_hidden = torch.stack(subgoals_hidden, dim=1)
        
        return subgoals_logits, subgoals_hidden
    
    
    def _enhance_context(self, enc_lang, subgoals_hidden):
        # enc_lang: (batch, seq_len, 256)
        # subgoals_hidden: (batch, max_subgoals, 128)
        
        subgoals_proj = self.project_subgoals(subgoals_hidden)
        # subgoals_proj: (batch, max_subgoals, 256)
        
        return torch.cat([enc_lang, subgoals_proj], dim=1)
        # ✓ Dimension 2: 256 vs 256
        # Résultat: (batch, seq_len + max_subgoals, 256)
    def compute_loss(self, out, batch, feat):
        """Loss avec CoT - Version optimale avec padding"""
        losses = super().compute_loss(out, batch, feat)
        
        if self.use_cot and 'subgoals_logits' in feat and 'action_high' in feat:
            sg_logits = feat['subgoals_logits']  # (batch, max_subgoals, vocab)
            sg_gt = feat['action_high']           # (batch, actual_length)
            
            batch_size, max_pred, vocab_size = sg_logits.shape
            max_gt = sg_gt.size(1)
            
            # ✓ Aligner les longueurs avec padding
            if max_gt < max_pred:
                # Padder sg_gt
                padding = torch.full(
                    (batch_size, max_pred - max_gt),
                    self.pad,
                    dtype=sg_gt.dtype,
                    device=sg_gt.device
                )
                sg_gt = torch.cat([sg_gt, padding], dim=1)
            elif max_gt > max_pred:
                # Tronquer sg_gt (rare)
                sg_gt = sg_gt[:, :max_pred]
            
            # Flatten
            sg_logits_flat = sg_logits.reshape(-1, vocab_size)
            sg_gt_flat = sg_gt.reshape(-1)
            
            # ✓ Loss avec ignore_index (ignore automatiquement le padding)
            cot_loss = F.cross_entropy(
                sg_logits_flat,
                sg_gt_flat,
                ignore_index=self.pad,
                reduction='mean'
            )
            
            losses['cot'] = cot_loss * self.cot_loss_weight
            
            # Accuracy
            pad_mask = (sg_gt_flat != self.pad)
            if pad_mask.sum() > 0:
                pred = sg_logits_flat.argmax(dim=-1)
                correct = (pred == sg_gt_flat) & pad_mask
                acc = correct.float().sum() / pad_mask.float().sum()
            else:
                acc = torch.tensor(0.0, device=sg_gt.device)
            
            losses['cot_acc'] = acc
        
        return losses
        
    def featurize(self, batch, load_mask=True, load_frames=True):
        """
        Ajoute high-level actions au features
        
        Version GPU-safe qui gère le transfert CPU/GPU correctement
        """
        import collections
        
        # Appeler le parent (crée tout sur CPU)
        feat = super().featurize(batch, load_mask, load_frames)
        
        # Ajouter action_high pour CoT
        if self.use_cot and not self.test_mode:
            # Créer sur CPU (comme le baseline)
            device = torch.device('cpu')
            
            high_actions = []
            
            for ex in batch:
                if 'plan' in ex and 'high_pddl' in ex['plan']:
                    acts = []
                    for a in ex['plan']['high_pddl']:
                        act_name = a['discrete_action']['action']
                        try:
                            act_idx = self.vocab['action_high'].word2index(act_name)
                        except:
                            act_idx = self.subgoal_start
                        acts.append(act_idx)
                    
                    acts.append(self.subgoal_stop)
                else:
                    acts = [self.subgoal_stop]
                
                high_actions.append(acts)
            
            # Créer les tenseurs sur CPU
            seqs = [torch.tensor(a, device=device, dtype=torch.long) 
                for a in high_actions]
            feat['action_high'] = pad_sequence(
                seqs, 
                batch_first=True, 
                padding_value=self.pad
            )
        
        return feat
        
    
    def step(self, feat, prev_action=None):
        """Inference temps réel avec CoT"""
        if self.r_state['cont_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)
            
            if self.use_cot:
                subgoals_logits, subgoals_hidden = self._generate_subgoals(
                    self.r_state['cont_lang'],
                    self.r_state['enc_lang'],
                    {}
                )
                self.r_state['subgoals'] = subgoals_logits.argmax(dim=-1)
                self.r_state['subgoals_hidden'] = subgoals_hidden
                
                self._print_cot_plan(self.r_state['subgoals'])
                
                self.r_state['enc_lang'] = self._enhance_context(
                    self.r_state['enc_lang'],
                    subgoals_hidden
                )
        
        return super().step(feat, prev_action)
    
    
    def _print_cot_plan(self, subgoals):
        """Affiche le plan CoT"""
        print("\n" + "="*60)
        print("CHAIN-OF-THOUGHT PLAN:")
        print("="*60)
        
        subgoals_list = subgoals[0].cpu().tolist()
        
        for i, sg_idx in enumerate(subgoals_list):
            if sg_idx == self.subgoal_stop:
                break
            
            try:
                subgoal_name = self.vocab['action_high'].index2word([sg_idx])[0]
            except:
                subgoal_name = f"<unknown_{sg_idx}>"
            
            print(f"  Step {i+1}: {subgoal_name}")
        
        print("="*60 + "\n")


# Alias pour ALFRED
Module = CoTModule