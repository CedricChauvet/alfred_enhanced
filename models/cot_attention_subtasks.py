"""
Chain-of-Thought avec Attention Masquée sur Subtask Actuel

Différences clés avec seq2seq_cot.py :
1. Masquage : Seul le subgoal ACTUEL est actif dans enc_lang
2. Tracking : Suivi du subgoal en cours d'exécution
3. Avancement : Détection automatique de complétion des subgoals
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


# Ajouter ALFRED au path
ALFRED_ROOT = Path.home() / "Bureau" / "Alfred" / "alfred"
if str(ALFRED_ROOT) not in sys.path:
    sys.path.insert(0, str(ALFRED_ROOT))
    
# Import du baseline
from models.model.seq2seq_im_mask import Module as BaseModule


class CoTAttentionSubtaskModule(BaseModule):
    """
    Seq2Seq + Chain-of-Thought avec Attention Masquée sur Subtask
    
    Architecture :
    - Génère un plan complet de 10 subgoals au départ
    - Enrichit enc_lang avec SEULEMENT le subgoal actuel (les autres sont masqués)
    - Détecte automatiquement quand avancer au subgoal suivant
    """
    
    def __init__(self, args, vocab):
    
        print("\n" + "="*70)
        print("INITIALIZING CHAIN-OF-THOUGHT WITH SUBTASK ATTENTION")
        print("="*70)
        
        # Initialiser le parent
        super().__init__(args, vocab)
        
        # Hyperparamètres CoT
        self.use_cot = getattr(args, 'use_cot', True)
        self.max_subgoals = getattr(args, 'max_subgoals', 12)
        self.cot_loss_weight = getattr(args, 'cot_loss_weight', 0.5)
        
        # ✅ NOUVEAU : Paramètres pour l'avancement de subgoal
        self.steps_per_subgoal = getattr(args, 'steps_per_subgoal', 15)
        self.use_completion_detector = getattr(args, 'use_completion_detector', True)
        
        if self.use_cot:
            print(f"\n✓ Chain-of-Thought with Subtask Attention ENABLED")
            print(f"  Max subgoals: {self.max_subgoals}")
            print(f"  CoT loss weight: {self.cot_loss_weight}")
            print(f"  Steps per subgoal: {self.steps_per_subgoal}")
            print(f"  Use completion detector: {self.use_completion_detector}")
            print(f"  High-level vocab size: {len(vocab['action_high'])}")
            
            # Projections
            self.project_cont = nn.Linear(2*args.dhid, args.dhid)
            self.project_subgoals = nn.Linear(args.dhid, 2*args.dhid)
            
            # Modules CoT
            self.subgoal_decoder = nn.LSTM(
                args.demb + args.dhid,
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
            
            # ✅ NOUVEAU : Compteur de steps (pour avancement automatique)
            self.step_counter = 0
            
        else:
            print("\n✗ Chain-of-Thought DISABLED")
        
        print("="*70 + "\n")


    def forward(self, feat, max_decode=300):
        """Forward avec CoT (training/validation)"""
        
        # ✓ FIX GPU : Déplacer features vers le device du modèle
        device = next(self.parameters()).device
        
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
            
            # ✅ TRAINING : Enrichir avec TOUS les subgoals
            # (En training, on ne masque pas car on a le ground truth)
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
        """
        Génère les subgoals avec CoT
        
        ✅ FIX : TOUJOURS générer max_subgoals (pas de early stop)
        """
        batch_size = cont_lang.size(0)
        device = cont_lang.device
        
        # Projeter cont_lang
        cont_lang_proj = self.project_cont(cont_lang)
        
        # État initial
        h_0 = cont_lang_proj.unsqueeze(0)
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
        
        # ✅ FIX CRITIQUE : TOUJOURS générer max_subgoals (pas de break)
        for t in range(self.max_subgoals):
            emb = self.emb_subgoal(input_tok)
            lstm_input = torch.cat([emb, cont_lang_proj], dim=-1).unsqueeze(1)
            
            output, state = self.subgoal_decoder(lstm_input, state)
            output = output.squeeze(1)
            
            logits = self.subgoal_classifier(output)
            
            subgoals_logits.append(logits)
            subgoals_hidden.append(output)
            
            # Préparer input suivant
            if not self.test_mode and 'action_high' in feat:
                # TRAINING : Teacher forcing
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
                # INFERENCE : Greedy
                input_tok = logits.argmax(dim=-1)
                # ✅ PAS DE BREAK - continuer jusqu'à max_subgoals
        
        # Stack exactement max_subgoals timesteps
        subgoals_logits = torch.stack(subgoals_logits, dim=1)
        subgoals_hidden = torch.stack(subgoals_hidden, dim=1)
        
        return subgoals_logits, subgoals_hidden
    
    
    
    def _enhance_context(self, enc_lang, subgoals_hidden, current_subgoal_idx=None):
        """
        ✅ MÉTHODE UNIFIÉE : Enrichir enc_lang avec les subgoals (avec masquage optionnel)
        
        Args:
            enc_lang: (batch, seq_len, 2*dhid)
            subgoals_hidden: (batch, max_subgoals, dhid)
            current_subgoal_idx: int or None
                - None : Tous les subgoals actifs (training)
                - int : Seulement ce subgoal actif (inference)
        """
        batch_size = enc_lang.size(0)
        device = enc_lang.device
        
        # Projeter subgoals
        subgoals_proj = self.project_subgoals(subgoals_hidden)
        
        # Masquage conditionnel
        if current_subgoal_idx is not None:
            mask = torch.zeros(batch_size, self.max_subgoals, 1, device=device)
            mask[:, current_subgoal_idx, :] = 1.0
            subgoals_proj = subgoals_proj * mask
        
        return torch.cat([enc_lang, subgoals_proj], dim=1)
    

    
    def compute_loss(self, out, batch, feat):
        """Loss avec CoT"""
        losses = super().compute_loss(out, batch, feat)
        
        if self.use_cot and 'subgoals_logits' in feat and 'action_high' in feat:
            sg_logits = feat['subgoals_logits']
            sg_gt = feat['action_high']
            
            batch_size, max_pred, vocab_size = sg_logits.shape
            max_gt = sg_gt.size(1)
            
            # Aligner les longueurs avec padding
            if max_gt < max_pred:
                padding = torch.full(
                    (batch_size, max_pred - max_gt),
                    self.pad,
                    dtype=sg_gt.dtype,
                    device=sg_gt.device
                )
                sg_gt = torch.cat([sg_gt, padding], dim=1)
            elif max_gt > max_pred:
                sg_gt = sg_gt[:, :max_pred]
            
            # Flatten
            sg_logits_flat = sg_logits.reshape(-1, vocab_size)
            sg_gt_flat = sg_gt.reshape(-1)
            
            # Loss avec ignore_index
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
        """Ajoute high-level actions au features"""
        feat = super().featurize(batch, load_mask, load_frames)
        
        if self.use_cot and not self.test_mode:
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
            
            seqs = [torch.tensor(a, device=device, dtype=torch.long) 
                for a in high_actions]
            feat['action_high'] = pad_sequence(
                seqs, 
                batch_first=True, 
                padding_value=self.pad
            )
        
        return feat
    
    
    def step(self, feat, prev_action=None):
        """
        ✅ NOUVEAU : Inference avec attention masquée sur subtask actuel
        """
        if self.r_state['cont_lang'] is None:
            # ========== INITIALISATION (premier step) ==========
            
            # Encoder l'instruction
            self.r_state['cont_lang'], enc_lang_base = self.encode_lang(feat)
            
            if self.use_cot:
                # Générer le plan complet (10 subgoals)
                subgoals_logits, subgoals_hidden = self._generate_subgoals(
                    self.r_state['cont_lang'],
                    enc_lang_base,
                    {}
                )
                
                # Stocker le plan complet
                self.r_state['subgoals'] = subgoals_logits.argmax(dim=-1)
                self.r_state['subgoals_hidden'] = subgoals_hidden
                
                # ✅ Initialiser le tracker à 0 (premier subgoal)
                self.r_state['current_subgoal_idx'] = 0
                self.step_counter = 0
                
                # Afficher le plan
                self._print_cot_plan(self.r_state['subgoals'])
                
                # ✅ Enrichir avec SEULEMENT le premier subgoal
                self.r_state['enc_lang'] = self._enhance_context(
                    enc_lang_base,
                    subgoals_hidden,
                    current_subgoal_idx=0
                )
                
                # Stocker enc_lang_base pour réutilisation
                self.r_state['enc_lang_base'] = enc_lang_base
            else:
                # Pas de CoT
                self.r_state['enc_lang'] = enc_lang_base
        
        else:
            # ========== STEPS SUIVANTS ==========
            
            if self.use_cot and prev_action is not None:
                # Incrémenter le compteur
                self.step_counter += 1
                
                # ✅ Détecter si on doit avancer au subgoal suivant
                should_advance = False
                
                if self.use_completion_detector and 'out_subgoal' in self.r_state:
                    # Option 1 : Utiliser la prédiction du baseline
                    subgoal_completion = torch.sigmoid(self.r_state['out_subgoal'][0, -1, 0])
                    if subgoal_completion > 0.75:
                        should_advance = True
                
                # Option 2 : Heuristique basée sur les steps
                if not should_advance and self.step_counter >= self.steps_per_subgoal:
                    should_advance = True
                
                # Avancer si nécessaire
                if should_advance:
                    old_idx = self.r_state['current_subgoal_idx']
                    self.r_state['current_subgoal_idx'] = min(
                        old_idx + 1,
                        self.max_subgoals - 1
                    )
                    
                    if self.r_state['current_subgoal_idx'] != old_idx:
                        # Reset le compteur
                        self.step_counter = 0
                        
                        # ✅ Re-enrichir enc_lang avec le NOUVEAU subgoal actuel
                        self.r_state['enc_lang'] = self._enhance_context(
                            self.r_state['enc_lang_base'],
                            self.r_state['subgoals_hidden'],
                            current_subgoal_idx=self.r_state['current_subgoal_idx']
                        )
                        
                        # Afficher le changement
                        new_subgoal_name = self._get_subgoal_name(
                            self.r_state['subgoals'][0, self.r_state['current_subgoal_idx']]
                        )
                        print(f"\n>>> Advancing: subgoal {old_idx} → {self.r_state['current_subgoal_idx']} ({new_subgoal_name})")
        
        # Appeler le parent pour décoder l'action
        m_out = super().step(feat, prev_action)
        
        # Stocker out_subgoal pour la prochaine itération (si disponible)
        if 'out_subgoal' in m_out:
            self.r_state['out_subgoal'] = m_out['out_subgoal']
        
        return m_out
    
    
    def reset(self):
        """Reset entre les épisodes"""
        super().reset()
        
        # ✅ Réinitialiser le tracker CoT
        self.step_counter = 0
        
        if 'current_subgoal_idx' in self.r_state:
            self.r_state['current_subgoal_idx'] = 0
        
        if 'enc_lang_base' in self.r_state:
            del self.r_state['enc_lang_base']
    
    
    def _get_subgoal_name(self, subgoal_idx):
        """Helper pour afficher le nom du subgoal"""
        try:
            return self.vocab['action_high'].index2word(subgoal_idx.item())
        except:
            return f"<unknown_{subgoal_idx.item()}>"
    
    
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
            
            marker = "✓" if i == 0 else " "
            print(f"  {marker} Step {i+1}: {subgoal_name}")
        
        print("="*60 + "\n")


# Alias pour ALFRED
Module = CoTAttentionSubtaskModule