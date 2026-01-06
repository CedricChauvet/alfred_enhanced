"""
VERSION G - BASE : Chain-of-Thought avec génération de subgoals
AVANT ajout de self.subtasks dans le décodeur

Cette version ajoute UNIQUEMENT :
- Génération de subgoals via un LSTM autoregressif
- Loss CoT pour entraîner la génération
- SANS modifier le décodeur d'actions
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from models.model.seq2seq_im_mask import Module as BaseModule


class CoTModuleSubgoalBase(BaseModule):
    """
    Module avec Chain-of-Thought : génération de subgoals AVANT le décodage d'actions
    """
    
    def __init__(self, args, vocab):
        """
        Initialisation avec ajout du module CoT
        """
        super().__init__(args, vocab)
        
        # Hyperparamètres CoT
        self.use_cot_subgoal = getattr(args, 'use_cot_subgoal', True)
        self.max_subgoals = getattr(args, 'max_subgoals', 12)
        self.cot_loss_weight = getattr(args, 'cot_loss_weight', 0.5)
        
        if self.use_cot_subgoal:
            print("\n" + "="*70)
            print("🚀 VERSION G BASE : CHAIN-OF-THOUGHT SUBGOALS")
            print("="*70)
            print(f"  ✓ CoT Subgoals: {self.max_subgoals} max")
            print(f"  ✓ Subgoal vocab size: {len(vocab['action_high'])}")
            print(f"  ✓ Subgoal embedding: {args.demb}")
            
            # Modules CoT (génération de subgoals)
            self.subgoal_decoder = nn.LSTM(
                args.demb + 2*args.dhid,  # emb + contexte bidirectionnel
                args.dhid,
                batch_first=True
            )
            
            # Projection pour l'état initial : 2*dhid → dhid
            self.cont_to_hidden = nn.Linear(2*args.dhid, args.dhid)
            
            self.subgoal_classifier = nn.Linear(args.dhid, len(vocab['action_high']))
            self.emb_subgoal = nn.Embedding(len(vocab['action_high']), args.demb)
            
            print("="*70 + "\n")
    
    
    def forward(self, feat, max_decode=150):
        """
        Forward avec génération CoT de subgoals
        """
        # ═══════════════════════════════════════════════════════════
        # 0. Encodage linguistique (comme le parent)
        # ═══════════════════════════════════════════════════════════
        
        # Encode language
        emb_lang = self.emb_word(feat['lang'])
        enc_lang, _ = self.enc(emb_lang)
        enc_lang, _ = self.enc_att(enc_lang)
        
        # State encoding (dernier hidden state)
        cont_lang = enc_lang[:, -1]
        
        # ═══════════════════════════════════════════════════════════
        # 1. Générer subgoals CoT (NOUVEAU)
        # ═══════════════════════════════════════════════════════════
        
        if self.use_cot_subgoal:
            try:
                subgoals_logits, subgoals_hidden, subgoal_mask = \
                    self._generate_subgoals(cont_lang, enc_lang, feat)
                
                feat['subgoals_logits'] = subgoals_logits
                feat['subgoals_hidden'] = subgoals_hidden
                feat['subgoal_mask'] = subgoal_mask
                
            except Exception as e:
                print(f"\n❌ ERROR in subgoal generation:")
                print(f"   {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        # ═══════════════════════════════════════════════════════════
        # 2. Décodage d'actions (comme le parent)
        # ═══════════════════════════════════════════════════════════
        
        frames = self.vis_dropout(feat['frames'])
        res = self.dec(
            enc_lang,
            frames,
            max_decode=max_decode,
            gold=feat['action_low'],
            state_0=cont_lang
        )
        
        # Ajouter subgoals_logits dans le résultat
        if self.use_cot_subgoal and 'subgoals_logits' in feat:
            res['subgoals_logits'] = feat['subgoals_logits']
        
        return res
    
    
    def _generate_subgoals(self, cont_lang, enc_lang, feat):
        """
        Génération autorégressive de subgoals via LSTM
        
        Args:
            cont_lang: (batch, 2*dhid) - État final de l'encodeur
            enc_lang: (batch, seq_len, 2*dhid) - États de l'encodeur
            feat: Dictionnaire avec action_high si disponible
        
        Returns:
            subgoals_logits: (batch, max_subgoals, vocab_size)
            subgoals_hidden: (batch, max_subgoals, dhid)
            subgoal_mask: (batch, max_subgoals)
        """
        batch_size = cont_lang.size(0)
        device = cont_lang.device
        
        # État initial du LSTM : projection de cont_lang
        cont_projected = self.cont_to_hidden(cont_lang)  # (batch, dhid)
        h_0 = cont_projected.unsqueeze(0)  # (1, batch, dhid)
        c_0 = torch.zeros_like(h_0)
        
        # GO embedding
        go_emb = torch.zeros(batch_size, self.args.demb, device=device)
        
        # Context répété
        cont_expanded = cont_lang.unsqueeze(1).expand(-1, self.max_subgoals, -1)
        
        # Gold subgoals si disponible (training)
        gold_subgoals = None
        if 'action_high' in feat and self.training:
            gold_subgoals = feat['action_high']
        
        # Générer
        outputs = []
        hiddens = []
        
        for t in range(self.max_subgoals):
            # Embedding précédent
            if t == 0:
                emb_t = go_emb
            elif gold_subgoals is not None and self.training:
                # Teacher forcing
                emb_t = self.emb_subgoal(gold_subgoals[:, t-1])
            else:
                # Auto-regressive
                prev_pred = outputs[-1].argmax(-1)
                emb_t = self.emb_subgoal(prev_pred)
            
            # Input LSTM
            lstm_input = torch.cat([emb_t, cont_expanded[:, t]], dim=-1).unsqueeze(1)
            
            # LSTM step
            output, (h_0, c_0) = self.subgoal_decoder(lstm_input, (h_0, c_0))
            
            # Classifier
            logits = self.subgoal_classifier(output.squeeze(1))
            outputs.append(logits)
            hiddens.append(output.squeeze(1))
        
        # Stack
        subgoals_logits = torch.stack(outputs, dim=1)  # (batch, max_subgoals, vocab)
        subgoals_hidden = torch.stack(hiddens, dim=1)  # (batch, max_subgoals, dhid)
        
        # Masque (tous valides pour l'instant)
        subgoal_mask = torch.ones(batch_size, self.max_subgoals, dtype=torch.bool, device=device)
        
        return subgoals_logits, subgoals_hidden, subgoal_mask
    
    
    def compute_loss(self, out, batch, feat):
        """
        Calcul de la loss avec ajout de la loss CoT
        """
        # Loss du parent (actions, masque, progress)
        losses = super().compute_loss(out, batch, feat)
        
        # Loss CoT subgoals
        if self.use_cot_subgoal and 'subgoals_logits' in out and 'action_high' in feat:
            try:
                subgoals_logits = out['subgoals_logits']  # (batch, max_subgoals, vocab)
                gold_subgoals = feat['action_high']  # (batch, max_subgoals)
                
                # Cross-entropy
                loss_cot = F.cross_entropy(
                    subgoals_logits.view(-1, subgoals_logits.size(-1)),
                    gold_subgoals.view(-1),
                    ignore_index=self.pad,
                    reduction='mean'
                )
                
                # Accuracy
                pred_subgoals = subgoals_logits.argmax(-1)
                mask_valid = (gold_subgoals != self.pad)
                acc_cot = (pred_subgoals == gold_subgoals).float()
                acc_cot = (acc_cot * mask_valid).sum() / mask_valid.sum()
                
                # Ajouter aux losses
                losses['cot_subgoal'] = loss_cot * self.cot_loss_weight
                losses['cot_acc'] = acc_cot
                
            except Exception as e:
                print(f"\n❌ ERROR in CoT loss computation:")
                print(f"   {str(e)}")
                import traceback
                traceback.print_exc()
        
        return losses
    
    
    def featurize(self, batch):
        """
        Featurize avec extraction de action_high pour CoT
        """
        # Appeler featurize du parent
        feat = super().featurize(batch)
        
        # Extraire action_high pour CoT
        if not self.test_mode and self.use_cot_subgoal:
            action_high_list = []
            
            for ex in batch:
                # Extraire action_high
                if 'num' in ex and 'action_high' in ex['num']:
                    action_high_raw = ex['num']['action_high']
                    
                    # Gérer différents formats
                    if isinstance(action_high_raw, list) and len(action_high_raw) > 0:
                        first_elem = action_high_raw[0]
                        
                        if isinstance(first_elem, dict):
                            action_high = [a['action'] for a in action_high_raw]
                        elif isinstance(first_elem, str):
                            action_high = [self.vocab['action_high'].word2index(a) 
                                          for a in action_high_raw]
                        elif isinstance(first_elem, (int, np.integer)):
                            action_high = [int(a) for a in action_high_raw]
                        else:
                            action_high = [int(a) if not isinstance(a, str) 
                                          else self.vocab['action_high'].word2index(a) 
                                          for a in action_high_raw]
                    else:
                        action_high = []
                else:
                    action_high = []
                
                # Padding/troncature à max_subgoals
                if len(action_high) > self.max_subgoals:
                    action_high = action_high[:self.max_subgoals]
                
                action_high_list.append(torch.tensor(action_high, dtype=torch.long))
            
            # Padder et convertir en tensor
            if action_high_list:
                feat['action_high'] = pad_sequence(
                    action_high_list, 
                    batch_first=True, 
                    padding_value=self.pad
                )
        
        return feat


# Export
Module = CoTModuleSubgoalBase