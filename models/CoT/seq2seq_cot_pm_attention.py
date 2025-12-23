"""
Chain-of-Thought extension pour ALFRED avec VRAIE attention PM-Subgoals
Version finale avec mécanisme d'attention Query-Key-Value
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
ALFRED_ROOT = Path("/media/cedrix/Ubuntu_2To/Alfred/alfred")
if str(ALFRED_ROOT) not in sys.path:
    sys.path.insert(0, str(ALFRED_ROOT))
    
# Import du baseline
from models.model.seq2seq_im_mask import Module as BaseModule


class PMSubgoalAttention(nn.Module):
    """
    VRAIE attention Query-Key-Value entre Progress Monitor et Subgoals
    
    Le PM génère une Query qui calcule la similarité avec chaque Subgoal (Keys)
    pour obtenir une représentation pondérée des Subgoals (Values)
    """
    
    def __init__(self, pm_dim, subgoal_dim, hidden_dim):
        super().__init__()
        
        # ═══════════════════════════════════════════════════════════
        # Projections Query-Key-Value
        # ═══════════════════════════════════════════════════════════
        
        # PM → Query
        self.query_projection = nn.Sequential(
            nn.Linear(pm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Subgoals → Keys
        self.key_projection = nn.Linear(subgoal_dim, hidden_dim)
        
        # Subgoals → Values
        self.value_projection = nn.Linear(subgoal_dim, hidden_dim)
        
        # Scaling factor pour stabilité numérique
        self.scale = hidden_dim ** -0.5
        
        # ═══════════════════════════════════════════════════════════
        # Biais positionnel (optionnel mais utile)
        # ═══════════════════════════════════════════════════════════
        self.use_positional_bias = True
        
    
    def compute_positional_bias(self, pm, n_subgoals, device):
        """
        Calcule un biais positionnel basé sur le PM
        
        Si PM=0.3 → Biais vers les premiers subgoals
        Si PM=0.7 → Biais vers les derniers subgoals
        """
        batch_size = pm.size(0)
        
        # Positions normalisées des subgoals [0, 1/(n-1), 2/(n-1), ..., 1]
        positions = torch.arange(n_subgoals, dtype=torch.float32, device=device)
        positions = positions / (n_subgoals - 1) if n_subgoals > 1 else positions
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (batch, n_subgoals)
        
        # PM étendu
        pm_expanded = pm.expand(-1, n_subgoals)  # (batch, n_subgoals)
        
        # Distance entre PM et chaque position
        distances = torch.abs(pm_expanded - positions)
        
        # Gaussienne centrée sur PM (sigma contrôle la largeur)
        sigma = 0.25  # Plus sigma est petit, plus l'attention est focalisée
        bias = torch.exp(- (distances ** 2) / (2 * sigma ** 2))
        
        return bias  # (batch, n_subgoals)
    
    
    def forward(self, pm, subgoals, mask=None):
        """
        Attention PM → Subgoals
        
        Args:
            pm: (batch, pm_dim) - Progress Monitor (généralement pm_dim=1)
            subgoals: (batch, n_subgoals, subgoal_dim) - Représentations des subgoals
            mask: (batch, n_subgoals) - 1 pour valide, 0 pour padding
            
        Returns:
            attended_output: (batch, hidden_dim) - Représentation agrégée des subgoals
            attention_weights: (batch, n_subgoals) - Poids d'attention
        """
        batch_size, n_subgoals, subgoal_dim = subgoals.shape
        device = subgoals.device
        
        # ═══════════════════════════════════════════════════════════
        # 1. Projections Query-Key-Value
        # ═══════════════════════════════════════════════════════════
        
        # Query depuis PM
        query = self.query_projection(pm)  # (batch, hidden_dim)
        query = query.unsqueeze(1)          # (batch, 1, hidden_dim)
        
        # Keys et Values depuis Subgoals
        keys = self.key_projection(subgoals)    # (batch, n_subgoals, hidden_dim)
        values = self.value_projection(subgoals)  # (batch, n_subgoals, hidden_dim)
        
        # ═══════════════════════════════════════════════════════════
        # 2. Calcul des scores d'attention (similarité Query-Keys)
        # ═══════════════════════════════════════════════════════════
        
        # Dot product: Query @ Keys^T
        scores = torch.matmul(query, keys.transpose(-2, -1))  # (batch, 1, n_subgoals)
        scores = scores.squeeze(1)  # (batch, n_subgoals)
        
        # Scaling (stabilité numérique)
        scores = scores * self.scale
        
        # ═══════════════════════════════════════════════════════════
        # 3. Biais positionnel (optionnel)
        # ═══════════════════════════════════════════════════════════
        
        if self.use_positional_bias:
            # PM est (batch, 1), extraire le scalar
            pm_scalar = pm.squeeze(-1) if pm.dim() > 1 else pm
            positional_bias = self.compute_positional_bias(
                pm_scalar.unsqueeze(-1), n_subgoals, device
            )
            
            # Combiner attention sémantique + biais positionnel
            # scores reflète QUOI (contenu)
            # positional_bias reflète OÙ (position dans la tâche)
            scores = scores + positional_bias
        
        # ═══════════════════════════════════════════════════════════
        # 4. Masquage du padding
        # ═══════════════════════════════════════════════════════════
        
        if mask is not None:
            # Mettre -inf sur les positions paddées
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # ═══════════════════════════════════════════════════════════
        # 5. Normalisation (Softmax)
        # ═══════════════════════════════════════════════════════════
        
        attention_weights = F.softmax(scores, dim=-1)  # (batch, n_subgoals)
        
        # ═══════════════════════════════════════════════════════════
        # 6. Agrégation pondérée des Values
        # ═══════════════════════════════════════════════════════════
        
        # attention_weights: (batch, 1, n_subgoals)
        # values: (batch, n_subgoals, hidden_dim)
        attention_weights_expanded = attention_weights.unsqueeze(1)
        attended_output = torch.matmul(attention_weights_expanded, values)  # (batch, 1, hidden_dim)
        attended_output = attended_output.squeeze(1)  # (batch, hidden_dim)
        
        return attended_output, attention_weights


class CoTModule(BaseModule):
    """
    Seq2Seq + Chain-of-Thought avec VRAIE attention PM-Subgoals
    """
    
    def __init__(self, args, vocab):
    
        print("\n" + "="*70)
        print("INITIALIZING CoT WITH TRUE PM-SUBGOAL ATTENTION")
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
            
            # ═══════════════════════════════════════════════════════════
            # Projections standard
            # ═══════════════════════════════════════════════════════════
            self.project_cont = nn.Linear(2*args.dhid, args.dhid)
            self.project_subgoals = nn.Linear(args.dhid, 2*args.dhid)
            
            # ═══════════════════════════════════════════════════════════
            # ✨ VRAIE ATTENTION PM → Subgoals
            # ═══════════════════════════════════════════════════════════
            self.pm_subgoal_attention = PMSubgoalAttention(
                pm_dim=1,              # PM est un scalar
                subgoal_dim=args.dhid, # Hidden state des subgoals
                hidden_dim=args.dhid   # Dimension de l'attention
            )
            print(f"  ✨ PM-Subgoal Attention (Query-Key-Value)")
            print(f"     - PM (1D) → Query ({args.dhid}D)")
            print(f"     - Subgoals ({args.dhid}D) → Keys & Values")
            print(f"     - Output: Weighted subgoal representation")
            print(f"     - Positional bias: ENABLED")
            # ═══════════════════════════════════════════════════════════
            
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
            
            print(f"  Subgoal tokens: start=<<seg>>, stop=<<stop>>")
            print(f"  Projections: {2*args.dhid}→{args.dhid}, {args.dhid}→{2*args.dhid}")
        else:
            print("\n✗ Chain-of-Thought DISABLED")
        
        print("="*70 + "\n")


    def forward(self, feat, max_decode=300):
        """Forward avec CoT et vraie attention PM-Subgoals"""
        
        # Déplacer vers device
        device = next(self.parameters()).device
        for key in feat:
            if isinstance(feat[key], torch.Tensor):
                feat[key] = feat[key].to(device)
        
        # Encoder instruction
        cont_lang, enc_lang = self.encode_lang(feat)
        
        # Générer subgoals (CoT)
        if self.use_cot and not self.test_mode:
            subgoals_logits, subgoals_hidden, subgoal_mask = self._generate_subgoals(
                cont_lang, enc_lang, feat
            )
            feat['subgoals_logits'] = subgoals_logits
            feat['subgoals_hidden'] = subgoals_hidden
            feat['subgoal_mask'] = subgoal_mask
            
            # Enrichir le contexte (sans PM pour l'instant, le PM vient après décodage)
            enc_lang = self._enhance_context_initial(enc_lang, subgoals_hidden, subgoal_mask)
        
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
        
        # ═══════════════════════════════════════════════════════════
        # ✨ POST-PROCESSING: Appliquer l'attention PM-Subgoals
        # ═══════════════════════════════════════════════════════════
        if self.use_cot and 'progress' in res and 'subgoals_hidden' in feat:
            # Extraire PM (moyenne sur la séquence)
            pm = res['progress']  # (batch, seq_len)
            pm_mean = pm.mean(dim=1, keepdim=True)  # (batch, 1)
            
            # Appliquer l'ATTENTION avec le masque
            subgoals_hidden = feat['subgoals_hidden']
            subgoal_mask = feat['subgoal_mask']
            
            attended_subgoal, attention_weights = self.pm_subgoal_attention(
                pm_mean, 
                subgoals_hidden, 
                mask=subgoal_mask
            )
            
            # Stocker pour analyse
            feat['attended_subgoal'] = attended_subgoal
            feat['pm_subgoal_attention'] = attention_weights
        # ═══════════════════════════════════════════════════════════
        
        feat.update(res)
        return feat


    def _generate_subgoals(self, cont_lang, enc_lang, feat):
        """Génère les subgoals et retourne le masque de validité"""
        batch_size = cont_lang.size(0)
        device = cont_lang.device
        
        # Initialisation LSTM
        h_0 = self.project_cont(cont_lang).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        
        input_tok = torch.full((batch_size,), self.subgoal_start, dtype=torch.long, device=device)
        
        subgoals_logits = []
        subgoals_hidden = []
        subgoal_mask = torch.ones(batch_size, self.max_subgoals, device=device)
        
        cont_lang_projected = self.project_cont(cont_lang)
        
        # Générer max_subgoals timesteps
        for t in range(self.max_subgoals):
            emb = self.emb_subgoal(input_tok)
            weighted_lang = cont_lang_projected
            lstm_input = torch.cat([emb, weighted_lang], dim=-1).unsqueeze(1)
            
            output, (h_0, c_0) = self.subgoal_decoder(lstm_input, (h_0, c_0))
            output = output.squeeze(1)
            
            logits = self.subgoal_classifier(output)
            
            subgoals_logits.append(logits)
            subgoals_hidden.append(output)
            
            # ✨ Masquage après <<stop>>
            if not self.test_mode and 'action_high' in feat:
                if t < feat['action_high'].size(1):
                    current_token = feat['action_high'][:, t]
                    is_stop_or_pad = (current_token == self.subgoal_stop) | (current_token == self.pad)
                    subgoal_mask[:, t] = (~is_stop_or_pad).float()
            else:
                pred_token = logits.argmax(dim=-1)
                is_stop = (pred_token == self.subgoal_stop)
                subgoal_mask[:, t] = (~is_stop).float()
            
            # Préparer prochain input
            if not self.test_mode and 'action_high' in feat:
                if t < feat['action_high'].size(1):
                    input_tok = feat['action_high'][:, t]
                else:
                    input_tok = torch.full((batch_size,), self.subgoal_stop, dtype=torch.long, device=device)
            else:
                input_tok = logits.argmax(dim=-1)
        
        subgoals_logits = torch.stack(subgoals_logits, dim=1)
        subgoals_hidden = torch.stack(subgoals_hidden, dim=1)
        
        return subgoals_logits, subgoals_hidden, subgoal_mask


    def _enhance_context_initial(self, enc_lang, subgoals_hidden, subgoal_mask):
        """
        Enrichissement initial du contexte SANS PM
        (Le PM n'est disponible qu'après décodage)
        
        Zéro out les subgoals paddés pour ne pas polluer le contexte
        """
        # Projeter les subgoals
        subgoals_proj = self.project_subgoals(subgoals_hidden)  # (batch, max_subgoals, 2*dhid)
        
        # Appliquer le masque
        if subgoal_mask is not None:
            mask_expanded = subgoal_mask.unsqueeze(-1)  # (batch, max_subgoals, 1)
            subgoals_proj = subgoals_proj * mask_expanded
        
        # Concaténer
        enhanced_context = torch.cat([enc_lang, subgoals_proj], dim=1)
        
        return enhanced_context


    def _enhance_context_with_pm(self, enc_lang, subgoals_hidden, pm, subgoal_mask=None):
        """
        ═══════════════════════════════════════════════════════════
        ✨ VRAIE ATTENTION PM → Subgoals
        ═══════════════════════════════════════════════════════════
        
        Utilise l'attention Query-Key-Value pour pondérer les subgoals
        selon le Progress Monitor
        
        Args:
            enc_lang: (batch, seq_len, 2*dhid) - Contexte linguistique
            subgoals_hidden: (batch, max_subgoals, dhid) - Hidden states des subgoals
            pm: (batch,) ou (batch, 1) - Progress Monitor [0-1]
            subgoal_mask: (batch, max_subgoals) - Masque de validité
            
        Returns:
            enhanced_context: (batch, seq_len + 1, 2*dhid)
                Le "+1" vient du subgoal attendu ajouté au contexte
        """
        
        # ═══════════════════════════════════════════════════════════
        # 1. Attention PM → Subgoals
        # ═══════════════════════════════════════════════════════════
        
        # S'assurer que PM est (batch, 1)
        if pm.dim() == 1:
            pm = pm.unsqueeze(-1)
        
        # Appliquer l'attention (retourne un subgoal agrégé)
        attended_subgoal, attention_weights = self.pm_subgoal_attention(
            pm, 
            subgoals_hidden, 
            mask=subgoal_mask
        )
        # attended_subgoal: (batch, dhid)
        # attention_weights: (batch, max_subgoals)
        
        # ═══════════════════════════════════════════════════════════
        # 2. Projeter le subgoal attendu vers la dimension du contexte
        # ═══════════════════════════════════════════════════════════
        
        attended_subgoal_proj = self.project_subgoals(attended_subgoal)  # (batch, 2*dhid)
        attended_subgoal_proj = attended_subgoal_proj.unsqueeze(1)  # (batch, 1, 2*dhid)
        
        # ═══════════════════════════════════════════════════════════
        # 3. Ajouter au contexte linguistique
        # ═══════════════════════════════════════════════════════════
        
        # Option A: Concaténer le subgoal attendu
        enhanced_context = torch.cat([enc_lang, attended_subgoal_proj], dim=1)
        # (batch, seq_len + 1, 2*dhid)
        
        # Option B (alternative): Ajouter à tous les tokens
        # attended_expanded = attended_subgoal_proj.expand(-1, enc_lang.size(1), -1)
        # enhanced_context = enc_lang + attended_expanded
        
        return enhanced_context
        
        
    def compute_loss(self, out, batch, feat):
        """Loss avec CoT"""
        losses = super().compute_loss(out, batch, feat)
        
        if self.use_cot and 'subgoals_logits' in feat and 'action_high' in feat:
            sg_logits = feat['subgoals_logits']
            sg_gt = feat['action_high']
            
            batch_size, max_pred, vocab_size = sg_logits.shape
            max_gt = sg_gt.size(1)
            
            # Aligner les longueurs
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
            
            # Loss
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
            
            seqs = [torch.tensor(a, device=device, dtype=torch.long) for a in high_actions]
            feat['action_high'] = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
        
        return feat


# Alias
Module = CoTModule