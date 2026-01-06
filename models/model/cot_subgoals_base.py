import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.model.seq2seq_im_mask import Module as BaseModule
from models.nn import vnn
import numpy as np


class Module(BaseModule):
    """
    Modèle avec prédiction Chain-of-Thought des subgoals
    Hérite du modèle baseline seq2seq_im_mask
    """
    
    def __init__(self, args, vocab):
        """
        Initialisation du modèle avec décodeur de subgoals
        """
        super().__init__(args, vocab)
        
        # Vocabulaire des subgoals (actions high-level)
        self.vocab_subgoal = vocab['action_high']
        
        # Dimension des embeddings de subgoals
        self.demb_subgoal = getattr(args, 'demb_subgoal', args.demb)
        
        # Embeddings des subgoals
        self.emb_subgoal = nn.Embedding(len(self.vocab_subgoal), self.demb_subgoal)
        
        # Décodeur LSTM pour les subgoals
        # Prend en entrée: embedding du subgoal précédent + contexte linguistique
        self.subgoal_decoder = nn.LSTM(
            input_size=self.demb_subgoal + 2*args.dhid,  # embedding + contexte bidirectionnel
            hidden_size=args.dhid,
            num_layers=1,
            batch_first=True
        )
        
        # Couche de projection pour prédire le subgoal suivant
        self.subgoal_proj = nn.Linear(args.dhid, len(self.vocab_subgoal))
        
        # Embedding GO pour démarrer la génération (comme dans le décodeur d'actions)
        self.subgoal_go = nn.Parameter(torch.Tensor(self.demb_subgoal))
        nn.init.normal_(self.subgoal_go)
        
        # Utiliser les tokens existants du vocabulaire parent
        # self.pad et self.stop_token sont déjà définis dans la classe Base
        
        # Dropout pour le décodeur de subgoals
        self.subgoal_dropout = nn.Dropout(getattr(args, 'subgoal_dropout', 0.1))
        
        # Attention pour combiner contexte linguistique et subgoals
        self.subgoal_attention = vnn.SelfAttn(args.dhid)
        
        # Paramètre pour activer/désactiver la génération de subgoals
        self.use_subgoals = getattr(args, 'use_subgoals', True)
        
        print(f"✅ Subgoal decoder initialized with vocab size: {len(self.vocab_subgoal)}")
    
    
    def generate_subgoals(self, enc_lang, cont_lang, max_subgoals=10, sampling=False, temperature=1.0):
        """
        Génère une séquence de subgoals avec approche Chain-of-Thought
        
        Args:
            enc_lang: Encodage linguistique complet [batch, seq_len, dhid*2]
            cont_lang: Contexte linguistique (dernier état caché) [batch, dhid*2]
            max_subgoals: Nombre maximum de subgoals à générer
            sampling: Si True, échantillonne; sinon prend argmax
            temperature: Température pour le sampling (plus élevé = plus aléatoire)
        
        Returns:
            subgoals: Liste de subgoals prédits [batch, num_subgoals]
            subgoal_logits: Logits pour chaque subgoal [batch, num_subgoals, vocab_size]
            subgoal_embeddings: Embeddings des subgoals [batch, num_subgoals, demb_subgoal]
        """
        device = next(self.parameters()).device
        batch_size = enc_lang.size(0)
        
        # Liste pour stocker les prédictions
        predicted_subgoals = []
        subgoal_logits_list = []
        subgoal_embeddings_list = []
        
        # État caché initial du décodeur (dérivé du contexte linguistique)
        # Réduire la dimension de cont_lang (dhid*2) vers dhid
        h_t = cont_lang[:, :self.args.dhid].unsqueeze(0).contiguous()  # [1, batch, dhid]
        c_t = torch.zeros_like(h_t)  # État de cellule initial
        
        # Embedding de démarrage GO pour tous les exemples du batch
        emb_t = self.subgoal_go.repeat(batch_size, 1)  # [batch, demb_subgoal]
        current_token = None  # Première itération utilise GO, pas un token
        
        # Génération autoregressif des subgoals
        for step in range(max_subgoals):
            # Embedding du token actuel (GO pour la première itération, sinon embedding du token précédent)
            if current_token is None:
                # Première itération: utiliser GO
                emb_current = emb_t
            else:
                # Itérations suivantes: utiliser l'embedding du token prédit
                emb_current = self.emb_subgoal(current_token)  # [batch, demb_subgoal]
            
            # Concaténer avec le contexte linguistique
            decoder_input = torch.cat([emb_current, cont_lang], dim=-1)  # [batch, demb_subgoal + dhid*2]
            decoder_input = decoder_input.unsqueeze(1)  # [batch, 1, demb_subgoal + dhid*2]
            
            # Passer dans le LSTM décodeur
            lstm_out, (h_t, c_t) = self.subgoal_decoder(decoder_input, (h_t, c_t))
            
            # Appliquer dropout
            lstm_out = self.subgoal_dropout(lstm_out.squeeze(1))  # [batch, dhid]
            
            # Prédire le prochain subgoal
            logits = self.subgoal_proj(lstm_out)  # [batch, vocab_size]
            
            # Échantillonnage ou argmax
            if sampling:
                # Échantillonnage avec température
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy decoding
                next_token = logits.argmax(dim=-1)
            
            # Stocker les résultats
            predicted_subgoals.append(next_token)
            subgoal_logits_list.append(logits)
            subgoal_embeddings_list.append(emb_current)
            
            # Vérifier si tous les exemples du batch ont généré <<stop>>
            if (next_token == self.stop_token).all():
                break
            
            # Mettre à jour le token actuel pour le prochain step
            current_token = next_token
        
        # Stacker les résultats
        subgoals = torch.stack(predicted_subgoals, dim=1)  # [batch, num_subgoals]
        subgoal_logits = torch.stack(subgoal_logits_list, dim=1)  # [batch, num_subgoals, vocab_size]
        subgoal_embeddings = torch.stack(subgoal_embeddings_list, dim=1)  # [batch, num_subgoals, demb_subgoal]
        
        return subgoals, subgoal_logits, subgoal_embeddings
    
    
    def forward(self, feat, max_decode=300):
        """
        Forward pass avec génération de subgoals CoT
        
        Processus:
        1. Encoder le langage (instructions + goal)
        2. Générer les subgoals avec CoT
        3. Utiliser les subgoals pour guider la génération d'actions bas niveau
        """
        # Encoder le langage (appel à la méthode parente)
        cont_lang, enc_lang = self.encode_lang(feat)
        
        # 🎯 GÉNÉRATION DES SUBGOALS (Chain-of-Thought)
        if self.use_subgoals and not self.test_mode:
            subgoals, subgoal_logits, subgoal_embeddings = self.generate_subgoals(
                enc_lang, cont_lang, 
                max_subgoals=self.max_subgoals
            )
            
            # Stocker dans feat pour utilisation ultérieure et calcul de loss
            feat['predicted_subgoals'] = subgoals
            feat['subgoal_logits'] = subgoal_logits
            feat['subgoal_embeddings'] = subgoal_embeddings
        
        # Appeler le forward du modèle parent pour générer les actions bas niveau
        # (Le parent va utiliser enc_lang et cont_lang)
        out = super().forward(feat, max_decode=max_decode)
        
        # Ajouter les subgoals à la sortie si disponibles
        if self.use_subgoals and not self.test_mode:
            out['predicted_subgoals'] = subgoals
            out['subgoal_logits'] = subgoal_logits
        
        return out
    
    
    def compute_subgoal_loss(self, predicted_logits, ground_truth_subgoals):
        """
        Calcule la loss pour la prédiction des subgoals
        
        Args:
            predicted_logits: Logits prédits [batch, num_subgoals, vocab_size]
            ground_truth_subgoals: Subgoals ground truth [batch, num_subgoals]
        
        Returns:
            loss: Cross-entropy loss pour les subgoals
        """
        device = next(self.parameters()).device
        
        # Reshape pour le calcul de cross-entropy
        pred_flat = predicted_logits.view(-1, predicted_logits.size(-1))  # [batch*num_subgoals, vocab_size]
        gt_flat = ground_truth_subgoals.view(-1)  # [batch*num_subgoals]
        
        # Masque pour ignorer le padding
        pad_mask = (gt_flat != self.pad)
        
        # Cross-entropy loss
        loss = F.cross_entropy(pred_flat, gt_flat, reduction='none')
        loss = loss * pad_mask.float()
        loss = loss.sum() / pad_mask.sum()
        
        return loss
    
    
    def compute_loss(self, out, batch, feat):
        """
        Calcule la loss totale incluant la loss des subgoals
        """
        # Loss du modèle parent (actions + masques)
        losses = super().compute_loss(out, batch, feat)
        
        # Ajouter la loss des subgoals si activée
        if self.use_subgoals and 'subgoal_logits' in out and 'action_high' in feat:
            subgoal_loss = self.compute_subgoal_loss(
                out['subgoal_logits'],
                feat['action_high']
            )
            
            # Poids pour la loss des subgoals
            subgoal_loss_weight = getattr(self.args, 'subgoal_loss_wt', 1.0)
            losses['subgoal'] = subgoal_loss * subgoal_loss_weight
        
        return losses


if __name__ == "__main__":
    print("✅ Subgoal CoT module créé avec succès!")
    print("📝 Fonctionnalités:")
    print("   - generate_subgoals(): Génération Chain-of-Thought des subgoals")
    print("   - Support du sampling et greedy decoding")
    print("   - Intégration avec le modèle baseline")