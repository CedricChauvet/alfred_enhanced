import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.model.seq2seq_im_mask import Module as BaseModule
from models.nn import vnn
import numpy as np


class Module(BaseModule):
    """
    Modèle avec génération Chain-of-Thought des subgoals
    Hérite directement de seq2seq_im_mask.Module
    
    Fonctionnalité:
    - Génère les subgoals avant de générer les actions
    - Calcule la loss de prédiction des subgoals
    """
    
    def __init__(self, args, vocab):
        """
        Initialisation avec génération CoT des subgoals
        """
        super().__init__(args, vocab)
        
        # ========================================
        # GÉNÉRATION DES SUBGOALS (CoT)
        # ========================================
        
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
        
        # Paramètre pour activer/désactiver la génération de subgoals
        self.use_subgoals = getattr(args, 'use_subgoals', True)
        
        # Paramètre pour activer/désactiver la loss du current subgoal
        # Mettre à False si cette loss ne converge pas bien
        self.use_current_subgoal_loss = getattr(args, 'use_current_subgoal_loss', True)
        
        # ========================================
        # PRÉDICTION DU SUBGOAL ACTIF
        # ========================================
        # À chaque timestep, prédit quel subgoal (parmi ceux générés) est actif
        # Input: CONTEXTE COMPLET + SIGNAUX DU DÉCODEUR PARENT
        #        - h_t (dhid)
        #        - visual features (dframe)
        #        - action embedding (demb)
        #        - temporal progress (1) - calculé (t/max_t)
        #        - 🆕 progress_t (1) - sigmoid(self.progress(cont_t)) du décodeur parent
        #        - 🆕 subgoal_t (1) - sigmoid(self.subgoal(cont_t)) du décodeur parent
        # Output: distribution sur les subgoals générés
        
        # Calculer la dimension du contexte complet pour le prédicteur
        # +3 car: +1 progress calculé, +1 progress_t du parent, +1 subgoal_t du parent
        context_dim = 2*args.dhid + args.dframe + args.demb + 3
        
        # MLP plus profond pour mieux capturer les patterns temporels et multimodaux
        self.current_subgoal_scorer = nn.Sequential(
            nn.Linear(context_dim, args.dhid),
            nn.ReLU(),
            nn.Dropout(getattr(args, 'current_subgoal_dropout', 0.1)),
            nn.Linear(args.dhid, args.dhid),
            nn.ReLU(),
            nn.Linear(args.dhid, args.dhid)  # Output: dhid
        )
        self.current_subgoal_dropout = nn.Dropout(getattr(args, 'current_subgoal_dropout', 0.1))
        
        print(f"✅ Subgoal CoT decoder initialized with vocab size: {len(self.vocab_subgoal)}")
        print(f"✅ Current subgoal predictor initialized with:")
        print(f"   - Multi-layer MLP (3 layers)")
        print(f"   - FULL CONTEXT: h_t + visual + action")
        print(f"   - 🆕 PARENT DECODER SIGNALS: progress_t + subgoal_t")
        print(f"   - Input dim: {2*args.dhid + args.dframe + args.demb + 3}")
        print(f"     (context={2*args.dhid + args.dframe + args.demb}, +3 signals)")
    
    
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
    
    
    def predict_current_subgoal(self, context, subgoal_embeddings, timestep=None, progress=None, 
                                progress_signal=None, subgoal_signal=None):
        """
        Prédit quel subgoal (parmi ceux générés) est actuellement actif
        
        UTILISE LE CONTEXTE COMPLET + SIGNAUX DU BASELINE:
        - Contexte multimodal (h_t + visual + action)
        - Information temporelle (progress)
        - 🆕 self.progress du décodeur baseline (% tâche accomplie)
        - 🆕 self.subgoal du décodeur baseline (% subgoals complétés)
        
        Args:
            context: Contexte complet du décodeur [batch, dhid+dhid+dframe+demb]
            subgoal_embeddings: Embeddings des subgoals générés [batch, num_subgoals, demb_subgoal]
            timestep: Timestep actuel (optionnel)
            progress: Progression dans la séquence 0-1 (optionnel)
            progress_signal: Signal self.progress du baseline [batch, 1] (optionnel)
            subgoal_signal: Signal self.subgoal du baseline [batch, 1] (optionnel)
        
        Returns:
            current_subgoal_logits: Scores pour chaque subgoal [batch, num_subgoals]
            current_subgoal_idx: Indice du subgoal le plus probable [batch]
            current_subgoal_probs: Probabilités pour chaque subgoal [batch, num_subgoals]
        """
        device = next(self.parameters()).device
        batch_size = context.size(0)
        num_subgoals = subgoal_embeddings.size(1)
        
        # ========================================
        # 1. ENRICHIR L'INPUT AVEC TOUTES LES INFOS
        # ========================================
        context_input = context  # [batch, dhid+dhid+dframe+demb]
        
        # Liste des features additionnelles à concaténer
        additional_features = []
        
        # Progression temporelle (calculée)
        if progress is not None:
            if isinstance(progress, (int, float)):
                progress_tensor = torch.full((batch_size, 1), progress, device=device)
            else:
                progress_tensor = progress.view(batch_size, 1)
            additional_features.append(progress_tensor)
        
        # 🆕 Signal self.progress du baseline (prédiction du décodeur)
        if progress_signal is not None:
            # progress_signal est déjà [batch, 1]
            additional_features.append(progress_signal)
        
        # 🆕 Signal self.subgoal du baseline (proportion de subgoals complétés)
        if subgoal_signal is not None:
            # subgoal_signal est déjà [batch, 1]
            additional_features.append(subgoal_signal)
        
        # Concaténer toutes les features
        if len(additional_features) > 0:
            context_input = torch.cat([context] + additional_features, dim=-1)
        
        # ========================================
        # 2. PROJETER AVEC MLP PLUS PROFOND
        # ========================================
        # Utiliser le scorer (MLP 3-layers)
        h_proj = self.current_subgoal_scorer(context_input)  # [batch, dhid]
        h_proj = self.current_subgoal_dropout(h_proj)
        
        # ========================================
        # 3. PROJETER LES SUBGOALS DANS LE MÊME ESPACE
        # ========================================
        # Si demb_subgoal != dhid, on doit projeter
        if self.demb_subgoal != self.args.dhid:
            # Créer une projection si elle n'existe pas
            if not hasattr(self, 'subgoal_emb_to_dhid'):
                self.subgoal_emb_to_dhid = nn.Linear(self.demb_subgoal, self.args.dhid).to(device)
            subgoal_proj = self.subgoal_emb_to_dhid(subgoal_embeddings)  # [batch, num_subgoals, dhid]
        else:
            subgoal_proj = subgoal_embeddings
        
        # ========================================
        # 4. CALCULER LA SIMILARITÉ
        # ========================================
        # Produit scalaire pour mesurer la similarité
        # h_proj: [batch, dhid] -> [batch, 1, dhid]
        h_expanded = h_proj.unsqueeze(1)  # [batch, 1, dhid]
        
        # Similarity: [batch, 1, dhid] @ [batch, dhid, num_subgoals] -> [batch, 1, num_subgoals]
        similarity = torch.bmm(h_expanded, subgoal_proj.transpose(1, 2))
        current_subgoal_logits = similarity.squeeze(1)  # [batch, num_subgoals]
        
        # Softmax pour obtenir des probabilités
        current_subgoal_probs = F.softmax(current_subgoal_logits, dim=-1)  # [batch, num_subgoals]
        
        # Subgoal le plus probable
        current_subgoal_idx = current_subgoal_probs.argmax(dim=-1)  # [batch]
        
        return current_subgoal_logits, current_subgoal_idx, current_subgoal_probs
    
    
    
    def forward(self, feat, max_decode=300):
        """
        Forward pass avec génération de subgoals CoT et tracking du subgoal actif
        
        Processus:
        1. Encoder le langage (instructions + goal)
        2. Générer les subgoals avec CoT
        3. Générer les actions bas niveau
        4. À chaque timestep: prédire quel subgoal est actif
        """
        device = next(self.parameters()).device
        
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
        
        # ========================================
        # GÉNÉRATION DES ACTIONS avec tracking du subgoal actif
        # ========================================
        if self.use_subgoals and not self.test_mode and 'subgoal_embeddings' in feat:
            # Mode avec tracking: on doit décoder manuellement pour tracker
            # Initialiser les états
            e_t = self.dec.go.repeat(enc_lang.size(0), 1)
            state_t = cont_lang, torch.zeros_like(cont_lang)
            
            # Listes pour stocker les outputs
            outputs = []
            masks = []
            current_subgoal_predictions = []  # 🎯 NOUVEAU: track des subgoals actifs
            subgoal_monitoring = []  # Pour self.subgoal
            progress_monitoring = []  # Pour self.progress
            
            # Séquence d'actions ground truth pour teacher forcing
            actions = feat['action_low'] if self.dec.teacher_forcing and 'action_low' in feat else None
            
            # Boucle de décodage
            max_t = actions.size(1) if actions is not None else min(max_decode, feat['frames'].shape[1])
            
            for t in range(max_t):
                # Frames à ce timestep
                frames_t = feat['frames'][:, t] if t < feat['frames'].size(1) else feat['frames'][:, -1]
                
                # Décoder une action
                out_t, mask_t, state_t, *extra = self.dec.step(enc_lang, frames_t, e_t=e_t, state_tm1=state_t)
                
                outputs.append(out_t)
                masks.append(mask_t)
                
                # 🎯 RÉCUPÉRER LES SIGNAUX DU DÉCODEUR PARENT
                # extra = [lang_attn_t, subgoal_t, progress_t]
                # subgoal_t et progress_t sont déjà calculés par le décodeur parent:
                #   progress_t = sigmoid(self.progress(cont_t))
                #   subgoal_t = sigmoid(self.subgoal(cont_t))
                progress_t = None
                subgoal_t = None
                if self.subgoal_monitoring and len(extra) >= 3:
                    subgoal_t = extra[1]  # sigmoid(self.subgoal(cont_t)) du parent
                    progress_t = extra[2]  # sigmoid(self.progress(cont_t)) du parent
                    subgoal_monitoring.append(subgoal_t)
                    progress_monitoring.append(progress_t)
                
                # 🎯 PRÉDIRE LE SUBGOAL ACTIF à ce timestep
                h_t, c_t = state_t
                
                # 🔍 ENCODER LES FEATURES VISUELLES
                # frames_t est [batch, channels, height, width] - il faut l'encoder !
                vis_feat_t = self.dec.vis_encoder(frames_t)  # [batch, dframe]
                
                # Créer le contexte complet (comme dans ConvFrameMaskDecoderProgressMonitor)
                # cont_t contient: h_t + visual_features + action_embedding
                cont_t = torch.cat([h_t, vis_feat_t, e_t], dim=1)  
                # Dimension: [batch, 2*dhid + dframe + demb] = [batch, 256+2048+100] = [batch, 2404]
                
                # Calculer la progression temporelle (calculée naïvement)
                temporal_progress = t / max_t  # Entre 0 et 1
                
                current_logits, current_idx, current_probs = self.predict_current_subgoal(
                    cont_t,  # Contexte complet
                    feat['subgoal_embeddings'],
                    timestep=t,
                    progress=temporal_progress,     # Calculé (t/max_t)
                    progress_signal=progress_t,     # 🆕 Du décodeur parent (déjà entraîné!)
                    subgoal_signal=subgoal_t        # 🆕 Du décodeur parent (déjà entraîné!)
                )
                current_subgoal_predictions.append(current_logits)
                
                # 🔍 LOGGING OCCASIONNEL (pour debug)
                # Afficher quelques prédictions pour vérifier que ça a du sens
                if self.training and t % 20 == 0 and torch.rand(1).item() < 0.005:  # 0.5% des cas
                    # FIXED: Format strings corrects (pas de :.2f dans if/else)
                    prog_val = progress_t[0].item() if progress_t is not None else None
                    subg_val = subgoal_t[0].item() if subgoal_t is not None else None
                    prog_str = f"{prog_val:.2f}" if prog_val is not None else "N/A"
                    subg_str = f"{subg_val:.2f}" if subg_val is not None else "N/A"
                    print(f"🔍 [t={t:3d}] Predicted subgoal: {current_idx[0].item()} "
                          f"probs={current_probs[0].tolist()} "
                          f"progress_t={prog_str} subgoal_t={subg_str}")
                
                # 🎨 CRÉER LE CONTEXTE DU SUBGOAL ACTIF (Hard Selection)
                # Sélectionner directement l'embedding du subgoal prédit (au lieu de moyenne pondérée)
                batch_indices = torch.arange(feat['subgoal_embeddings'].size(0), device=device)
                subgoal_ctx = feat['subgoal_embeddings'][batch_indices, current_idx]  # [batch, demb_subgoal]
                # Pour l'instant on le calcule juste, on l'injectera dans le décodeur plus tard
                
                # Teacher forcing ou utiliser la prédiction
                if self.dec.teacher_forcing and actions is not None and t < actions.size(1):
                    e_t = self.dec.emb(actions[:, t])
                else:
                    e_t = self.dec.emb(out_t.max(1)[1])
                
                # Arrêter si toutes les séquences ont généré stop
                if not self.dec.teacher_forcing:
                    predictions = out_t.max(1)[1]
                    if (predictions == self.stop_token).all():
                        break
            
            # Préparer les outputs
            out = {}
            out['out_action_low'] = torch.stack(outputs, dim=1)  # [batch, seq_len, vocab_size]
            out['out_action_low_mask'] = torch.stack(masks, dim=1)  # [batch, seq_len, h, w]
            out['predicted_subgoals'] = subgoals
            out['subgoal_logits'] = subgoal_logits
            out['current_subgoal_logits'] = torch.stack(current_subgoal_predictions, dim=1)  # [batch, seq_len, num_subgoals]
            
            # ✅ IMPORTANT: Copier vers feat pour extract_preds
            feat['out_action_low'] = out['out_action_low']
            feat['out_action_low_mask'] = out['out_action_low_mask']
            
            # Ajouter les outputs de monitoring si activé
            if self.subgoal_monitoring and len(subgoal_monitoring) > 0:
                out['out_subgoal'] = torch.stack(subgoal_monitoring, dim=1)
                out['out_progress'] = torch.stack(progress_monitoring, dim=1)
                feat['out_subgoal'] = out['out_subgoal']
                feat['out_progress'] = out['out_progress']
        
        else:
            # Mode sans tracking: appeler simplement le parent
            out = super().forward(feat, max_decode=max_decode)
            
            # Ajouter les subgoals à la sortie si disponibles
            if self.use_subgoals and not self.test_mode:
                out['predicted_subgoals'] = subgoals
                out['subgoal_logits'] = subgoal_logits
        
        return out
    
    
    def compute_loss(self, out, batch, feat):
        """
        Calcule la loss totale incluant:
        - Loss des actions (baseline)
        - Loss des masques (baseline)  
        - Loss des subgoals (CoT)
        - Loss du current subgoal (tracking)
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
        
        # 🎯 NOUVEAU: Loss pour la prédiction du subgoal actif
        if (self.use_subgoals and self.use_current_subgoal_loss and 
            'current_subgoal_logits' in out and 'low_to_high_idx' in feat):
            current_subgoal_loss = self.compute_current_subgoal_loss(
                out['current_subgoal_logits'],
                feat['low_to_high_idx']
            )
            
            # Poids pour la loss du current subgoal (réduit pour éviter de dominer)
            current_subgoal_loss_weight = getattr(self.args, 'current_subgoal_loss_wt', 0.1)  # 0.1 au lieu de 0.5
            losses['current_subgoal'] = current_subgoal_loss * current_subgoal_loss_weight
            
            # 📊 NOUVEAU: Calculer l'accuracy comme métrique (pas de gradient)
            with torch.no_grad():
                # Prédictions: argmax des logits
                pred_idx = out['current_subgoal_logits'].argmax(dim=-1)  # [batch, seq_len]
                gt_idx = feat['low_to_high_idx']  # [batch, seq_len]
                
                # Masque pour ignorer le padding
                valid_mask = (gt_idx != self.pad)
                
                # Accuracy
                correct = (pred_idx == gt_idx) & valid_mask
                accuracy = correct.sum().float() / (valid_mask.sum().float() + 1e-8)
                
                # ENHANCED: Stocker comme tensor (cohérent avec les autres losses)
                # Note: seq2seq.py appelle .item() sur toutes les valeurs, donc on garde le tensor
                losses['current_subgoal_accuracy'] = accuracy  # Tensor, pas .item()
        
        return losses
    
    
    def compute_current_subgoal_loss(self, predicted_logits, ground_truth_idx):
        """
        Loss pour la prédiction du subgoal actif à chaque timestep
        
        Args:
            predicted_logits: Logits pour chaque subgoal [batch, seq_len, num_subgoals]
            ground_truth_idx: Indice du subgoal qui devrait être actif [batch, seq_len]
        
        Returns:
            loss: Cross-entropy loss
        """
        # Reshape pour le calcul de cross-entropy
        batch_size, seq_len, num_subgoals = predicted_logits.size()
        pred_flat = predicted_logits.view(-1, num_subgoals)  # [batch*seq_len, num_subgoals]
        gt_flat = ground_truth_idx.view(-1)  # [batch*seq_len]
        
        # Masque pour ignorer le padding
        pad_mask = (gt_flat != self.pad)
        
        # Cross-entropy loss
        loss = F.cross_entropy(pred_flat, gt_flat, reduction='none')
        loss = loss * pad_mask.float()
        loss = loss.sum() / (pad_mask.sum() + 1e-8)
        
        return loss
    
    
    def featurize(self, batch, load_mask=True, load_frames=True):
        """
        Tensorize batch avec ajout de low_to_high_idx pour la loss du current subgoal
        """
        # Appeler la méthode parente
        feat = super().featurize(batch, load_mask=load_mask, load_frames=load_frames)
        
        # Ajouter l'alignement action -> subgoal pour la loss
        if not self.test_mode and self.use_subgoals:
            device = next(self.parameters()).device
            low_to_high_indices = []
            
            for ex in batch:
                if 'low_to_high_idx' in ex['num']:
                    # Déjà disponible dans les données
                    low_to_high_indices.append(ex['num']['low_to_high_idx'])
                else:
                    # Créer un mapping par défaut (chaque action appartient au subgoal 0)
                    num_actions = len(ex['num']['action_low'])
                    low_to_high_indices.append([0] * num_actions)
            
            # Tensorize et pad
            seqs = [torch.tensor(indices, device=device, dtype=torch.long) for indices in low_to_high_indices]
            feat['low_to_high_idx'] = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
        
        return feat



if __name__ == "__main__":
    print("✅ CoT Subgoals module créé avec succès!")
    print("📝 Fonctionnalités:")
    print("   - generate_subgoals(): Génération Chain-of-Thought des subgoals")
    print("   - compute_subgoal_loss(): Loss de prédiction des subgoals")
    print("   - Intégration avec le modèle baseline seq2seq_im_mask")