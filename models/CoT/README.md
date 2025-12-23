


**Auteur**: Cedrix  
**Date**: 2025  
**Base**: ALFRED Benchmark




## 📊 Expériences


### CoT v1
- Config: `configs/cot_v1.yaml`
- Modèle: `alfred_experiments.models.seq2seq_cot`
- Ajouts: Génération explicite de subgoals
- Objectif: Améliorer planning et décomposition

### CoT v2
- Config: `configs/cot_v2_exploration.yaml`
- Variations: Plus de subgoals, loss weight plus élevé
- Objectif: Explorer hyperparamètres CoT

## 📝 Notes

- **IMPORTANT**: Le dossier `alfred/` n'est JAMAIS modifié
- Tous les modèles héritent des classes originales
- Résultats trackés dans `experiments/`
- Chaque expérience a son dossier avec timestamp

## 🔧 Développement
```bash
# Lancer
./scripts/train.sh configs/cot_v1.yaml.yaml
```

## 📈 Résultats Attendus

**Baseline**: SR ~3.5-4.5%, GC ~9-12%

**CoT v1**: SR ~5-7% (+30-50%), GC ~12-16%

---


╔═══════════════════════════════════════════════════════════════════════════╗
║                    LISTE COMPLÈTE DES LAYERS CoT                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

1. self.project_cont
   ══════════════════
   Type : nn.Linear(2*dhid, dhid)
   Ligne : 55
   Dimensions : 256 → 128 (si dhid=128)
   Rôle : Projeter cont_lang pour initialiser subgoal_decoder
   Paramètres : 256*128 + 128 = 32,896


2. self.project_subgoals
   ══════════════════════
   Type : nn.Linear(dhid, 2*dhid)
   Ligne : 56
   Dimensions : 128 → 256 (si dhid=128)
   Rôle : Projeter subgoals_hidden pour concaténer avec enc_lang
   Paramètres : 128*256 + 256 = 33,024


3. self.subgoal_decoder
   ═════════════════════
   Type : nn.LSTM(demb + dhid, dhid, batch_first=True)
   Lignes : 58-62
   Input : 228 (100 + 128, si demb=100, dhid=128)
   Hidden : 128
   Rôle : Générer les subgoals de manière auto-régressive
   Paramètres : 4 * ((228+128)*128 + 128) = ~182,784
   
   Détails LSTM :
   - Input gate
   - Forget gate  
   - Cell gate
   - Output gate
   Chacun a : (input_size + hidden_size) * hidden_size + hidden_size


4. self.subgoal_classifier
   ════════════════════════
   Type : nn.Linear(dhid, len(vocab['action_high']))
   Lignes : 64-67
   Dimensions : 128 → ~15 (si vocab_high a 15 actions)
   Rôle : Classifier la sortie du LSTM en subgoal
   Paramètres : 128*15 + 15 = 1,935


5. self.emb_subgoal
   ═════════════════
   Type : nn.Embedding(len(vocab['action_high']), demb)
   Lignes : 69-72
   Dimensions : 15 → 100 (si vocab_high=15, demb=100)
   Rôle : Embedder les indices de subgoals en vecteurs
   Paramètres : 15*100 = 1,500
"""




╔═══════════════════════════════════════════════════════════════════════════╗
║                    FLUX COMPLET AVEC COULEURS                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

Instructions "Put heated apple in fridge"
    │
    ↓
⚫ emb_word(tokens)
    │
    ↓
⚫ enc (LSTM bidirectional)
    │
    ↓
⚫ enc_att (self-attention)
    │
    ├────────────┬────────────────┐
    ↓            ↓                │
cont_lang    enc_lang             │
(256)        (seq,256)            │
    │            │                │
    ↓            │                │
🟢 project_cont  │                │
    │            │                │
    ↓            │                │
cont_lang_proj   │                │
(128)            │                │
    │            │                │
    ↓            │                │
Initialize       │                │
LSTM state       │                │
(h_0, c_0)       │                │
    │            │                │
    ↓            │                │
Loop t=0..9:     │                │
  ↓              │                │
🟢 emb_subgoal   │                │
  ↓              │                │
concat(emb,cont) │                │
  ↓              │                │
🟢 subgoal_decoder (LSTM)         │
  ↓              │                │
🟢 subgoal_classifier             │
  ↓              │                │
subgoal_t        │                │
    │            │                │
    ↓            │                │
subgoals_hidden  │                │
(10,128)         │                │
    │            │                │
    ↓            │                │
🟢 project_subgoals               │
    │            │                │
    ↓            │                │
subgoals_proj    │                │
(10,256)         │                │
    │            │                │
    └────────┬───┴────────────────┘
             ↓
    concat([enc_lang, subgoals_proj])
             ↓
    enc_lang_enhanced
    (seq+10, 256)
             │
             ├───────────┐
             ↓           ↓
    ⚫ dec (decoder) ← frames
             ↓
    ┌────────┴────────┐
    ↓                 ↓
actions            masks
(low-level)    (interaction)


🟢 = Nouveau (CoT)  : 5 layers, 252K params
⚫ = Hérité (Baseline) : ~10M params
