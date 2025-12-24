# ALFRED Enhanced

Projet basé sur [ALFRED](https://github.com/askforalfred/alfred) - Action Learning From Realistic Environments and Directives.

**Repository:** https://github.com/CedricChauvet/alfred_enhanced/

---

## 📋 Configuration initiale

### Activation de l'environnement

```bash
conda activate alfred_env
```

### Chargement des chemins

**Voir le contenu du fichier** [`.env`](.env)

```bash
cd /my_path/alfred_enhanced/
source .env
```

Vous devriez voir :
```
✓ ALFRED environment loaded
  ALFRED_ROOT: /media/cedrix/Ubuntu_2To/Alfred/alfred
  ALFRED_EXP_ROOT: /media/cedrix/Ubuntu_2To/Alfred/alfred_experiments
  PYTHONPATH: /media/cedrix/Ubuntu_2To/Alfred/alfred:/media/cedrix/Ubuntu_2To/Alfred/alfred_experiments:
```

**Important :** 
- Travaillez exclusivement dans `ALFRED_EXP_ROOT`
- `ALFRED_ROOT` doit être une copie exacte du repo GitHub

---

## 📦 Téléchargement du dataset

Deux options disponibles :

```bash
# Option 1 : Dataset JSON (léger, sans images RGB)
sh download_data.sh json_feat

# Option 2 : Dataset complet (avec images RGB)
sh download_data.sh full
```

**Note :** Le dataset `json_feat` est suffisant tant que vous n'entraînez pas la partie visuelle. Des problèmes ont été rencontrés avec `full`.

---

## ⚙️ Preprocessing

Lors du premier lancement d'un script, il faut préprocesser le dataset :

```bash
--preprocess
```

Cette étape complète le dataset avec un dossier `pp` pour chaque trajectoire, contenant des fichiers `ann_*.json`.

---

## 🎯 Modèle Baseline

### Téléchargement du modèle pré-entraîné

```bash
wget https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/seq2seq_pm_chkpt.zip
```

Ce modèle permet de vérifier que les scripts d'évaluation fonctionnent correctement.

### Évaluation du modèle téléchargé

```bash
python models/eval/eval_seq2seq.py \
  --model_path /media/cedrix/Ubuntu_2To/Alfred/alfred_experiments/experiments/Baseline/best_seen.pth \
  --eval_split valid_seen \
  --data data/json_feat_2.1.0 \
  --model models.model.seq2seq_im_mask \
  --gpu \
  --num_threads 2 \
  --preprocess  # (à exécuter une seule fois)
```

**Résultats du modèle téléchargé :**
```
SR: 8/820 = 0.010
GC: 140/2109 = 0.066
PLW SR: 0.003
PLW GC: 0.038
```

---

## 🚀 Entraînement du Baseline

### Commande d'entraînement

Utilisez le script `train.sh` avec le fichier de configuration YAML :

```bash
cd $ALFRED_EXP_ROOT
./scripts/train.sh ./config/baseline_reproduction.yaml
```

Les résultats du training sont stockés dans `$ALFRED_EXP_ROOT/experiments/`

**Résultats après entraînement :**
```
SR: 19/820 = 0.023
GC: 194/2109 = 0.092
PLW SR: 0.018
PLW GC: 0.073
```

---

## 💡 Améliorations du modèle

### Chain of Thoughts (CoT)

Voir le README détaillé : `models/model/CoT/README.md`

---

## 📊 Métriques

- **SR** : Success Rate (Taux de réussite)
- **GC** : Goal Condition (Conditions d'objectif atteintes)
- **PLW SR** : Path Length Weighted Success Rate
- **PLW GC** : Path Length Weighted Goal Condition