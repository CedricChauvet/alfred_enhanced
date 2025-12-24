# ALFRED Enhanced

Projet basé sur [ALFRED](https://github.com/askforalfred/alfred) - Action Learning From Realistic Environments and Directives.

![ALFRED Teaser](https://github.com/askforalfred/alfred/raw/master/media/instr_teaser.png)

## À propos d'ALFRED (issu du README officiel)

ALFRED (Action Learning From Realistic Environments and Directives) est un nouveau benchmark pour l'apprentissage d'une correspondance entre les instructions en langage naturel et la vision égocentrique vers des séquences d'actions pour des tâches domestiques. Les longues compositions de déroulements avec des changements d'état non réversibles font partie des phénomènes que nous incluons pour réduire l'écart entre les benchmarks de recherche et les applications du monde réel.

---

## 📋 Configuration initiale



#### Création de l'environnement ALFRED

```bash
# Créer l'environnement avec Python 3.6 (requis pour ALFRED)
conda create -n alfred_env python=3.6

# Activer l'environnement
conda activate alfred_env

# Installer les dépendances requises
pip install -r requirements.txt
```

**Note :** ALFRED nécessite Python 3.6 pour assurer la compatibilité avec toutes les dépendances.

### Activation de l'environnement

```bash
conda activate alfred_env
```

### Chargement des chemins

**Voir le contenu du fichier** [`.env`](.env)
Editer le fichier .env a la ligne 7:

export ALFRED_ROOT="/my_path/alfred_enhanced"

Changer my_path par le repertoire actuel de votre repo.

```bash
cd /my_path/alfred_enhanced/
source .env
```

Vous devriez voir :
```
✓ ALFRED environment loaded
  ALFRED_ROOT: /media/cedrix/Ubuntu_2To/Alfred/alfred_enhanced
  PYTHONPATH: /media/cedrix/Ubuntu_2To/Alfred/alfred_enhanced:
```

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

[Documentation complète des scripts d'entraînement](https://github.com/CedricChauvet/alfred_enhanced/blob/main/scripts/training/README.md)

Utilisez le script `train.sh` avec le fichier de configuration YAML :

```bash
cd $ALFRED_EXP_ROOT
./scripts/train.sh ./config/baseline_reproduction.yaml
```

Les résultats du training sont stockés dans `$ALFRED_ROOT/experiments/`

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

Voir le README détaillé :[models/CoT](https://github.com/CedricChauvet/alfred_enhanced/tree/main/models/CoT)


---

## 📄 Licence

Ce projet est basé sur [ALFRED](https://github.com/askforalfred/alfred) qui est sous licence MIT.

### Licence MIT

Copyright (c) 2020 ALFRED Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📚 Citation


```bibtex
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

---


## 🎓 Lectures recommandées


### Article fondateur

**ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks**
- Auteurs : Mohit Shridhar, Jesse Thomason, Daniel Gordon, et al.
- Conférence : CVPR 2020
- [Paper](https://arxiv.org/abs/1912.01734) | [Site officiel](https://askforalfred.com)
