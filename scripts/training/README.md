🚀 Scripts d'Entraînement ALFRED

Scripts pour entraîner les modèles ALFRED Enhanced.

⭐ Fichiers Essentiels

Les deux fichiers les plus importants de ce dossier sont :

run_experiment.py - Script Python pour lancer les expériences


train.sh - Wrapper bash qui simplifie l'utilisation


Ces deux scripts gèrent automatiquement :

✅ Le chargement de l'environnement (.env)

✅ La création des dossiers d'expérience

✅ Les logs complets (TensorBoard, train.log, etc.)

✅ La sauvegarde des checkpoints

✅ La reprise d'entraînement (resume)

✅ Le arg parser complet


-------------------------
## Checkpoints disponibles

- **latest.pth** - Dernier checkpoint (pour reprendre)
- **best_seen.pth** - Meilleur sur validation seen
- **best_unseen.pth** - Meilleur sur validation unseen

-----------------------

## 🔄 Reprise d'Entraînement (Resume)

### Pourquoi reprendre ?

- 💾 Entraînement interrompu (panne, erreur, Ctrl+C)
- 🎯 Continuer avec plus d'epochs
- 🔧 Ajuster les hyperparamètres

### Comment faire

Ajoutez ces deux lignes à votre fichier YAML :

```yaml
resume: experiments/mon_exp_20251220_140532/checkpoints/latest.pth
dout: experiments/mon_exp_20251220_140532/checkpoints
```
--------------


## 🔧 Liste complète des paramètres

#### Paramètres obligatoires

| Paramètre | Type | Description |
|-----------|------|-------------|
| `exp_name` | string | Nom de l'expérience (pour le dossier) |
| `model` | string | Module Python du modèle (ex: `seq2seq_cot`) |
| `splits` | string | Chemin vers le fichier splits JSON |

#### Paramètres de base

| Paramètre | Type | Description | Défaut |
|-----------|------|-------------|--------|
| `data` | string | Chemin vers les données | `data/json_feat_2.1.0` |
| `batch` | int | Taille du batch | 8 |
| `epoch` | int | Nombre d'epochs | 20 |
| `lr` | float | Learning rate | 0.0001 |
| `seed` | int | Random seed | 1 |
| `decay_epoch` | int | Epoch pour decay du learning rate | - |

#### Reprise d'entraînement

| Paramètre | Type | Description | Défaut |
|-----------|------|-------------|--------|
| `resume` | string | Chemin vers checkpoint .pth pour reprendre | - |
| `dout` | string | Dossier de sortie (obligatoire avec resume) | - |

#### Options booléennes

| Paramètre | Type | Description | Défaut |
|-----------|------|-------------|--------|
| `gpu` | bool | Utiliser le GPU | false |
| `preprocess` | bool | Préprocesser le dataset (première fois) | false |
| `fast_epoch` | bool | Mode test rapide (sous-ensemble) | false |
| `save_every_epoch` | bool | Sauvegarder checkpoint à chaque epoch | false |
| `dec_teacher_forcing` | bool | Teacher forcing pour le decoder | false |

#### Architecture du réseau

| Paramètre | Type | Description | Défaut |
|-----------|------|-------------|--------|
| `dhid` | int | Dimension des états cachés | 512 |
| `demb` | int | Dimension des embeddings | 100 |
| `dframe` | int | Dimension des features visuelles | 2500 |
| `pframe` | int | Nombre de frames par pas | 300 |

#### Loss weights

| Paramètre | Type | Description | Défaut |
|-----------|------|-------------|--------|
| `action_loss_wt` | float | Poids de la loss action | 1.0 |
| `mask_loss_wt` | float | Poids de la loss mask | 1.0 |
| `subgoal_aux_loss_wt` | float | Poids de la loss subgoals (CoT) | 0.0 |
| `pm_aux_loss_wt` | float | Poids de la loss progress monitor | 0.0 |

#### Dropout

| Paramètre | Type | Description | Défaut |
|-----------|------|-------------|--------|
| `vis_dropout` | float | Dropout vision | 0.0 |
| `lang_dropout` | float | Dropout language | 0.0 |
| `input_dropout` | float | Dropout input | 0.0 |
| `hstate_dropout` | float | Dropout hidden state | 0.0 |
| `attn_dropout` | float | Dropout attention | 0.0 |
| `actor_dropout` | float | Dropout actor | 0.0 |
