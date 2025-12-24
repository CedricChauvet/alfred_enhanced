🚀 Scripts d'Entraînement ALFRED
Scripts pour entraîner les modèles ALFRED Enhanced.

⭐ Fichiers Essentiels
Les deux fichiers les plus importants de ce dossier sont :

run_experiment.py - Script Python principal pour lancer les expériences
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
