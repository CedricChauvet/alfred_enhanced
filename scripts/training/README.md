# Training Scripts

Scripts pour entraîner les modèles ALFRED.

## run_experiment.py

Script principal pour lancer une expérience complète.

**Usage:**
```bash
python training/run_experiment.py --config ../configs/react/react_light_v1.yaml
```

**Ce qu'il fait:**
- Charge la configuration
- Crée le dossier d'expérience
- Lance l'entraînement
- Sauvegarde les logs et checkpoints

## train.sh

Script bash wrapper pour lancer plusieurs trainings.

**Usage:**
```bash
./training/train.sh
```
