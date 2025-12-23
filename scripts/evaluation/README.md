# Evaluation Scripts

Scripts pour évaluer les modèles entraînés.

## eval_best.sh

Évalue le meilleur checkpoint d'une expérience.

**Usage:**
```bash
./evaluation/eval_best.sh <exp_name>
```

## eval_with_env.sh

Évalue un modèle avec l'environnement AI2-THOR.

**Usage:**
```bash
./evaluation/eval_with_env.sh <checkpoint_path> <split>
```

**Exemple:**
```bash
./evaluation/eval_with_env.sh \
    ../experiments/react/react_light_v1_*/checkpoints/best_seen.pth \
    valid_seen
```
