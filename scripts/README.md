# Scripts

Scripts d'orchestration pour ALFRED experiments.

## 📁 Structure

```
scripts/
├── training/          # Entraînement de modèles
│   ├── run_experiment.py
│   └── train.sh
├── evaluation/        # Évaluation de modèles
│   ├── eval_best.sh
│   └── eval_with_env.sh
├── analysis/          # Analyse de résultats
│   ├── compare.sh
│   └── analyze.sh
├── maintenance/       # Maintenance du projet
│   ├── patches/       # Patches pour ALFRED
│   └── cleanup/       # Nettoyage
└── utils/             # Utilitaires
    ├── list_experiments.sh
    └── check_status.sh
```

## 🚀 Workflows Typiques

### 1. Entraîner un Modèle

```bash
# Avec Python (recommandé)
python training/run_experiment.py --config ../configs/react/react_light_v1.yaml

# Avec bash wrapper
./training/train.sh
```

### 2. Évaluer un Modèle

```bash
# Évaluer le meilleur checkpoint
./evaluation/eval_best.sh react_light_v1_20241209_150000

# Évaluer avec environnement complet
./evaluation/eval_with_env.sh \
    ../experiments/react/*/checkpoints/best_seen.pth \
    valid_seen
```

### 3. Analyser les Résultats

```bash
# Comparer deux expériences
./analysis/compare.sh \
    ../experiments/cot/test_quick_gpu_* \
    ../experiments/react/react_light_v1_*

# Analyse détaillée
./analysis/analyze.sh react_light_v1_20241209_150000
```

### 4. Maintenance

```bash
# Appliquer patches ALFRED
./maintenance/patches/patch_baseline_gpu.sh

# Nettoyer expériences échouées
./maintenance/cleanup/clean_failed.sh

# Vérifier status
./utils/check_status.sh
```

## 🔗 Relation avec tools/

Les scripts dans `scripts/` orchestrent des workflows complets.
Ils utilisent les outils dans `tools/` comme composants.

**Exemple:**
```bash
# scripts/analysis/compare.sh appelle:
python ../tools/analysis/compare_cot_react.py
```

## 📚 Documentation

Chaque sous-dossier contient un README.md détaillé.
