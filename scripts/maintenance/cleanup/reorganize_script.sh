#!/bin/bash

# Réorganisation du dossier scripts/
# Structure claire : training/ evaluation/ analysis/ maintenance/

set -e

SCRIPTS_DIR="$HOME/Bureau/Alfred/alfred_experiments/scripts"

echo "════════════════════════════════════════════════════════════════════"
echo "  Réorganisation de scripts/"
echo "════════════════════════════════════════════════════════════════════"
echo ""

cd "$SCRIPTS_DIR"

# ════════════════════════════════════════════════════════════════════
# 1. Créer la nouvelle structure
# ════════════════════════════════════════════════════════════════════

echo "📦 Création de la structure..."

mkdir -p training/
mkdir -p evaluation/
mkdir -p analysis/
mkdir -p maintenance/{patches,cleanup}
mkdir -p utils/

echo "✓ Structure créée"
echo ""

# ════════════════════════════════════════════════════════════════════
# 2. Organiser les scripts de TRAINING
# ════════════════════════════════════════════════════════════════════

echo "🎓 Organisation des scripts de training..."

# Script principal d'entraînement
if [ -f "run_experiment.py" ]; then
    mv run_experiment.py training/
    echo "  ✓ run_experiment.py → training/"
fi

if [ -f "train.sh" ]; then
    mv train.sh training/
    echo "  ✓ train.sh → training/"
fi

# Créer un wrapper si besoin
cat > training/README.md << 'EOF'
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
EOF

echo ""

# ════════════════════════════════════════════════════════════════════
# 3. Organiser les scripts d'EVALUATION
# ════════════════════════════════════════════════════════════════════

echo "📊 Organisation des scripts d'évaluation..."

if [ -f "eval_best.sh" ]; then
    mv eval_best.sh evaluation/
    echo "  ✓ eval_best.sh → evaluation/"
fi

if [ -f "eval_with_env.sh" ]; then
    mv eval_with_env.sh evaluation/
    echo "  ✓ eval_with_env.sh → evaluation/"
fi

# Créer README
cat > evaluation/README.md << 'EOF'
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
EOF

echo ""

# ════════════════════════════════════════════════════════════════════
# 4. Organiser les scripts d'ANALYSE
# ════════════════════════════════════════════════════════════════════

echo "🔍 Organisation des scripts d'analyse..."

# Dossier analysis/ existant
if [ -d "analysis" ]; then
    # Déplacer le contenu
    if [ -f "analysis/analyze.sh" ]; then
        mv analysis/analyze.sh analysis_old.sh
        echo "  ✓ analysis/analyze.sh → analysis_old.sh (temporaire)"
    fi
    
    # Supprimer l'ancien dossier s'il est vide
    rmdir analysis 2>/dev/null || true
fi

# Créer le nouveau dossier
mkdir -p analysis/

# Déplacer les scripts
if [ -f "compare.sh" ]; then
    mv compare.sh analysis/
    echo "  ✓ compare.sh → analysis/"
fi

if [ -f "analysis_old.sh" ]; then
    mv analysis_old.sh analysis/analyze.sh
    echo "  ✓ analyze.sh → analysis/"
fi

# README
cat > analysis/README.md << 'EOF'
# Analysis Scripts

Scripts pour analyser les résultats d'expériences.

## compare.sh

Compare deux expériences (CoT vs ReAct).

**Usage:**
```bash
./analysis/compare.sh <exp_cot> <exp_react>
```

## analyze.sh

Analyse détaillée d'une expérience.

**Usage:**
```bash
./analysis/analyze.sh <exp_name>
```

**Note:** Ces scripts utilisent les outils Python dans `tools/`:
- `compare_cot_react.py`
- `analyze_failures.py`
- `visualize_trajectory.py`
EOF

echo ""

# ════════════════════════════════════════════════════════════════════
# 5. Organiser les scripts de MAINTENANCE
# ════════════════════════════════════════════════════════════════════

echo "🔧 Organisation des scripts de maintenance..."

# Patches
if [ -d "patchs" ]; then
    # Note: "patchs" avec faute d'orthographe
    mv patchs/* maintenance/patches/ 2>/dev/null || true
    rmdir patchs 2>/dev/null || true
    echo "  ✓ patchs/* → maintenance/patches/"
fi

# Cleanup
if [ -f "clean_failed.sh" ]; then
    mv clean_failed.sh maintenance/cleanup/
    echo "  ✓ clean_failed.sh → maintenance/cleanup/"
fi

# README patches
cat > maintenance/patches/README.md << 'EOF'
# Patches

Patches pour corriger des bugs dans le code ALFRED original.

## patch_alfred_baseline.sh

Patch le baseline ALFRED pour compatibilité.

## patch_baseline_gpu.sh

Corrige les problèmes GPU dans le baseline.

## patch_compute_loss_gpu.sh

Corrige le calcul de loss sur GPU.

## patch_eval_imports.sh

Corrige les imports dans eval_seq2seq.py.

**Usage:**
```bash
cd ~/Bureau/Alfred/alfred
../alfred_experiments/scripts/maintenance/patches/patch_baseline_gpu.sh
```

**⚠️ IMPORTANT:** 
Appliquer ces patches après avoir cloné ALFRED et avant le premier training.
EOF

# README cleanup
cat > maintenance/cleanup/README.md << 'EOF'
# Cleanup Scripts

Scripts pour nettoyer les expériences échouées.

## clean_failed.sh

Supprime les expériences avec status FAILED.

**Usage:**
```bash
./maintenance/cleanup/clean_failed.sh
```

**⚠️ Attention:** Crée un backup avant de supprimer.
EOF

echo ""

# ════════════════════════════════════════════════════════════════════
# 6. Organiser les scripts UTILITAIRES
# ════════════════════════════════════════════════════════════════════

echo "🛠️  Organisation des utilitaires..."

if [ -f "list_experiments.sh" ]; then
    mv list_experiments.sh utils/
    echo "  ✓ list_experiments.sh → utils/"
fi

# Créer des utilitaires manquants
cat > utils/check_status.sh << 'EOF'
#!/bin/bash
# Vérifie le status de toutes les expériences

EXPERIMENTS_DIR="../experiments"

echo "════════════════════════════════════════════════════════════════════"
echo "  Status des Expériences"
echo "════════════════════════════════════════════════════════════════════"
echo ""

for exp_dir in "$EXPERIMENTS_DIR"/{cot,react}/*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        status_file="$exp_dir/status.txt"
        
        if [ -f "$status_file" ]; then
            status=$(cat "$status_file")
            if [ "$status" = "SUCCESS" ]; then
                echo "✓ $exp_name"
            else
                echo "✗ $exp_name ($status)"
            fi
        else
            echo "? $exp_name (no status)"
        fi
    fi
done

echo ""
EOF

chmod +x utils/check_status.sh
echo "  ✓ check_status.sh créé → utils/"

# README utils
cat > utils/README.md << 'EOF'
# Utilities

Scripts utilitaires pour la gestion du projet.

## list_experiments.sh

Liste toutes les expériences disponibles.

**Usage:**
```bash
./utils/list_experiments.sh
```

## check_status.sh

Vérifie le status (SUCCESS/FAILED) de toutes les expériences.

**Usage:**
```bash
./utils/check_status.sh
```
EOF

echo ""

# ════════════════════════════════════════════════════════════════════
# 7. Créer un README principal
# ════════════════════════════════════════════════════════════════════

echo "📝 Création du README principal..."

cat > README.md << 'EOF'
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
EOF

echo "✓ README.md créé"
echo ""

# ════════════════════════════════════════════════════════════════════
# 8. Rendre les scripts exécutables
# ════════════════════════════════════════════════════════════════════

echo "🔑 Configuration des permissions..."

find training/ evaluation/ analysis/ maintenance/ utils/ -name "*.sh" -exec chmod +x {} \;

echo "✓ Scripts rendus exécutables"
echo ""

# ════════════════════════════════════════════════════════════════════
# 9. Afficher le résumé
# ════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ Réorganisation Terminée!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "📂 Nouvelle structure:"
echo ""
tree -L 2 -F --dirsfirst 2>/dev/null || find . -maxdepth 2 -type d | sort
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  📖 Documentation"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "README créés dans:"
echo "  • scripts/README.md"
echo "  • training/README.md"
echo "  • evaluation/README.md"
echo "  • analysis/README.md"
echo "  • maintenance/patches/README.md"
echo "  • maintenance/cleanup/README.md"
echo "  • utils/README.md"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  🚀 Prochaines Étapes"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Vérifier que le training en cours fonctionne toujours"
echo "2. Mettre à jour vos commandes selon la nouvelle structure:"
echo ""
echo "   Avant: python run_experiment.py --config ..."
echo "   Après: python training/run_experiment.py --config ..."
echo ""
echo "3. Lire les README pour comprendre chaque dossier:"
echo "   cat training/README.md"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""