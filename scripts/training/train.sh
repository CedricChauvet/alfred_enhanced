#!/bin/bash
# ============================================================================
# ALFRED Training Script
# Charge automatiquement l'environnement depuis .env
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Détecter le répertoire racine du projet
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================================================
# Charger les variables d'environnement depuis .env
# ============================================================================
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "✓ Loading environment from: $PROJECT_ROOT/.env"
    source "$PROJECT_ROOT/.env"
else
    echo "⚠️  Warning: .env file not found at $PROJECT_ROOT/.env"
    echo "   Using default paths..."
    export ALFRED_ROOT="$PROJECT_ROOT"
fi

# ============================================================================
# Vérifier/Activer l'environnement conda
# ============================================================================
if [ -n "$ALFRED_CONDA_ENV" ]; then
    echo "✓ Activating conda environment: $ALFRED_CONDA_ENV"
    
    # Initialiser conda si nécessaire
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    
    conda activate "$ALFRED_CONDA_ENV" 2>/dev/null || {
        echo "⚠️  Could not activate conda environment: $ALFRED_CONDA_ENV"
        echo "   Make sure it exists: conda env list"
    }
else
    echo "⚠️  ALFRED_CONDA_ENV not set, using current Python environment"
fi

# ============================================================================
# Vérifier l'argument config
# ============================================================================
if [ -z "$1" ]; then
    echo ""
    echo "=================================="
    echo "ALFRED Experiment Runner"
    echo "=================================="
    echo ""
    echo "Usage: ./scripts/training/train.sh <config.yaml>"
    echo ""
    echo "Available configs:"
    ls -1 "$PROJECT_ROOT/configs/"*.yaml 2>/dev/null || echo "  No configs found in $PROJECT_ROOT/configs/"
    echo ""
    exit 1
fi

CONFIG=$1

# Résoudre le chemin de la config (absolu ou relatif)
if [ ! -f "$CONFIG" ]; then
    # Si le chemin n'existe pas tel quel, essayer depuis PROJECT_ROOT
    if [ -f "$PROJECT_ROOT/$CONFIG" ]; then
        CONFIG="$PROJECT_ROOT/$CONFIG"
    else
        echo "❌ Error: Config file not found: $CONFIG"
        exit 1
    fi
fi

echo ""
echo "=================================="
echo "ALFRED Experiment Runner"
echo "=================================="
echo "Config: $CONFIG"
echo "ALFRED_ROOT: $ALFRED_ROOT"
echo "Experiments dir: $ALFRED_ROOT/experiments"
echo "=================================="
echo ""

# ============================================================================
# Lancer l'expérience
# ============================================================================
cd "$PROJECT_ROOT"
python scripts/training/run_experiment.py --config "$CONFIG"

echo ""