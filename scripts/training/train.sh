#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# Activer environnement
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alfred_env

# Vérifier argument
if [ -z "$1" ]; then
    echo "Usage: ./scripts/training/train.sh <config.yaml>"
    echo ""
    echo "Available configs:"
    ls -1 configs/*.yaml
    exit 1
fi

CONFIG=$1

echo "=================================="
echo "ALFRED Experiment Runner"
echo "=================================="
echo "Config: $CONFIG"
echo "=================================="

# Lancer
python scripts/training/run_experiment.py --config "$CONFIG"

echo ""