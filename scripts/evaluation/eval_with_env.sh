#!/bin/bash
# Wrapper pour évaluation avec ALFRED_ROOT

# Définir ALFRED_ROOT
export ALFRED_ROOT=/home/cedrix/Bureau/Alfred/alfred

# Arguments
checkpoint=${1:-/home/cedrix/Bureau/Alfred/alfred_experiments/experiments/test_quick_gpu_20251209_110322/best_unseen.pth}
split=${2:-valid_unseen}
mode=${3:-full}

if [ ! -f "$checkpoint" ]; then
    echo "✗ Checkpoint non trouvé: $checkpoint"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ÉVALUATION ALFRED                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "💾 Checkpoint:  $(basename $checkpoint)"
echo "📊 Split:       $split"
echo "🔧 Mode:        $mode"
echo "📁 ALFRED_ROOT: $ALFRED_ROOT"
echo ""

cd "$ALFRED_ROOT"

if [ "$mode" = "subgoals" ]; then
    echo "🎯 Évaluation subgoals (CoT)"
    echo ""
    
    python models/eval/eval_subgoals.py \
        --model_path "$checkpoint" \
        --eval_split "$split" \
        --data data/json_feat_2.1.0 \
        --model models.model.seq2seq_cot \
        --gpu
else
    echo "🎯 Évaluation complète"
    echo ""
    
    # Créer dossier résultats
    exp_dir=$(dirname $(dirname $checkpoint))
    results_dir="$exp_dir/eval_${split}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$results_dir"
    
    echo "📂 Résultats: $results_dir"
    echo ""
    
    python models/eval/eval_seq2seq.py \
        --model_path "$checkpoint" \
        --eval_split "$split" \
        --data data/json_feat_2.1.0 \
        --model models.model.seq2seq_cot \
        --gpu \
        --num_threads 4 \
        --max_steps 1000 \
        --max_fails 10 \
        --results_path "$results_dir"
    
    echo ""
    echo "✓ Résultats: $results_dir"
    
    # Afficher métriques
    if [ -f "$results_dir/results.json" ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║                    RÉSULTATS                               ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        
        python3 << EOFPYTHON
import json

with open("$results_dir/results.json") as f:
    results = json.load(f)

print("📊 MÉTRIQUES:")
print("")
print(f"  Success Rate:     {results.get('success', {}).get('all', 0):.2%}")
print(f"  Goal Condition:   {results.get('goal_condition_success', {}).get('all', 0):.2%}")
print(f"  Path Length:      {results.get('path_length_weighted', 0):.4f}")

if 'subgoal' in results:
    print(f"  Subgoal Accuracy: {results.get('subgoal', {}).get('all', 0):.2%}")

EOFPYTHON
    fi
fi

echo ""
echo "✓ Évaluation terminée"

