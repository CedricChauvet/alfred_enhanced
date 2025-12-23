#!/bin/bash
# Évaluation best_unseen.pth

export ALFRED_ROOT=/home/cedrix/Bureau/Alfred/alfred
export PYTHONPATH=$ALFRED_ROOT:$ALFRED_ROOT/gen:$PYTHONPATH

latest_exp=$(ls -td ../experiments/*/ | head -1)
checkpoint="$latest_exp/checkpoints/best_unseen.pth"
split=${1:-valid_unseen}

if [ ! -f "$checkpoint" ]; then
    echo "✗ Checkpoint non trouvé"
    exit 1
fi

results_dir="$latest_exp/eval_${split}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$results_dir"

echo "Évaluation: $(basename $latest_exp)"
echo "Checkpoint: best_unseen.pth"
echo "Split: $split"
echo ""

cd $ALFRED_ROOT

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
