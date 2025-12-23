#!/bin/bash
# Analyse détaillée d'une expérience

set -e

# Détecter le chemin des experiments selon la structure du projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Si on est dans scripts/analysis/, experiments/ est à ../../experiments/
if [[ "$SCRIPT_DIR" == */scripts/analysis ]]; then
    EXPERIMENTS_DIR="$SCRIPT_DIR/../../experiments"
# Sinon, essayer le chemin relatif classique
else
    EXPERIMENTS_DIR="../experiments"
fi

# Convertir en chemin absolu
EXPERIMENTS_DIR=$(cd "$EXPERIMENTS_DIR" 2>/dev/null && pwd || echo "$EXPERIMENTS_DIR")

# Déterminer le chemin de l'expérience
if [ $# -eq 0 ]; then
    # Pas d'argument : prendre la dernière
    exp_name=$(ls -t "$EXPERIMENTS_DIR" | head -1)
    exp_dir="$EXPERIMENTS_DIR/$exp_name"
else
    exp_input="$1"
    
    # Vérifier si c'est un chemin absolu ou existant
    if [ -d "$exp_input" ]; then
        # Chemin direct (absolu ou relatif valide)
        exp_dir="$exp_input"
    else
        # Nom relatif : chercher dans experiments/ avec support wildcards
        exp_dir=$(ls -td "$EXPERIMENTS_DIR"/${exp_input}* 2>/dev/null | head -1)
    fi
fi

# Vérifier que l'expérience existe
if [ -z "$exp_dir" ] || [ ! -d "$exp_dir" ]; then
    echo "✗ Expérience non trouvée: ${1:-dernière}"
    echo ""
    echo "Expériences disponibles:"
    ls -1 "$EXPERIMENTS_DIR" 2>/dev/null | head -10
    echo ""
    echo "Usage:"
    echo "  $0                                    # Dernière expérience"
    echo "  $0 test_quick_gpu_20251209_110322    # Par nom"
    echo "  $0 test_quick_gpu_*                  # Avec wildcard"
    echo "  $0 ../experiments/test_quick_gpu_*   # Chemin relatif"
    echo "  $0 /chemin/absolu/vers/experience    # Chemin absolu"
    exit 1
fi

exp_basename=$(basename "$exp_dir")
exp_dir=$(cd "$exp_dir" && pwd)  # Convertir en chemin absolu

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ANALYSE: $exp_basename"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ════════════════════════════════════════════════════════════════════
# 1. Status
# ════════════════════════════════════════════════════════════════════

echo "📊 STATUS:"
if [ -f "$exp_dir/status.txt" ]; then
    status=$(cat "$exp_dir/status.txt")
    case "$status" in
        "SUCCESS") echo "  ✓ SUCCESS" ;;
        "FAILED") echo "  ✗ FAILED" ;;
        "RUNNING") echo "  ⚙ RUNNING" ;;
        *) echo "  ? $status" ;;
    esac
else
    echo "  ? UNKNOWN"
fi
echo ""

# ════════════════════════════════════════════════════════════════════
# 2. Configuration
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# 2. Configuration
# ════════════════════════════════════════════════════════════════════

echo "⚙️  CONFIGURATION:"
if [ -f "$exp_dir/config.yaml" ]; then
    python3 - "$exp_dir/config.yaml" << 'PYTHON_EOF'
import yaml
import sys

config_file = sys.argv[1]

with open(config_file) as f:
    config = yaml.safe_load(f)

# Afficher les paramètres clés
keys = [
    'model', 'batch', 'epoch', 'lr', 'gpu',
    'use_cot', 'max_subgoals', 'use_react',
    'replan_threshold', 'max_replans'
]

for key in keys:
    if key in config:
        print(f"  {key:20s}: {config[key]}")
PYTHON_EOF
else
    echo "  (config.yaml non trouvé)"
fi
echo ""

# ════════════════════════════════════════════════════════════════════
# 3. Métriques d'Entraînement
# ════════════════════════════════════════════════════════════════════

echo "📈 MÉTRIQUES D'ENTRAÎNEMENT:"

if [ -f "$exp_dir/logs/train.log" ]; then
    # Vérifier si le training est en cours
    last_line=$(tail -1 "$exp_dir/logs/train.log")
    
    if echo "$last_line" | grep -q "batch:.*%"; then
        # Training en cours
        current_batch=$(echo "$last_line" | grep -oP '\d+/\d+' | head -1)
        progress=$(echo "$last_line" | grep -oP '\d+%' | head -1)
        
        echo "  ⚙️  Training EN COURS"
        echo "  Progression: $progress (batch $current_batch)"
        echo ""
        echo "  ⚠️  Les métriques de validation ne sont pas encore disponibles."
        echo "     Elles seront écrites à la fin de chaque epoch."
        echo ""
    else
        # Training terminé ou pausé
        # Dernière epoch
        last_epoch=$(grep -a "'epoch':" "$exp_dir/logs/train.log" 2>/dev/null | tail -1 | grep -oP "'epoch':\s*\K\d+")
        
        if [ -n "$last_epoch" ]; then
            echo "  Dernière epoch: $last_epoch"
        else
            echo "  Dernière epoch: (non trouvée)"
        fi
        echo ""
        
        # Trouver le script de parsing
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PARSE_SCRIPT="$SCRIPT_DIR/parse_alfred_log.py"
        
        # Fallback : chercher dans tools/utils/
        if [ ! -f "$PARSE_SCRIPT" ]; then
            PARSE_SCRIPT="$SCRIPT_DIR/../../tools/utils/parse_alfred_log.py"
        fi
        
        if [ -f "$PARSE_SCRIPT" ]; then
            # Parser valid_seen
            echo "  Valid Seen (dernière epoch):"
            python3 "$PARSE_SCRIPT" "$exp_dir/logs/train.log" "valid_seen"
            echo ""
            
            # Parser valid_unseen
            echo "  Valid Unseen (dernière epoch):"
            python3 "$PARSE_SCRIPT" "$exp_dir/logs/train.log" "valid_unseen"
        else
            echo "  ⚠️  Parser non trouvé: $PARSE_SCRIPT"
            echo "     Copiez parse_alfred_log.py dans scripts/analysis/"
        fi
    fi
    
else
    echo "  (train.log non trouvé)"
fi

echo ""

# ════════════════════════════════════════════════════════════════════
# 4. Résultats d'Évaluation (si disponibles)
# ════════════════════════════════════════════════════════════════════

echo "🎯 RÉSULTATS D'ÉVALUATION:"

eval_found=false

for split in valid_seen valid_unseen tests_seen tests_unseen; do
    result_file="$exp_dir/results/${split}_results.json"
    
    if [ -f "$result_file" ]; then
        eval_found=true
        echo "  $split:"
        python3 - "$result_file" << 'PYTHON_EOF'
import json
import sys

result_file = sys.argv[1]

with open(result_file) as f:
    data = json.load(f)
    
if 'success' in data:
    print(f"    Success Rate:      {data['success']:.2%}")
if 'goal_condition_success' in data:
    print(f"    Goal Condition:    {data['goal_condition_success']:.2%}")
if 'path_len_weighted' in data:
    print(f"    Path Efficiency:   {data['path_len_weighted']:.3f}")
PYTHON_EOF
        echo ""
    fi
done

if [ "$eval_found" = false ]; then
    echo "  (Aucune évaluation effectuée)"
    echo "  Lancer: python models/eval/eval_seq2seq.py --model_path $exp_dir/checkpoints/best_seen.pth --eval_split valid_seen --gpu"
fi

echo ""

# ════════════════════════════════════════════════════════════════════
# 5. Checkpoints
# ════════════════════════════════════════════════════════════════════

echo "💾 CHECKPOINTS:"
if [ -d "$exp_dir/checkpoints" ]; then
    checkpoint_count=$(ls "$exp_dir/checkpoints/"*.pth 2>/dev/null | wc -l)
    
    if [ $checkpoint_count -gt 0 ]; then
        ls -lh "$exp_dir/checkpoints/"*.pth 2>/dev/null | \
            awk '{printf "  %-25s %8s  %s %s %s\n", $9, $5, $6, $7, $8}' | \
            sed 's|.*/||'
    else
        echo "  (Aucun checkpoint trouvé)"
    fi
else
    echo "  (Dossier checkpoints/ non trouvé)"
fi

echo ""

# ════════════════════════════════════════════════════════════════════
# 6. Durée
# ════════════════════════════════════════════════════════════════════

echo "⏱️  DURÉE:"
if [ -f "$exp_dir/logs/train.log" ]; then
    start=$(head -50 "$exp_dir/logs/train.log" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1)
    end=$(tail -50 "$exp_dir/logs/train.log" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | tail -1)
    
    if [ -n "$start" ] && [ -n "$end" ]; then
        echo "  Début:    $start"
        echo "  Fin:      $end"
        
        # Calculer durée
        start_sec=$(date -d "$start" +%s 2>/dev/null || echo "0")
        end_sec=$(date -d "$end" +%s 2>/dev/null || echo "0")
        
        if [ $start_sec -gt 0 ] && [ $end_sec -gt 0 ]; then
            duration=$((end_sec - start_sec))
            hours=$((duration / 3600))
            minutes=$(((duration % 3600) / 60))
            echo "  Durée:    ${hours}h ${minutes}min"
        fi
    else
        echo "  (Timestamps non trouvés)"
    fi
else
    echo "  (train.log non trouvé)"
fi

echo ""

# ════════════════════════════════════════════════════════════════════
# 7. Erreurs (si FAILED)
# ════════════════════════════════════════════════════════════════════

if [ -f "$exp_dir/status.txt" ] && [ "$(cat $exp_dir/status.txt)" = "FAILED" ]; then
    echo "❌ ERREURS:"
    if [ -f "$exp_dir/logs/train.log" ]; then
        echo "  Dernières lignes du log:"
        tail -20 "$exp_dir/logs/train.log" | sed 's/^/    /'
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════════════════
# 8. Suggestions
# ════════════════════════════════════════════════════════════════════

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  PROCHAINES ÉTAPES                                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Vérifier si évaluation faite
eval_done=$(ls "$exp_dir/results/"*_results.json 2>/dev/null | wc -l)

if [ $eval_done -eq 0 ]; then
    echo "📊 Évaluer le modèle:"
    echo "  cd ~/Bureau/Alfred/alfred"
    echo "  python models/eval/eval_seq2seq.py \\"
    echo "      --model_path $exp_dir/checkpoints/best_seen.pth \\"
    echo "      --eval_split valid_seen \\"
    echo "      --gpu"
    echo ""
fi

echo "🔍 Analyser les échecs:"
echo "  python scripts/analysis/analyze_failures.py \\"
echo "      --results $exp_dir/checkpoints/valid_seen.debug.preds.json"
echo ""

echo "📈 Comparer avec une autre expérience:"
echo "  python scripts/analysis/compare_cot_react.py \\"
echo "      --cot experiments/cot/baseline_* \\"
echo "      --react $exp_dir"
echo ""

echo "🎨 Visualiser une trajectoire:"
echo "  python scripts/analysis/visualize_trajectory.py \\"
echo "      --split valid_seen \\"
echo "      --task trial_T20190909_003841_461863"
echo ""

echo "╚════════════════════════════════════════════════════════════════╝"