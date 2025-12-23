
#!/bin/bash
# Liste toutes les expériences avec leurs métriques

echo "=== LISTE DES EXPÉRIENCES ==="
echo ""

# Header
printf "%-40s %-12s %-10s %-10s %-10s %-10s\n" \
    "NOM" "STATUS" "EPOCHS" "F1" "CoT ACC" "DEVICE"
printf "%s\n" "$(printf '=%.0s' {1..100})"

for exp_dir in experiments/*/; do
    exp_name=$(basename "$exp_dir")
    
    # Status
    if [ -f "$exp_dir/status.txt" ]; then
        status=$(cat "$exp_dir/status.txt" | head -c 10)
    else
        status="unknown"
    fi
    
    # Epochs
    if [ -f "$exp_dir/config.yaml" ]; then
        epochs=$(grep "^epoch:" "$exp_dir/config.yaml" | awk '{print $2}')
        device=$(grep "^gpu:" "$exp_dir/config.yaml" | grep -q "true" && echo "GPU" || echo "CPU")
    else
        epochs="?"
        device="?"
    fi
    
    # Métriques
    f1="?"
    cot_acc="?"
    
    if [ -f "$exp_dir/logs/train.log" ]; then
        # Extraire dernières métriques
        last_metrics=$(grep "valid_unseen" "$exp_dir/logs/train.log" | tail -1)
        
        if [ ! -z "$last_metrics" ]; then
            f1=$(echo "$last_metrics" | grep -oP "action_low_f1': [0-9.]+" | grep -oP "[0-9.]+" | head -1)
            cot_acc=$(echo "$last_metrics" | grep -oP "loss_cot_acc': [0-9.]+" | grep -oP "[0-9.]+" | head -1)
            
            # Formater
            [ ! -z "$f1" ] && f1=$(printf "%.4f" "$f1")
            [ ! -z "$cot_acc" ] && cot_acc=$(printf "%.4f" "$cot_acc")
        fi
    fi
    
    # Afficher
    printf "%-40s %-12s %-10s %-10s %-10s %-10s\n" \
        "$exp_name" "$status" "$epochs" "$f1" "$cot_acc" "$device"
done

echo ""
echo "Total: $(ls -d experiments/*/ 2>/dev/null | wc -l) expériences"
echo ""

# Espace disque
echo "💾 Espace disque:"
du -sh experiments/ 2>/dev/null | awk '{print "   Total: " $1}'
du -sh experiments/*/checkpoints/ 2>/dev/null | awk '{sum+=$1} END {print "   Checkpoints: " sum/1024 " MB"}'
