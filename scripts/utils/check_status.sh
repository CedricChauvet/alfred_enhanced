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
