#!/bin/bash
# Nettoie les expériences échouées ou incomplètes

echo "=== NETTOYAGE EXPÉRIENCES ÉCHOUÉES ==="
echo ""

failed_count=0
incomplete_count=0

for exp_dir in experiments/*/; do
    exp_name=$(basename "$exp_dir")
    
    # Vérifier status
    if [ -f "$exp_dir/status.txt" ]; then
        status=$(cat "$exp_dir/status.txt")
        
        if [[ "$status" == *"failed"* ]] || [[ "$status" == *"error"* ]]; then
            echo "❌ Failed: $exp_name"
            echo "   Status: $status"
            
            # Demander confirmation
            read -p "   Supprimer ? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$exp_dir"
                echo "   ✓ Supprimé"
                ((failed_count++))
            fi
        fi
    else
        # Pas de status = probablement crashé
        if [ ! -f "$exp_dir/logs/train.log" ]; then
            echo "⚠️  Incomplete: $exp_name (pas de logs)"
            read -p "   Supprimer ? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$exp_dir"
                echo "   ✓ Supprimé"
                ((incomplete_count++))
            fi
        fi
    fi
done

echo ""
echo "=== RÉSUMÉ ==="
echo "Expériences failed supprimées: $failed_count"
echo "Expériences incomplètes supprimées: $incomplete_count"
echo ""

# Espace disque libéré
du -sh experiments/ 2>/dev/null | awk '{print "Espace utilisé: " $1}'
