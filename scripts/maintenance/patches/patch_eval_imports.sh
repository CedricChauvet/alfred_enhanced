#!/bin/bash
# Patch les scripts d'évaluation pour fix les imports

echo "=== PATCH IMPORTS ÉVALUATION ==="
echo ""

for script in eval_subgoals.py eval_seq2seq.py; do
    file="../alfred/models/eval/$script"
    
    if [ ! -f "$file" ]; then
        echo "⚠️  $script non trouvé"
        continue
    fi
    
    echo "Patching $script..."
    
    # Backup
    cp "$file" "${file}.backup_imports"
    
    # Vérifier si déjà patché
    if grep -q "Fix imports ALFRED" "$file"; then
        echo "  ✓ Déjà patché"
        continue
    fi
    
    # Créer version patchée
    cat > "${file}.tmp" << 'EOFPATCH'
import sys
import os
from pathlib import Path

# Fix imports ALFRED
ALFRED_ROOT = Path(os.environ.get('ALFRED_ROOT', os.path.join(os.path.dirname(__file__), '..', '..')))
ALFRED_ROOT = ALFRED_ROOT.resolve()
if str(ALFRED_ROOT) not in sys.path:
    sys.path.insert(0, str(ALFRED_ROOT))

EOFPATCH
    
    # Ajouter le contenu original (en enlevant les 3 premières lignes)
    tail -n +4 "$file" >> "${file}.tmp"
    
    # Remplacer
    mv "${file}.tmp" "$file"
    
    echo "  ✓ Patché"
done

echo ""
echo "✓ Patch terminé"

