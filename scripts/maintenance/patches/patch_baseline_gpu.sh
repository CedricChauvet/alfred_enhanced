#!/bin/bash

echo "=== PATCH BASELINE ALFRED POUR GPU ==="
echo ""

BASELINE="../alfred/models/model/seq2seq_im_mask.py"

# Backup
if [ ! -f "${BASELINE}.backup" ]; then
    echo "✓ Création backup..."
    cp "$BASELINE" "${BASELINE}.backup"
else
    echo "✓ Backup existe déjà"
fi

# Vérifier si déjà patché
if grep -q "pad_seq.to(next(self.parameters()).device)" "$BASELINE"; then
    echo "✓ Déjà patché !"
    exit 0
fi

echo "Application du patch..."

# Trouver la ligne
LINE=$(grep -n "embed_seq = self.emb_word(pad_seq)" "$BASELINE" | head -1 | cut -d: -f1)

if [ -z "$LINE" ]; then
    echo "✗ Ligne non trouvée"
    exit 1
fi

echo "✓ Ligne $LINE trouvée"

# Appliquer patch
sed -i "${LINE}i\\        # Fix GPU\\n        pad_seq = pad_seq.to(next(self.parameters()).device)" "$BASELINE"

echo "✓ Patch appliqué !"
echo ""
echo "Vérification:"
sed -n "$((LINE-1)),$((LINE+3))p" "$BASELINE"

