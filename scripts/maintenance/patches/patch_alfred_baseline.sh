#!/bin/bash

echo "=== PATCH ALFRED BASELINE POUR GPU ==="

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
    echo "✓ Déjà patché"
    exit 0
fi

# Trouver la ligne avec embed_seq = self.emb_word(pad_seq)
line_num=$(grep -n "embed_seq = self.emb_word(pad_seq)" "$BASELINE" | head -1 | cut -d: -f1)

if [ -z "$line_num" ]; then
    echo "✗ Ligne non trouvée"
    exit 1
fi

echo "✓ Ligne trouvée: $line_num"

# Créer fichier temporaire avec le patch
awk -v line="$line_num" '
NR == line {
    print "        # Fix GPU: déplacer vers le device du modèle"
    print "        pad_seq = pad_seq.to(next(self.parameters()).device)"
}
{ print }
' "$BASELINE" > "${BASELINE}.tmp"

# Remplacer
mv "${BASELINE}.tmp" "$BASELINE"

echo "✓ Patch appliqué"
echo ""
echo "Vérification (lignes $((line_num-1)) à $((line_num+2))):"
sed -n "$((line_num-1)),$((line_num+2))p" "$BASELINE"
echo ""
echo "=== PATCH TERMINÉ ==="

