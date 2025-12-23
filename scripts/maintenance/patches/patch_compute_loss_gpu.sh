#!/bin/bash

echo "=== PATCH COMPUTE_LOSS POUR GPU ==="

BASELINE="../alfred/models/model/seq2seq_im_mask.py"

# Vérifier si déjà patché
if grep -q "# Fix GPU: compute_loss device" "$BASELINE"; then
    echo "✓ Déjà patché"
    exit 0
fi

echo "Application du patch..."

# Trouver la ligne de début de compute_loss
LINE=$(grep -n "def compute_loss(self, out, batch, feat):" "$BASELINE" | cut -d: -f1)

if [ -z "$LINE" ]; then
    echo "✗ Fonction compute_loss non trouvée"
    exit 1
fi

echo "✓ compute_loss trouvée ligne $LINE"

# Insérer après la définition de la fonction (ligne suivante)
INSERT_LINE=$((LINE + 2))

# Ajouter le fix
sed -i "${INSERT_LINE}i\\        # Fix GPU: compute_loss device\\n        device = next(self.parameters()).device\\n        # Déplacer tous les tenseurs feat vers device\\n        for key in feat:\\n            if isinstance(feat[key], torch.Tensor):\\n                feat[key] = feat[key].to(device)" "$BASELINE"

echo "✓ Patch appliqué !"

