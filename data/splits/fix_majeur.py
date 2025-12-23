#!/usr/bin/env python3
"""
Crée oct21_trainonly.json en excluant les splits tests_* corrompus
"""

import json
from pathlib import Path

print("="*70)
print("CRÉATION oct21_trainonly.json")
print("="*70)

# Charger oct21.json original
input_file = 'oct21.json'  # Adapter le chemin si besoin

with open(input_file) as f:
    data = json.load(f)

# Extraire seulement train et valid
trainonly = {
    'train': data['train'],
    'valid_seen': data['valid_seen'],
    'valid_unseen': data['valid_unseen']
}

# Parser le format task pour extraire task_name et trial
print("\n🔄 Parsing format task...")
for split_name, items in trainonly.items():
    print(f"\n{split_name}:")
    for item in items:
        # Format: "TASK_NAME/trial_XXXXX"
        task_path = item['task']
        
        if '/' in task_path:
            # Séparer task et trial
            parts = task_path.split('/')
            task_name = parts[0]
            trial_name = parts[1]
            
            # Extraire repeat_idx du trial_name
            # trial_T20190909_070538_437648
            trial_id = trial_name.replace('trial_', '')
            
            # Mettre à jour l'item
            item['task'] = task_name
            item['repeat_idx'] = trial_id
        else:
            print(f"  ⚠️  Format inattendu: {task_path}")

# Sauvegarder
output_file = 'oct21_trainonly.json'
with open(output_file, 'w') as f:
    json.dump(trainonly, f, indent=2)

print(f"\n✓ Créé {output_file}")

# Statistiques
print("\n📊 Résumé:")
for split_name, items in trainonly.items():
    print(f"  {split_name}: {len(items)} trajectoires")

print("\n" + "="*70)
print("UTILISATION:")
print("  Copier ce fichier dans /media/cedrix/Ubuntu_2To/Alfred/alfred/data/splits/")
print("  Puis lancer training avec: --splits data/splits/oct21_trainonly.json")
print("="*70)