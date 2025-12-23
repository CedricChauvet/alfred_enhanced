"""
remap_thought_ids.py

Remappe les thought_ids non-contigus (0-62 avec gaps)
vers des IDs contigus [0-36] pour le classifier
"""

import json
import sys
from pathlib import Path

# Mapping original (avec gaps)
ORIGINAL_IDS = [
    0, 1, 2, 3,           # Navigation (4)
    10, 11, 12, 13, 14, 15, 16,  # Pickup (7)
    20, 21, 22, 23, 24,   # Placement (5)
    30, 31, 32, 33, 34, 35, 36,  # Containers (7)
    40, 41, 42, 43, 44, 45, 46, 47,  # State changes (8)
    50, 51, 52,           # Progress (3)
    60, 61, 62            # Errors (3)
]  # Total: 37 classes

# Créer mapping: original_id → new_id
ID_REMAP = {old_id: new_id for new_id, old_id in enumerate(ORIGINAL_IDS)}

print(f"ID Remapping (37 classes):")
print(f"Original range: 0-62 (with gaps)")
print(f"New range: 0-36 (contiguous)")
print()

def remap_dataset(input_file, output_file):
    """Remap thought_ids dans le dataset"""
    print(f"Reading: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Stats
    total_thoughts = 0
    remapped_count = 0
    
    # Remap
    for traj in data:
        for thought in traj['thoughts']:
            old_id = thought['thought_id']
            new_id = ID_REMAP.get(old_id, None)
            
            if new_id is None:
                print(f"⚠️  Unknown thought_id: {old_id}")
                continue
            
            thought['thought_id'] = new_id
            thought['thought_id_original'] = old_id  # Garder trace
            total_thoughts += 1
            
            if old_id != new_id:
                remapped_count += 1
    
    # Save
    print(f"Writing: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Processed {total_thoughts:,} thoughts")
    print(f"✓ Remapped {remapped_count:,} IDs")
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python remap_thought_ids.py <dataset_file>")
        print()
        print("Example:")
        print("  python remap_thought_ids.py data/thoughts/thoughts_dataset_train.json")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        sys.exit(1)
    
    # Output: ajouter _remapped
    output_file = input_file.parent / (input_file.stem + '_remapped.json')
    
    remap_dataset(input_file, output_file)
    
    print(f"✓ Done! New file: {output_file}")
    print()
    print("Next steps:")
    print(f"  1. Use remapped file for training:")
    print(f"     --data {output_file}")
    print(f"  2. Model will now expect IDs [0-36] instead of [0-62]")