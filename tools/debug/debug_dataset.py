#!/usr/bin/env python3
"""
debug_dataset.py

Vérifier que les features et instructions sont bien chargées
"""

import sys
sys.path.insert(0, '/home/cedrix/Bureau/Alfred/alfred')

import torch
import json
from pathlib import Path

# Charger le dataset
dataset_path = '/home/cedrix/Bureau/Alfred/alfred/data/thoughts/thoughts_dataset_train_remapped.json'
feat_root = Path('/home/cedrix/Bureau/Alfred/alfred/data/json_feat_2.1.0')
data_root = Path('/home/cedrix/Bureau/Alfred/alfred/data/json_2.1.0')

with open(dataset_path, 'r') as f:
    data = json.load(f)

print("="*70)
print("DATASET DEBUG")
print("="*70)

# Test sur 10 premiers samples
samples_to_test = 10
success_feat = 0
success_inst = 0
feat_values = []
inst_values = []

for i in range(min(samples_to_test, len(data))):
    traj = data[i]
    task_id = traj['task_id']
    
    print(f"\n{i+1}. Task: {task_id[:60]}...")
    
    # Chercher feat_conv.pt
    feat_path = None
    for split in ['train', 'valid_seen', 'valid_unseen']:
        candidate = feat_root / split / task_id / 'feat_conv.pt'
        if candidate.exists():
            feat_path = candidate
            print(f"   ✓ Features found in {split}/")
            break
    
    if feat_path:
        try:
            feats = torch.load(feat_path, map_location='cpu')
            print(f"   ✓ Loaded: shape={feats.shape}, mean={feats.mean():.3f}, std={feats.std():.3f}")
            success_feat += 1
            feat_values.append(feats.mean().item())
        except Exception as e:
            print(f"   ❌ Error loading features: {e}")
    else:
        print(f"   ❌ Features NOT found")
    
    # Chercher traj_data.json
    traj_path = None
    for split in ['train', 'valid_seen', 'valid_unseen']:
        candidate = data_root / split / task_id / 'traj_data.json'
        if candidate.exists():
            traj_path = candidate
            break
    
    if traj_path:
        try:
            with open(traj_path, 'r') as f:
                traj_data = json.load(f)
                anns = traj_data.get('turk_annotations', {}).get('anns', [])
                if anns:
                    instruction = anns[0].get('task_desc', '')
                    print(f"   ✓ Instruction: '{instruction[:50]}...'")
                    success_inst += 1
                    inst_values.append(len(instruction))
                else:
                    print(f"   ⚠️  No annotations")
        except Exception as e:
            print(f"   ❌ Error loading instruction: {e}")
    else:
        print(f"   ❌ traj_data.json NOT found")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Features loaded successfully: {success_feat}/{samples_to_test}")
print(f"Instructions loaded successfully: {success_inst}/{samples_to_test}")

if feat_values:
    print(f"\nFeature statistics:")
    print(f"  Mean values: {sum(feat_values)/len(feat_values):.3f}")
    print(f"  All same? {len(set(feat_values)) == 1}")
    if len(set(feat_values)) == 1:
        print(f"  ⚠️  All features have SAME mean → Probably dummy!")

if inst_values:
    print(f"\nInstruction statistics:")
    print(f"  Avg length: {sum(inst_values)/len(inst_values):.1f} chars")
    print(f"  All empty? {sum(inst_values) == 0}")

print("="*70)

if success_feat < samples_to_test * 0.8:
    print("❌ PROBLEM: Less than 80% features loaded!")
    print("   → Features are probably NOT being loaded correctly")

if success_inst < samples_to_test * 0.8:
    print("❌ PROBLEM: Less than 80% instructions loaded!")
    print("   → Instructions are probably NOT being loaded correctly")

if success_feat >= samples_to_test * 0.8 and success_inst >= samples_to_test * 0.8:
    print("✓ Dataset loading looks OK")
    print("\nIf accuracy is still stuck at 50%, the problem is likely:")
    print("  1. Language tokenization not working (hash collision)")
    print("  2. Model architecture issue (gradients not flowing)")
    print("  3. Features too weak (need better features or more training)")