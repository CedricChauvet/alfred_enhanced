#!/usr/bin/env python3
"""
Script rapide pour inspecter le format du fichier splits
"""

import json
import sys
from pathlib import Path

ALFRED_ROOT = Path.home() / "Bureau" / "Alfred" / "alfred"
split_file = ALFRED_ROOT / "data" / "splits" / "oct21.json"

print(f"Inspecting: {split_file}\n")

with open(split_file, 'r') as f:
    splits = json.load(f)

print("Keys in splits file:", list(splits.keys()))
print()

for split_name in ['train', 'valid_seen', 'valid_unseen']:
    if split_name in splits:
        split_data = splits[split_name]
        print(f"\n{'='*70}")
        print(f"Split: {split_name}")
        print(f"{'='*70}")
        print(f"Type: {type(split_data)}")
        print(f"Length: {len(split_data)}")
        
        if len(split_data) > 0:
            print(f"\nFirst item type: {type(split_data[0])}")
            print(f"First item: {split_data[0]}")
            
            if isinstance(split_data[0], dict):
                print(f"Keys in first item: {list(split_data[0].keys())}")
        
        if len(split_data) > 1:
            print(f"\nSecond item: {split_data[1]}")