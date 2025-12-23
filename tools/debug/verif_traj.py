#!/usr/bin/env python3
"""
verify_dataset_alignment.py

Vérifie que json_2.1.0 et json_feat_2.1.0 ont les mêmes task_ids
"""

import os
import sys
from pathlib import Path

def get_task_ids(root_path, split='train'):
    """Récupérer tous les task_ids d'un dataset"""
    split_path = Path(root_path) / split
    
    if not split_path.exists():
        print(f"❌ Path not found: {split_path}")
        return set()
    
    task_ids = set()
    
    # Parcourir les task_dirs
    for task_dir in split_path.iterdir():
        if not task_dir.is_dir():
            continue
        
        # Parcourir les trials
        for trial_dir in task_dir.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
                continue
            
            # task_id = task_dir/trial_dir
            task_id = f"{task_dir.name}/{trial_dir.name}"
            task_ids.add(task_id)
    
    return task_ids


def main():
    # Paths
    json_2_path = "/home/cedrix/Bureau/Alfred/alfred/data/json_2.1.0"
    json_feat_path = "/home/cedrix/Bureau/Alfred/alfred/data/json_feat_2.1.0"
    
    print("="*70)
    print("DATASET ALIGNMENT VERIFICATION")
    print("="*70)
    
    for split in ['train', 'valid_seen', 'valid_unseen']:
        print(f"\n{split.upper()}:")
        print("-" * 70)
        
        # Get task_ids from both datasets
        json_2_ids = get_task_ids(json_2_path, split)
        json_feat_ids = get_task_ids(json_feat_path, split)
        
        print(f"json_2.1.0:      {len(json_2_ids):,} task_ids")
        print(f"json_feat_2.1.0: {len(json_feat_ids):,} task_ids")
        
        # Compare
        only_in_json_2 = json_2_ids - json_feat_ids
        only_in_json_feat = json_feat_ids - json_2_ids
        common = json_2_ids & json_feat_ids
        
        print(f"\nCommon: {len(common):,}")
        print(f"Only in json_2.1.0: {len(only_in_json_2)}")
        print(f"Only in json_feat_2.1.0: {len(only_in_json_feat)}")
        
        if len(only_in_json_2) > 0:
            print(f"\n⚠️  Examples only in json_2.1.0:")
            for task_id in list(only_in_json_2)[:3]:
                print(f"   {task_id}")
        
        if len(only_in_json_feat) > 0:
            print(f"\n⚠️  Examples only in json_feat_2.1.0:")
            for task_id in list(only_in_json_feat)[:3]:
                print(f"   {task_id}")
        
        # Alignment percentage
        if len(json_2_ids) > 0:
            alignment = len(common) / len(json_2_ids) * 100
            print(f"\n{'✓' if alignment == 100 else '⚠️ '} Alignment: {alignment:.1f}%")
        
        # Sample verification
        if len(common) > 0:
            print(f"\nSample verification (checking actual files):")
            sample_id = list(common)[0]
            
            traj_path = Path(json_2_path) / split / sample_id / 'traj_data.json'
            feat_path = Path(json_feat_path) / split / sample_id / 'feat_conv.pt'
            
            print(f"  Task: {sample_id}")
            print(f"  traj_data.json exists: {traj_path.exists()}")
            print(f"  feat_conv.pt exists: {feat_path.exists()}")
            
            if traj_path.exists() and feat_path.exists():
                print(f"  ✓ Both files accessible!")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("If alignment is 100% for all splits, the datasets are perfectly aligned.")
    print("If < 100%, you may have missing features or trajectories.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()