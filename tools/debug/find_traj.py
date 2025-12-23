#!/usr/bin/env python3
"""Test de recherche trajectoires"""

from pathlib import Path
import sys

def test_find():
    # Depuis alfred_experiments
    data_path = Path('../alfred/data/json_2.1.0')
    
    print("="*70)
    print("TEST RECHERCHE TRAJECTOIRES")
    print("="*70)
    print(f"Input: {data_path}")
    print(f"Resolved: {data_path.resolve()}")
    print(f"Exists: {data_path.exists()}")
    print()
    
    split_path = data_path / 'train'
    print(f"Split path: {split_path}")
    print(f"Exists: {split_path.exists()}")
    print()
    
    if not split_path.exists():
        print("❌ Split path does not exist!")
        sys.exit(1)
    
    # Méthode 1: iterdir
    print("Method 1: iterdir()")
    count = 0
    for item in split_path.iterdir():
        if item.is_dir():
            count += 1
            if count <= 3:
                print(f"  Task dir: {item.name}")
    print(f"Total task dirs: {count}")
    print()
    
    # Méthode 2: glob
    print("Method 2: glob('*/')")
    task_dirs = list(split_path.glob('*/'))
    print(f"Found: {len(task_dirs)} task dirs")
    print()
    
    # Chercher trials
    print("Method 3: Find trials in first task")
    if task_dirs:
        first_task = task_dirs[0]
        print(f"First task: {first_task.name}")
        
        trials = list(first_task.glob('trial_*'))
        print(f"Trials: {len(trials)}")
        
        if trials:
            first_trial = trials[0]
            print(f"First trial: {first_trial.name}")
            
            traj_file = first_trial / 'traj_data.json'
            print(f"traj_data.json exists: {traj_file.exists()}")
    
    print()
    
    # Total
    print("Method 4: Count all traj_data.json")
    total = 0
    for task_dir in split_path.iterdir():
        if not task_dir.is_dir():
            continue
        for trial_dir in task_dir.glob('trial_*'):
            if not trial_dir.is_dir():
                continue
            if (trial_dir / 'traj_data.json').exists():
                total += 1
    
    print(f"✓ Total found: {total}")
    print("="*70)

if __name__ == '__main__':
    test_find()