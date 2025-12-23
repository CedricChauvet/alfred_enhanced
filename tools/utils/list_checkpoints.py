#!/usr/bin/env python3
"""
Liste les checkpoints disponibles pour évaluation
"""

import os
from pathlib import Path
from datetime import datetime

ALFRED_EXP_ROOT = Path.home() / "Bureau" / "Alfred" / "alfred_experiments"
experiments_dir = ALFRED_EXP_ROOT / "experiments"

print("="*70)
print("AVAILABLE CHECKPOINTS FOR EVALUATION")
print("="*70)
print()

if not experiments_dir.exists():
    print("❌ No experiments directory found!")
    print(f"Expected: {experiments_dir}")
    exit(1)

# Trouver toutes les expériences
experiments = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])

if not experiments:
    print("❌ No experiments found!")
    print("\nYou need to train a model first:")
    print("  python scripts/run_experiment.py --config configs/react_light_v1.yaml")
    exit(1)

print(f"Found {len(experiments)} experiment(s):\n")

available_checkpoints = []

for exp_dir in experiments:
    exp_name = exp_dir.name
    checkpoints_dir = exp_dir / "checkpoints"
    
    if not checkpoints_dir.exists():
        continue
    
    # Chercher les checkpoints
    best_seen = checkpoints_dir / "best_seen.pth"
    best_unseen = checkpoints_dir / "best_unseen.pth"
    latest = checkpoints_dir / "latest.pth"
    
    # Chercher aussi net_epoch_*.pth
    epoch_checkpoints = sorted(checkpoints_dir.glob("net_epoch_*.pth"))
    
    print(f"{'─'*70}")
    print(f"📁 {exp_name}")
    print(f"{'─'*70}")
    
    # Lire le status
    status_file = exp_dir / "status.txt"
    if status_file.exists():
        status = status_file.read_text().strip()
        status_icon = "✓" if status == "SUCCESS" else "⚠"
        print(f"   Status: {status_icon} {status}")
    
    # Lire la config
    config_file = exp_dir / "config.yaml"
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        model_type = config.get('model', 'unknown')
        epochs = config.get('epoch', '?')
        print(f"   Model:  {model_type}")
        print(f"   Epochs: {epochs}")
    
    print()
    
    # Lister les checkpoints disponibles
    has_checkpoints = False
    
    if best_seen.exists():
        size = best_seen.stat().st_size / (1024*1024)
        mtime = datetime.fromtimestamp(best_seen.stat().st_mtime)
        print(f"   ✓ best_seen.pth ({size:.1f} MB) - {mtime.strftime('%Y-%m-%d %H:%M')}")
        available_checkpoints.append(('best_seen', str(best_seen), exp_name))
        has_checkpoints = True
    
    if best_unseen.exists():
        size = best_unseen.stat().st_size / (1024*1024)
        mtime = datetime.fromtimestamp(best_unseen.stat().st_mtime)
        print(f"   ✓ best_unseen.pth ({size:.1f} MB) - {mtime.strftime('%Y-%m-%d %H:%M')}")
        available_checkpoints.append(('best_unseen', str(best_unseen), exp_name))
        has_checkpoints = True
    
    if latest.exists():
        size = latest.stat().st_size / (1024*1024)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        print(f"   ✓ latest.pth ({size:.1f} MB) - {mtime.strftime('%Y-%m-%d %H:%M')}")
        available_checkpoints.append(('latest', str(latest), exp_name))
        has_checkpoints = True
    
    for epoch_ckpt in epoch_checkpoints:
        size = epoch_ckpt.stat().st_size / (1024*1024)
        mtime = datetime.fromtimestamp(epoch_ckpt.stat().st_mtime)
        print(f"   ✓ {epoch_ckpt.name} ({size:.1f} MB) - {mtime.strftime('%Y-%m-%d %H:%M')}")
        available_checkpoints.append((epoch_ckpt.name, str(epoch_ckpt), exp_name))
        has_checkpoints = True
    
    if not has_checkpoints:
        print(f"   ❌ No checkpoints found")
    
    print()

print("="*70)
print("EVALUATION COMMANDS")
print("="*70)
print()

if not available_checkpoints:
    print("❌ No checkpoints available for evaluation!")
    print("\nTrain a model first:")
    print("  python scripts/run_experiment.py --config configs/react_light_v1.yaml")
else:
    print("Use these commands to evaluate:\n")
    
    # CoT experiments
    cot_experiments = [c for c in available_checkpoints if 'cot' in c[2].lower() and 'react' not in c[2].lower()]
    if cot_experiments:
        print("🔷 CoT BASELINE:\n")
        for ckpt_type, ckpt_path, exp_name in cot_experiments[:2]:
            print(f"# {exp_name} - {ckpt_type}")
            print(f"cd ~/Bureau/Alfred/alfred")
            print(f"python models/eval/eval_seq2seq.py \\")
            print(f"    --model_path {ckpt_path} \\")
            print(f"    --eval_split valid_seen \\")
            print(f"    --gpu")
            print()
    
    # ReAct experiments
    react_experiments = [c for c in available_checkpoints if 'react' in c[2].lower()]
    if react_experiments:
        print("🔶 REACT-LIGHT:\n")
        for ckpt_type, ckpt_path, exp_name in react_experiments[:2]:
            print(f"# {exp_name} - {ckpt_type}")
            print(f"cd ~/Bureau/Alfred/alfred")
            print(f"python models/eval/eval_seq2seq.py \\")
            print(f"    --model_path {ckpt_path} \\")
            print(f"    --eval_split valid_seen \\")
            print(f"    --gpu")
            print()
    
    # Si pas de séparation claire
    if not cot_experiments and not react_experiments:
        print("📦 ALL CHECKPOINTS:\n")
        for ckpt_type, ckpt_path, exp_name in available_checkpoints[:3]:
            print(f"# {exp_name} - {ckpt_type}")
            print(f"cd ~/Bureau/Alfred/alfred")
            print(f"python models/eval/eval_seq2seq.py \\")
            print(f"    --model_path {ckpt_path} \\")
            print(f"    --eval_split valid_seen \\")
            print(f"    --gpu")
            print()

print("="*70)
print("EVALUATION OPTIONS")
print("="*70)
print("""
--eval_split:
  • valid_seen     : Validation sur scènes vues (rapide, ~5-10 min)
  • valid_unseen   : Validation sur scènes nouvelles (plus long, ~15-20 min)
  • tests_seen     : Test final sur scènes vues
  • tests_unseen   : Test final sur scènes nouvelles

--gpu              : Utiliser GPU (recommandé)
--num_threads 1    : Nombre de threads (1 pour debug, 4+ pour vitesse)

EXEMPLE COMPLET:

python models/eval/eval_seq2seq.py \\
    --model_path ../alfred_experiments/experiments/react_light_v1_*/checkpoints/best_seen.pth \\
    --eval_split valid_seen \\
    --gpu \\
    --num_threads 1

""")

print("="*70)