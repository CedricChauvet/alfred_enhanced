#!/usr/bin/env python3
"""
fix_checkpoint_for_eval.py

Ajoute la clé 'optim' manquante à un checkpoint pour eval_seq2seq.py
"""

import torch
import sys
from pathlib import Path

def fix_checkpoint(checkpoint_path, output_path=None):
    """
    Ajoute 'optim' valide à un checkpoint s'il manque
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ File not found: {checkpoint_path}")
        return
    
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    
    # Vérifier si 'optim' manque ou est vide
    needs_fix = False
    if 'optim' not in checkpoint:
        print("⚠️  'optim' key missing")
        needs_fix = True
    elif not checkpoint['optim'] or 'param_groups' not in checkpoint['optim']:
        print("⚠️  'optim' exists but is empty or invalid")
        needs_fix = True
    
    if needs_fix:
        print("Creating valid optimizer state...")
        
        # Créer un optimizer state valide (minimal)
        # Adam optimizer structure
        checkpoint['optim'] = {
            'state': {},
            'param_groups': [
                {
                    'lr': 0.0001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0,
                    'amsgrad': False,
                    'params': []
                }
            ]
        }
        
        # Sauvegarder
        if output_path is None:
            # Backup original
            backup = checkpoint_path.parent / (checkpoint_path.stem + '_backup.pth')
            if not backup.exists():
                print(f"Backing up original to: {backup}")
                torch.save(torch.load(checkpoint_path, map_location='cpu'), backup)
            
            # Overwrite
            output_path = checkpoint_path
        
        print(f"Saving fixed checkpoint to: {output_path}")
        torch.save(checkpoint, output_path)
        print("✓ Done!")
    else:
        print("✓ 'optim' key already valid, no fix needed")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to checkpoint to fix')
    parser.add_argument('--output', default=None, help='Output path (default: overwrite)')
    args = parser.parse_args()
    
    fix_checkpoint(args.checkpoint, args.output)