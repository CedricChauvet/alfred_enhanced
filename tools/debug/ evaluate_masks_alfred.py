#!/usr/bin/env python3
"""
evaluate_masks_alfred.py

Évalue les masques prédits par le modèle ALFRED baseline
"""

import os
import sys
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np

# Ajouter le path ALFRED
sys.path.append(os.environ.get('ALFRED_ROOT', '.'))

from models.eval.eval import Eval
from models.model.seq2seq_im_mask import Module
from gen.utils.image_util import decompress_mask


class ALFREDMaskEvaluator:
    """
    Évaluateur spécifique pour ALFRED
    """
    
    def __init__(self, args):
        self.args = args
        self.results = []
        
    def load_model(self):
        """Charge le modèle entraîné"""
        print(f"Loading model from {self.args.model_path}")
        checkpoint = torch.load(self.args.model_path, map_location='cpu')
        
        # Charger architecture
        from importlib import import_module
        M = import_module(self.args.model)
        model, _ = M.Module.load(self.args.model_path)
        model.eval()
        model.test_mode = True
        
        return model
    
    def decompress_gt_mask(self, compressed_mask):
        """Décompresse le masque ground truth"""
        mask = np.array(decompress_mask(compressed_mask))
        return torch.from_numpy(mask).float()
    
    def evaluate_single_trajectory(self, model, traj_data):
        """
        Évalue une trajectoire complète
        """
        # Préparer les features
        feat = model.featurize([traj_data], load_mask=True, load_frames=False)
        
        # Prédire (mode test - autoregressive)
        model.reset()
        pred_masks = []
        
        for t in range(len(traj_data['plan']['low_actions'])):
            # Step-by-step prediction
            prev_action = traj_data['plan']['low_actions'][t-1]['api_action']['action'] if t > 0 else None
            
            # Feature pour ce step (sans images pour plus rapide)
            step_feat = {
                'lang_goal_instr': feat['lang_goal_instr'],
                'frames': torch.zeros(1, 1, 512, 7, 7)  # Dummy frames
            }
            
            out = model.step(step_feat, prev_action)
            pred_mask = torch.sigmoid(out['out_action_low_mask'][0, 0])
            pred_masks.append(pred_mask)
        
        # Ground truth masks
        gt_masks = []
        for action in traj_data['num']['action_low']:
            if action['mask'] is not None:
                gt_mask = self.decompress_gt_mask(action['mask'])
                gt_masks.append(gt_mask)
            else:
                gt_masks.append(None)
        
        # Évaluer chaque masque
        traj_results = []
        for t, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            if gt is None or gt.sum() == 0:
                continue  # Pas d'interaction à ce step
            
            iou = self.compute_iou(pred, gt)
            dice = self.compute_dice(pred, gt)
            
            traj_results.append({
                'step': t,
                'action': traj_data['plan']['low_actions'][t]['api_action']['action'],
                'iou': iou,
                'dice': dice
            })
        
        return traj_results
    
    def compute_iou(self, pred, gt, threshold=0.5):
        """IoU"""
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * gt).sum()
        union = pred_binary.sum() + gt.sum() - intersection
        
        if union == 0:
            return 1.0
        return (intersection / union).item()
    
    def compute_dice(self, pred, gt, threshold=0.5):
        """Dice coefficient"""
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * gt).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + gt.sum() + 1e-8)
        return dice.item()
    
    def run_evaluation(self):
        """
        Évalue sur tout le dataset
        """
        # Charger modèle
        model = self.load_model()
        
        # Charger split
        with open(self.args.splits) as f:
            splits = json.load(f)
        
        eval_files = splits[self.args.eval_split]
        
        # Limiter pour test rapide
        if self.args.fast_epoch:
            eval_files = eval_files[:self.args.num_eval]
        
        print(f"\nEvaluating {len(eval_files)} trajectories...")
        
        all_ious = []
        all_dices = []
        
        for traj_file in tqdm(eval_files):
            # Charger trajectoire
            with open(traj_file) as f:
                traj_data = json.load(f)
            
            # Évaluer
            try:
                results = self.evaluate_single_trajectory(model, traj_data)
                
                for r in results:
                    all_ious.append(r['iou'])
                    all_dices.append(r['dice'])
                    self.results.append(r)
            
            except Exception as e:
                print(f"Error evaluating {traj_file}: {e}")
                continue
        
        # Résumé
        print("\n" + "="*60)
        print("MASK EVALUATION RESULTS")
        print("="*60)
        print(f"Total interactions evaluated: {len(all_ious)}")
        print(f"\nIoU:")
        print(f"  Mean:   {np.mean(all_ious):.4f} ± {np.std(all_ious):.4f}")
        print(f"  Median: {np.median(all_ious):.4f}")
        print(f"  Min:    {np.min(all_ious):.4f}")
        print(f"  Max:    {np.max(all_ious):.4f}")
        
        print(f"\nDice:")
        print(f"  Mean:   {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")
        print(f"  Median: {np.median(all_dices):.4f}")
        
        # Sauvegarde
        output_file = os.path.join(self.args.model_path.replace('best_seen.pth', ''), 
                                   'mask_eval_results.json')
        
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'iou_mean': float(np.mean(all_ious)),
                    'iou_std': float(np.std(all_ious)),
                    'iou_median': float(np.median(all_ious)),
                    'dice_mean': float(np.mean(all_dices)),
                    'dice_std': float(np.std(all_dices)),
                    'num_evaluated': len(all_ious)
                },
                'detailed_results': self.results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        return all_ious, all_dices


def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask',
                       help='Model architecture')
    parser.add_argument('--data', type=str, default='data/json_feat_2.1.0',
                       help='Path to data')
    parser.add_argument('--splits', type=str, default='data/splits/oct21.json',
                       help='Path to splits file')
    
    # Evaluation
    parser.add_argument('--eval_split', type=str, default='valid_seen',
                       choices=['train', 'valid_seen', 'valid_unseen', 'tests_seen', 'tests_unseen'])
    parser.add_argument('--fast_epoch', action='store_true',
                       help='Evaluate on subset only')
    parser.add_argument('--num_eval', type=int, default=50,
                       help='Number of trajectories for fast evaluation')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ALFREDMaskEvaluator(args)
    evaluator.run_evaluation()


if __name__ == '__main__':
    main()