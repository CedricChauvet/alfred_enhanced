#!/usr/bin/env python3
"""
Parser pour train.log ALFRED
Extrait les métriques de validation pour un split donné
"""

import sys
import re
import ast

def parse_alfred_log(log_file, split_name):
    """Parse metrics from ALFRED train.log"""
    
    try:
        with open(log_file, 'r', errors='ignore') as f:
            content = f.read()
    except FileNotFoundError:
        print("    (Fichier non trouvé)")
        return
    
    # Chercher tous les blocs epoch
    # Format: {'epoch': N, 'valid_seen': {...}, 'valid_unseen': {...}}
    
    # Pattern pour trouver les blocs epoch complets
    pattern = r"\{'epoch':\s*\d+,\s*'valid_seen':\s*\{[^}]+\},\s*'valid_unseen':\s*\{[^}]+\}\}"
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    epochs_data = []
    
    for match in matches:
        block = match.group(0)
        
        try:
            # Nettoyer et parser
            # Le bloc peut être multilignes, on doit le reconstruire
            clean_block = re.sub(r'\s+', ' ', block)  # Enlever les newlines
            
            # Essayer d'évaluer comme dictionnaire Python
            data = ast.literal_eval(clean_block)
            epochs_data.append(data)
        except:
            continue
    
    if not epochs_data:
        print("    (Pas de données trouvées)")
        return
    
    # Prendre la dernière epoch
    last_epoch_data = epochs_data[-1]
    
    if split_name not in last_epoch_data:
        print(f"    (Split '{split_name}' non trouvé)")
        return
    
    metrics = last_epoch_data[split_name]
    
    # Afficher les métriques principales
    if 'action_low_f1' in metrics:
        print(f"    F1 Score:       {metrics['action_low_f1']:.4f}")
    
    if 'loss_cot_acc' in metrics:
        print(f"    CoT Accuracy:   {metrics['loss_cot_acc']:.4f}")
    
    if 'total_loss' in metrics:
        print(f"    Total Loss:     {metrics['total_loss']:.4f}")
    
    # Métriques supplémentaires
    if 'action_low_em' in metrics:
        print(f"    Exact Match:    {metrics['action_low_em']:.4f}")
    
    if 'loss_cot' in metrics:
        print(f"    CoT Loss:       {metrics['loss_cot']:.6f}")
    
    # ReAct metrics (si présents)
    if 'react_thought_acc' in metrics:
        print(f"    Thought Acc:    {metrics['react_thought_acc']:.4f}")
    
    if 'react_replan_acc' in metrics:
        print(f"    Replan Acc:     {metrics['react_replan_acc']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: parse_alfred_log.py <log_file> <split_name>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    split_name = sys.argv[2]
    
    parse_alfred_log(log_file, split_name)