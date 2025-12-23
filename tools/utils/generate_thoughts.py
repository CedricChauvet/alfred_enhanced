"""
Génération automatique de thoughts pour ReAct

Ce script génère des annotations "thoughts" à partir des trajectoires ALFRED
en utilisant des heuristiques simples basées sur:
- Le type d'action
- Le succès/échec de l'action
- Le contexte (objet en main, objectif, etc.)

Usage:
    python generate_thoughts.py --data data/json_feat_2.1.0 --split train --output data/thoughts_train.json

Architecture:
    1. Charger trajectoires ALFRED
    2. Pour chaque step:
        - Analyser action + observation
        - Générer thought approprié
    3. Sauvegarder annotations
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# VOCABULAIRE DE THOUGHTS (cohérent avec ReActLightModule)
# ═══════════════════════════════════════════════════════════

THOUGHT_VOCAB = {
    # Navigation
    "need_to_navigate": 0,
    "location_reached": 1,
    "location_not_found": 2,
    
    # Manipulation
    "need_to_pickup": 3,
    "object_picked_up": 4,
    "object_not_found": 5,
    "need_to_place": 6,
    "object_placed": 7,
    
    # State changes
    "need_to_open": 8,
    "container_opened": 9,
    "need_to_close": 10,
    "need_to_toggle": 11,
    
    # Temperature
    "need_to_heat": 12,
    "object_heated": 13,
    "need_to_cool": 14,
    "object_cooled": 15,
    
    # Cleaning
    "need_to_clean": 16,
    "object_cleaned": 17,
    
    # Slicing
    "need_to_slice": 18,
    "object_sliced": 19,
    
    # Errors & Replanning
    "action_failed": 20,
    "replanning_required": 21,
    "trying_alternative": 22,
    
    # Success
    "subgoal_completed": 23,
    "task_completed": 24
}

# Reverse mapping
THOUGHT_IDX = {v: k for k, v in THOUGHT_VOCAB.items()}


# ═══════════════════════════════════════════════════════════
# HEURISTIQUES DE GÉNÉRATION
# ═══════════════════════════════════════════════════════════

def generate_thought_for_action(action, prev_action, next_action, subgoal_idx, total_subgoals):
    """
    Génère un thought basé sur l'action high-level
    
    Args:
        action: dict avec 'discrete_action' et éventuellement 'planner_action'
        prev_action: action précédente (ou None)
        next_action: action suivante (ou None)
        subgoal_idx: index du subgoal actuel
        total_subgoals: nombre total de subgoals
    
    Returns:
        thought_name: str (clé dans THOUGHT_VOCAB)
    """
    
    action_name = action['discrete_action']['action']
    
    # ═══════════════════════════════════════════════════════════
    # RÈGLES PAR TYPE D'ACTION
    # ═══════════════════════════════════════════════════════════
    
    # Navigation
    if action_name == 'GotoLocation':
        # Si c'est la première fois qu'on va quelque part
        if prev_action is None or prev_action['discrete_action']['action'] != 'GotoLocation':
            return 'need_to_navigate'
        else:
            return 'location_reached'
    
    # Pickup
    elif action_name == 'PickupObject':
        # Avant pickup
        if next_action and next_action['discrete_action']['action'] == 'PickupObject':
            return 'need_to_pickup'
        else:
            return 'object_picked_up'
    
    # Place
    elif action_name == 'PutObject':
        return 'object_placed'
    
    # Open
    elif action_name == 'OpenObject':
        return 'container_opened'
    
    # Close
    elif action_name == 'CloseObject':
        return 'need_to_close'
    
    # Toggle
    elif action_name == 'ToggleObjectOn' or action_name == 'ToggleObjectOff':
        return 'need_to_toggle'
    
    # Heat
    elif action_name == 'HeatObject':
        return 'object_heated'
    
    # Cool
    elif action_name == 'CoolObject':
        return 'object_cooled'
    
    # Clean
    elif action_name == 'CleanObject':
        return 'object_cleaned'
    
    # Slice
    elif action_name == 'SliceObject':
        return 'object_sliced'
    
    # Fin de tâche
    elif subgoal_idx == total_subgoals - 1:
        return 'task_completed'
    
    # Par défaut
    else:
        return 'subgoal_completed'


def generate_thought_for_low_action(low_action, frame_idx, total_frames, last_high_action):
    """
    Génère thought pour action low-level
    
    Args:
        low_action: dict avec 'api_action'
        frame_idx: index du frame
        total_frames: nombre total de frames dans le subgoal
        last_high_action: dernière action high-level
    
    Returns:
        thought_name: str
    """
    
    action_name = low_action['api_action']['action']
    
    # Navigation low-level
    if action_name in ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']:
        if frame_idx < total_frames * 0.3:
            return 'need_to_navigate'
        else:
            return 'location_reached'
    
    # Pickup low-level
    elif action_name == 'PickupObject':
        return 'object_picked_up'
    
    # Put low-level
    elif action_name == 'PutObject':
        return 'object_placed'
    
    # Open/Close
    elif action_name in ['OpenObject', 'CloseObject']:
        return 'container_opened'
    
    # Toggle
    elif action_name in ['ToggleObjectOn', 'ToggleObjectOff']:
        return 'need_to_toggle'
    
    # Par défaut: utiliser le high-level action
    if last_high_action:
        return generate_thought_for_action(
            last_high_action, 
            None, 
            None, 
            0, 
            1
        )
    
    return 'subgoal_completed'


def add_error_thoughts(thoughts, actions_success):
    """
    Ajoute des thoughts d'erreur là où actions échouent
    
    Args:
        thoughts: list of thought names
        actions_success: list of bool (True si action réussie)
    
    Returns:
        thoughts: list modifiée
    """
    
    for i, success in enumerate(actions_success):
        if not success:
            # Remplacer par action_failed
            thoughts[i] = 'action_failed'
            
            # Marquer besoin de replanning si échecs répétés
            if i > 0 and not actions_success[i-1]:
                thoughts[i] = 'replanning_required'
    
    return thoughts


# ═══════════════════════════════════════════════════════════
# TRAITEMENT DES TRAJECTOIRES
# ═══════════════════════════════════════════════════════════

def process_trajectory(traj):
    """
    Génère thoughts pour une trajectoire complète
    
    Args:
        traj: dict avec 'plan' et 'images'
    
    Returns:
        dict avec:
            - 'high_thoughts': list of thought names pour high-level
            - 'high_thought_indices': list of thought indices
            - 'low_thoughts': list of thought names pour low-level
            - 'low_thought_indices': list of thought indices
    """
    
    high_thoughts = []
    low_thoughts = []
    
    # ═══════════════════════════════════════════════════════════
    # THOUGHTS HIGH-LEVEL
    # ═══════════════════════════════════════════════════════════
    
    if 'high_pddl' in traj['plan']:
        high_actions = traj['plan']['high_pddl']
        total_high = len(high_actions)
        
        for i, action in enumerate(high_actions):
            prev_action = high_actions[i-1] if i > 0 else None
            next_action = high_actions[i+1] if i < total_high - 1 else None
            
            thought = generate_thought_for_action(
                action,
                prev_action,
                next_action,
                i,
                total_high
            )
            
            high_thoughts.append(thought)
    
    # ═══════════════════════════════════════════════════════════
    # THOUGHTS LOW-LEVEL
    # ═══════════════════════════════════════════════════════════
    
    if 'low_actions' in traj['plan']:
        low_actions = traj['plan']['low_actions']
        
        current_high_action = None
        current_high_idx = 0
        
        for i, low_action in enumerate(low_actions):
            # Trouver le high-level action correspondant
            if 'high_idx' in low_action:
                high_idx = low_action['high_idx']
                if high_idx != current_high_idx:
                    current_high_idx = high_idx
                    if high_idx < len(traj['plan']['high_pddl']):
                        current_high_action = traj['plan']['high_pddl'][high_idx]
            
            # Compter frames dans ce subgoal
            subgoal_frames = sum(
                1 for la in low_actions 
                if la.get('high_idx') == current_high_idx
            )
            frame_in_subgoal = sum(
                1 for la in low_actions[:i] 
                if la.get('high_idx') == current_high_idx
            )
            
            thought = generate_thought_for_low_action(
                low_action,
                frame_in_subgoal,
                subgoal_frames,
                current_high_action
            )
            
            low_thoughts.append(thought)
    
    # ═══════════════════════════════════════════════════════════
    # AJOUTER ERREURS (si info disponible)
    # ═══════════════════════════════════════════════════════════
    
    # ALFRED n'a pas toujours les infos de succès
    # On peut les simuler ou les laisser pour plus tard
    
    # Convertir en indices
    high_thought_indices = [THOUGHT_VOCAB[t] for t in high_thoughts]
    low_thought_indices = [THOUGHT_VOCAB[t] for t in low_thoughts]
    
    return {
        'high_thoughts': high_thoughts,
        'high_thought_indices': high_thought_indices,
        'low_thoughts': low_thoughts,
        'low_thought_indices': low_thought_indices
    }


def process_split(split_file, data_path, split_name):
    """
    Traite un split complet (train/valid_seen/valid_unseen)
    
    Args:
        split_file: chemin vers splits/oct21.json
        data_path: chemin vers data/json_feat_2.1.0
        split_name: 'train', 'valid_seen', ou 'valid_unseen'
    
    Returns:
        annotations: dict avec task_id → thoughts
    """
    
    # Charger split
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    # Les splits ALFRED peuvent avoir différents formats
    split_data = splits.get(split_name, [])
    
    if not split_data:
        print(f"Warning: No data found for split '{split_name}'")
        return {}
    
    # Extraire les task_ids selon le format
    task_ids = []
    
    first_item = split_data[0]
    
    if isinstance(first_item, str):
        # Format simple: liste de strings directement
        task_ids = split_data
        print(f"Format: List of strings (simple)")
        
    elif isinstance(first_item, dict):
        # Format dict: besoin d'extraire les task_ids
        print(f"Format: List of dicts")
        print(f"Sample keys: {list(first_item.keys())}")
        
        # Essayer différentes clés possibles
        possible_keys = ['task', 'task_id', 'task_name', 'id', 'name']
        key_to_use = None
        
        for key in possible_keys:
            if key in first_item:
                key_to_use = key
                break
        
        if key_to_use:
            print(f"Using key: '{key_to_use}'")
            task_ids = [item[key_to_use] for item in split_data]
        else:
            # Dernier recours: prendre la première valeur qui ressemble à un task_id
            print(f"No standard key found, trying heuristic...")
            for item in split_data:
                found = False
                for key, value in item.items():
                    if isinstance(value, str) and ('trial' in value.lower() or 'task' in value.lower()):
                        task_ids.append(value)
                        found = True
                        break
                if not found:
                    # Si vraiment rien ne marche, prendre la première string
                    for key, value in item.items():
                        if isinstance(value, str):
                            task_ids.append(value)
                            break
    else:
        print(f"Error: Unknown format for split data")
        print(f"First item type: {type(first_item)}")
        print(f"First item: {first_item}")
        return {}
    
    if not task_ids:
        print(f"Error: Could not extract task_ids from split '{split_name}'")
        return {}
    
    annotations = {}
    stats = defaultdict(int)
    
    print(f"\n{'='*70}")
    print(f"Processing {split_name}: {len(task_ids)} tasks")
    print(f"{'='*70}\n")
    
    for task_id in tqdm(task_ids, desc=f"{split_name}"):
        # Charger trajectoire
        traj_path = Path(data_path) / split_name / task_id / 'traj_data.json'
        
        if not traj_path.exists():
            stats['missing'] += 1
            continue
        
        try:
            with open(traj_path, 'r') as f:
                traj = json.load(f)
            
            # Générer thoughts
            thoughts = process_trajectory(traj)
            
            annotations[task_id] = thoughts
            
            # Stats
            stats['processed'] += 1
            stats['high_thoughts'] += len(thoughts['high_thoughts'])
            stats['low_thoughts'] += len(thoughts['low_thoughts'])
            
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            stats['errors'] += 1
    
    # Afficher stats
    print(f"\n{'='*70}")
    print(f"Statistics for {split_name}:")
    print(f"{'='*70}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    return annotations


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate thought annotations for ALFRED'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/json_feat_2.1.0',
        help='Path to ALFRED data'
    )
    parser.add_argument(
        '--splits',
        type=str,
        default='data/splits/oct21.json',
        help='Path to splits file'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'valid_seen', 'valid_unseen', 'all'],
        default='all',
        help='Which split to process'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/thoughts_annotations.json',
        help='Output file'
    )
    args = parser.parse_args()
    
    # Setup paths
    ALFRED_ROOT = Path.home() / "Bureau" / "Alfred" / "alfred"
    data_path = ALFRED_ROOT / args.data
    split_file = ALFRED_ROOT / args.splits
    output_path = ALFRED_ROOT / args.output
    
    print("\n" + "="*70)
    print("THOUGHT ANNOTATION GENERATOR")
    print("="*70)
    print(f"Data: {data_path}")
    print(f"Splits: {split_file}")
    print(f"Output: {output_path}")
    print("="*70 + "\n")
    
    # Process splits
    all_annotations = {}
    
    splits_to_process = (
        ['train', 'valid_seen', 'valid_unseen'] 
        if args.split == 'all' 
        else [args.split]
    )
    
    for split_name in splits_to_process:
        annotations = process_split(split_file, data_path, split_name)
        all_annotations[split_name] = annotations
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Annotations saved to: {output_path}")
    print(f"{'='*70}\n")
    
    # Summary
    total_tasks = sum(len(annotations) for annotations in all_annotations.values())
    total_high = sum(
        sum(len(t['high_thoughts']) for t in annotations.values())
        for annotations in all_annotations.values()
    )
    total_low = sum(
        sum(len(t['low_thoughts']) for t in annotations.values())
        for annotations in all_annotations.values()
    )
    
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total tasks annotated: {total_tasks}")
    print(f"Total high-level thoughts: {total_high}")
    print(f"Total low-level thoughts: {total_low}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()