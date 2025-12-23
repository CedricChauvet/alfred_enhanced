"""
generate_thought_dataset_v2.py - VERSION FINALE TESTÉE

Structure:
  data/json_2.1.0/train/
    └── task_dir/
        └── trial_*/
            └── traj_data.json

Usage:
    python generate_thought_dataset_v2.py \
        --data /path/to/json_2.1.0 \
        --split train \
        --output /path/to/output
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


THOUGHT_VOCAB = {
    0: "navigating_to_target_object",
    1: "navigating_to_receptacle",
    2: "navigating_with_object_in_hand",
    3: "reached_destination",
    10: "picking_up_object_to_slice",
    11: "picking_up_object_to_cook",
    12: "picking_up_object_to_clean",
    13: "picking_up_object_to_place",
    14: "picking_up_sliced_object",
    15: "picking_up_cooked_object",
    16: "picked_up_target_object",
    20: "placing_object_in_receptacle",
    21: "placing_object_for_cooking",
    22: "placing_object_for_cooling",
    23: "placing_sliced_object",
    24: "object_placed_successfully",
    30: "opening_container_to_retrieve",
    31: "opening_container_to_place",
    32: "opening_fridge_to_cool",
    33: "opening_microwave_to_heat",
    34: "container_opened",
    35: "closing_container_after_placing",
    36: "closing_fridge_after_cooling",
    40: "slicing_target_object",
    41: "object_sliced_successfully",
    42: "heating_object_in_microwave",
    43: "object_heated_successfully",
    44: "cooling_object_in_fridge",
    45: "object_cooled_successfully",
    46: "cleaning_object_in_sink",
    47: "object_cleaned_successfully",
    50: "subgoal_completed",
    51: "approaching_task_completion",
    52: "task_completed_successfully",
    60: "action_failed",
    61: "object_not_found",
    62: "replanning_required"
}

THOUGHT_NAME_TO_ID = {v: k for k, v in THOUGHT_VOCAB.items()}


def infer_thought(action, next_action, obj_type, obj_states, task_type, holding_obj):
    action_type = action.get('action', '')
    
    if action_type in ['LookDown', 'LookUp', 'RotateLeft', 'RotateRight', 'MoveAhead']:
        if holding_obj:
            return THOUGHT_NAME_TO_ID["navigating_with_object_in_hand"]
        if next_action:
            next_type = next_action.get('action', '')
            if next_type == 'PickupObject':
                return THOUGHT_NAME_TO_ID["navigating_to_target_object"]
            elif next_type in ['PutObject', 'OpenObject', 'CloseObject']:
                return THOUGHT_NAME_TO_ID["navigating_to_receptacle"]
        return THOUGHT_NAME_TO_ID["navigating_to_target_object"]
    
    if action_type == 'PickupObject':
        if 'slice' in task_type.lower():
            if 'sliced' in obj_states:
                return THOUGHT_NAME_TO_ID["picking_up_sliced_object"]
            return THOUGHT_NAME_TO_ID["picking_up_object_to_slice"]
        elif 'heat' in task_type.lower() or 'cook' in task_type.lower():
            if 'cooked' in obj_states or 'heated' in obj_states:
                return THOUGHT_NAME_TO_ID["picking_up_cooked_object"]
            return THOUGHT_NAME_TO_ID["picking_up_object_to_cook"]
        elif 'clean' in task_type.lower():
            return THOUGHT_NAME_TO_ID["picking_up_object_to_clean"]
        return THOUGHT_NAME_TO_ID["picking_up_object_to_place"]
    
    if action_type == 'PutObject':
        if 'Microwave' in obj_type:
            return THOUGHT_NAME_TO_ID["placing_object_for_cooking"]
        elif 'Fridge' in obj_type:
            return THOUGHT_NAME_TO_ID["placing_object_for_cooling"]
        elif 'sliced' in obj_states:
            return THOUGHT_NAME_TO_ID["placing_sliced_object"]
        return THOUGHT_NAME_TO_ID["placing_object_in_receptacle"]
    
    if action_type == 'OpenObject':
        if 'Fridge' in obj_type:
            if 'cool' in task_type.lower():
                return THOUGHT_NAME_TO_ID["opening_fridge_to_cool"]
            return THOUGHT_NAME_TO_ID["opening_container_to_retrieve"]
        elif 'Microwave' in obj_type:
            return THOUGHT_NAME_TO_ID["opening_microwave_to_heat"]
        else:
            if next_action and next_action.get('action') == 'PutObject':
                return THOUGHT_NAME_TO_ID["opening_container_to_place"]
            return THOUGHT_NAME_TO_ID["opening_container_to_retrieve"]
    
    if action_type == 'CloseObject':
        if 'Fridge' in obj_type:
            return THOUGHT_NAME_TO_ID["closing_fridge_after_cooling"]
        return THOUGHT_NAME_TO_ID["closing_container_after_placing"]
    
    if action_type == 'SliceObject':
        return THOUGHT_NAME_TO_ID["slicing_target_object"]
    
    if action_type == 'ToggleObject':
        if 'Microwave' in obj_type:
            if next_action and next_action.get('action') == 'ToggleObject':
                return THOUGHT_NAME_TO_ID["heating_object_in_microwave"]
            return THOUGHT_NAME_TO_ID["object_heated_successfully"]
        elif 'Faucet' in obj_type or 'Sink' in obj_type:
            return THOUGHT_NAME_TO_ID["cleaning_object_in_sink"]
    
    return THOUGHT_NAME_TO_ID["navigating_to_target_object"]


def generate_thoughts_for_trajectory(traj_data):
    task_id = traj_data['task_id']
    task_type = traj_data['task_type']
    low_actions = traj_data['plan']['low_actions']
    num_steps = len(low_actions)
    
    thoughts = []
    holding_object = False
    
    for i, action in enumerate(low_actions):
        api_action = action['api_action']
        action_type = api_action['action']
        
        next_action = None
        if i + 1 < num_steps:
            next_action = low_actions[i + 1]['api_action']
        
        object_id = api_action.get('objectId', '')
        obj_type = object_id.split('|')[0] if '|' in object_id else 'Unknown'
        
        obj_states = []
        if 'Sliced' in object_id:
            obj_states.append('sliced')
        
        thought_id = infer_thought(
            api_action, next_action, obj_type, obj_states, task_type, holding_object
        )
        
        if action_type == 'PickupObject':
            holding_object = True
        elif action_type == 'PutObject':
            holding_object = False
        
        thoughts.append({
            'step': i,
            'action': action_type,
            'objectId': object_id,
            'object_type': obj_type,
            'object_states': obj_states,
            'thought': THOUGHT_VOCAB[thought_id],
            'thought_id': thought_id,
            'holding_object': holding_object
        })
    
    return {
        'task_id': task_id,
        'task_type': task_type,
        'num_steps': num_steps,
        'thoughts': thoughts
    }


def find_traj_files(split_path):
    """
    TESTÉ ET VALIDÉ avec votre structure!
    """
    split_path = Path(split_path).resolve()
    
    print(f"Searching in: {split_path}")
    
    if not split_path.exists():
        print(f"❌ Path does not exist!")
        return []
    
    traj_files = []
    
    # Lister tous les task dirs
    task_dirs = [x for x in split_path.iterdir() if x.is_dir()]
    print(f"Found {len(task_dirs)} task directories")
    
    # Pour chaque task dir
    for task_dir in task_dirs:
        # Chercher trial_* dedans
        trial_dirs = list(task_dir.glob('trial_*'))
        
        for trial_dir in trial_dirs:
            if not trial_dir.is_dir():
                continue
            
            # Chercher traj_data.json
            traj_file = trial_dir / 'traj_data.json'
            if traj_file.exists():
                traj_files.append(traj_file)
    
    return traj_files


def generate_dataset(data_path, split, output_path):
    print(f"\n{'='*70}")
    print(f"GENERATING THOUGHT DATASET")
    print(f"{'='*70}")
    print(f"Data: {data_path}")
    print(f"Split: {split}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")
    
    data_path = Path(data_path).resolve()
    split_path = data_path / split
    
    if not split_path.exists():
        raise ValueError(f"Split path not found: {split_path}")
    
    # Trouver trajectoires
    trajectories = find_traj_files(split_path)
    print(f"\n✓ Found {len(trajectories)} trajectories\n")
    
    if len(trajectories) == 0:
        print("❌ No trajectories found!")
        return None
    
    results = []
    thought_stats = defaultdict(int)
    
    for traj_file in tqdm(trajectories, desc="Generating thoughts"):
        try:
            with open(traj_file, 'r') as f:
                traj_data = json.load(f)
            
            # IMPORTANT: Reconstruire task_id depuis le PATH, pas depuis traj_data!
            # Path format: .../task_dir/trial_dir/traj_data.json
            trial_dir = traj_file.parent.name  # trial_T...
            task_dir = traj_file.parent.parent.name  # pick_and_place-...
            correct_task_id = f"{task_dir}/{trial_dir}"
            
            # Override task_id avec le bon chemin
            traj_data['task_id'] = correct_task_id
            
            result = generate_thoughts_for_trajectory(traj_data)
            results.append(result)
            
            for thought in result['thoughts']:
                thought_stats[thought['thought_id']] += 1
        
        except Exception as e:
            print(f"\n⚠️  Error: {traj_file}: {e}")
            continue
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f'thoughts_dataset_{split}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"COMPLETED")
    print(f"{'='*70}")
    print(f"✓ Trajectories: {len(results)}")
    print(f"✓ Total thoughts: {sum(thought_stats.values()):,}")
    print(f"✓ Unique classes: {len(thought_stats)}")
    print(f"✓ Output: {output_file}")
    print(f"{'='*70}\n")
    
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    generate_dataset(args.data, args.split, args.output)


if __name__ == '__main__':
    main()