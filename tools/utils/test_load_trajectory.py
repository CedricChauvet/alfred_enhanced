#!/usr/bin/env python
import os
import sys
import json

# Configuration
ALFRED_ROOT = os.environ.get('ALFRED_ROOT', os.getcwd())
sys.path.insert(0, ALFRED_ROOT)

from env.thor_env import ThorEnv

def test_trajectory():
    """Teste le chargement d'une trajectoire ALFRED"""
    
    # Chemin vers les données
    data_path = os.path.join(ALFRED_ROOT, 'data', 'json_feat_2.1.0')
    
    # Vérifier que les données existent
    if not os.path.exists(data_path):
        print(f"✗ Données non trouvées dans {data_path}")
        print("Téléchargez d'abord les données avec: cd data && sh download_data.sh json_feat")
        return
    
    print(f"✓ Données trouvées dans {data_path}")
    
    # Trouver une trajectoire d'exemple
    train_path = os.path.join(data_path, 'train')
    if os.path.exists(train_path):
        # Prendre la première trajectoire
        tasks = os.listdir(train_path)
        if tasks:
            task_path = os.path.join(train_path, tasks[0])
            trials = os.listdir(task_path)
            if trials:
                traj_path = os.path.join(task_path, trials[0], 'traj_data.json')
                
                print(f"\nChargement de la trajectoire: {traj_path}")
                
                with open(traj_path, 'r') as f:
                    traj_data = json.load(f)
                
                print("\n=== Informations sur la trajectoire ===")
                print(f"Task type: {traj_data['task_type']}")
                print(f"Scene: {traj_data['scene']['floor_plan']}")
                
                # Instructions
                if 'turk_annotations' in traj_data:
                    ann = traj_data['turk_annotations']['anns'][0]
                    print(f"\nGoal: {ann['task_desc']}")
                    print(f"\nInstructions ({len(ann['high_descs'])} steps):")
                    for i, instr in enumerate(ann['high_descs'][:5]):  # 5 premières
                        print(f"  {i+1}. {instr}")
                    if len(ann['high_descs']) > 5:
                        print(f"  ... ({len(ann['high_descs']) - 5} étapes supplémentaires)")
                
                # Actions
                if 'plan' in traj_data:
                    actions = traj_data['plan']['low_actions']
                    print(f"\nNombre d'actions: {len(actions)}")
                    print(f"Première action: {actions[0]['discrete_action']['action']}")
                
                print("\n✓ Trajectoire chargée avec succès!")
                return True
    
    print("✗ Aucune trajectoire trouvée")
    return False

if __name__ == "__main__":
    test_trajectory()