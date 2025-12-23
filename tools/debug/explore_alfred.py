#!/usr/bin/env python
"""
Script interactif pour explorer ALFRED
"""
import os
import sys

ALFRED_ROOT = os.environ.get('ALFRED_ROOT', os.getcwd())
sys.path.insert(0, ALFRED_ROOT)

from env.thor_env import ThorEnv
import json

def explore_scene():
    """Exploration interactive d'une scène"""
    print("=== Exploration interactive ALFRED ===\n")
    
    # Créer l'environnement
    env = ThorEnv()
    
    # Liste des scènes disponibles
    scenes = [f"FloorPlan{i}" for i in range(1, 31)]
    
    print(f"Scènes disponibles: {scenes[0]} à {scenes[-1]}")
    scene_name = input(f"Choisir une scène (défaut: FloorPlan1): ").strip() or "FloorPlan1"
    
    print(f"\nChargement de {scene_name}...")
    env.reset(scene_name)
    
    print("✓ Scène chargée!")
    print(f"Position de l'agent: {env.last_event.metadata['agent']}")
    
    # Actions disponibles
    actions = [
        "MoveAhead", "RotateLeft", "RotateRight", 
        "LookUp", "LookDown", "Done"
    ]
    
    print("\nActions disponibles:")
    for i, action in enumerate(actions):
        print(f"  {i+1}. {action}")
    
    # Boucle interactive
    step = 0
    while True:
        step += 1
        print(f"\n--- Step {step} ---")
        choice = input("Action (1-6, ou 'q' pour quitter): ").strip()
        
        if choice.lower() == 'q':
            break
        
        try:
            action_idx = int(choice) - 1
            if 0 <= action_idx < len(actions):
                action_name = actions[action_idx]
                
                if action_name == "Done":
                    print("Terminé!")
                    break
                
                # Exécuter l'action
                event = env.step({'action': action_name})
                
                if event.metadata['lastActionSuccess']:
                    print(f"✓ {action_name} réussi")
                    print(f"Position: {event.metadata['agent']['position']}")
                    print(f"Rotation: {event.metadata['agent']['rotation']}")
                else:
                    print(f"✗ {action_name} échoué")
            else:
                print("Choix invalide")
        except ValueError:
            print("Entrée invalide")
    
    print("\nFin de l'exploration")
    env.stop()

if __name__ == "__main__":
    explore_scene()

