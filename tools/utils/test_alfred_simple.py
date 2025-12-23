"""
Teste un environnement ALFRED simple en Python. verifie que tout est installé correctement et que l'environnement peut être initialisé.
"""

# test_alfred_simple.py
import os
import sys

# Ajouter le chemin ALFRED aux imports
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from env.thor_env import ThorEnv

# Créer un environnement
print("Initialisation de l'environnement THOR...")
env = ThorEnv()

# Réinitialiser avec une scène
print("Chargement d'une scène...")
env.reset('FloorPlan1')

# Obtenir une observation
print("Récupération d'une frame...")
event = env.last_event
frame = event.frame
print(f"Taille de la frame : {frame.shape}")

print("Test réussi !")