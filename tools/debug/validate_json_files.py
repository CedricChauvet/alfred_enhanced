import os
import json
from pathlib import Path

def validate_json_files(data_dir):
    """Valide tous les fichiers JSON dans le répertoire de données"""
    corrupted_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    corrupted_files.append({
                        'file': filepath,
                        'error': str(e)
                    })
                    print(f"❌ Fichier corrompu: {filepath}")
                    print(f"   Erreur: {e}\n")
    
    if not corrupted_files:
        print("✅ Tous les fichiers JSON sont valides!")
    else:
        print(f"\n⚠️  {len(corrupted_files)} fichier(s) corrompu(s) trouvé(s)")
    
    return corrupted_files

if __name__ == "__main__":
    data_dir = "data/full_2.1.0"  # Ajustez selon votre config
    corrupted = validate_json_files(data_dir)