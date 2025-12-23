"""
Loader custom pour charger les modèles ALFRED
Sans modifier alfred/
"""

import sys
import os
from pathlib import Path
import importlib.util

# Paths
ALFRED_ROOT = Path.home() / "Bureau" / "Alfred" / "alfred"
EXP_ROOT = Path(__file__).parent.parent

# Ajouter alfred au path
if str(ALFRED_ROOT) not in sys.path:
    sys.path.insert(0, str(ALFRED_ROOT))


def load_baseline_module():
    """
    Charge le module baseline en corrigeant les imports à la volée
    """
    baseline_path = ALFRED_ROOT / "models" / "model" / "seq2seq_im_mask.py"
    
    # Lire le fichier
    with open(baseline_path, 'r') as f:
        code = f.read()
    
    # Corriger l'import problématique
    code = code.replace('import nn.vnn as vnn', 'from models.nn import vnn')
    
    # Créer un module temporaire
    spec = importlib.util.spec_from_loader('seq2seq_im_mask_fixed', loader=None)
    module = importlib.util.module_from_spec(spec)
    
    # Exécuter le code corrigé
    exec(code, module.__dict__)
    
    return module.Module


def get_baseline_class():
    """
    Retourne la classe BaseModule corrigée
    """
    try:
        return load_baseline_module()
    except Exception as e:
        print(f"Error loading baseline with fix: {e}")
        print("Trying direct import...")
        from models.model.seq2seq_im_mask import Module
        return Module
