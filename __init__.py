# alfred_experiments/__init__.py

import os
import sys

# Ajouter alfred/ au PYTHONPATH
ALFRED_ROOT = os.path.join(os.path.dirname(__file__), '..', 'alfred')
if ALFRED_ROOT not in sys.path:
    sys.path.insert(0, ALFRED_ROOT)

# Variables d'environnement
os.environ['ALFRED_ROOT'] = ALFRED_ROOT
os.environ['ALFRED_EXP_ROOT'] = os.path.dirname(__file__)

print(f"✓ ALFRED_ROOT set to: {ALFRED_ROOT}")
print(f"✓ ALFRED_EXP_ROOT set to: {os.environ['ALFRED_EXP_ROOT']}")

__version__ = '0.1.0'