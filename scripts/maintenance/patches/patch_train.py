#!/usr/bin/env python3
"""
Patch minimal pour run_experiment.py
Corrige UNIQUEMENT le chemin experiments (pas alfred/)
"""
import sys
from pathlib import Path
import re

script_path = Path.home() / "Bureau/Alfred/alfred_experiments/scripts/training/run_experiment.py"

if not script_path.exists():
    print(f"❌ {script_path} non trouvé")
    sys.exit(1)

print("Patching run_experiment.py (version safe)...")

content = script_path.read_text()

# Backup si pas déjà fait
backup = script_path.with_suffix('.py.backup_original')
if not backup.exists():
    backup.write_text(content)
    print(f"✓ Backup: {backup}")

# Fix SEULEMENT la ligne qui crée exp_dir avec "experiments"
# On cherche spécifiquement: exp_dir = Path("experiments") / quelque_chose
# Et on remplace par un chemin relatif correct

# Pattern: exp_dir = Path("experiments") / exp_name
# Remplacer par: exp_dir = Path("../../experiments") / exp_name

old_pattern = r'exp_dir\s*=\s*Path\(["\']experiments["\']\)'
new_code = 'exp_dir = Path("../../experiments")'

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_code, content)
    print("✓ Fix: exp_dir corrigé (experiments → ../../experiments)")
else:
    print("⚠️  Pattern exp_dir non trouvé, peut-être déjà patché?")

# Fix 2: TypeError avec dict (safe)
if 'f.write(" ".join(cmd)' in content and 'str(x) for x' not in content:
    content = content.replace(
        '" ".join(cmd)',
        '" ".join(str(x) for x in cmd)'
    )
    print("✓ Fix: TypeError dict corrigé")

# Sauvegarder
script_path.write_text(content)

print("")
print("✅ Patch appliqué!")
print("")
print("Structure attendue:")
print("  alfred_experiments/")
print("  ├── scripts/training/run_experiment.py  ← Vous êtes ici")
print("  ├── experiments/                        ← Créé ici (../../)")
print("  └── ...")
print("")
print("Testez:")
print("  ./scripts/training/train.sh configs/react_light_v1.yaml")