# Patches

Patches pour corriger des bugs dans le code ALFRED original.

## patch_alfred_baseline.sh

Patch le baseline ALFRED pour compatibilité.

## patch_baseline_gpu.sh

Corrige les problèmes GPU dans le baseline.

## patch_compute_loss_gpu.sh

Corrige le calcul de loss sur GPU.

## patch_eval_imports.sh

Corrige les imports dans eval_seq2seq.py.

**Usage:**
```bash
cd ~/Bureau/Alfred/alfred
../alfred_experiments/scripts/maintenance/patches/patch_baseline_gpu.sh
```

**⚠️ IMPORTANT:** 
Appliquer ces patches après avoir cloné ALFRED et avant le premier training.
