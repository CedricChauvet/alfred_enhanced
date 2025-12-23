cat > configs/README.md << 'EOF'
# Configurations ALFRED

## Configs de Test

### test_mini.yaml
**Durée**: ~2-3 minutes  
**Usage**: Vérifier que tout fonctionne (imports, training loop, etc.)  
**Specs**: 1 epoch, batch=2, dhid=128, fast_epoch
```bash
./scripts/train.sh configs/test_mini.yaml
```

### test_quick.yaml
**Durée**: ~5-10 minutes  
**Usage**: Test plus complet avant entraînement long  
**Specs**: 2 epochs, batch=4, dhid=128, fast_epoch
```bash
./scripts/train.sh configs/test_quick.yaml
```

---

## Configs Complètes

### cot_v1.yaml
**Durée**: ~8-10 heures  
**Usage**: Entraînement complet du modèle CoT  
**Specs**: 20 epochs, batch=8, dhid=512, CoT enabled
```bash
./scripts/train.sh configs/cot_v1.yaml
```

### baseline.yaml
**Durée**: ~6-8 heures  
**Usage**: Reproduction du baseline pour comparaison  
**Specs**: 20 epochs, batch=8, dhid=512, CoT disabled
```bash
./scripts/train.sh configs/baseline.yaml
```

---

## Workflow Recommandé

1. **Test mini** (obligatoire) : Vérifier setup
```bash
   ./scripts/train.sh configs/test_mini.yaml
```

2. **Test quick** (recommandé) : Vérifier convergence
```bash
   ./scripts/train.sh configs/test_quick.yaml
```

3. **Entraînement complet** : Lancer la vraie expérience
```bash
   ./scripts/train.sh configs/cot_v1.yaml
```

---

## Paramètres Importants

### Architecture
- `dhid`: Taille cachée LSTM (128=test, 512=standard)
- `demb`: Taille embeddings (50=test, 100=standard)
- `dframe`: Taille features visuelles (2500 fixe)

### Chain-of-Thought
- `use_cot`: true/false
- `max_subgoals`: Nombre max de sous-buts (5-15)
- `cot_loss_weight`: Poids de la loss CoT (0.3-0.8)

### Training
- `batch`: Taille du batch (2-8)
- `epoch`: Nombre d'epochs (1-20)
- `fast_epoch`: Mode rapide (subset dataset)
EOF

echo "✓ README.md créé dans configs/"