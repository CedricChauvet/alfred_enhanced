# Chain of Thoughts (CoT) pour ALFRED

**Auteur** : Cedrix  
**Date** : 2025  
**Base** : ALFRED Benchmark

---

## 🎯 Objectif

Améliorer les performances du modèle baseline ALFRED en introduisant une génération explicite de **subgoals** (sous-objectifs) pour améliorer le planning et la décomposition des tâches.

---

## 📊 Expériences

### CoT v1
- **Configuration** : `configs/cot_v1.yaml`
- **Modèle** : `alfred_experiments.models.seq2seq_cot`
- **Ajouts** : Génération explicite de subgoals
- **Objectif** : Améliorer planning et décomposition des tâches

### CoT v2
- **Configuration** : `configs/cot_v2_exploration.yaml`
- **Variations** : Plus de subgoals, loss weight plus élevé
- **Objectif** : Explorer les hyperparamètres CoT

---

## 🏗️ Architecture

### Vue d'ensemble CoT v1
Le modèle CoT v1 ajoute **5 nouveaux layers** (~252K paramètres) au modèle baseline (~10M paramètres) :

1. **project_cont** - Projection du contexte linguistique
2. **project_subgoals** - Projection des subgoals
3. **subgoal_decoder** - LSTM pour génération auto-régressive
4. **subgoal_classifier** - Classification des subgoals
5. **emb_subgoal** - Embedding des indices de subgoals

En résumé, l'ajout par rapport a la baseline et de créer la CoT qui est une liste d'actions de ce type:

Exemple 1 : "Put a heated apple in the fridge"
Subgoals (high-level actions) :

GotoLocation (Counter/Table)
PickupObject (Apple)
GotoLocation (Microwave)
PutObject (Apple in Microwave)
ToggleObject (Microwave ON)
ToggleObject (Microwave OFF)
PickupObject (Apple from Microwave)
GotoLocation (Fridge)
OpenObject (Fridge)
PutObject (Apple in Fridge)


Cette liste est concaténée avec la sortie de l'encoder

### Vue d'ensemble CoT_ProgressMonitor with Attention
Le modele CoT_pm_attention est un peu plus élaboré:
il utilise