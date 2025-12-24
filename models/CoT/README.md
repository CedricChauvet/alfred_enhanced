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

----------------------------------------

## 🚀 Training :


Voici le Yaml pour un entrainement du CoT: 
[Configuration CoT v1](https://github.com/CedricChauvet/alfred_enhanced/blob/main/configs/cot_v1.yaml)



```bash

# Pour un train:
cd $ALFRED_ROOT
./scripts/train.sh ./config/cot_v1.yaml
```
---------------------------------------
## 📊 Monitoring avec TensorBoard

### Qu'est-ce que TensorBoard ?

TensorBoard est l'outil de visualisation de TensorFlow/PyTorch qui permet de suivre en temps réel l'entraînement de vos modèles. Il affiche :

- **Courbes de loss** : Évolution des pertes d'entraînement et de validation
- **Métriques** : Accuracy, Success Rate, Goal Condition, etc.
- **Graphes** : Architecture du réseau de neurones
- **Histogrammes** : Distribution des poids et gradients
- **Images** : Visualisation des prédictions (optionnel)

### Lancement de TensorBoard
```bash
# Depuis n'importe quel terminal
tensorboard --logdir /chemin/vers/experiments/nom_experience/tensorboard
```

### Accès à l'interface

Une fois lancé, TensorBoard affiche :
```
TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
```
--------------------------------------

## 🧪 Évaluation

### Évaluation sur validation seen
```bash
python models/eval/eval_seq2seq.py 
--model_path experiments/cot_v1/best_seen.pth 
--eval_split valid_seen 
--data data/json_feat_2.1.0 
--model alfred_experiments.models.seq2seq_cot 
--gpu 
--num_threads 2
```

Devrait lancer Thor


----------------------------------------
## 🏗️ Architecture

----------------------------------------

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

----------------------------------------



### Vue d'ensemble CoT_ProgressMonitor with Attention

Le modele CoT_pm_attention est un peu plus élaboré:

Il utilise l'apprentissage du progress monitor qui indique l'avancée des subgaols en pourcentage.

Par exemple 0% aucun subgoals atteints 50% la moitié de la tache est remplie.

couplé avec le CoT, ce modele est capable de predire quelle tache actuelle l'IA doit resoudre.


----------------------------------------


### Differences entre v1 et pm_attention
Tout est  dans la taille de l'encodage, v1 concatène un vecteur de taille max_subgoals=12

pm_attention concatene lui aussi a la sortie de l'encodeur mais seulement un élément (par exemple go to location, ou pickup)

En résume le modele sait ce qu'il doit faire a chaque instant.



----------------------------------------