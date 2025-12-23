"""
ReAct (Reasoning + Acting) Models for ALFRED

Ce package contient les modèles basés sur ReAct (Yao et al. 2022)
qui combinent raisonnement et action avec feedback environnemental.

Usage:
    from models.React import Module
    from models.React import MODEL_INFO
"""

import sys
from pathlib import Path

# ════════════════════════════════════════════════════════════════════
# Configuration des Chemins
# ════════════════════════════════════════════════════════════════════

# Ajouter le dossier actuel au PYTHONPATH pour imports relatifs
REACT_DIR = Path(__file__).parent
if str(REACT_DIR) not in sys.path:
    sys.path.insert(0, str(REACT_DIR))

# ════════════════════════════════════════════════════════════════════
# Import du Module Principal
# ════════════════════════════════════════════════════════════════════

# Import avec gestion d'erreur
Module = None
seq2seq_react_light = None

try:
    # Import du module seq2seq_react_light
    from . import seq2seq_react_light
    
    # Import de la classe Module principale
    from .seq2seq_react_light import Module
    
    _IMPORT_SUCCESS = True
    _IMPORT_ERROR = None
    
except ImportError as e:
    _IMPORT_SUCCESS = False
    _IMPORT_ERROR = str(e)
    
    # Warning mais pas d'erreur fatale
    import warnings
    warnings.warn(
        f"Could not import seq2seq_react_light Module: {e}\n"
        f"ReAct model may not be available for training.",
        ImportWarning
    )

# ════════════════════════════════════════════════════════════════════
# Métadonnées du Modèle
# ════════════════════════════════════════════════════════════════════

__version__ = "1.0.0"
__author__ = "ALFRED Experiments"

MODEL_INFO = {
    'name': 'ReAct-Light (Reasoning + Acting)',
    'version': __version__,
    'paper': 'Yao et al. (2022) - ReAct: Synergizing Reasoning and Acting in Language Models',
    'description': (
        'Modèle seq2seq avec observation environnementale après chaque action, '
        'génération de thoughts explicites, et replanning dynamique en cas d\'échec. '
        'Extension du modèle CoT avec feedback loop.'
    ),
    'features': {
        'high_level_planning': True,
        'environment_feedback': True,
        'observation_encoding': True,
        'thought_generation': True,
        'dynamic_replanning': True,
        'error_recovery': True,
    },
    'architecture': {
        'base': 'seq2seq_cot (hérité)',
        'observation_encoder': 'LSTM 128→128 dimensions',
        'thought_generator': 'LSTM + classifier (25 types)',
        'replan_detector': 'Binary classifier (seuil: 0.5)',
        'replanner': 'LSTM avec contexte observation',
        'max_subgoals': 5,
        'max_replans': 3,
    },
    'thoughts': {
        'vocab_size': 25,
        'categories': [
            'Navigation (need_to_navigate, location_reached, location_not_found)',
            'Manipulation (pickup, place, open, close, toggle)',
            'Temperature (heat, cool)',
            'Cleaning/Slicing (clean, slice)',
            'Errors (action_failed, replanning_required, trying_alternative)',
            'Success (subgoal_completed, task_completed)',
        ],
    },
    'training': {
        'inherits_cot': True,
        'cot_loss_weight': 0.5,
        'action_loss_weight': 1.0,
        'react_thought_loss_weight': 0.3,
        'react_replan_loss_weight': 0.3,
        'total_loss': 'action + 0.5*cot + 0.3*thought + 0.3*replan',
    },
    'hyperparameters': {
        'replan_threshold': 0.5,
        'max_replans': 3,
        'observation_dim': 128,
        'hidden_dim': 128,
        'batch_size': 4,
        'learning_rate': 0.001,
        'epochs': 3,
    },
    'performance': {
        'baseline_cot': {
            'success_rate': '0-5%',
            'goal_condition': '20-30%',
            'cot_accuracy': '95%+',
        },
        'react_light_v1': {
            'success_rate': '15-25% (phase 1)',
            'goal_condition': '40-50% (phase 1)',
            'recovery_rate': '50-60%',
            'replanning_activation': 'Correct',
        },
        'react_light_optimized': {
            'success_rate': '30-40% (TARGET)',
            'goal_condition': '50-60%',
            'recovery_rate': '60-70%',
            'with_thoughts': True,
        },
    },
    'improvements_over_cot': [
        'Environment feedback après chaque action',
        'Replanning dynamique (max 3x par épisode)',
        'Génération de thoughts explicites (25 types)',
        'Recovery des erreurs (60%+ taux de récupération)',
        'Amélioration success rate: 0% → 30-40%',
    ],
}

# Vocabulaire de thoughts (pour référence)
THOUGHT_VOCAB = {
    0: "need_to_navigate",
    1: "location_reached",
    2: "location_not_found",
    3: "need_to_pickup",
    4: "object_picked_up",
    5: "object_not_found",
    6: "need_to_place",
    7: "object_placed",
    8: "need_to_open",
    9: "container_opened",
    10: "need_to_close",
    11: "need_to_toggle",
    12: "need_to_heat",
    13: "object_heated",
    14: "need_to_cool",
    15: "object_cooled",
    16: "need_to_clean",
    17: "object_cleaned",
    18: "need_to_slice",
    19: "object_sliced",
    20: "action_failed",
    21: "replanning_required",
    22: "trying_alternative",
    23: "subgoal_completed",
    24: "task_completed",
}

# Phases de développement
ROADMAP = {
    'v1.0.0': {
        'status': 'Current',
        'features': ['Basic observation', 'Simple replanning', 'Heuristic thoughts'],
        'target_success': '15-25%',
    },
    'v1.1.0': {
        'status': 'Planned (weeks 2-3)',
        'features': ['Thought annotations', 'Tuned hyperparameters', 'Better observation encoding'],
        'target_success': '30-40%',
    },
    'v2.0.0': {
        'status': 'Future (week 4+)',
        'features': ['Memory mechanism', 'Multi-step replanning', 'Rich observations'],
        'target_success': '45-55%',
    },
    'v3.0.0': {
        'status': 'Long-term',
        'features': ['Transformer-based', 'Multi-modal observations', 'Advanced reasoning'],
        'target_success': '70%+',
    },
}

# ════════════════════════════════════════════════════════════════════
# Exports Publics
# ════════════════════════════════════════════════════════════════════

__all__ = [
    'Module',
    'seq2seq_react_light',
    'MODEL_INFO',
    'THOUGHT_VOCAB',
    'ROADMAP',
    '__version__',
]