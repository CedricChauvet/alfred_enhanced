import sys
from pathlib import Path

# ════════════════════════════════════════════════════════════════════
# Configuration des Chemins
# ════════════════════════════════════════════════════════════════════

# Ajouter le dossier actuel au PYTHONPATH pour imports relatifs
COT_DIR = Path(__file__).parent
if str(COT_DIR) not in sys.path:
    sys.path.insert(0, str(COT_DIR))

# ════════════════════════════════════════════════════════════════════
# Import du Module Principal
# ════════════════════════════════════════════════════════════════════

# Import avec gestion d'erreur
Module = None
seq2seq_cot = None

try:
    # Import du module seq2seq_cot
    from . import seq2seq_cot
    
    # Import de la classe Module principale
    from .seq2seq_cot import Module
    
    _IMPORT_SUCCESS = True
    _IMPORT_ERROR = None
    
except ImportError as e:
    _IMPORT_SUCCESS = False
    _IMPORT_ERROR = str(e)
    
    # Warning mais pas d'erreur fatale
    import warnings
    warnings.warn(
        f"Could not import seq2seq_cot Module: {e}\n"
        f"CoT model may not be available for training.",
        ImportWarning
    )

# ════════════════════════════════════════════════════════════════════
# Métadonnées du Modèle
# ════════════════════════════════════════════════════════════════════

__version__ = "1.0.0"
__author__ = "ALFRED Experiments"

MODEL_INFO = {
    'name': 'Chain-of-Thought (CoT)',
    'version': __version__,
    'paper': 'Wei et al. (2022) - Chain-of-Thought Prompting Elicits Reasoning in Large Language Models',
    'description': (
        'Modèle seq2seq avec génération de plan high-level (subgoals) '
        'avant l\'exécution des actions low-level. '
        'Pas de feedback environnemental.'
    ),
    'features': {
        'high_level_planning': True,
        'environment_feedback': False,
        'dynamic_replanning': False,
        'thought_generation': False,
    },
    'architecture': {
        'encoder': 'LSTM bidirectionnel',
        'decoder': 'LSTM avec attention',
        'subgoal_decoder': 'LSTM séparé pour high-level',
        'max_subgoals': 5,
    },
    'training': {
        'two_stage': True,
        'cot_loss_weight': 1.0,
        'action_loss_weight': 1.0,
    },
    'performance': {
        'cot_accuracy': '~95%',
        'success_rate': '~5-10% (baseline)',
        'goal_condition': '~20-30%',
    }
}

__all__ = ['Module', 'seq2seq_cot', 'MODEL_INFO', '__version__']