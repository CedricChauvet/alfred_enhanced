"""
Script pour lancer des expériences ALFRED
Avec logs complets dans experiments/*/logs/
Version avec support des variables d'environnement (.env)
"""

import os
import sys
import subprocess
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# ============================================================================
# Charger les variables d'environnement
# ============================================================================
def get_env_paths():
    """Récupère les chemins depuis les variables d'environnement ou utilise les valeurs par défaut"""
    
    # Essayer de récupérer depuis les variables d'environnement
    alfred_root = os.environ.get('ALFRED_ROOT')
    
    # Si pas défini, auto-détecter ALFRED_ROOT
    if not alfred_root:
        # Remonter de 2 niveaux depuis ce script (scripts/training/run_experiment.py -> racine)
        script_dir = Path(__file__).resolve().parent
        alfred_root = script_dir.parent.parent
        print(f"⚠️  ALFRED_ROOT non défini, auto-détecté: {alfred_root}")
        print(f"   Pour éviter ce message, faites: source .env")
    
    return Path(alfred_root)

# Récupérer les chemins
ALFRED_ROOT = get_env_paths()
EXP_ROOT = ALFRED_ROOT / "experiments"

print("="*70)
print("ALFRED EXPERIMENT RUNNER")
print("="*70)
print(f"ALFRED_ROOT: {ALFRED_ROOT}")
print(f"Experiments will be saved in: {EXP_ROOT}")
print("="*70 + "\n")

def create_experiment_dir(exp_name):
    """Crée un dossier d'expérience avec timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name_full = f"{exp_name}_{timestamp}"
    
    exp_dir = EXP_ROOT / exp_name_full
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'tensorboard').mkdir(exist_ok=True)
    
    return exp_dir


def load_config(config_path):
    """Charge config YAML et résout les variables d'environnement"""
    
    # S'assurer que ALFRED_ROOT est défini pour expandvars AVANT de l'utiliser
    if 'ALFRED_ROOT' not in os.environ:
        os.environ['ALFRED_ROOT'] = str(ALFRED_ROOT)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Résoudre les variables d'environnement dans tous les chemins
    path_keys = ['data', 'splits', 'dout', 'resume', 'model_path', 'checkpoint']
    for key in path_keys:
        if key in config and isinstance(config[key], str):
            original = config[key]
            # Expand variables like $ALFRED_ROOT
            expanded = os.path.expandvars(config[key])
            config[key] = expanded
            
            # Log si une variable a été étendue
            if original != expanded and '$' in original:
                print(f"   Resolved: {key}")
                print(f"      {original}")
                print(f"   -> {expanded}")
    
    return config

def build_train_command(config, exp_dir):
    """Construit la ligne de commande pour train_seq2seq.py"""
    cmd = [
        'python',
        'models/train/train_seq2seq.py',
    ]
    
    # Gérer data (peut être dict ou string)
    if 'data' in config:
        data = config['data']
        if isinstance(data, dict):
            # Si c'est un dict, utiliser le chemin par défaut
            data_path = ALFRED_ROOT / 'data/json_feat_2.1.0'
        else:
            # Le chemin peut déjà contenir $ALFRED_ROOT qui a été résolu par load_config
            # Ou être un chemin relatif/absolu
            data_str = str(data)
            
            # Si le chemin n'est toujours pas résolu (contient encore $), le résoudre
            if '$' in data_str:
                data_str = os.path.expandvars(data_str)
            
            # Si chemin relatif, résoudre depuis ALFRED_ROOT
            if not os.path.isabs(data_str):
                data_path = ALFRED_ROOT / data_str
            else:
                data_path = Path(data_str)
        cmd.extend(['--data', str(data_path)])
    else:
        data_path = ALFRED_ROOT / 'data/json_feat_2.1.0'
        cmd.extend(['--data', str(data_path)])
    
    # Résoudre splits path
    splits = config['splits']
    splits_str = str(splits)
    
    # Si le chemin contient encore $, le résoudre
    if '$' in splits_str:
        splits_str = os.path.expandvars(splits_str)
    
    # Si chemin relatif, résoudre depuis ALFRED_ROOT
    if not os.path.isabs(splits_str):
        splits_path = ALFRED_ROOT / splits_str
    else:
        splits_path = Path(splits_str)
    
    # Reste de la commande
    cmd.extend([
        '--splits', str(splits_path),
        '--model', str(config['model']),
        '--dout', str(exp_dir / 'checkpoints'),
        '--batch', str(config.get('batch', 8)),
        '--epoch', str(config.get('epoch', 20)),
        '--lr', str(config.get('lr', 0.0001)),
        '--seed', str(config.get('seed', 1)),
    ])
    
    # Ajouter le chemin TensorBoard
    tensorboard_dir = exp_dir / 'tensorboard'
    cmd.extend(['--tensorboard_dir', str(tensorboard_dir)])
    
    # Gérer resume (checkpoint pour continuer l'entraînement)
    if 'resume' in config and config['resume']:
        resume_path = str(config['resume'])
        
        # Résoudre les variables d'environnement si présentes
        if '$' in resume_path:
            resume_path = os.path.expandvars(resume_path)
        
        # Si le chemin n'est pas absolu, le résoudre depuis ALFRED_ROOT
        if not os.path.isabs(resume_path):
            resume_path = ALFRED_ROOT / resume_path
        cmd.extend(['--resume', str(resume_path)])
    
    # Options booléennes
    if config.get('gpu', False):
        cmd.append('--gpu')
    
    if config.get('fast_epoch', False):
        cmd.append('--fast_epoch')
    
    if config.get('save_every_epoch', False):
        cmd.append('--save_every_epoch')
    
    if config.get('preprocess', False):
        cmd.append('--preprocess')
    
    if config.get('dec_teacher_forcing', False):
        cmd.append('--dec_teacher_forcing')
    
    # Hyperparamètres
    standard_args = [
        'dhid', 'demb', 'dframe', 'pframe', 'decay_epoch',
        'action_loss_wt', 'mask_loss_wt', 'pm_aux_loss_wt', 
        'subgoal_aux_loss_wt', 'vis_dropout', 'lang_dropout',
        'input_dropout', 'hstate_dropout', 'attn_dropout',
        'actor_dropout'
    ]
    
    for key in standard_args:
        if key in config:
            cmd.extend([f'--{key}', str(config[key])])
    
    return cmd


def run_experiment(config_path):
    """Lance une expérience complète"""
    
    config = load_config(config_path)
    exp_name = config.get('exp_name', 'unnamed')
    model_name = config.get('model', 'N/A')
    
    # ✓✓✓ FIX: Si resume, utiliser le dossier existant ✓✓✓
    if 'resume' in config and config['resume'] and 'dout' in config and config['dout']:
        # Mode RESUME: utiliser le dossier existant spécifié dans dout
        exp_dir = Path(config['dout'])
        print(f"\n✓ RESUME MODE: Using existing directory")
        print(f"  {exp_dir}")
    else:
        # Mode NOUVEAU: créer un nouveau dossier avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name_full = f"{exp_name}_{timestamp}"
        exp_dir = EXP_ROOT / exp_name_full
        print(f"\n✓ NEW EXPERIMENT: Creating directory")
        print(f"  {exp_dir}")
    
    # Créer les sous-dossiers si nécessaire
    exp_dir.mkdir(parents=True, exist_ok=True)

    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {exp_name}")
    print("="*70)
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Model: {model_name}")
    print(f"CoT: {config.get('use_cot', False)}")
    
    # Afficher info sur resume si présent
    if 'resume' in config and config['resume']:
        resume_path = config['resume']
        if not os.path.isabs(resume_path):
            resume_path = ALFRED_ROOT / resume_path
        print(f"Resume from: {resume_path}")
        
        # Vérifier que le checkpoint existe
        if not os.path.exists(resume_path):
            print(f"❌ ERROR: Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        else:
            print(f"✓ Checkpoint found")
    
    print("="*70 + "\n")
    
    config_save_path = exp_dir / "config.yaml"

    # Créer le dossier si nécessaire
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"✓ Experiment dir: {exp_dir}")
    print(f"✓ Config saved: {config_save_path}\n")
    
    # Sauvegarder args pour le modèle
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    args_file = exp_dir / 'checkpoints' / 'args.json'
    with open(args_file, 'w') as f:
        import json
        json.dump(config, f, indent=2)
    
    cmd = build_train_command(config, exp_dir)
    
    # ════════════════════════════════════════════════════════════════════════
    # VERIFY PATHS BEFORE STARTING
    # ════════════════════════════════════════════════════════════════════════
    
    print("="*70)
    print("VERIFYING PATHS")
    print("="*70)
    
    # Vérifier data path
    data_arg_idx = cmd.index('--data') + 1
    data_path = Path(cmd[data_arg_idx])
    if not data_path.exists():
        print(f"❌ ERROR: Data directory not found: {data_path}")
        print("   Please download the data first:")
        print("   cd data && sh download_data.sh json_feat")
        sys.exit(1)
    else:
        print(f"✓ Data directory: {data_path}")
    
    # Vérifier splits path
    splits_arg_idx = cmd.index('--splits') + 1
    splits_path = Path(cmd[splits_arg_idx])
    if not splits_path.exists():
        print(f"❌ ERROR: Splits file not found: {splits_path}")
        sys.exit(1)
    else:
        print(f"✓ Splits file: {splits_path}")
    
    print("="*70 + "\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # LOGS FILES
    # ════════════════════════════════════════════════════════════════════════
    
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    train_log = log_dir / 'train.log'
    command_log = log_dir / 'command.txt'
    summary_log = log_dir / 'summary.txt'
    
    # Sauvegarder la commande
    with open(command_log, 'w') as f:
        f.write("TRAINING COMMAND\n")
        f.write("="*70 + "\n")
        f.write(" ".join(str(x) for x in cmd) + "\n")
        f.write("="*70 + "\n\n")
        f.write(f"Started: {datetime.now()}\n")
    
    print("="*70)
    print("TRAINING COMMAND")
    print("="*70)
    print(" ".join(str(x) for x in cmd))
    print("="*70 + "\n")
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"✓ Logs will be saved to: {log_dir}")
    print(f"  - train.log: Full training output")
    print(f"  - command.txt: Command used")
    print(f"  - summary.txt: Final summary")
    print(f"✓ TensorBoard logs: {exp_dir / 'tensorboard'}")
    print(f"  To view: tensorboard --logdir={exp_dir / 'tensorboard'}")
    print("="*70 + "\n")
    
    try:
        # Setup environnement
        env = os.environ.copy()
        
        # CRITIQUE: S'assurer que ALFRED_ROOT est défini pour train_seq2seq.py
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        env['ALFRED_ROOT'] = str(ALFRED_ROOT)
        env['PYTHONPATH'] = f"{ALFRED_ROOT}:{env.get('PYTHONPATH', '')}"
        
        # Debug: afficher les variables d'environnement critiques
        print(f"Environment variables set:")
        print(f"  ALFRED_ROOT={env['ALFRED_ROOT']}")
        print(f"  PYTHONPATH={env['PYTHONPATH']}")
        print()
        
        # ✓✓✓ LANCER AVEC LOGS ✓✓✓
        with open(train_log, 'w', buffering=1) as log_file:
            # Header du log
            log_file.write("="*70 + "\n")
            log_file.write(f"ALFRED TRAINING LOG\n")
            log_file.write("="*70 + "\n")
            log_file.write(f"Experiment: {exp_name}\n")
            log_file.write(f"Model: {model_name}\n")
            log_file.write(f"Started: {datetime.now()}\n")
            log_file.write("="*70 + "\n\n")
            log_file.flush()
            
            # Lancer le subprocess, on test avec subprocess.run d'abord 
            """
            process = subprocess.Popen(
                cmd,
                cwd=ALFRED_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            """
            process = subprocess.run(cmd, cwd=ALFRED_ROOT, env=env) # la on perd le log mais au moins ca marche
            
            
            
            # Lire et afficher en temps réel
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Écrire dans le log
                    log_file.write(line)
                    log_file.flush()
                    
                    # Afficher aussi dans le terminal
                    print(line, end='', flush=True)
            
            # Attendre la fin
            process.wait()
            returncode = process.returncode
            
            # Footer du log
            log_file.write("\n" + "="*70 + "\n")
            log_file.write(f"Ended: {datetime.now()}\n")
            log_file.write(f"Exit code: {returncode}\n")
            log_file.write("="*70 + "\n")
        
        # Vérifier le code de sortie
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)
        
        # ════════════════════════════════════════════════════════════════════
        # SUCCESS
        # ════════════════════════════════════════════════════════════════════
        
        with open(exp_dir / 'status.txt', 'w') as f:
            f.write('SUCCESS')
        
        # Créer le résumé
        with open(summary_log, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*70 + "\n")
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Status: SUCCESS\n")
            f.write(f"Completed: {datetime.now()}\n")
            f.write("="*70 + "\n\n")
            
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Results:\n")
            f.write(f"  Experiment directory: {exp_dir}\n")
            f.write(f"  Logs: {log_dir}\n")
            f.write(f"  Checkpoints: {exp_dir / 'checkpoints'}\n")
        
        print("\n" + "="*70)
        print(f"✅ EXPERIMENT COMPLETED: {exp_name}")
        print("="*70)
        print(f"✓ Results in: {exp_dir}")
        print(f"✓ Full log: {train_log}")
        print(f"✓ Summary: {summary_log}")
        print(f"✓ TensorBoard: tensorboard --logdir={exp_dir / 'tensorboard'}")
        print("="*70 + "\n")
        
        return exp_dir
        
    except subprocess.CalledProcessError as e:
        # ════════════════════════════════════════════════════════════════════
        # FAILURE
        # ════════════════════════════════════════════════════════════════════
        
        with open(exp_dir / 'status.txt', 'w') as f:
            f.write(f'FAILED: exit code {e.returncode}')
        
        # Résumé d'échec
        with open(summary_log, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT FAILED\n")
            f.write("="*70 + "\n")
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Exit code: {e.returncode}\n")
            f.write(f"Failed: {datetime.now()}\n")
            f.write("="*70 + "\n\n")
            f.write("See train.log for full error details.\n")
        
        print("\n" + "="*70)
        print(f"❌ EXPERIMENT FAILED: {exp_name}")
        print("="*70)
        print(f"✗ Exit code: {e.returncode}")
        print(f"✗ Check logs: {train_log}")
        print("="*70 + "\n")
        
        raise
    
    except Exception as e:
        with open(exp_dir / 'status.txt', 'w') as f:
            f.write(f'FAILED: {str(e)}')
        
        print("\n" + "="*70)
        print(f"❌ EXPERIMENT FAILED: {exp_name}")
        print("="*70)
        print(f"✗ Error: {str(e)}")
        print("="*70 + "\n")
        
        import traceback
        traceback.print_exc()
        
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Run ALFRED experiment with full logging'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to config YAML file'
    )
    args = parser.parse_args()
    
    run_experiment(args.config)


if __name__ == '__main__':
    main()