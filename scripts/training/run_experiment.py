"""
Script pour lancer des expériences ALFRED
Avec logs complets dans experiments/*/logs/
"""

import os
import sys
import subprocess
import argparse
import yaml
from datetime import datetime
from pathlib import Path

ALFRED_ROOT = Path("/media/cedrix/Ubuntu_2To/Alfred/alfred")
EXP_ROOT = Path("/media/cedrix/Ubuntu_2To/Alfred/alfred_experiments")

print("="*70)
print("ALFRED EXPERIMENT RUNNER")
print("="*70)
print(f"ALFRED_ROOT: {ALFRED_ROOT}")
print(f"EXP_ROOT: {EXP_ROOT}")
print("="*70 + "\n")

def create_experiment_dir(exp_name):
    """Crée un dossier d'expérience avec timestamp"""
    from datetime import datetime  # Import si pas déjà fait
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name_full = f"{exp_name}_{timestamp}"  # ← CORRECTION ICI
    
    exp_dir = EXP_ROOT / "experiments" / exp_name_full
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'tensorboard').mkdir(exist_ok=True)  # ← AJOUT TENSORBOARD
    
    return exp_dir


def load_config(config_path):
    """Charge config YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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
            cmd.extend(['--data', 'data/json_feat_2.1.0'])
        else:
            cmd.extend(['--data', str(data)])
    else:
        cmd.extend(['--data', 'data/json_feat_2.1.0'])
    
    # Reste de la commande
    cmd.extend([
        '--splits', str(config['splits']),
        '--model', str(config['model']),
        '--dout', str(exp_dir / 'checkpoints'),
        '--batch', str(config.get('batch', 8)),
        '--epoch', str(config.get('epoch', 20)),
        '--lr', str(config.get('lr', 0.0001)),
        '--seed', str(config.get('seed', 1)),
    ])
    
    # ✅ AJOUT: Passer le chemin TensorBoard
    tensorboard_dir = exp_dir / 'tensorboard'
    cmd.extend(['--tensorboard_dir', str(tensorboard_dir)])
    
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
    # Ajouter timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name_full = f"{exp_name}_{timestamp}"

    # Créer exp_dir
    from pathlib import Path
    exp_dir = EXP_ROOT / "experiments" / exp_name_full   # ✅ Utilise la variable définie en haut
    exp_dir.mkdir(parents=True, exist_ok=True) 
    model_name = config.get('model', 'N/A')

    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {exp_name}")
    print("="*70)
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Model: {model_name}")
    print(f"CoT: {config.get('use_cot', False)}")
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
    
    # ═══════════════════════════════════════════════════════════
    # LOGS FILES
    # ═══════════════════════════════════════════════════════════
    
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)  # ← AJOUTER CETTE LIGNE
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
        env['ALFRED_ROOT'] = str(ALFRED_ROOT)
        env['PYTHONPATH'] = f"{ALFRED_ROOT}:{EXP_ROOT}:{env.get('PYTHONPATH', '')}"
        
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
            
            # Lancer le subprocess
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
        
        # ═══════════════════════════════════════════════════════════
        # SUCCESS
        # ═══════════════════════════════════════════════════════════
        
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
        # ═══════════════════════════════════════════════════════════
        # FAILURE
        # ═══════════════════════════════════════════════════════════
        
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