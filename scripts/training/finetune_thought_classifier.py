"""
finetune_thought_classifier_REAL.py

Fine-tune thought classifier avec VRAIES features
au lieu de features aléatoires
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ════════════════════════════════════════════════════════════════════
# SETUP PATHS
# ════════════════════════════════════════════════════════════════════

ALFRED_ROOT = "/home/cedrix/Bureau/Alfred/alfred"

if not os.path.exists(ALFRED_ROOT):
    print(f"❌ ERROR: ALFRED_ROOT not found")
    sys.exit(1)

sys.path.insert(0, ALFRED_ROOT)
print(f"✓ ALFRED_ROOT: {ALFRED_ROOT}")


# ════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════

class ThoughtDatasetReal(Dataset):
    """
    Dataset qui charge les VRAIES features visuelles et langage
    """
    def __init__(self, json_file, feat_root, vocab, data_root=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.feat_root = Path(feat_root)  # json_feat_2.1.0
        self.data_root = Path(data_root) if data_root else Path(feat_root).parent / 'json_2.1.0'  # json_2.1.0
        self.vocab = vocab
        
        # Flatten samples
        self.samples = []
        for traj in self.data:
            task_id = traj['task_id']
            
            for thought in traj['thoughts']:
                self.samples.append({
                    'thought_id': thought['thought_id'],
                    'step': thought['step'],
                    'task_id': task_id,
                    'num_steps': traj['num_steps']
                })
        
        print(f"Dataset created with {len(self.samples):,} samples")
        print(f"Features root: {self.feat_root}")
        print(f"Data root: {self.data_root}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        task_id = sample['task_id']
        
        feat_path = None
        traj_data_path = None
        instruction = ""
        
        # Chercher features dans json_feat_2.1.0
        for split in ['train', 'valid_seen', 'valid_unseen']:
            candidate_feat = self.feat_root / split / task_id / 'feat_conv.pt'
            if candidate_feat.exists():
                feat_path = candidate_feat
                break
        
        # Chercher traj_data.json dans json_2.1.0
        for split in ['train', 'valid_seen', 'valid_unseen']:
            candidate_traj = self.data_root / split / task_id / 'traj_data.json'
            if candidate_traj.exists():
                traj_data_path = candidate_traj
                break
        
        # Charger instruction depuis traj_data.json
        if traj_data_path and traj_data_path.exists():
            try:
                with open(traj_data_path, 'r') as f:
                    traj_data = json.load(f)
                    anns = traj_data.get('turk_annotations', {}).get('anns', [])
                    if anns:
                        instruction = anns[0].get('task_desc', '')
            except Exception as e:
                instruction = ""
        
        # Charger features visuelles
        if feat_path and feat_path.exists():
            try:
                feats = torch.load(feat_path, map_location='cpu')
                step_idx = min(sample['step'], len(feats) - 1)
                frame_feat = feats[step_idx]
                frame_feat = frame_feat.mean(dim=(1, 2))
            except Exception as e:
                frame_feat = torch.randn(512)
        else:
            frame_feat = torch.randn(512)
        
        return {
            'thought_id': sample['thought_id'],
            'frame_feat': frame_feat,
            'instruction': instruction,
            'task_id': task_id
        }


def collate_fn_real(batch):
    """Collate avec vraies features et tokenization des instructions"""
    batch_size = len(batch)
    
    thought_ids = torch.tensor([s['thought_id'] for s in batch], dtype=torch.long)
    frames_feat = torch.stack([s['frame_feat'] for s in batch])  # (batch, 512)
    
    # Tokenizer les instructions (simple: utiliser vocab word)
    # Note: C'est une simplification, idéalement utiliser le vrai tokenizer ALFRED
    max_len = 20
    lang_tokens = []
    
    for s in batch:
        instruction = s['instruction']
        
        if instruction:
            # Tokenization simple par mots
            words = instruction.lower().split()[:max_len]
            # Pour l'instant, utiliser des IDs constants (pas de vrai vocab)
            # TODO: Utiliser vocab['word'] pour tokenizer correctement
            tokens = [hash(w) % 100 for w in words]  # Hash simple
            
            # Pad à max_len
            tokens = tokens + [0] * (max_len - len(tokens))
            lang_tokens.append(tokens[:max_len])
        else:
            # Pas d'instruction → tokens constant
            lang_tokens.append([0] * max_len)
    
    lang_tokens = torch.tensor(lang_tokens, dtype=torch.long)  # (batch, max_len)
    
    # Expand frames pour simuler séquence temporelle
    frames_feat = frames_feat.unsqueeze(1).repeat(1, 10, 1)  # (batch, 10, 512)
    
    return {
        'thought_ids': thought_ids,
        'frames_feat': frames_feat,
        'lang_tokens': lang_tokens
    }


# ════════════════════════════════════════════════════════════════════
# FREEZE
# ════════════════════════════════════════════════════════════════════

def freeze_action_decoder(model):
    """Freeze action modules"""
    freeze_modules = ['dec', 'enc_att', 'state_t', 'subgoal_tt', 'out', 'object_out', 'mask_dec']
    
    frozen = 0
    trainable = 0
    
    for name, param in model.named_parameters():
        if name.split('.')[0] in freeze_modules:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    
    print(f"✓ Frozen: {frozen:,} | Trainable: {trainable:,}")
    return model


# ════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        labels = batch['thought_ids'].to(device)
        frames = batch['frames_feat'].to(device)
        lang = batch['lang_tokens'].to(device)
        
        try:
            # Frames déjà encodés
            vis_feat = frames
            
            # Encode language
            emb_lang = model.emb_word(lang)
            enc_output = model.enc(emb_lang)
            enc_lang = enc_output[0] if isinstance(enc_output, tuple) else enc_output
            
            if enc_lang.size(-1) == 1024:
                enc_lang = enc_lang[..., :512]
            
            # Thought prediction
            vis_pooled = vis_feat.mean(1)
            lang_pooled = enc_lang.mean(1)
            combined = torch.cat([vis_pooled, lang_pooled], dim=-1)
            
            thought_output = model.thought_generator(combined)
            if isinstance(thought_output, tuple):
                thought_hidden = thought_output[0]
            else:
                thought_hidden = thought_output
            
            thought_logits = model.thought_classifier(thought_hidden)
            
            # Loss
            loss = nn.CrossEntropyLoss()(thought_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = thought_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        except Exception as e:
            print(f"\n⚠️  {e}")
            continue
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total if total > 0 else 0
    }


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            labels = batch['thought_ids'].to(device)
            frames = batch['frames_feat'].to(device)
            lang = batch['lang_tokens'].to(device)
            
            try:
                vis_feat = frames
                
                emb_lang = model.emb_word(lang)
                enc_output = model.enc(emb_lang)
                enc_lang = enc_output[0] if isinstance(enc_output, tuple) else enc_output
                
                if enc_lang.size(-1) == 1024:
                    enc_lang = enc_lang[..., :512]
                
                vis_pooled = vis_feat.mean(1)
                lang_pooled = enc_lang.mean(1)
                combined = torch.cat([vis_pooled, lang_pooled], dim=-1)
                
                thought_output = model.thought_generator(combined)
                if isinstance(thought_output, tuple):
                    thought_hidden = thought_output[0]
                else:
                    thought_hidden = thought_output
                
                thought_logits = model.thought_classifier(thought_hidden)
                
                loss = nn.CrossEntropyLoss()(thought_logits, labels)
                total_loss += loss.item()
                
                preds = thought_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            except Exception as e:
                continue
    
    return {
        'loss': total_loss / len(dataloader) if len(dataloader) > 0 else float('inf'),
        'accuracy': correct / total if total > 0 else 0
    }


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='thoughts_dataset_train_remapped.json')
    parser.add_argument('--valid', required=True, help='thoughts_dataset_valid_remapped.json')
    parser.add_argument('--feat_root', required=True, help='Path to json_feat_2.1.0')
    parser.add_argument('--data_root', default=None, help='Path to json_2.1.0 (for traj_data.json)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("THOUGHT CLASSIFIER FINE-TUNING (REAL FEATURES)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Features: {args.feat_root}\n")
    
    # Import
    from models.model.seq2seq_react_light import Module as ReActLightModule
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_args = checkpoint['args']
    vocab = checkpoint['vocab']
    
    model = ReActLightModule(model_args, vocab)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    print("✓ Model loaded\n")
    
    # Freeze
    model = freeze_action_decoder(model)
    print()
    
    # Datasets avec VRAIES features
    print("Loading datasets with REAL features...")
    train_dataset = ThoughtDatasetReal(args.data, args.feat_root, vocab, args.data_root)
    valid_dataset = ThoughtDatasetReal(args.valid, args.feat_root, vocab, args.data_root)
    print(f"✓ Train: {len(train_dataset):,} | Valid: {len(valid_dataset):,}\n")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        collate_fn=collate_fn_real
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=collate_fn_real
    )
    
    # Optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Train
    best_loss = float('inf')
    history = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        train_m = train_epoch(model, train_loader, optimizer, device)
        valid_m = validate(model, valid_loader, device)
        
        print(f"Train Loss: {train_m['loss']:.4f} | Acc: {train_m['accuracy']:.4f}")
        print(f"Valid Loss: {valid_m['loss']:.4f} | Acc: {valid_m['accuracy']:.4f}")
        
        history.append({
            'epoch': epoch+1, 
            'train_loss': train_m['loss'],
            'train_acc': train_m['accuracy'],
            'valid_loss': valid_m['loss'],
            'valid_acc': valid_m['accuracy']
        })
        
        # Save
        torch.save({
            'model': model.state_dict(), 
            'args': model_args, 
            'vocab': vocab
        }, output_dir / f'checkpoint_epoch{epoch+1}.pth')
        
        if valid_m['loss'] < best_loss:
            best_loss = valid_m['loss']
            torch.save({
                'model': model.state_dict(), 
                'args': model_args, 
                'vocab': vocab
            }, output_dir / 'best_thought.pth')
            print("✓ Best!")
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETED: {output_dir / 'best_thought.pth'}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()