import os
from pyexpat import model
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # Utiliser torch.utils.tensorboard au lieu de tensorboardX
from tqdm import trange

class Module(nn.Module):

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)

        # summary self.writer
        self.summary_writer = None

    def run_train(self, splits, args=None, optimizer=None):
        '''
        training loop with enhanced TensorBoard logging
        '''

        # args
        args = args or self.args
        
        # PATCH: Si de nouveaux args sont fournis, mettre à jour self.args
        # Ceci est crucial pour les reprises (resume) où self.args contient
        # les anciens chemins du checkpoint
        if args is not self.args:
            # Mettre à jour les chemins dans self.args
            self.args.dout = args.dout
            self.args.data = args.data
            self.args.splits = args.splits
            if hasattr(args, 'tensorboard_dir'):
                self.args.tensorboard_dir = args.tensorboard_dir
            print(f"✓ Updated model args with new paths:")
            print(f"    dout: {args.dout}")
            if hasattr(args, 'tensorboard_dir'):
                print(f"    tensorboard_dir: {args.tensorboard_dir}")

        # splits
        train = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_train_size = int(self.args.dataset_fraction * 0.7)
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train = train[:small_train_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[:16]
            valid_seen = valid_seen[:16]
            valid_unseen = valid_unseen[:16]

        # ══════════════════════════════════════════════════════════════
        # Initialize TensorBoard writer
        # ══════════════════════════════════════════════════════════════
        # Utiliser tensorboard_dir si disponible, sinon utiliser dout
        tensorboard_dir = getattr(args, 'tensorboard_dir', args.dout)
        self.summary_writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"✓ TensorBoard logs: {tensorboard_dir}")

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        # display dout
        print("="*70)
        print(f"Training Configuration:")
        print(f"  Model output: {self.args.dout}")
        print(f"  Epochs: {args.epoch}")
        print(f"  Batch size: {args.batch}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Train samples: {len(train)}")
        print(f"  Valid seen: {len(valid_seen)}")
        print(f"  Valid unseen: {len(valid_unseen)}")
        print("="*70)
        
        best_loss = {'train': 1e10, 'valid_seen': 1e10, 'valid_unseen': 1e10}
        train_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0
        
        # ══════════════════════════════════════════════════════════════
        # Detect starting epoch for resume
        # ══════════════════════════════════════════════════════════════
        start_epoch = 0
        if hasattr(args, 'resume') and args.resume:
            # Extract epoch number from checkpoint filename
            # e.g., "net_epoch_20.pth" -> 20
            import re
            match = re.search(r'net_epoch_(\d+)', args.resume)
            if match:
                start_epoch = int(match.group(1))
                print(f"✓ Resuming from epoch {start_epoch}")
                print(f"  Will train epochs {start_epoch + 1} to {args.epoch}")
            else:
                print(f"⚠️  Could not detect epoch from checkpoint name, starting from 0")
        
        # ══════════════════════════════════════════════════════════════
        # Training Loop
        # ══════════════════════════════════════════════════════════════
        for epoch in trange(start_epoch, args.epoch, desc='epoch'):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{args.epoch}")
            print(f"{'='*70}")
            
            # ─────────────────────────────────────────────────────────
            # TRAINING PHASE
            # ─────────────────────────────────────────────────────────
            m_train = collections.defaultdict(list)
            self.train()
            self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            current_lr = optimizer.param_groups[0]['lr']
            
            total_train_loss = list()
            random.shuffle(train)  # shuffle every epoch
            
            for batch_idx, (batch, feat) in enumerate(self.iterate(train, args.batch)):
                out = self.forward(feat)
                preds = self.extract_preds(out, batch, feat)
                loss = self.compute_loss(out, batch, feat)
                
                # Accumulate losses for logging
                for k, v in loss.items():
                    ln = 'loss_' + k
                    m_train[ln].append(v.item())
                    self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                # optimizer backward pass
                optimizer.zero_grad()
                sum_loss = sum(loss.values())
                sum_loss.backward()
                optimizer.step()

                self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                
                # ⬇️ TensorBoard: Log learning rate
                if batch_idx % 10 == 0:
                    self.summary_writer.add_scalar('train/learning_rate', current_lr, train_iter)
                
                sum_loss = sum_loss.detach().cpu()
                total_train_loss.append(float(sum_loss))
                train_iter += self.args.batch
                
                # Print progress
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx} - Loss: {sum_loss:.4f} - LR: {current_lr:.6f}")

            # Average training loss
            avg_train_loss = sum(total_train_loss) / len(total_train_loss)
            self.summary_writer.add_scalar('train/avg_loss_epoch', avg_train_loss, epoch)
            
            print(f"\n  Training - Avg Loss: {avg_train_loss:.4f}")

            # ─────────────────────────────────────────────────────────
            # VALIDATION PHASE: VALID SEEN
            # ─────────────────────────────────────────────────────────
            print(f"\n  Validating on seen environments...")
            p_valid_seen, valid_seen_iter, total_valid_seen_loss, m_valid_seen = self.run_pred(
                valid_seen, args=args, name='valid_seen', iter=valid_seen_iter
            )
            m_valid_seen.update(self.compute_metric(p_valid_seen, valid_seen))
            m_valid_seen['total_loss'] = float(total_valid_seen_loss)
            self.summary_writer.add_scalar('valid_seen/total_loss', m_valid_seen['total_loss'], valid_seen_iter)
            
            # ⬇️ Logger TOUTES les métriques individuelles
            for metric_name, metric_value in m_valid_seen.items():
                if metric_name not in ['total_loss']:  # total_loss déjà loggé
                    self.summary_writer.add_scalar(f'valid_seen/{metric_name}', metric_value, epoch)

            # ─────────────────────────────────────────────────────────
            # VALIDATION PHASE: VALID UNSEEN
            # ─────────────────────────────────────────────────────────
            print(f"  Validating on unseen environments...")
            p_valid_unseen, valid_unseen_iter, total_valid_unseen_loss, m_valid_unseen = self.run_pred(
                valid_unseen, args=args, name='valid_unseen', iter=valid_unseen_iter
            )
            m_valid_unseen.update(self.compute_metric(p_valid_unseen, valid_unseen))
            m_valid_unseen['total_loss'] = float(total_valid_unseen_loss)
            self.summary_writer.add_scalar('valid_unseen/total_loss', m_valid_unseen['total_loss'], valid_unseen_iter)
            
            # ⬇️ Logger TOUTES les métriques individuelles
            for metric_name, metric_value in m_valid_unseen.items():
                if metric_name not in ['total_loss']:  # total_loss déjà loggé
                    self.summary_writer.add_scalar(f'valid_unseen/{metric_name}', metric_value, epoch)

            # ⬇️ TensorBoard: Compare losses across splits
            self.summary_writer.add_scalars('loss_comparison', {
                'train': avg_train_loss,
                'valid_seen': total_valid_seen_loss,
                'valid_unseen': total_valid_unseen_loss
            }, epoch)

            stats = {'epoch': epoch,
                     'valid_seen': m_valid_seen,
                     'valid_unseen': m_valid_unseen}

            # ─────────────────────────────────────────────────────────
            # SAVE BEST MODELS
            # ─────────────────────────────────────────────────────────
            # new best valid_seen loss
            if total_valid_seen_loss < best_loss['valid_seen']:
                print('\n  ✓ Found new best valid_seen!! Saving...')
                fsave = os.path.join(args.dout, 'best_seen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_seen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(args.dout, 'valid_seen.debug.preds.json')
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_valid_seen, valid_seen), f, indent=2)
                best_loss['valid_seen'] = total_valid_seen_loss
                
                # ⬇️ TensorBoard: Log best loss
                self.summary_writer.add_scalar('best/valid_seen_loss', best_loss['valid_seen'], epoch)

            # new best valid_unseen loss
            if total_valid_unseen_loss < best_loss['valid_unseen']:
                print('  ✓ Found new best valid_unseen!! Saving...')
                fsave = os.path.join(args.dout, 'best_unseen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_unseen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(args.dout, 'valid_unseen.debug.preds.json')
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_valid_unseen, valid_unseen), f, indent=2)

                best_loss['valid_unseen'] = total_valid_unseen_loss
                
                # ⬇️ TensorBoard: Log best loss
                self.summary_writer.add_scalar('best/valid_unseen_loss', best_loss['valid_unseen'], epoch)

            # save the latest checkpoint
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')
            torch.save({
                'metric': stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

            # write stats to tensorboard
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
            
            # Pretty print stats
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1} SUMMARY:")
            print(f"{'='*70}")
            pprint.pprint(stats)
            print(f"{'='*70}\n")

        # ══════════════════════════════════════════════════════════════
        # END OF TRAINING
        # ══════════════════════════════════════════════════════════════
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Best Losses:")
        print(f"  Valid Seen:   {best_loss['valid_seen']:.4f}")
        print(f"  Valid Unseen: {best_loss['valid_unseen']:.4f}")
        print(f"{'='*70}")
        
        # ⬇️ TensorBoard: Log final hyperparameters
        hparams = {
            'lr': args.lr,
            'batch_size': args.batch,
            'hidden_size': args.dhid,
            'embedding_size': args.demb,
            'decay_epoch': args.decay_epoch,
        }
        
        final_metrics = {
            'best_valid_seen_loss': best_loss['valid_seen'],
            'best_valid_unseen_loss': best_loss['valid_unseen'],
        }
        
        self.summary_writer.add_hparams(hparams, final_metrics)
        
        # ⬇️ TensorBoard: Close writer
        self.summary_writer.close()
        print(f"✓ TensorBoard logs saved to: {tensorboard_dir}")
        print(f"  View with: tensorboard --logdir={tensorboard_dir}")
        print(f"{'='*70}\n")

    def run_pred(self, dev, args=None, name='dev', iter=0):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        for batch, feat in self.iterate(dev, args.batch):
            out = self.forward(feat)
            preds = self.extract_preds(out, batch, feat)
            p_dev.update(preds)
            loss = self.compute_loss(out, batch, feat)
            for k, v in loss.items():
                ln = 'loss_' + k
                m_dev[ln].append(v.item())
                self.summary_writer.add_scalar("%s/%s" % (name, ln), v.item(), dev_iter)
            sum_loss = sum(loss.values())
            self.summary_writer.add_scalar("%s/loss" % (name), sum_loss, dev_iter)
            total_loss.append(float(sum_loss.detach().cpu()))
            dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['repeat_idx']))

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[i]['action_low'].split(),
            }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def iterate(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        for i in trange(0, len(data), batch_size, desc='batch'):
            tasks = data[i:i+batch_size]
            batch = [self.load_task_json(task) for task in tasks]
            feat = self.featurize(batch)
            yield batch, feat

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave):
        '''
        load pth model from disk
        '''
        checkpoint = torch.load(fsave, map_location='cpu')
        save = torch.load(fsave)
        model = cls(save['args'], save['vocab'])

        model.load_state_dict(save['model'], strict=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Load optimizer state if available FIXed
        if 'optim' in save and save['optim']:
            try:
                optimizer.load_state_dict(save['optim'])
            except Exception as e:
                print(f"Warning: could not load optimizer ({e})")
        else:
            print("Warning: no optimizer state, using fresh optimizer")

        return model, optimizer

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True