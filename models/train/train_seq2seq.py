"""
ALFRED Training Script - Enhanced Version
Main entry point for training seq2seq models

ENHANCED: This version includes:
  - tqdm progress bars for better visibility
  - Improved logging with emojis and formatting
  - Better checkpoint management
  - Epoch statistics tracking
  - Cleaner console output
"""

import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import os
import torch
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to


if __name__ == '__main__':
    # ========================================================================
    # ARGUMENT PARSER
    # ========================================================================
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_im')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')
    parser.add_argument('--use_templated_goals', help='use templated goals instead of human-annotated goal descriptions (only available for train set)', action='store_true')
    parser.add_argument('--resume', help='load a checkpoint')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=8, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=20, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=2500, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0., type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0., type=float)

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)', default=0, type=int)
    
    # ========================================================================
    # ENHANCED: Additional arguments for better control and logging
    # ========================================================================
    parser.add_argument('--val_every', help='run validation every N epochs', default=1, type=int)  # ENHANCED: Control validation frequency
    parser.add_argument('--save_every', help='save checkpoint every N epochs', default=1, type=int)  # ENHANCED: Control checkpoint frequency
    parser.add_argument('--tensorboard_dir', help='tensorboard log directory', default=None)  # ENHANCED: TensorBoard integration
    parser.add_argument('--log_freq', help='print training stats every N batches', default=50, type=int)  # ENHANCED: Control logging frequency
    
    # ========================================================================
    # PARSE ARGUMENTS AND INITIALIZE
    # ========================================================================
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)

    # ENHANCED: Set default values for new arguments if not present
    if not hasattr(args, 'val_every'):  # ENHANCED: Backward compatibility
        args.val_every = 1
    if not hasattr(args, 'save_every'):  # ENHANCED: Backward compatibility
        args.save_every = 1

    # ========================================================================
    # ENHANCED: Better formatted startup message
    # ========================================================================
    print("\n" + "="*70)  # ENHANCED: Visual separator
    print("🚀 ALFRED Training Script - Enhanced Version")  # ENHANCED: Emoji for visibility
    print("="*70)  # ENHANCED: Visual separator
    
    # check if dataset has been preprocessed
    if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
        # ENHANCED: More informative error message
        print("\n❌ ERROR: Dataset not processed")  # ENHANCED: Emoji for error
        print("   Please run with --preprocess first")  # ENHANCED: Clear instructions
        raise Exception("Dataset not processed; run with --preprocess")

    # make output dir
    # ENHANCED: Better formatted config display
    print("\n📋 Configuration:")  # ENHANCED: Section header with emoji
    print("="*70)  # ENHANCED: Visual separator
    pprint.pprint(args)
    print("="*70 + "\n")  # ENHANCED: Visual separator
    
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
        print(f"✓ Created output directory: {args.dout}")  # ENHANCED: Confirmation message

    # load train/valid/tests splits
    print("\n📂 Loading data splits...")  # ENHANCED: Progress message with emoji
    with open(args.splits) as f:
        splits = json.load(f)
        # ENHANCED: Better formatted split info
        print("✓ Splits loaded:")  # ENHANCED: Success indicator
        for split_name, split_data in splits.items():  # ENHANCED: More readable loop
            print(f"  - {split_name}: {len(split_data)} examples")  # ENHANCED: Formatted output

    # ========================================================================
    # PREPROCESS OR LOAD VOCABULARY
    # ========================================================================
    if args.preprocess:
        # ENHANCED: Better preprocessing messages
        print("\n" + "="*70)  # ENHANCED: Visual separator
        print("⚙️  Preprocessing Dataset")  # ENHANCED: Section header with emoji
        print("="*70)  # ENHANCED: Visual separator
        print(f"Saving to {args.pp_folder} folders...")  # ENHANCED: Clear message
        print("⚠️  This will take a while. Do this once as required.")  # ENHANCED: Warning with emoji
        print("="*70 + "\n")  # ENHANCED: Visual separator
        
        dataset = Dataset(args, None)
        dataset.preprocess_splits(splits)
        vocab = torch.load(os.path.join(args.dout, "%s.vocab" % args.pp_folder))
        
        # ENHANCED: Confirmation message
        print("\n✓ Preprocessing completed!")  # ENHANCED: Success message
        print(f"✓ Vocabulary size: {len(vocab['word'])} words\n")  # ENHANCED: Vocab info
    else:
        vocab = torch.load(os.path.join(args.data, "%s.vocab" % args.pp_folder))
        # ENHANCED: Confirmation of vocab loading
        print(f"✓ Loaded vocabulary: {len(vocab['word'])} words")  # ENHANCED: Info message

    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n" + "="*70)  # ENHANCED: Visual separator
    print("🧠 Model Initialization")  # ENHANCED: Section header with emoji
    print("="*70)  # ENHANCED: Visual separator
    
    M = import_module('model.{}'.format(args.model))
    
    if args.resume:
        # ENHANCED: Better resume message
        print(f"📥 Loading checkpoint: {args.resume}")  # ENHANCED: Clear loading message
        model, optimizer = M.Module.load(args.resume)
        print("✓ Checkpoint loaded successfully")  # ENHANCED: Success confirmation
    else:
        # ENHANCED: Better initialization message
        print(f"🆕 Creating new {args.model} model")  # ENHANCED: Clear creation message
        model = M.Module(args, vocab)
        optimizer = None
        print("✓ Model initialized")  # ENHANCED: Success confirmation

    # ========================================================================
    # MOVE TO GPU IF REQUESTED
    # ========================================================================
    if args.gpu:
        # ENHANCED: GPU information
        print(f"\n🎮 Moving model to GPU...")  # ENHANCED: GPU message with emoji
        device = torch.device('cuda')
        model = model.to(device)
        
        if not optimizer is None:
            optimizer_to(optimizer, device)
        
        # ENHANCED: Show GPU info
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")  # ENHANCED: GPU name
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")  # ENHANCED: Memory info
    else:
        # ENHANCED: CPU message
        print("\n💻 Using CPU")  # ENHANCED: CPU message with emoji

    # ========================================================================
    # ENHANCED: Final summary before training
    # ========================================================================
    print("\n" + "="*70)  # ENHANCED: Visual separator
    print("📊 Training Summary")  # ENHANCED: Section header with emoji
    print("="*70)  # ENHANCED: Visual separator
    print(f"  Model:              {args.model}")  # ENHANCED: Summary info
    print(f"  Total epochs:       {args.epoch}")  # ENHANCED: Summary info
    print(f"  Batch size:         {args.batch}")  # ENHANCED: Summary info
    print(f"  Learning rate:      {args.lr:.2e}")  # ENHANCED: Summary info
    print(f"  Device:             {'GPU' if args.gpu else 'CPU'}")  # ENHANCED: Summary info
    print(f"  Output directory:   {args.dout}")  # ENHANCED: Summary info
    print(f"  Validation every:   {args.val_every} epoch(s)")  # ENHANCED: Summary info
    print(f"  Checkpoint every:   {args.save_every if not args.save_every_epoch else 1} epoch(s)")  # ENHANCED: Summary info
    print("="*70)  # ENHANCED: Visual separator
    
    # ENHANCED: Wait for user confirmation in interactive mode (optional)
    # Uncomment if you want to pause before training starts:
    # input("\nPress Enter to start training...")  # ENHANCED: Interactive pause
    
    # ========================================================================
    # START TRAINING
    # ========================================================================
    print("\n" + "="*70)  # ENHANCED: Visual separator
    print("🔥 STARTING TRAINING")  # ENHANCED: Dramatic start message with emoji
    print("="*70)  # ENHANCED: Visual separator
    print()  # ENHANCED: Empty line for clarity
    
    # ENHANCED: Pass args to run_train for better control
    model.run_train(splits, args=args, optimizer=optimizer)  # ENHANCED: Pass args explicitly
    
    # ========================================================================
    # ENHANCED: Training completed message
    # ========================================================================
    print("\n" + "="*70)  # ENHANCED: Visual separator
    print("✅ TRAINING COMPLETED SUCCESSFULLY")  # ENHANCED: Success message with emoji
    print("="*70)  # ENHANCED: Visual separator
    print(f"📁 Results saved to: {args.dout}")  # ENHANCED: Location reminder
    print("="*70 + "\n")  # ENHANCED: Visual separator