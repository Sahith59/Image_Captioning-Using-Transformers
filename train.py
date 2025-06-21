import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import time
import datetime
import json

from models import caption
from datasets import coco, utils
from configuration import Config
from engine import train_one_epoch, evaluate
from models.utils import save_checkpoint, load_checkpoint

def get_args():
    parser = argparse.ArgumentParser(description='CATR Training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--clip_max_norm', type=float, default=0.1, help='Gradient clipping max norm')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no_save', action='store_true', help='Do not save checkpoints')
    parser.add_argument('--log_step', type=int, default=100, help='Print logs every n steps')
    return parser.parse_args()

def main():
    # Parse arguments
    args = get_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = Config()
    
    # Update configuration from arguments
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.clip_max_norm = args.clip_max_norm
    
    # Create output directories
    checkpoint_dir = config.checkpoint_path
    log_dir = config.log_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Build model
    print("Building model...")
    model, criterion = caption.build_model(config)
    model = model.to(device)
    
    # Setup optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr * 0.1,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint, device)
        print(f"Loaded checkpoint from {args.checkpoint}, starting from epoch {start_epoch}")
    
    # Build data loaders
    print("Building data loaders...")
    train_loader = utils.build_data_loader(config, mode='train')
    val_loader = utils.build_data_loader(config, mode='val')
    
    # Training loop
    print("Starting training...")
    
    # Log file
    log_file = os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Store training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': -1
    }
    
    # Start training
    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch {epoch}/{config.epochs - 1}")
        
        # Train
        train_loss = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=config.clip_max_norm
        )
        
        # Evaluate
        val_loss = evaluate(
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            device=device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log results
        log_message = f"Epoch {epoch}/{config.epochs - 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        print(log_message)
        
        # Write to log file
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')
        
        # Save checkpoint
        if not args.no_save:
            is_best = val_loss < history['best_val_loss']
            
            if is_best:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
                
                # Save best model
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    filename=os.path.join(checkpoint_dir, 'best_model.pth')
                )
            
            # Save latest model
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                filename=os.path.join(checkpoint_dir, 'latest_model.pth')
            )
            
            # Save epoch model
            if epoch % 5 == 0:  # Save every 5 epochs
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    filename=os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
                )
        
        # Save history to JSON
        with open(os.path.join(log_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
    
    print("Training complete!")
    print(f"Best validation loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")

if __name__ == "__main__":
    main()