import torch
import time
import datetime
from tqdm import tqdm
import math

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm=0.1, log_step=100):
    """
    Train the model for one epoch
    
    Args:
        model: CATR model
        criterion: Loss function
        data_loader: Training data loader
        optimizer: Model optimizer
        device: Device to run training on
        epoch: Current epoch number
        max_norm: Max gradient norm for clipping
        log_step: How often to print logs
        
    Returns:
        float: Average loss for this epoch
    """
    model.train()
    criterion.train()
    
    total_loss = 0
    n_steps = len(data_loader)
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), 
                desc=f"Epoch {epoch} - Training", ncols=100)
    
    for step, (images, _, captions, cap_masks) in pbar:
        # Move data to device
        images = images.to(device)
        captions = captions.to(device)
        cap_masks = cap_masks.to(device)
        
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, captions[:, :-1], cap_masks[:, :-1])
        
        # Calculate loss (shift targets because we're predicting next token)
        targets = captions[:, 1:].reshape(-1)
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Update weights
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Print logs
        if (step + 1) % log_step == 0:
            # Calculate time per batch
            elapsed = time.time() - start_time
            time_per_batch = elapsed / (step + 1)
            remaining = time_per_batch * (n_steps - step - 1)
            
            # Format remaining time
            remaining_str = str(datetime.timedelta(seconds=int(remaining)))
            
            # Log progress
            print(f"Epoch {epoch} [{step+1}/{n_steps}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Time left: {remaining_str}")
    
    # Calculate average loss
    avg_loss = total_loss / n_steps
    
    print(f"Epoch {epoch} - Training completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, criterion, data_loader, device):
    """
    Evaluate the model on validation data
    
    Args:
        model: CATR model
        criterion: Loss function
        data_loader: Validation data loader
        device: Device to run evaluation on
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    criterion.eval()
    
    total_loss = 0
    n_steps = len(data_loader)
    
    # Create progress bar
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), 
                desc="Validation", ncols=100)
    
    with torch.no_grad():
        for step, (images, _, captions, cap_masks) in pbar:
            # Move data to device
            images = images.to(device)
            captions = captions.to(device)
            cap_masks = cap_masks.to(device)
            
            # Forward pass
            outputs = model(images, captions[:, :-1], cap_masks[:, :-1])
            
            # Calculate loss
            targets = captions[:, 1:].reshape(-1)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets)
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate average loss
    avg_loss = total_loss / n_steps
    
    print(f"Validation completed. Average Loss: {avg_loss:.4f}")
    return avg_loss