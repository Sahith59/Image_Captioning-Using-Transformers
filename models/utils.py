import torch
import os

def save_checkpoint(model, optimizer, epoch, filename):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        filename: Path to save checkpoint
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename, device):
    """
    Load model checkpoint
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filename: Path to checkpoint
        device: Device to load model on
        
    Returns:
        int: Epoch number from checkpoint
    """
    if not os.path.exists(filename):
        print(f"Checkpoint {filename} does not exist.")
        return 0
    
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Handle case where epoch is not saved in checkpoint
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded from {filename} at epoch {epoch}")
    return epoch + 1  # Return next epoch

def disable_grad(module):
    """
    Disable gradients for a module
    
    Args:
        module: PyTorch module to disable gradients for
    """
    for param in module.parameters():
        param.requires_grad = False

def enable_grad(module):
    """
    Enable gradients for a module
    
    Args:
        module: PyTorch module to enable gradients for
    """
    for param in module.parameters():
        param.requires_grad = True
        
def print_model_summary(model):
    """
    Print model summary with number of parameters
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
    
    # Print modules
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"Module: {name}, Parameters: {params:,}")