import torch
from torch.utils.data import DataLoader, DistributedSampler
from .coco import build_dataset

def collate_fn(batch):
    """
    Custom collate function for the COCO dataset
    
    Args:
        batch: List of samples
        
    Returns:
        tuple: (images, masks, captions, caption_masks)
    """
    images, masks, captions, caption_masks = zip(*batch)
    
    # Stack inputs
    images = torch.stack(images)
    masks = torch.stack(masks)
    captions = torch.stack(captions)
    caption_masks = torch.stack(caption_masks)
    
    return images, masks, captions, caption_masks

def build_data_loader(config, mode='train', distributed=False):
    """
    Build data loader for training or validation
    
    Args:
        config: Dataset configuration
        mode (str): 'train' or 'val'
        distributed (bool): Whether to use distributed training
        
    Returns:
        DataLoader: Data loader
    """
    # Build datasets
    train_dataset, val_dataset = build_dataset(config)
    
    # Choose dataset based on mode
    if mode == 'train':
        dataset = train_dataset
        shuffle = True
        batch_size = config.batch_size
    else:  # val
        dataset = val_dataset
        shuffle = False
        batch_size = config.batch_size
    
    # Create sampler for distributed training
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers if hasattr(config, 'num_workers') else 4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return data_loader