import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer

class CocoCaption(Dataset):
    """
    COCO Caption dataset for image captioning
    """
    def __init__(self, root, ann_file, max_length=128, limit=-1, transform=None, mode='training'):
        """
        Initialize the CocoCaption dataset
        
        Args:
            root (str): Root directory for COCO images
            ann_file (str): Path to annotation file
            max_length (int): Maximum caption length
            limit (int): Maximum number of samples to use (-1 for no limit)
            transform: Image transforms
            mode (str): 'training' or 'validation'
        """
        super().__init__()
        
        self.root = root
        self.max_length = max_length
        self.mode = mode
        
        if transform is not None:
            self.transform = transform
        elif mode == 'training':
            self.transform = train_transform
        else:
            self.transform = val_transform
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create image id to filename mapping
        self.image_id_to_filename = {}
        for image in self.annotations['images']:
            self.image_id_to_filename[image['id']] = image['file_name']
        
        # Create caption list
        self.captions = []
        for annotation in self.annotations['annotations']:
            if annotation['image_id'] in self.image_id_to_filename:
                self.captions.append({
                    'image_id': annotation['image_id'],
                    'caption': annotation['caption']
                })
        
        # Limit dataset size if requested
        if limit > 0:
            self.captions = self.captions[:limit]
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            int: Number of samples
        """
        return len(self.captions)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (image, mask, caption, caption_mask)
        """
        # Get caption and image id
        caption_data = self.captions[idx]
        image_id = caption_data['image_id']
        caption = caption_data['caption']
        
        # Get image path and load image
        image_path = os.path.join(self.root, self.image_id_to_filename[image_id])
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
        
        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # Get caption tokens and mask
        caption = tokenized.input_ids.squeeze()
        caption_mask = tokenized.attention_mask.squeeze().bool()
        
        # Create image mask (no padding)
        mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.bool)
        
        return image, mask, caption, caption_mask


# Normalization constants for ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transform for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Transform for validation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def build_dataset(config):
    """
    Build COCO dataset for training and validation
    
    Args:
        config: Dataset configuration
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Get dataset paths
    train_dir = os.path.join(config.coco_path, config.train_images)
    train_file = os.path.join(config.coco_path, config.train_json)
    val_dir = os.path.join(config.coco_path, config.val_images)
    val_file = os.path.join(config.coco_path, config.val_json)
    
    # Create datasets
    train_dataset = CocoCaption(
        root=train_dir,
        ann_file=train_file,
        max_length=config.max_position_embeddings,
        limit=config.limit,
        transform=train_transform,
        mode='training'
    )
    
    val_dataset = CocoCaption(
        root=val_dir,
        ann_file=val_file,
        max_length=config.max_position_embeddings,
        limit=config.limit,
        transform=val_transform,
        mode='validation'
    )
    
    return train_dataset, val_dataset