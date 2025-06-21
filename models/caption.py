import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizer
from .transformer import Transformer
from .backbone import build_backbone

class Caption(nn.Module):
    """
    CATR Caption model using Transformer architecture
    """
    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        """
        Initialize the Caption model
        
        Args:
            backbone: ResNet backbone for image feature extraction
            transformer: Transformer model for sequence processing
            hidden_dim: Hidden dimension size
            vocab_size: Size of the vocabulary
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, images, captions, cap_masks):
        """
        Forward pass of the Caption model
        
        Args:
            images: Input images or image features
            captions: Input captions (token IDs)
            cap_masks: Caption attention masks
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Extract features from images
        features, masks = self.backbone(images)
        
        # Create visual embeddings
        src = features.flatten(2).permute(2, 0, 1)
        masks = masks.flatten(1)
        
        # Transform sequence
        tgt = captions.permute(1, 0, 2).contiguous()  # [L, B, D]
        memory = self.transformer.encoder(src, src_key_padding_mask=masks)
        out = self.transformer.decoder(tgt, memory, memory_key_padding_mask=masks)
        
        # Generate logits
        logits = self.linear(out)
        
        return logits

def build_model(config):
    """
    Build the CATR Caption model
    
    Args:
        config: Model configuration
        
    Returns:
        tuple: (model, criterion)
    """
    # Build backbone
    backbone = build_backbone()
    
    # Create Transformer
    transformer = Transformer(
        d_model=config.hidden_dim,
        nhead=config.encoder_heads,
        num_encoder_layers=config.encoder_layer,
        num_decoder_layers=config.decoder_layer,
        dim_feedforward=config.encoder_ff_dim,
        dropout=config.dropout
    )
    
    # Create model
    model = Caption(
        backbone=backbone,
        transformer=transformer,
        hidden_dim=config.hidden_dim,
        vocab_size=config.vocab_size
    )
    
    # Set criterion
    criterion = nn.CrossEntropyLoss()
    
    return model, criterion