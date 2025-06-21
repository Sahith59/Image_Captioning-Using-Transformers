import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

class PositionEmbeddingSine(nn.Module):
    """
    Positional encoding using sine and cosine functions
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        Initialize the PositionEmbeddingSine
        
        Args:
            num_pos_feats (int): Number of positional features
            temperature (float): Temperature for scaling
            normalize (bool): Whether to normalize coordinates
            scale (float): Optional scaling factor
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale

    def forward(self, mask):
        """
        Forward pass of the PositionEmbeddingSine
        
        Args:
            mask: Input mask (B, H, W)
            
        Returns:
            torch.Tensor: Position embeddings
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where running_mean and running_var are fixed
    """
    def __init__(self, n):
        """
        Initialize the FrozenBatchNorm2d
        
        Args:
            n (int): Number of channels
        """
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        """
        Forward pass of the FrozenBatchNorm2d
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Normalized output
        """
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        
        return x * scale + bias

class BackboneBase(nn.Module):
    """
    Base class for backbones
    """
    def __init__(self, backbone, train_backbone, num_channels, return_interm_layers):
        """
        Initialize the BackboneBase
        
        Args:
            backbone: Backbone network
            train_backbone (bool): Whether to train the backbone
            num_channels (int): Number of output channels
            return_interm_layers (bool): Whether to return intermediate layers
        """
        super().__init__()
        
        # Set trainable parameters
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        
        self.body = backbone
        self.num_channels = num_channels
        
        # Initialize position embeddings
        hidden_dim = 256
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    
    def forward(self, tensor_list):
        """
        Forward pass of the BackboneBase
        
        Args:
            tensor_list: Input tensor
            
        Returns:
            tuple: (features, position embeddings)
        """
        xs = self.body.conv1(tensor_list)
        xs = self.body.bn1(xs)
        xs = self.body.relu(xs)
        xs = self.body.maxpool(xs)

        xs = self.body.layer1(xs)
        xs = self.body.layer2(xs)
        xs = self.body.layer3(xs)
        xs = self.body.layer4(xs)
        
        # Project features
        features = self.input_proj(xs)
        
        # Generate masks
        masks = torch.zeros_like(features[:, 0], dtype=torch.bool)
        
        # Generate position embeddings
        pos = self.position_embedding(masks)
        
        return features, masks

class Backbone(BackboneBase):
    """
    ResNet backbone with frozen BatchNorm
    """
    def __init__(self, name, train_backbone, return_interm_layers, dilation):
        """
        Initialize the Backbone
        
        Args:
            name (str): Backbone name
            train_backbone (bool): Whether to train the backbone
            return_interm_layers (bool): Whether to return intermediate layers
            dilation (bool): Whether to replace stride with dilation
        """
        # Replace BatchNorm with FrozenBatchNorm
        norm_layer = FrozenBatchNorm2d
        
        # Create backbone
        if name == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, norm_layer=norm_layer)
        else:
            raise ValueError(f"Backbone {name} not supported")
        
        # Set dilation
        if dilation:
            backbone.layer4.apply(lambda m: setattr(m, 'dilation', 2))
            backbone.layer4.apply(lambda m: setattr(m, 'padding', 2))
        
        num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

def build_backbone():
    """
    Build the ResNet backbone for feature extraction
    
    Returns:
        Backbone: ResNet backbone
    """
    train_backbone = True
    return_interm_layers = False
    dilation = False
    
    backbone = Backbone('resnet50', train_backbone, return_interm_layers, dilation)
    
    return backbone