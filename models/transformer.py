import copy
import torch
import torch.nn.functional as F
from torch import nn

class Transformer(nn.Module):
    """
    Transformer model for image captioning
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Initialize the Transformer model
        
        Args:
            d_model (int): Hidden dimension size
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Create encoder and decoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        
        # Stack layers
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Initialize parameters
        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        """
        Initialize model parameters
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module
    """
    def __init__(self, encoder_layer, num_layers):
        """
        Initialize the TransformerEncoder
        
        Args:
            encoder_layer: Single encoder layer to be stacked
            num_layers (int): Number of encoder layers
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
    
    def forward(self, src, src_key_padding_mask=None):
        """
        Forward pass of the TransformerEncoder
        
        Args:
            src: Source sequence
            src_key_padding_mask: Source padding mask
            
        Returns:
            torch.Tensor: Encoded output
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
            
        return output

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder module
    """
    def __init__(self, decoder_layer, num_layers):
        """
        Initialize the TransformerDecoder
        
        Args:
            decoder_layer: Single decoder layer to be stacked
            num_layers (int): Number of decoder layers
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the TransformerDecoder
        
        Args:
            tgt: Target sequence
            memory: Memory from encoder
            tgt_mask: Target mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
            
        Returns:
            torch.Tensor: Decoded output
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
            
        return output

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        Initialize the TransformerEncoderLayer
        
        Args:
            d_model (int): Hidden dimension size
            nhead (int): Number of attention heads
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Self-attention and normalization
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feedforward network and normalization
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation function
        self.activation = F.relu
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass of the TransformerEncoderLayer
        
        Args:
            src: Source sequence
            src_mask: Source mask
            src_key_padding_mask: Source padding mask
            
        Returns:
            torch.Tensor: Encoded output
        """
        # Self-attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src

class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        Initialize the TransformerDecoderLayer
        
        Args:
            d_model (int): Hidden dimension size
            nhead (int): Number of attention heads
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
        """
        super().__init__()
        
        # Self-attention and normalization
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention and normalization
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network and normalization
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Activation function
        self.activation = F.relu
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass of the TransformerDecoderLayer
        
        Args:
            tgt: Target sequence
            memory: Memory from encoder
            tgt_mask: Target mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
            
        Returns:
            torch.Tensor: Decoded output
        """
        # Self-attention block
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention block
        tgt2, attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Save attention weights for visualization
        self.attn = attn
        
        # Feedforward block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt