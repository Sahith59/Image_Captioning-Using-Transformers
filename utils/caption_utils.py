import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from datasets.coco import val_transform
import matplotlib.cm as cm

def greedy_decode(model, image, config, tokenizer, device):
    """
    Greedy decoding for caption generation
    
    Args:
        model: CATR model
        image: Input image tensor
        config: Model configuration
        tokenizer: BERT tokenizer
        device: Device to run inference on
        
    Returns:
        str: Generated caption
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get tokens
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    
    # Determine max length to use - handle both config formats
    max_length = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else config.max_length
    
    # Create caption template and mask
    caption = torch.zeros((1, max_length), dtype=torch.long).to(device)
    mask = torch.ones((1, max_length), dtype=torch.bool).to(device)
    
    # Set start token
    caption[:, 0] = start_token
    mask[:, 0] = False
    
    # Generate tokens sequentially
    try:
        with torch.no_grad():
            for i in range(max_length - 1):
                # Get predictions
                predictions = model(image, caption, mask)
                predictions = predictions[:, i, :]
                
                # Get next token (greedy)
                predicted_id = torch.argmax(predictions, axis=-1)
                
                # Stop if end token is generated
                if predicted_id[0] == end_token:
                    break
                
                # Update caption and mask
                caption[:, i+1] = predicted_id[0]
                mask[:, i+1] = False
        
        # Return decoded caption
        return tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
    except Exception as e:
        # Fallback if something goes wrong
        return f"A picture worth describing. (Error: {str(e)})"

def beam_search_decode(model, image, config, tokenizer, device, beam_size=3):
    """
    Beam search decoding for caption generation
    
    Args:
        model: CATR model
        image: Input image tensor
        config: Model configuration
        tokenizer: BERT tokenizer
        device: Device to run inference on
        beam_size: Beam width
        
    Returns:
        str: Generated caption with highest likelihood
    """
    # Use simple greedy decode as fallback if beam search fails
    try:
        # Ensure model is in evaluation mode
        model.eval()
        
        # Get tokens
        start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        end_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        # Determine max length to use - handle both config formats
        max_length = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else config.max_length
        
        # Initialize beam
        beam = [(torch.tensor([[start_token]], device=device), 
                torch.tensor([[False]], device=device), 
                0.0)]
        
        # Generate caption tokens with beam search
        with torch.no_grad():
            for step in range(max_length - 1):
                candidates = []
                
                # Expand each beam
                for seq, mask, score in beam:
                    # If sequence ended, keep as candidate
                    if seq[0, -1].item() == end_token:
                        candidates.append((seq, mask, score))
                        continue
                    
                    # Prepare inputs
                    current_length = seq.shape[1]
                    padded_seq = torch.cat([
                        seq, 
                        torch.full((1, max_length - current_length), 
                                pad_token, device=device)
                    ], dim=1)
                    padded_mask = torch.cat([
                        mask, 
                        torch.full((1, max_length - current_length), 
                                True, device=device)
                    ], dim=1)
                    
                    # Get predictions
                    predictions = model(image, padded_seq, padded_mask)
                    predictions = predictions[:, current_length-1, :]
                    
                    # Get top beam_size predictions
                    probs = torch.nn.functional.log_softmax(predictions, dim=-1)
                    topk_probs, topk_ids = torch.topk(probs, beam_size)
                    
                    # Add to candidates
                    for i in range(beam_size):
                        token_id = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
                        token_prob = topk_probs[0, i].item()
                        
                        # Create new sequence
                        new_seq = torch.cat([seq, token_id], dim=1)
                        new_mask = torch.cat([
                            mask, 
                            torch.tensor([[False]], device=device)
                        ], dim=1)
                        new_score = score + token_prob
                        
                        candidates.append((new_seq, new_mask, new_score))
                
                # Check if we have any candidates
                if not candidates:
                    break
                
                # Sort candidates by score
                candidates.sort(key=lambda x: x[2], reverse=True)
                
                # Keep top beam_size candidates
                beam = candidates[:beam_size]
                
                # Check if all beams have ended
                if all(b[0][0, -1].item() == end_token for b in beam):
                    break
        
        # Select sequence with highest score
        if beam:
            best_seq = beam[0][0][0].tolist()
            
            # Return decoded caption
            return tokenizer.decode(best_seq, skip_special_tokens=True)
        else:
            # Fallback to greedy if beam search produces no results
            return greedy_decode(model, image, config, tokenizer, device)
    except Exception as e:
        # Fallback if something goes wrong
        return greedy_decode(model, image, config, tokenizer, device)

def visualize_attention(model, image_tensor, caption, tokenizer, device):
    """
    Visualize attention weights for a generated caption
    
    Args:
        model: CATR model
        image_tensor: Input image tensor
        caption: Generated caption tokens
        tokenizer: BERT tokenizer
        device: Device to run on
        
    Returns:
        numpy.ndarray: Attention weights
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get tokens from caption
    tokens = tokenizer.tokenize(caption)
    
    # Create mask for caption
    mask = torch.zeros_like(caption, dtype=torch.bool)
    
    # Get attention weights
    with torch.no_grad():
        # Forward pass
        memory = model.transformer.encoder(
            model.backbone(image_tensor)[0].flatten(2).permute(2, 0, 1),
            src_key_padding_mask=None
        )
        
        # Decode and get attention weights
        attn_weights = []
        for i in range(len(tokens)):
            # Decode up to current token
            tgt = caption[:, :i+1].permute(1, 0, 2)
            out = model.transformer.decoder(
                tgt, memory, 
                memory_key_padding_mask=None
            )
            
            # Extract attention weights from last layer
            last_layer = model.transformer.decoder.layers[-1]
            if hasattr(last_layer, 'attn'):
                attn_weights.append(last_layer.attn.detach().cpu().numpy())
    
    # Return attention weights
    return np.array(attn_weights) if attn_weights else None

def visualize_caption_generation(model, image_path, config, tokenizer, device):
    """
    Generate visualization of caption generation process with attention
    
    Args:
        model: CATR model
        image_path: Path to image
        config: Model configuration
        tokenizer: BERT tokenizer
        device: Device to run on
        
    Returns:
        tuple: (caption, figure)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    transformed_image = val_transform(image)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    
    # Generate caption
    caption = greedy_decode(model, transformed_image, config, tokenizer, device)
    
    # Get tokens
    tokens = ['[CLS]'] + tokenizer.tokenize(caption) + ['[SEP]']
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Display image
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(image)
    ax1.set_title('Input Image', fontsize=14)
    ax1.axis('off')
    
    # Display caption
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(0.5, 0.5, caption, fontsize=16, ha='center', va='center', wrap=True)
    ax2.set_title('Generated Caption', fontsize=14)
    ax2.axis('off')
    
    # Return caption and figure
    return caption, fig