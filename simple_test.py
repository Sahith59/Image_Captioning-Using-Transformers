#!/usr/bin/env python3
"""
Simplified test script for CATR - minimal dependencies, easy to run
"""
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from torchvision import transforms

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model from PyTorch Hub
    print("Loading model...")
    try:
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True, trust_repo=True)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load and process image - using a sample image path
    image_path = 'temp/sample.jpg'  # Change this to your image path
    print(f"Loading image: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transformed_image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Generate caption
    print("Generating caption...")
    caption_text = generate_caption(model, transformed_image, tokenizer, device)
    print(f"Generated caption: {caption_text}")
    
    # Display image with caption
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(caption_text)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('caption_result.png')
    print("Result saved as caption_result.png")

def generate_caption(model, image, tokenizer, device, max_length=128):
    """Generate image caption using greedy decoding"""
    # Get tokens
    start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    
    # Create caption and mask
    caption = torch.zeros((1, max_length), dtype=torch.long).to(device)
    mask = torch.ones((1, max_length), dtype=torch.bool).to(device)
    
    # Set start token
    caption[:, 0] = start_token
    mask[:, 0] = False
    
    # Generate tokens
    with torch.no_grad():
        for i in range(max_length - 1):
            # Forward pass
            predictions = model(image, caption, mask)
            # Get prediction for current token position
            predictions = predictions[:, i, :]
            # Get most likely token
            predicted_id = torch.argmax(predictions, axis=-1)
            
            # Stop if end token is generated
            if predicted_id[0] == end_token:
                break
            
            # Add predicted token to caption and update mask
            caption[:, i+1] = predicted_id[0]
            mask[:, i+1] = False
    
    # Convert token IDs to text
    caption_text = tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
    return caption_text

if __name__ == '__main__':
    main()