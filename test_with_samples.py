#!/usr/bin/env python3
"""
Test CATR model with various sample images
"""
import torch
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from torchvision import transforms
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Test CATR with sample images')
    parser.add_argument('--directory', type=str, default='test_images',
                        help='Directory containing test images')
    parser.add_argument('--version', type=str, default='v3',
                        choices=['v1', 'v2', 'v3'],
                        help='Model version to use')
    parser.add_argument('--output', type=str, default='sample_captions',
                        help='Output directory for visualizations')
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model from PyTorch Hub
    print(f"Loading {args.version} model from PyTorch Hub...")
    try:
        model = torch.hub.load('saahiluppal/catr', args.version, pretrained=True, trust_repo=True)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Get all images in the directory
    image_files = glob.glob(os.path.join(args.directory, "*.jpg")) + \
                  glob.glob(os.path.join(args.directory, "*.jpeg")) + \
                  glob.glob(os.path.join(args.directory, "*.png"))
    
    if not image_files:
        print(f"No image files found in {args.directory}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create a figure for all images
    num_images = len(image_files)
    fig = plt.figure(figsize=(15, 5 * num_images))
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing image: {image_path}")
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            transformed_image = transform(image).unsqueeze(0).to(device)
            
            # Generate caption
            caption_text = generate_caption(model, transformed_image, tokenizer, device)
            print(f"Caption: {caption_text}")
            
            # Save to results file
            with open(os.path.join(args.output, "results.txt"), "a") as f:
                f.write(f"Image: {os.path.basename(image_path)}\n")
                f.write(f"Caption: {caption_text}\n\n")
            
            # Add to visualization
            ax = fig.add_subplot(num_images, 1, i+1)
            ax.imshow(image)
            ax.set_title(f"Caption: {caption_text}")
            ax.axis('off')
            
            # Save individual image with caption
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.title(f"Caption: {caption_text}")
            plt.axis('off')
            plt.tight_layout()
            output_filename = os.path.join(args.output, f"{os.path.splitext(os.path.basename(image_path))[0]}_captioned.png")
            plt.savefig(output_filename)
            plt.close()
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    # Save combined visualization
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "all_results.png"))
    print(f"Results saved to {args.output} directory")

if __name__ == '__main__':
    main()