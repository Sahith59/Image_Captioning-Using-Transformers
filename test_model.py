#!/usr/bin/env python3
"""
Quick test script for CATR model
"""
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BertTokenizer

from models import caption
from datasets import coco
from configuration import Config
from utils.caption_utils import greedy_decode

def parse_args():
    parser = argparse.ArgumentParser(description='Test CATR image captioning model')
    parser.add_argument('--image', type=str, default='temp/sample.jpg', 
                        help='Path to test image')
    parser.add_argument('--output', type=str, default='output_caption.txt',
                        help='Output file to save caption')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the result')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = Config()
    
    # Initialize model
    print("Building model from scratch...")
    model, _ = caption.build_model(config)
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load and process image
    print(f"Loading image: {args.image}")
    try:
        image = Image.open(args.image).convert('RGB')
        transformed_image = coco.val_transform(image)
        transformed_image = transformed_image.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Generate caption
    print("Generating caption...")
    try:
        caption_text = greedy_decode(model, transformed_image, config, tokenizer, device)
        print(f"Generated caption: {caption_text}")
        
        # Save caption to file
        with open(args.output, 'w') as f:
            f.write(caption_text)
        print(f"Caption saved to {args.output}")
        
        # Visualize if requested
        if args.visualize:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(image)
            plt.title(caption_text)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('output_visualization.png')
            print("Visualization saved to output_visualization.png")
            plt.show()
            
    except Exception as e:
        print(f"Error generating caption: {e}")
        return

if __name__ == '__main__':
    main()