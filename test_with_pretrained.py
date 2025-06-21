#!/usr/bin/env python3
"""
Test CATR model with pre-trained weights from PyTorch Hub
"""
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test CATR with pre-trained weights')
    parser.add_argument('--image', type=str, default='temp/sample.jpg', 
                        help='Path to test image')
    parser.add_argument('--version', type=str, default='v3',
                        choices=['v1', 'v2', 'v3'],
                        help='Model version to use')
    parser.add_argument('--output', type=str, default='output_caption_pretrained.txt',
                        help='Output file to save caption')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the result')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Load and process image
    print(f"Loading image: {args.image}")
    try:
        image = Image.open(args.image).convert('RGB')
        
        # Transform image (as expected by the model)
        from torchvision import transforms
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
    try:
        # Get tokens
        start_token = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        end_token = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        
        # Create caption template and mask
        max_length = 128  # Maximum caption length
        caption = torch.zeros((1, max_length), dtype=torch.long).to(device)
        mask = torch.ones((1, max_length), dtype=torch.bool).to(device)
        
        # Set start token
        caption[:, 0] = start_token
        mask[:, 0] = False
        
        # Generate tokens sequentially
        with torch.no_grad():
            for i in range(max_length - 1):
                predictions = model(transformed_image, caption, mask)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)
                
                if predicted_id[0] == end_token:
                    break
                
                caption[:, i+1] = predicted_id[0]
                mask[:, i+1] = False
        
        caption_text = tokenizer.decode(caption[0].tolist(), skip_special_tokens=True)
        print(f"Generated caption: {caption_text}")
        
        # Save caption to file
        with open(args.output, 'w') as f:
            f.write(caption_text)
        print(f"Caption saved to {args.output}")
        
        # Visualize if requested
        if args.visualize:
            plt.figure(figsize=(12, 6))
            plt.imshow(image)
            plt.title(caption_text)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('output_visualization_pretrained.png')
            print("Visualization saved to output_visualization_pretrained.png")
            plt.show()
            
    except Exception as e:
        print(f"Error generating caption: {e}")
        print("Full error:")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()