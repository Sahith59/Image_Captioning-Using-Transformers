import argparse
import os
import torch
from PIL import Image
import io
from transformers import BertTokenizer
from gtts import gTTS
import matplotlib.pyplot as plt

from models import caption
from datasets import coco, utils
from configuration import Config
from utils.caption_utils import greedy_decode, beam_search_decode, visualize_caption_generation
from utils.audio_utils import generate_speech_from_text, save_audio_to_file

def get_args():
    parser = argparse.ArgumentParser(description='CATR Image Captioning')
    parser.add_argument('--path', type=str, help='Path to image', required=True)
    parser.add_argument('--v', type=str, help='Model version', default='v3', choices=['v1', 'v2', 'v3', 'custom'])
    parser.add_argument('--checkpoint', type=str, help='Path to custom checkpoint', default=None)
    parser.add_argument('--output', type=str, help='Output directory', default='./output')
    parser.add_argument('--no-audio', action='store_true', help='Disable audio generation')
    parser.add_argument('--beam-size', type=int, help='Beam size for caption generation', default=3)
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    return parser.parse_args()

def main():
    # Parse arguments
    args = get_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Get image path and version
    image_path = args.path
    version = args.v
    
    # Image filename (for saving)
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load configuration
    config = Config()
    
    # Load model
    print("Loading model...")
    if version in ['v1', 'v2', 'v3']:
        try:
            # Use trust_repo=True to avoid interactive prompts
            model = torch.hub.load('saahiluppal/catr', version, pretrained=True, trust_repo=True)
            print(f"Loaded pretrained {version} model from torch hub")
        except Exception as e:
            print(f"Failed to load model from torch hub: {e}")
            print("Building model from scratch...")
            model, _ = caption.build_model(config)
            if args.checkpoint and os.path.exists(args.checkpoint):
                try:
                    checkpoint = torch.load(args.checkpoint, map_location='cpu')
                    model.load_state_dict(checkpoint['model'])
                    print(f"Loaded checkpoint from {args.checkpoint}")
                except Exception as load_error:
                    print(f"Error loading checkpoint: {load_error}")
                    print("Using randomly initialized model.")
            else:
                print("No checkpoint provided. Using randomly initialized model.")
    else:  # custom model
        model, _ = caption.build_model(config)
        if args.checkpoint and os.path.exists(args.checkpoint):
            try:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
                print(f"Loaded checkpoint from {args.checkpoint}")
            except Exception as load_error:
                print(f"Error loading checkpoint: {load_error}")
                print("Using randomly initialized model.")
        else:
            print("No checkpoint provided for custom model. Using randomly initialized model.")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load and process image
    print(f"Processing image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
        transformed_image = coco.val_transform(image)
        transformed_image = transformed_image.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Generate caption
    print("Generating caption...")
    if args.beam_size > 1:
        caption_text = beam_search_decode(model, transformed_image, config, tokenizer, device, beam_size=args.beam_size)
    else:
        caption_text = greedy_decode(model, transformed_image, config, tokenizer, device)
    
    # Print and save caption
    print(f"Generated caption: {caption_text}")
    caption_path = os.path.join(args.output, f"{image_filename}_caption.txt")
    with open(caption_path, 'w') as f:
        f.write(caption_text)
    print(f"Caption saved to {caption_path}")
    
    # Generate and save audio if enabled
    if not args.no_audio:
        print("Generating audio...")
        try:
            audio_bytes = generate_speech_from_text(caption_text)
            audio_path = os.path.join(args.output, f"{image_filename}_audio.mp3")
            
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes.getvalue())
            
            print(f"Audio saved to {audio_path}")
        except Exception as e:
            print(f"Error generating audio: {e}")
    
    # Generate visualization if enabled
    if args.visualize:
        print("Generating visualization...")
        try:
            _, fig = visualize_caption_generation(model, image_path, config, tokenizer, device)
            vis_path = os.path.join(args.output, f"{image_filename}_visualization.png")
            fig.savefig(vis_path)
            plt.close(fig)
            print(f"Visualization saved to {vis_path}")
        except Exception as e:
            print(f"Error generating visualization: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()