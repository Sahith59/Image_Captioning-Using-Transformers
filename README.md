# CATR: Running Instructions

This guide provides step-by-step instructions to run the CATR (Caption Transformer) project.

## Setup Environment

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv catr-env

# Activate the environment (macOS/Linux)
source catr-env/bin/activate

# Activate the environment (Windows)
catr-env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Model Limitations and Workarounds

The project uses pre-trained models from PyTorch Hub, but there may be compatibility issues with the latest PyTorch versions. The code has been updated to handle these issues:

1. The default implementation uses a randomly initialized model which will generate placeholder captions
2. For proper image captioning, you need to train your own model using the training script

## Using the Streamlit Web Interface

Run the Streamlit application to interact with the model through a web interface:

```bash
streamlit run app.py
```

This starts a local server and opens the application in your web browser, where you can:
1. Upload images
2. Generate captions
3. Hear audio versions of the captions

## Command-Line Usage

For batch processing or headless operations, use the predict.py script:

```bash
python predict.py --path path/to/image.jpg --v v3 --output ./output
```

Options:
- `--path`: Path to the input image (required)
- `--v`: Model version (v1, v2, v3)
- `--output`: Output directory for captions and audio
- `--no-audio`: Skip audio generation
- `--beam-size`: Beam size for caption generation (default: 3)

## Training Your Own Model

To train a model on the COCO dataset:

1. Download the COCO dataset:
```bash
# Create data directory
mkdir -p data/coco
cd data/coco

# Download images and annotations
# (See software manual for detailed download instructions)
```

2. Run the training script:
```bash
python train.py --batch_size 32 --epochs 30 --lr 1e-4
```

## Troubleshooting

### Model Loading Issues
- If you see errors about model loading from PyTorch Hub, the code will fall back to a randomly initialized model
- This will produce generic captions until you train your own model

### TokenizerError
- If you see tokenizer errors, make sure you're using the correct attribute names for BERT tokenizer
- Recent versions use tokenizer.cls_token instead of tokenizer._cls_token

### CUDA/GPU Issues
- The code automatically falls back to CPU if CUDA is not available
- For faster performance, use a CUDA-capable GPU if possible

### Audio Generation Issues
- If audio generation fails, check that the gTTS library is properly installed
- Ensure you have an internet connection (required for gTTS)
- Use the `--no-audio` flag to skip audio generation

## Project Structure
```
catr-project/
├── app.py                  # Streamlit web interface
├── predict.py              # Command-line prediction script
├── train.py                # Training script
├── setup.py                # Setup and installation
├── configuration.py        # Model configuration
├── models/                 # Model architecture
├── datasets/               # Dataset handling
├── utils/                  # Utility functions
├── checkpoints/            # Saved models
└── output/                 # Generated captions
```

## Next Steps

1. Train your own model on the COCO dataset
2. Experiment with different model configurations
3. Use the trained model for generating captions

## Author Information

Developed by Sahith Reddy Thummala and Manish Yerram at Georgia State University.