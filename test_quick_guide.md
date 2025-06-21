# Quick Guide for Testing the CATR Model

This guide provides several ways to quickly test the CATR image captioning model with both pre-trained weights and your own trained models.

## 1. Simplest Test (Pre-trained Model)

Run the simplified test script to quickly see results:

```bash
# Make sure you're in the virtual environment
source catr-env/bin/activate

# Run the simplified test
python simple_test.py
```

This will:
- Load a pre-trained model from PyTorch Hub
- Generate a caption for the default image (temp/sample.jpg)
- Save the result as caption_result.png

To test on your own image, edit line 24 in `simple_test.py` to point to your image file.

## 2. Testing the Pre-trained Model with Options

For more control, use the pre-trained model test script:

```bash
# Test with a custom image
python test_with_pretrained.py --image path/to/your/image.jpg

# Try different model versions
python test_with_pretrained.py --version v1
python test_with_pretrained.py --version v2
python test_with_pretrained.py --version v3

# Save visualization
python test_with_pretrained.py --visualize
```

## 3. Testing a Custom-Trained Model

After training your own model, test it with:

```bash
# Basic test with default settings
python test_model.py

# Test with your trained model (replace path as needed)
python predict.py --path path/to/image.jpg --v custom --checkpoint checkpoints/best_model.pth
```

## 4. Using the Web Interface

The Streamlit web interface provides the most user-friendly way to test:

```bash
streamlit run app.py
```

In the web interface:
1. Select the model version (v1, v2, v3, or custom)
2. Upload an image
3. If using a custom model, upload your checkpoint file
4. Get the caption and audio output

## 5. Batch Testing

To test multiple images at once, you can create a simple batch script:

```python
import os
import subprocess
from glob import glob

# Path to your images
image_dir = "path/to/your/images"

# Get all image files
image_files = glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))

# Run prediction on each image
for img_path in image_files:
    output_name = os.path.basename(img_path).split('.')[0]
    cmd = f"python predict.py --path {img_path} --output ./output/{output_name}_caption.txt"
    subprocess.run(cmd, shell=True)
```

## Understanding the Results

The pre-trained models from PyTorch Hub may produce generic or repetitive captions if they don't recognize specific objects in your images. For better results, you should train your own model using the COCO dataset as described in the training guide.

Common issues with the pre-trained models:
- Repetitive captions like "a blue sky with a blue sky"
- Generic descriptions that don't capture the image details
- Missing objects or activities shown in the image

If you see these issues, it's recommended to train your own model or fine-tune the existing one to get more accurate and specific captions.

## Next Steps

- If the pre-trained models work well for your use case, you can continue using them
- For better results, train your own model following the detailed instructions in training_guide.md
- Experiment with different model configurations to improve caption quality