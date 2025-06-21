# CATR: Training Your Own Image Captioning Model

This guide provides detailed instructions on how to train your own transformer-based image captioning model using the CATR framework.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Configuration Settings](#configuration-settings)
4. [Training Process](#training-process)
5. [Monitoring and Evaluation](#monitoring-and-evaluation)
6. [Fine-tuning](#fine-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Using Custom Checkpoints](#using-custom-checkpoints)

## Prerequisites

Before starting the training process, ensure you have:

- A system with adequate computational resources (preferably with a GPU)
- Python 3.7+ installed
- The CATR repository cloned and dependencies installed
- At least 16GB of RAM (32GB+ recommended for larger batch sizes)
- At least 100GB of free disk space for the COCO dataset

## Dataset Preparation

### 1. Download the COCO Dataset

The CATR model uses the COCO (Common Objects in Context) dataset, which contains over 120,000 images and 5 captions per image.

```bash
# Create a directory for the COCO dataset
mkdir -p data/coco
cd data/coco

# Download the 2017 training images (~18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Download the 2017 validation images (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Download the annotations (~241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Remove the zip files to save space (optional)
rm *.zip
```

### 2. Set Up Dataset Configuration

Update the configuration settings in `configuration.py` to point to your COCO dataset location:

```python
# Dataset configuration
self.train_json = 'data/coco/annotations/captions_train2017.json'
self.train_images = 'data/coco/train2017'
self.val_json = 'data/coco/annotations/captions_val2017.json'
self.val_images = 'data/coco/val2017'
```

If you want to train on a smaller subset of the data for faster iteration (useful for debugging or testing changes), you can set the `limit` parameter in the configuration:

```python
# Limit the number of training samples (set to -1 for full dataset)
self.limit = 10000  # Use only 10,000 samples for training
```

## Configuration Settings

Before starting training, review and adjust the configuration settings in `configuration.py`:

### Model Architecture Settings

```python
# Model configuration
self.vocab_size = 30522  # BERT vocabulary size
self.max_position_embeddings = 128  # Maximum caption length
self.hidden_dim = 256  # Hidden dimension size
self.encoder_layer = 3  # Number of encoder layers
self.decoder_layer = 6  # Number of decoder layers
self.encoder_heads = 8  # Number of encoder attention heads
self.decoder_heads = 8  # Number of decoder attention heads
self.encoder_ff_dim = 2048  # Encoder feed-forward dimension
self.decoder_ff_dim = 2048  # Decoder feed-forward dimension
self.dropout = 0.1  # Dropout rate
```

### Training Settings

```python
# Training configuration
self.lr = 1e-4  # Learning rate
self.weight_decay = 1e-4  # Weight decay for regularization
self.clip_max_norm = 0.1  # Gradient clipping norm
self.batch_size = 32  # Batch size
self.epochs = 30  # Number of training epochs
self.patience = 3  # Early stopping patience
```

### Adjustments for Different Hardware

- For systems with less memory, reduce the batch size (e.g., 8 or 16)
- For faster training but potentially lower quality, reduce encoder/decoder layers
- For better quality at the cost of training time, increase hidden dimensions or layers

## Training Process

### 1. Basic Training Command

To start training with default settings:

```bash
python train.py
```

### 2. Training with Custom Parameters

You can customize various training parameters via command-line arguments:

```bash
python train.py --batch_size 16 --epochs 50 --lr 5e-5 --weight_decay 1e-5
```

Full list of available parameters:

```
--batch_size: Batch size (default: 32)
--epochs: Number of training epochs (default: 30)
--lr: Learning rate (default: 1e-4)
--weight_decay: Weight decay (default: 1e-4)
--clip_max_norm: Gradient clipping max norm (default: 0.1)
--checkpoint: Path to checkpoint to resume from
--no_save: Do not save checkpoints
--log_step: Print logs every n steps (default: 100)
```

### 3. Resuming Training from a Checkpoint

If your training was interrupted, you can resume from a saved checkpoint:

```bash
python train.py --checkpoint checkpoints/model_epoch_10.pth
```

## Monitoring and Evaluation

### Training Logs

During training, progress is logged to the console and saved to log files in the `logs` directory. Each training run creates a new log file with a timestamp:

```
logs/training_log_20250426_123456.txt
```

The logs contain information about training and validation loss for each epoch:

```
Epoch 0/29, Train Loss: 3.2456, Val Loss: 2.9876
Epoch 1/29, Train Loss: 2.8765, Val Loss: 2.6543
...
```

### Training History

A JSON file (`logs/history.json`) stores the complete training history, including:
- Training loss per epoch
- Validation loss per epoch
- Best validation loss
- Best epoch

You can visualize this history using tools like matplotlib to create learning curves.

### Model Checkpoints

The training script automatically saves:
- The latest model after each epoch (`checkpoints/latest_model.pth`)
- The best model based on validation loss (`checkpoints/best_model.pth`)
- Periodic checkpoints every 5 epochs (`checkpoints/model_epoch_X.pth`)

## Fine-tuning

After initial training, you may want to fine-tune the model to improve performance:

### Lower Learning Rate

```bash
python train.py --checkpoint checkpoints/best_model.pth --lr 1e-5 --epochs 10
```

### Unfreeze Backbone

By default, the ResNet backbone is partially frozen during training. To unfreeze it for fine-tuning, modify the `train.py` script:

```python
# In train.py, change the param_dicts setting to use the same learning rate for all parameters
param_dicts = [
    {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
]
```

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce the batch size: `--batch_size 8`
2. Reduce model size by editing `configuration.py`:
   ```python
   self.hidden_dim = 128  # Reduced from 256
   self.encoder_layer = 2  # Reduced from 3
   self.decoder_layer = 4  # Reduced from 6
   ```

### Slow Training

1. Check that you're using a GPU: Output should show `Using device: cuda`
2. Increase batch size if memory allows: `--batch_size 64`
3. Use mixed precision training (add to `train.py`):
   ```python
   # Near the top of train.py
   scaler = torch.cuda.amp.GradScaler()
   
   # In the training loop, replace the forward and backward pass with:
   with torch.cuda.amp.autocast():
       outputs = model(images, captions, caption_masks)
       loss = criterion(outputs.permute(0, 2, 1), targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

### Overfitting

If your training loss is much lower than validation loss:

1. Increase dropout: Edit `dropout` in `configuration.py` to 0.2 or 0.3
2. Add weight decay: `--weight_decay 1e-4`
3. Use early stopping: The code already implements this with the `patience` parameter

## Using Custom Checkpoints

After training, you can use your custom model with the prediction script:

```bash
python predict.py --path path/to/image.jpg --v custom --checkpoint checkpoints/best_model.pth
```

Or with the web interface by selecting "Use custom checkpoint" in the sidebar.

## Advanced Training Techniques

### Learning Rate Scheduling

Add a learning rate scheduler to your training script:

```python
# In train.py, after creating the optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# After validation in the epoch loop
scheduler.step(val_loss)
```

### Data Augmentation

Enhance dataset variety by adding more transformations in `datasets/coco.py`:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
```

### Hyperparameter Tuning

For systematic hyperparameter optimization, consider using tools like Optuna:

```bash
pip install optuna
```

Then create a script that runs multiple training runs with different parameters and tracks the results.

## Final Tips

1. **Start Small**: Begin with a small subset of data to ensure your pipeline works
2. **Validate Early**: Check validation loss frequently to catch issues early
3. **Save Frequently**: Create checkpoints often in case training is interrupted
4. **Monitor Resources**: Keep an eye on GPU memory and disk space
5. **Evaluate Qualitatively**: Look at generated captions on test images, not just loss values

With these detailed steps and tips, you should be able to successfully train your own CATR image captioning model that can generate high-quality captions for your images.