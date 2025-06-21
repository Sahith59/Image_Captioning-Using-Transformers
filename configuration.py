class Config(object):
    def __init__(self):
        # Model configuration
        self.vocab_size = 30522  # BERT vocabulary size
        self.max_position_embeddings = 128
        self.hidden_dim = 256     # Changed from 768 to match PyTorch Hub model dimensions
        self.encoder_layer = 3
        self.decoder_layer = 6
        self.encoder_heads = 8
        self.decoder_heads = 8
        self.encoder_ff_dim = 2048
        self.decoder_ff_dim = 2048
        self.dropout = 0.1
        
        # Training configuration
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.clip_max_norm = 0.1
        self.batch_size = 32
        self.epochs = 30
        self.patience = 3
        
        # Dataset configuration
        self.train_json = 'annotations/captions_train2017.json'
        self.train_images = 'train2017'
        self.val_json = 'annotations/captions_val2017.json'
        self.val_images = 'val2017'
        self.limit = -1 # No limit
        
        # Path configuration
        self.checkpoint_path = 'checkpoints'
        self.log_path = 'logs'
        
        # Inference configuration
        self.beam_size = 3
        self.max_length = 20