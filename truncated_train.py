
import os
import torch
from config import get_config
from train import train_model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    cfg = get_config()
    cfg.update({'batch_size': 1, 'num_epochs': 1, 'seq_len': 100, 'd_model': 256, 'lr': 0.0001, 'warmup_steps': 500, 'preload': None, 'model_folder': 'weights', 'tokenizer_file': 'tokenizer_{0}.json'})
    
    print("Starting training with truncated dataset:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    
    train_model(cfg)
    print("✓ Training completed successfully!")
    
except Exception as e:
    print(f"✗ Training failed: {e}")
    exit(1)
