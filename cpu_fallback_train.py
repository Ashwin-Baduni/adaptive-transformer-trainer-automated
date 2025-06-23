
import os
import torch
from config import get_config
from train import train_model

try:
    # Force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    cfg = get_config()
    cfg.update({'batch_size': 1, 'num_epochs': 3, 'seq_len': 128, 'd_model': 256, 'lr': 0.001, 'warmup_steps': 100, 'preload': None, 'model_folder': 'weights', 'tokenizer_file': 'tokenizer_{0}.json'})
    
    print("Starting CPU training (minimal configuration):")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    
    train_model(cfg)
    print("✓ CPU Training completed successfully!")
    
except Exception as e:
    print(f"✗ CPU Training failed: {e}")
    exit(1)
