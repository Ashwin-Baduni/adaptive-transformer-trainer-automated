
from config import get_config
from train import train_model
import sys

try:
    cfg = get_config()
    cfg.update({'batch_size': 8, 'num_epochs': 30, 'lr': 0.0001, 'warmup_steps': 4000, 'preload': None, 'model_folder': 'weights', 'tokenizer_file': 'tokenizer_{0}.json'})
    
    print("Starting training with configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    
    train_model(cfg)
    print("✓ Training completed successfully!")
    
except Exception as e:
    print(f"✗ Training failed: {e}")
    sys.exit(1)
