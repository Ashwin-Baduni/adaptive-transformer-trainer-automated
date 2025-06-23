
import os
import torch
from config import get_config
from train import train_model

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    cfg = get_config()
    cfg.update({'batch_size': 1, 'num_epochs': 1, 'seq_len': 64, 'd_model': 128, 'lr': 0.001, 'warmup_steps': 100})
    
    print("Starting training with memory-optimized configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    
    # Enable memory efficient training
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    train_model(cfg)
    print("✓ Training completed successfully!")
    
except torch.cuda.OutOfMemoryError as e:
    print(f"✗ CUDA OOM Error: {e}")
    print("Suggestion: Try reducing batch_size further or use CPU training")
    exit(2)  # Special exit code for OOM
except Exception as e:
    print(f"✗ Training failed: {e}")
    exit(1)
