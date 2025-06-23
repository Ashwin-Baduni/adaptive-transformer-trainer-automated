#!/usr/bin/env python3
"""
Transformer Model Training Script

This script configures and trains a transformer model with customizable parameters
and proper logging/monitoring capabilities.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TransformerTrainer:
    """Class to handle transformer model training with configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to config file (optional)
        """
        try:
            from config import get_config
            from train import train_model
            
            self.get_config = get_config
            self.train_model = train_model
            self.base_config = self.get_config() if config_path is None else self.get_config(config_path)
            
            logger.info("✓ Successfully imported training modules")
            
        except ImportError as e:
            logger.error(f"✗ Failed to import required modules: {e}")
            raise
    
    def create_training_config(self, **overrides) -> Dict[str, Any]:
        """
        Create training configuration with custom overrides.
        
        Args:
            **overrides: Configuration parameters to override
            
        Returns:
            Training configuration dictionary
        """
        config = self.base_config.copy()
        config.update(overrides)
        
        # Log configuration changes
        logger.info("Training Configuration:")
        for key, value in config.items():
            if key in overrides:
                logger.info(f"  {key}: {value} (overridden)")
            else:
                logger.info(f"  {key}: {value}")
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate training configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
        """
        # Check required parameters
        required_params = ['batch_size', 'num_epochs', 'lr', 'seq_len']
        missing_params = [param for param in required_params if param not in config]
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Validate parameter ranges
        if config['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")
        
        if config['num_epochs'] <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if config['lr'] <= 0:
            raise ValueError("Learning rate must be positive")
        
        logger.info("✓ Configuration validation passed")
    
    def setup_training_environment(self, config: Dict[str, Any]) -> None:
        """
        Setup training environment and directories.
        
        Args:
            config: Training configuration
        """
        # Create model directory if it doesn't exist
        if 'model_folder' in config:
            model_dir = Path(config['model_folder'])
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Model directory ready: {model_dir}")
        
        # Create tokenizer directory if specified
        if 'tokenizer_file' in config:
            tokenizer_path = Path(config['tokenizer_file'])
            tokenizer_dir = tokenizer_path.parent
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Tokenizer directory ready: {tokenizer_dir}")
    
    def train(self, **config_overrides) -> None:
        """
        Train the model with specified configuration.
        
        Args:
            **config_overrides: Configuration parameters to override
        """
        try:
            # Create and validate configuration
            config = self.create_training_config(**config_overrides)
            self.validate_config(config)
            
            # Setup environment
            self.setup_training_environment(config)
            
            # Start training
            logger.info("🚀 Starting model training...")
            logger.info("=" * 60)
            
            self.train_model(config)
            
            logger.info("✓ Training completed successfully!")
            
        except Exception as e:
            logger.error(f"✗ Training failed: {e}")
            raise


def main():
    """Main execution function."""
    try:
        # Initialize trainer
        trainer = TransformerTrainer()
        
        # Define training configuration
        training_config = {
            'batch_size': 6,
            'preload': None,
            'num_epochs': 30,
            # Add other parameters as needed
            # 'lr': 1e-4,
            # 'warmup_steps': 4000,
        }
        
        # Start training
        trainer.train(**training_config)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        sys.exit(1)


# Alternative: Direct execution function for simple use cases
def quick_train(batch_size: int = 6, num_epochs: int = 30, preload: Optional[str] = None):
    """
    Quick training function for simple use cases.
    
    Args:
        batch_size: Training batch size
        num_epochs: Number of training epochs
        preload: Path to preload weights (optional)
    """
    try:
        from config import get_config
        from train import train_model
        
        cfg = get_config()
        cfg['batch_size'] = batch_size
        cfg['preload'] = preload
        cfg['num_epochs'] = num_epochs
        
        print(f"Starting training with batch_size={batch_size}, num_epochs={num_epochs}")
        train_model(cfg)
        print("Training completed!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
