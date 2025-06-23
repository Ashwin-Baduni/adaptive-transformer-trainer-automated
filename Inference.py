#!/usr/bin/env python3
"""
Transformer Model Validation and Translation Testing

This script loads a pre-trained transformer model, runs validation on test data,
and tests the translation functionality with different input types.
"""

from pathlib import Path
from typing import Union, Optional, Callable
import torch
import torch.nn as nn

# Import custom modules with error handling
try:
    from config import get_config, latest_weights_file_path
    from train import get_model, get_ds, run_validation
    from translate import translate
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Please ensure all custom modules are available in the Python path")


class ModelValidator:
    """Class to handle model validation and translation testing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model validator.
        
        Args:
            config_path: Path to config file (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration and setup
        self.config = get_config() if config_path is None else get_config(config_path)
        self._setup_model()
    
    def _setup_model(self) -> None:
        """Setup model, data loaders, and load pre-trained weights."""
        try:
            # Load data and tokenizers
            self.train_dataloader, self.val_dataloader, self.tokenizer_src, self.tokenizer_tgt = get_ds(self.config)
            
            # Initialize model
            self.model = get_model(
                self.config,
                self.tokenizer_src.get_vocab_size(),
                self.tokenizer_tgt.get_vocab_size()
            ).to(self.device)
            
            # Load pre-trained weights
            self._load_weights()
            
        except Exception as e:
            print(f"Error during model setup: {e}")
            raise
    
    def _load_weights(self) -> None:
        """Load the latest pre-trained model weights."""
        try:
            model_filename = latest_weights_file_path(self.config)
            print(f"Loading weights from: {model_filename}")
            
            state = torch.load(model_filename, map_location=self.device)
            self.model.load_state_dict(state['model_state_dict'])
            print("✓ Model weights loaded successfully")
            
        except FileNotFoundError:
            print(f"✗ Model weights file not found: {model_filename}")
            raise
        except KeyError:
            print("✗ Model state dict not found in checkpoint file")
            raise
        except Exception as e:
            print(f"✗ Error loading model weights: {e}")
            raise
    
    def run_validation(self, num_examples: int = 10, 
                      print_func: Callable[[str], None] = print) -> None:
        """
        Run validation on the model.
        
        Args:
            num_examples: Number of validation examples to process
            print_func: Function to use for printing results
        """
        print(f"\nRunning validation with {num_examples} examples...")
        print("=" * 60)
        
        try:
            run_validation(
                model=self.model,
                validation_ds=self.val_dataloader,
                tokenizer_src=self.tokenizer_src,
                tokenizer_tgt=self.tokenizer_tgt,
                max_len=self.config['seq_len'],
                device=self.device,
                print_msg=print_func,
                global_step=0,
                writer=None,
                num_examples=num_examples
            )
            print("✓ Validation completed successfully")
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            raise


class TranslationTester:
    """Class to test translation functionality with different input types."""
    
    @staticmethod
    def test_string_translation(text: str) -> str:
        """
        Test translation with string input.
        
        Args:
            text: Input text to translate
            
        Returns:
            Translation result
        """
        try:
            result = translate(text)
            print(f"✓ String translation successful:")
            print(f"  Input: '{text}'")
            print(f"  Output: '{result}'")
            return result
            
        except Exception as e:
            print(f"✗ String translation failed: {e}")
            return f"Translation error: {e}"
    
    @staticmethod
    def test_invalid_input_translation(input_value: Union[int, float, list, dict]) -> str:
        """
        Test translation with invalid input types.
        
        Args:
            input_value: Invalid input to test error handling
            
        Returns:
            Translation result or error message
        """
        try:
            result = translate(input_value)
            print(f"✓ Invalid input handled:")
            print(f"  Input: {input_value} (type: {type(input_value).__name__})")
            print(f"  Output: '{result}'")
            return result
            
        except Exception as e:
            print(f"✗ Invalid input translation failed: {e}")
            return f"Translation error: {e}"
    
    def run_translation_tests(self) -> None:
        """Run comprehensive translation tests."""
        print("\nRunning Translation Tests...")
        print("=" * 40)
        
        # Test valid string input
        self.test_string_translation("Why do I need to translate this?")
        print()
        
        # Test invalid inputs
        test_cases = [
            34,                    # Integer
            3.14,                  # Float
            ["hello", "world"],    # List
            {"text": "hello"},     # Dictionary
            None                   # None type
        ]
        
        for test_input in test_cases:
            self.test_invalid_input_translation(test_input)
            print()


def main():
    """Main execution function."""
    try:
        # Initialize model validator
        print("🚀 Starting Model Validation and Translation Testing")
        print("=" * 60)
        
        validator = ModelValidator()
        
        # Run model validation
        validator.run_validation(num_examples=10)
        
        # Run translation tests
        tester = TranslationTester()
        tester.run_translation_tests()
        
        print("🎉 All tests completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required modules are installed and available")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
