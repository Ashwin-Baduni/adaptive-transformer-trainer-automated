#!/usr/bin/env python3
"""
Direct Dataset Modification Transformer Runner

This script directly modifies the existing dataset.py file to handle sentence truncation.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
import torch


class DirectDatasetModifier:
    """Runner that directly modifies the dataset.py file to handle long sentences."""
    
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize the direct modifier."""
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.setup_logging()
        self.gpu_memory_gb = self.detect_gpu_memory()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('transformer_runner.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def detect_gpu_memory(self) -> float:
        """Detect available GPU memory."""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"GPU Memory detected: {gpu_memory:.2f} GB")
                return gpu_memory
            else:
                self.logger.info("No CUDA GPU detected")
                return 0.0
        except Exception as e:
            self.logger.warning(f"Could not detect GPU memory: {e}")
            return 0.0
    
    def backup_original_dataset(self) -> bool:
        """Backup the original dataset.py file."""
        try:
            dataset_path = self.repo_path / 'dataset.py'
            backup_path = self.repo_path / 'dataset_original.py'
            
            if dataset_path.exists() and not backup_path.exists():
                shutil.copy2(dataset_path, backup_path)
                self.logger.info("✓ Original dataset.py backed up")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup dataset.py: {e}")
            return False
    
    def modify_dataset_file(self) -> bool:
        """Directly modify the dataset.py file to handle sentence truncation."""
        try:
            dataset_path = self.repo_path / 'dataset.py'
            
            # Read the original file
            with open(dataset_path, 'r') as f:
                content = f.read()
            
            # Find the __getitem__ method and modify it
            modified_content = self.inject_truncation_logic(content)
            
            # Write the modified content back
            with open(dataset_path, 'w') as f:
                f.write(modified_content)
            
            self.logger.info("✓ Dataset.py modified to handle sentence truncation")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to modify dataset.py: {e}")
            return False
    
    def inject_truncation_logic(self, content: str) -> str:
        """Inject truncation logic into the dataset code."""
        # Find the line where tokens are encoded
        lines = content.split('\n')
        modified_lines = []
        
        for i, line in enumerate(lines):
            modified_lines.append(line)
            
            # After encoding tokens, add truncation logic
            if 'enc_input_tokens = self.tokenizer_src.encode(src_text).ids' in line:
                modified_lines.append('        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids')
                modified_lines.append('')
                modified_lines.append('        # TRUNCATE LONG SEQUENCES TO FIT seq_len')
                modified_lines.append('        max_src_len = self.seq_len - 2  # Account for SOS/EOS tokens')
                modified_lines.append('        max_tgt_len = self.seq_len - 1  # Account for SOS token')
                modified_lines.append('        ')
                modified_lines.append('        if len(enc_input_tokens) > max_src_len:')
                modified_lines.append('            enc_input_tokens = enc_input_tokens[:max_src_len]')
                modified_lines.append('            ')
                modified_lines.append('        if len(dec_input_tokens) > max_tgt_len:')
                modified_lines.append('            dec_input_tokens = dec_input_tokens[:max_tgt_len]')
                modified_lines.append('')
                # Skip the next line since we already added dec_input_tokens
                continue
            elif 'dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids' in line:
                # Skip this line since we already added it above
                continue
        
        return '\n'.join(modified_lines)
    
    def create_simple_truncated_dataset(self) -> bool:
        """Create a completely new, simple dataset.py with truncation."""
        try:
            dataset_content = '''import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Encode the text
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # TRUNCATE LONG SEQUENCES TO FIT seq_len - THIS IS THE KEY FIX
        max_src_len = self.seq_len - 2  # Account for SOS/EOS tokens
        max_tgt_len = self.seq_len - 1  # Account for SOS token
        
        if len(enc_input_tokens) > max_src_len:
            enc_input_tokens = enc_input_tokens[:max_src_len]
            
        if len(dec_input_tokens) > max_tgt_len:
            dec_input_tokens = dec_input_tokens[:max_tgt_len]

        # Add SOS, EOS and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # These should never be negative now due to truncation
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Padding calculation error: enc_pad={enc_num_padding_tokens}, dec_pad={dec_num_padding_tokens}")

        # Add <s> and </s> token
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Add only <s> token
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Add only </s> token
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Double check the size of the tensors
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    print(f'Using seq_len: {config["seq_len"]} (truncating sentences longer than {config["seq_len"]-2} tokens)')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
'''
            
            # Write the new dataset.py
            with open(self.repo_path / 'dataset.py', 'w') as f:
                f.write(dataset_content)
            
            self.logger.info("✓ Created new dataset.py with truncation support")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create truncated dataset.py: {e}")
            return False
    
    def run_training(self, config: Dict[str, Any]) -> bool:
        """Run training with the modified dataset."""
        self.logger.info("🚀 Starting training with truncated dataset...")
        
        try:
            training_code = f"""
import os
import torch
from config import get_config
from train import train_model

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    cfg = get_config()
    cfg.update({config})
    
    print("Starting training with truncated dataset:")
    for key, value in cfg.items():
        print(f"  {{key}}: {{value}}")
    
    train_model(cfg)
    print("✓ Training completed successfully!")
    
except Exception as e:
    print(f"✗ Training failed: {{e}}")
    exit(1)
"""
            
            with open('truncated_train.py', 'w') as f:
                f.write(training_code)
            
            result = subprocess.run([sys.executable, 'truncated_train.py'])
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
    
    def restore_original_dataset(self) -> bool:
        """Restore the original dataset.py file."""
        try:
            backup_path = self.repo_path / 'dataset_original.py'
            dataset_path = self.repo_path / 'dataset.py'
            
            if backup_path.exists():
                shutil.copy2(backup_path, dataset_path)
                self.logger.info("✓ Original dataset.py restored")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore dataset.py: {e}")
            return False
    
    def run_complete_pipeline(self, config_overrides: Optional[Dict[str, Any]] = None) -> bool:
        """Run complete pipeline with direct dataset modification."""
        config_overrides = config_overrides or {}
        
        self.logger.info("🚀 Starting Direct Dataset Modification Pipeline")
        self.logger.info("=" * 60)
        
        # Setup
        os.chdir(self.repo_path)
        
        # Backup and modify dataset
        if not self.backup_original_dataset():
            return False
        
        if not self.create_simple_truncated_dataset():
            return False
        
        # Create configuration
        config = {
            'batch_size': 1,
            'num_epochs': 10,
            'seq_len': 100,  # Conservative for GTX 1050
            'd_model': 256,
            'lr': 1e-4,
            'warmup_steps': 500,
            'preload': None,
            'model_folder': 'weights',
            'tokenizer_file': 'tokenizer_{0}.json'
        }
        
        config.update(config_overrides)
        
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Run training
        success = self.run_training(config)
        
        if success:
            self.logger.info("🎉 Training completed successfully!")
        else:
            self.logger.error("❌ Training failed")
        
        return success


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Direct Dataset Modifier')
    parser.add_argument('--repo-path', type=str, help='Path to repository')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--seq-len', type=int, default=100, help='Sequence length')
    parser.add_argument('--restore', action='store_true', help='Restore original dataset.py')
    
    args = parser.parse_args()
    
    modifier = DirectDatasetModifier(args.repo_path)
    
    if args.restore:
        modifier.restore_original_dataset()
        return
    
    config_overrides = {
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'seq_len': args.seq_len
    }
    
    success = modifier.run_complete_pipeline(config_overrides)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
