#!/usr/bin/env python3
"""
Transformer Attention Visualization Tool

This script loads a pre-trained transformer model and visualizes attention maps
for encoder self-attention, decoder self-attention, and encoder-decoder cross-attention.
"""

import warnings
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import altair as alt
import pandas as pd
import numpy as np

from model import Transformer
from config import get_config, get_weights_file_path
from train import get_model, get_ds, greedy_decode

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class AttentionVisualizer:
    """Class to handle attention visualization for transformer models."""
    
    def __init__(self, config_path: Optional[str] = None, model_epoch: str = "29"):
        """
        Initialize the attention visualizer.
        
        Args:
            config_path: Path to config file (optional)
            model_epoch: Epoch number of the model to load
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration and data
        self.config = get_config() if config_path is None else get_config(config_path)
        self.train_dataloader, self.val_dataloader, self.vocab_src, self.vocab_tgt = get_ds(self.config)
        
        # Initialize and load model
        self.model = self._load_model(model_epoch)
        
    def _load_model(self, epoch: str) -> nn.Module:
        """Load pre-trained model weights."""
        model = get_model(
            self.config, 
            self.vocab_src.get_vocab_size(), 
            self.vocab_tgt.get_vocab_size()
        ).to(self.device)
        
        model_filename = get_weights_file_path(self.config, epoch)
        try:
            state = torch.load(model_filename, map_location=self.device)
            model.load_state_dict(state['model_state_dict'])
            print(f"Successfully loaded model from epoch {epoch}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_filename}")
        except KeyError:
            raise KeyError("Model state dict not found in checkpoint file")
            
        return model
    
    def load_validation_batch(self) -> Tuple[dict, List[str], List[str]]:
        """
        Load a sample batch from the validation set.
        
        Returns:
            Tuple of (batch_data, encoder_tokens, decoder_tokens)
        """
        batch = next(iter(self.val_dataloader))
        
        # Move tensors to device
        encoder_input = batch["encoder_input"].to(self.device)
        encoder_mask = batch["encoder_mask"].to(self.device)
        decoder_input = batch["decoder_input"].to(self.device)
        decoder_mask = batch["decoder_mask"].to(self.device)
        
        # Convert token IDs to tokens
        encoder_input_tokens = [
            self.vocab_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()
        ]
        decoder_input_tokens = [
            self.vocab_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()
        ]
        
        # Validate batch size
        if encoder_input.size(0) != 1:
            raise ValueError("Batch size must be 1 for validation")
        
        # Generate model output
        model_out = greedy_decode(
            self.model, encoder_input, encoder_mask, 
            self.vocab_src, self.vocab_tgt, 
            self.config['seq_len'], self.device
        )
        
        return batch, encoder_input_tokens, decoder_input_tokens
    
    def _matrix_to_dataframe(self, matrix: torch.Tensor, max_row: int, max_col: int, 
                           row_tokens: List[str], col_tokens: List[str]) -> pd.DataFrame:
        """
        Convert attention matrix to DataFrame for visualization.
        
        Args:
            matrix: Attention matrix
            max_row: Maximum number of rows to include
            max_col: Maximum number of columns to include
            row_tokens: List of row token labels
            col_tokens: List of column token labels
            
        Returns:
            DataFrame with attention values and token labels
        """
        data = []
        for r in range(min(matrix.shape[0], max_row)):
            for c in range(min(matrix.shape[1], max_col)):
                row_label = f"{r:03d} {row_tokens[r] if r < len(row_tokens) else '<blank>'}"
                col_label = f"{c:03d} {col_tokens[c] if c < len(col_tokens) else '<blank>'}"
                
                data.append({
                    "row": r,
                    "column": c,
                    "value": float(matrix[r, c]),
                    "row_token": row_label,
                    "col_token": col_label
                })
        
        return pd.DataFrame(data)
    
    def _get_attention_matrix(self, attn_type: str, layer: int, head: int) -> torch.Tensor:
        """
        Extract attention matrix from model.
        
        Args:
            attn_type: Type of attention ('encoder', 'decoder', 'encoder-decoder')
            layer: Layer index
            head: Attention head index
            
        Returns:
            Attention matrix tensor
        """
        if attn_type == "encoder":
            return self.model.encoder.layers[layer].self_attention_block.attention_scores[0, head].data
        elif attn_type == "decoder":
            return self.model.decoder.layers[layer].self_attention_block.attention_scores[0, head].data
        elif attn_type == "encoder-decoder":
            return self.model.decoder.layers[layer].cross_attention_block.attention_scores[0, head].data
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")
    
    def create_attention_heatmap(self, attn_type: str, layer: int, head: int,
                               row_tokens: List[str], col_tokens: List[str], 
                               max_sentence_len: int) -> alt.Chart:
        """
        Create attention heatmap visualization.
        
        Args:
            attn_type: Type of attention
            layer: Layer index
            head: Head index
            row_tokens: Row token labels
            col_tokens: Column token labels
            max_sentence_len: Maximum sentence length to display
            
        Returns:
            Altair chart object
        """
        matrix = self._get_attention_matrix(attn_type, layer, head)
        df = self._matrix_to_dataframe(
            matrix, max_sentence_len, max_sentence_len, row_tokens, col_tokens
        )
        
        chart = (
            alt.Chart(data=df)
            .mark_rect()
            .encode(
                x=alt.X("col_token", axis=alt.Axis(title="")),
                y=alt.Y("row_token", axis=alt.Axis(title="")),
                color=alt.Color("value", scale=alt.Scale(scheme="viridis")),
                tooltip=["row", "column", "value", "row_token", "col_token"],
            )
            .properties(
                height=400, 
                width=400, 
                title=f"Layer {layer} Head {head}"
            )
            .interactive()
        )
        
        return chart
    
    def visualize_attention_maps(self, attn_type: str, layers: List[int], heads: List[int],
                               row_tokens: List[str], col_tokens: List[str], 
                               max_sentence_len: int) -> alt.Chart:
        """
        Create comprehensive attention visualization across layers and heads.
        
        Args:
            attn_type: Type of attention to visualize
            layers: List of layer indices
            heads: List of head indices
            row_tokens: Row token labels
            col_tokens: Column token labels
            max_sentence_len: Maximum sentence length to display
            
        Returns:
            Combined Altair chart
        """
        layer_charts = []
        
        for layer in layers:
            head_charts = []
            for head in heads:
                chart = self.create_attention_heatmap(
                    attn_type, layer, head, row_tokens, col_tokens, max_sentence_len
                )
                head_charts.append(chart)
            layer_charts.append(alt.hconcat(*head_charts))
        
        return alt.vconcat(*layer_charts)


def main():
    """Main execution function."""
    # Initialize visualizer
    visualizer = AttentionVisualizer(model_epoch="29")
    
    # Load validation batch
    batch, encoder_input_tokens, decoder_input_tokens = visualizer.load_validation_batch()
    
    # Display source and target sentences
    print(f"Source: {batch['src_text'][0]}")
    print(f"Target: {batch['tgt_text'][0]}")
    
    # Calculate sentence length (up to first PAD token)
    try:
        sentence_len = encoder_input_tokens.index("[PAD]")
    except ValueError:
        sentence_len = len(encoder_input_tokens)
    
    # Configuration for visualization
    layers = [0, 1, 2]
    heads = [0, 1, 2, 3, 4, 5, 6, 7]
    max_display_len = min(20, sentence_len)
    
    # Generate attention visualizations
    print("\nGenerating attention visualizations...")
    
    # Encoder Self-Attention
    encoder_self_attn = visualizer.visualize_attention_maps(
        "encoder", layers, heads, encoder_input_tokens, encoder_input_tokens, max_display_len
    )
    
    # Decoder Self-Attention
    decoder_self_attn = visualizer.visualize_attention_maps(
        "decoder", layers, heads, decoder_input_tokens, decoder_input_tokens, max_display_len
    )
    
    # Encoder-Decoder Cross-Attention
    cross_attn = visualizer.visualize_attention_maps(
        "encoder-decoder", layers, heads, encoder_input_tokens, decoder_input_tokens, max_display_len
    )
    
    return encoder_self_attn, decoder_self_attn, cross_attn


if __name__ == "__main__":
    encoder_attn, decoder_attn, cross_attn = main()
    
    # Display the visualizations
    print("Attention visualizations generated successfully!")
    print("Use the returned chart objects to display or save the visualizations.")