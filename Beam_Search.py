#!/usr/bin/env python3
"""
Transformer Model Inference and Validation

This script implements beam search decoding and validation for transformer models,
comparing greedy decoding vs beam search results.
"""

from pathlib import Path
from typing import List, Tuple, Callable, Optional
import torch
import torch.nn as nn

from config import get_config, get_weights_file_path
from train import get_model, get_ds, causal_mask


class TransformerInference:
    """Class to handle transformer model inference with different decoding strategies."""
    
    def __init__(self, model_epoch: str = "19", config_path: Optional[str] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_epoch: Epoch number of the model to load
            config_path: Path to config file (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load configuration and data
        self.config = get_config() if config_path is None else get_config(config_path)
        self.train_dataloader, self.val_dataloader, self.tokenizer_src, self.tokenizer_tgt = get_ds(self.config)
        
        # Initialize and load model
        self.model = self._load_model(model_epoch)
        
        # Cache special token IDs
        self.sos_idx = self.tokenizer_tgt.token_to_id('[SOS]')
        self.eos_idx = self.tokenizer_tgt.token_to_id('[EOS]')
        
    def _load_model(self, epoch: str) -> nn.Module:
        """Load pre-trained model weights."""
        model = get_model(
            self.config,
            self.tokenizer_src.get_vocab_size(),
            self.tokenizer_tgt.get_vocab_size()
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
    
    def greedy_decode(self, source: torch.Tensor, source_mask: torch.Tensor, 
                     max_len: int) -> torch.Tensor:
        """
        Perform greedy decoding.
        
        Args:
            source: Source sequence tensor
            source_mask: Source mask tensor
            max_len: Maximum generation length
            
        Returns:
            Generated sequence tensor
        """
        # Precompute encoder output
        encoder_output = self.model.encode(source, source_mask)
        
        # Initialize decoder input with SOS token
        decoder_input = torch.empty(1, 1).fill_(self.sos_idx).type_as(source).to(self.device)
        
        for _ in range(max_len - 1):  # -1 because we already have SOS token
            # Build causal mask for current sequence
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(self.device)
            
            # Get model output
            out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            
            # Get next token probabilities and select most likely
            prob = self.model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            # Append next token to sequence
            next_token = torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(self.device)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_word.item() == self.eos_idx:
                break
                
        return decoder_input.squeeze(0)
    
    def beam_search_decode(self, source: torch.Tensor, source_mask: torch.Tensor,
                          beam_size: int, max_len: int) -> torch.Tensor:
        """
        Perform beam search decoding.
        
        Args:
            source: Source sequence tensor
            source_mask: Source mask tensor
            beam_size: Number of beams to maintain
            max_len: Maximum generation length
            
        Returns:
            Best generated sequence tensor
        """
        # Precompute encoder output
        encoder_output = self.model.encode(source, source_mask)
        
        # Initialize with SOS token
        decoder_initial_input = torch.empty(1, 1).fill_(self.sos_idx).type_as(source).to(self.device)
        
        # Initialize candidates: (sequence, cumulative_log_prob)
        candidates = [(decoder_initial_input, 0.0)]
        
        while True:
            # Check termination conditions
            if any(cand[0].size(1) == max_len for cand, _ in candidates):
                break
                
            if all(cand[0][0][-1].item() == self.eos_idx for cand, _ in candidates):
                break
            
            new_candidates = []
            
            for candidate_seq, candidate_score in candidates:
                # Skip if sequence already ended with EOS
                if candidate_seq[0][-1].item() == self.eos_idx:
                    new_candidates.append((candidate_seq, candidate_score))
                    continue
                
                # Build causal mask for current candidate
                candidate_mask = causal_mask(candidate_seq.size(1)).type_as(source_mask).to(self.device)
                
                # Get model output
                out = self.model.decode(encoder_output, source_mask, candidate_seq, candidate_mask)
                
                # Get next token probabilities
                prob = self.model.project(out[:, -1])
                log_prob = torch.log_softmax(prob, dim=1)
                
                # Get top-k candidates
                topk_log_prob, topk_idx = torch.topk(log_prob, beam_size, dim=1)
                
                for i in range(beam_size):
                    token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                    token_log_prob = topk_log_prob[0][i].item()
                    
                    # Create new candidate sequence
                    new_candidate_seq = torch.cat([candidate_seq, token], dim=1)
                    new_score = candidate_score + token_log_prob
                    
                    new_candidates.append((new_candidate_seq, new_score))
            
            # Sort by score and keep top beam_size candidates
            candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Return the best candidate
        return candidates[0][0].squeeze()
    
    def validate_model(self, num_examples: int = 2, beam_size: int = 3, 
                      max_len: int = 20, print_func: Callable = print) -> None:
        """
        Run validation on the model comparing greedy and beam search decoding.
        
        Args:
            num_examples: Number of validation examples to process
            beam_size: Beam size for beam search
            max_len: Maximum generation length
            print_func: Function to use for printing results
        """
        self.model.eval()
        console_width = 80
        
        with torch.no_grad():
            for count, batch in enumerate(self.val_dataloader, 1):
                # Move inputs to device
                encoder_input = batch["encoder_input"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                
                # Validate batch size
                if encoder_input.size(0) != 1:
                    raise ValueError("Batch size must be 1 for validation")
                
                # Generate outputs using both methods
                greedy_output = self.greedy_decode(encoder_input, encoder_mask, max_len)
                beam_output = self.beam_search_decode(encoder_input, encoder_mask, beam_size, max_len)
                
                # Decode outputs to text
                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                greedy_text = self.tokenizer_tgt.decode(greedy_output.detach().cpu().numpy())
                beam_text = self.tokenizer_tgt.decode(beam_output.detach().cpu().numpy())
                
                # Print results
                print_func('-' * console_width)
                print_func(f"{'SOURCE:':>20} {source_text}")
                print_func(f"{'TARGET:':>20} {target_text}")
                print_func(f"{'PREDICTED GREEDY:':>20} {greedy_text}")
                print_func(f"{'PREDICTED BEAM:':>20} {beam_text}")
                
                if count >= num_examples:
                    print_func('-' * console_width)
                    break


def main():
    """Main execution function."""
    # Initialize inference engine
    inference = TransformerInference(model_epoch="19")
    
    # Run validation
    print("Running model validation...")
    inference.validate_model(num_examples=2, beam_size=3, max_len=20)


if __name__ == "__main__":
    main()
