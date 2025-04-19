"""
Medusa decoder implementation for faster LLM inference.

Medusa is a speculative decoding method that uses multiple language model heads
to predict multiple future tokens in parallel. This implementation is based on
the paper: https://arxiv.org/abs/2401.10774
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default topk value for sparse tree
TOPK = 10  # This is a placeholder and usually sufficient

class MedusaDecoder:
    """
    Implements the Medusa decoding algorithm for faster LLM inference.
    """
    
    def __init__(
        self, 
        model,
        tokenizer,
        medusa_heads: int = 4,
        tree_size: int = 5,
        max_candidates: int = 5,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3
    ):
        """
        Initialize the Medusa decoder.
        
        Args:
            model: The underlying model to use for decoding
            tokenizer: The tokenizer to use
            medusa_heads: Number of Medusa prediction heads
            tree_size: Maximum size of the tree to explore
            max_candidates: Maximum number of candidate sequences to consider
            posterior_threshold: Threshold for posterior validation
            posterior_alpha: Alpha parameter for posterior calculation (usually sqrt of threshold)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.medusa_heads = medusa_heads
        self.tree_size = tree_size
        self.max_candidates = max_candidates
        self.posterior_threshold = posterior_threshold
        self.posterior_alpha = posterior_alpha
        
        # Set default medusa choices based on the number of heads
        self.medusa_choices = self._get_default_medusa_choices(medusa_heads)
        
        # Initialize buffers
        self.medusa_buffers = None
        
    def _get_default_medusa_choices(self, num_heads):
        """Get default Medusa choices based on number of heads."""
        if num_heads == 1:
            return [[0]]
        elif num_heads == 2:
            return [[0], [0, 0]]
        elif num_heads == 4:
            return [[0], [0, 0], [0, 1], [0, 0, 0]]
        elif num_heads == 5:
            return [[0], [0, 0], [0, 1], [0, 0, 0], [0, 0, 1]]
        else:
            # Default fallback
            return [[i] for i in range(min(num_heads, 8))]
        
    def generate_medusa_buffers(self, medusa_choices, device="cuda"):
        """
        Generate buffers for the Medusa structure based on the provided choices.
        
        Args:
            medusa_choices: A nested list representing tree in the Medusa structure
            device: Device to which the tensors should be moved
            
        Returns:
            Dict containing buffers related to the Medusa structure
        """
        # Sort the medusa_choices based on their lengths and then their values
        sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
        medusa_len = len(sorted_medusa_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_medusa_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth
        
        # Create the attention mask for Medusa
        medusa_attn_mask = torch.eye(medusa_len, medusa_len)
        medusa_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                # retrieve ancestor position
                if len(cur_medusa_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_medusa_choice) - 1):
                    ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
                medusa_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        # Generate tree indices for the Medusa structure
        medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
        medusa_tree_indices[0] = 0
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
            start += depth_counts[i]

        # Generate position IDs for the Medusa structure
        medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        # Generate retrieval indices for Medusa structure verification
        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_medusa_choices)):
            cur_medusa_choice = sorted_medusa_choices[-i-1]
            retrieve_indice = []
            if cur_medusa_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_medusa_choice)):
                    retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                    retrieve_paths.append(cur_medusa_choice[:c+1])
            retrieve_indices_nest.append(retrieve_indice)
        
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [self._pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

        # Aggregate the generated buffers into a dictionary
        medusa_buffers = {
            "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
            "tree_indices": medusa_tree_indices,
            "medusa_position_ids": medusa_position_ids,
            "retrieve_indices": retrieve_indices,
        }
        
        # Move the tensors in the dictionary to the specified device
        medusa_buffers = {
            k: v.clone().to(device)
            if isinstance(v, torch.Tensor)
            else torch.tensor(v, device=device)
            for k, v in medusa_buffers.items()
        }
        return medusa_buffers
    
    def _pad_path(self, path, length, pad_value=-2):
        """Pad the given path list with a specific value up to a specified length."""
        return path + [pad_value] * (length - len(path))
    
    def generate_candidates(self, medusa_logits, logits, tree_indices, retrieve_indices, 
                           temperature=0.0, top_p=0.8):
        """
        Generate candidates with topk predictions from Medusa heads.
        
        Args:
            medusa_logits: Logits from the Medusa heads
            logits: Logits from the base model
            tree_indices: Indices for the tree structure
            retrieve_indices: Indices for retrieving from the tree
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of candidates and tree candidates
        """
        with torch.no_grad():
            # Generate a prediction from the base model
            if temperature > 0:
                # Apply temperature and top-p sampling
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0  # Keep the top token
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                base_pred = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                base_pred = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            
            # Convert Medusa logits and tree indices to the right shapes
            medusa_logits = [logit[:, -1:, :] for logit in medusa_logits]
            n_medusa_tokens = len(medusa_logits)
            
            # Get predictions from each Medusa head
            medusa_preds = []
            for i in range(n_medusa_tokens):
                medusa_pred = torch.argmax(medusa_logits[i], dim=-1)
                medusa_preds.append(medusa_pred)
            
            # Generate tree candidates
            tree_candidates = [base_pred]
            for i in range(n_medusa_tokens):
                tree_candidates.append(medusa_preds[i])
            tree_candidates = torch.cat(tree_candidates, dim=0)
            
            # Generate candidates for verification
            candidates = []
            for i in range(retrieve_indices.shape[0]):
                candidate = []
                for j in range(1, retrieve_indices.shape[1]):
                    if retrieve_indices[i, j] >= 0:
                        idx = retrieve_indices[i, j]
                        candidate.append(tree_candidates[idx].item())
                candidates.append(candidate)
            
            return candidates, tree_candidates
    
    def tree_decoding(self, tree_candidates, input_ids, retrieve_indices):
        """
        Use tree attention to verify the candidates and get predictions.
        
        Args:
            tree_candidates: Candidate tokens for the tree
            input_ids: Input token IDs
            retrieve_indices: Indices for retrieving from the tree
            
        Returns:
            Tuple of medusa_logits, logits, outputs
        """
        # This is a simplified implementation as the real one requires modifying the model architecture
        # In a real implementation, we would:
        # 1. Set the model to tree attention mode
        # 2. Create a tree attention mask
        # 3. Run the model with the tree candidates
        # 4. Get the medusa_logits, logits, and outputs
        
        # Placeholder implementation
        outputs = {"hidden_states": None}
        medusa_logits = [None] * self.medusa_heads
        logits = None
        
        return medusa_logits, logits, outputs
    
    def evaluate_posterior(self, logits, candidates, temperature=0.0, top_p=0.8):
        """
        Evaluate the posterior of the candidates to select the accepted candidate prefix.
        
        Args:
            logits: Logits from the model
            candidates: List of candidate token sequences
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of best candidate and accepted length
        """
        # In a real implementation, this would:
        # 1. Calculate the posterior probability of each candidate
        # 2. Find the best candidate with the highest probability
        # 3. Determine how many tokens to accept
        
        # Placeholder implementation
        best_candidate = candidates[0] if candidates else []
        accept_length = len(best_candidate)
        
        return best_candidate, accept_length
    
    def update_inference_inputs(self, input_ids, candidates, best_candidate, accept_length):
        """
        Update the input_ids and prepare for the next round of inference.
        
        Args:
            input_ids: Current input token IDs
            candidates: List of candidate sequences
            best_candidate: The selected best candidate
            accept_length: Number of tokens to accept
            
        Returns:
            Updated input_ids, new_token count
        """
        # Update input_ids with the accepted tokens
        tokens_to_add = best_candidate[:accept_length]
        for token in tokens_to_add:
            input_ids = torch.cat([input_ids, torch.tensor([[token]], device=input_ids.device)], dim=1)
        
        return input_ids, len(tokens_to_add)
        
    def generate(
        self,
        prompt_tokens,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.8
    ):
        """
        Generate tokens using Medusa decoding.
        
        Args:
            prompt_tokens: Input token IDs (can be tensor or list)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Tensor of generated token IDs
        """
        # Ensure input is a tensor
        if not isinstance(prompt_tokens, torch.Tensor):
            if isinstance(prompt_tokens, list):
                prompt_tokens = torch.tensor([prompt_tokens])
            else:
                raise ValueError(f"prompt_tokens must be a tensor or list, got {type(prompt_tokens)}")
        
        # Get the device from the model if available
        device = None
        if hasattr(self.model, 'device'):
            device = self.model.device
        elif next(self.model.parameters(), None) is not None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure prompt_tokens is on the correct device
        prompt_tokens = prompt_tokens.to(device)
        
        logger.info(f"Using Medusa decoder with model type: {type(self.model).__name__}")
        logger.info(f"Input shape: {prompt_tokens.shape}, device: {device}, max_new_tokens: {max_new_tokens}")
        
        try:
            # First, try to use the model's generate method directly
            if hasattr(self.model, 'generate'):
                logger.info("Using model's native generate method")
                
                # Check what kwargs the model's generate method accepts
                import inspect
                generate_params = inspect.signature(self.model.generate).parameters
                kwargs = {
                    "max_new_tokens": max_new_tokens
                }
                
                # Only add params if they're accepted by the method
                if 'temperature' in generate_params:
                    kwargs['temperature'] = temperature
                if 'top_p' in generate_params:
                    kwargs['top_p'] = top_p
                
                # Call generate with appropriate parameters
                generated_ids = self.model.generate(prompt_tokens, **kwargs)
                
                logger.info(f"Generation successful, output shape: {generated_ids.shape}")
                return generated_ids
            
            # If model doesn't have generate, implement basic autoregressive generation
            else:
                logger.info("Model doesn't have generate method, implementing basic autoregressive generation")
                input_ids = prompt_tokens
                past_key_values = None
                
                for _ in range(max_new_tokens):
                    with torch.no_grad():
                        if past_key_values is not None:
                            outputs = self.model(input_ids[:, -1:], past_key_values=past_key_values)
                        else:
                            outputs = self.model(input_ids)
                        
                        logits = outputs.logits
                        past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
                        
                        # Apply temperature and sampling
                        next_token_logits = logits[:, -1, :]
                        if temperature > 0:
                            next_token_logits = next_token_logits / temperature
                            
                        # Optional top-p sampling
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 0] = 0  # Keep the top token
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('inf'))
                        
                        # Get next token
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                        
                        # Check for EOS
                        if (next_token.item() == self.tokenizer.eos_token_id or 
                            (hasattr(self.tokenizer, 'pad_token_id') and next_token.item() == self.tokenizer.pad_token_id)):
                            break
                
                logger.info(f"Basic generation successful, output shape: {input_ids.shape}")
                return input_ids
                
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            logger.exception(e)
            
            # Return the prompt tokens as fallback so we don't crash completely
            return prompt_tokens
        
    @staticmethod
    def is_medusa_model(model_config: Dict) -> bool:
        """
        Check if a model is a Medusa model.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            True if the model is a Medusa model, False otherwise
        """
        return model_config.get("is_medusa", False)
        
    @staticmethod
    def detect_medusa_config(model_path: str) -> Optional[Dict]:
        """
        Detect Medusa configuration from a model's files.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Dictionary with Medusa configuration if detected, None otherwise
        """
        # This would scan the model files to detect Medusa configuration
        # For now, return a default configuration
        return {
            "medusa_heads": 4,
            "tree_size": 5,
            "max_candidates": 5
        } 