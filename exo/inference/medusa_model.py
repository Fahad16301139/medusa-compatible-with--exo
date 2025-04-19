"""
Medusa model adapter that adds Medusa heads to existing models for faster inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
import os
import json

logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    """
    A Residual Block module for Medusa heads.
    
    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the LLaMA model
        self.act = nn.SiLU()

    def forward(self, x):
        """Forward pass of the ResBlock."""
        return x + self.act(self.linear(x))

class MedusaHead(nn.Module):
    """
    A Medusa head that predicts tokens at specific future positions.
    """
    def __init__(self, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.layers = nn.Sequential(
            *([ResBlock(hidden_size)] * num_layers),
            nn.Linear(hidden_size, vocab_size, bias=False)
        )
        
    def forward(self, x):
        """Forward pass of the Medusa head."""
        return self.layers(x)

class MedusaModel(nn.Module):
    """
    MedusaModel adapter that adds multiple prediction heads to a base model.
    """
    def __init__(
        self,
        base_model,
        hidden_size,
        vocab_size,
        medusa_num_heads=4,
        medusa_num_layers=1
    ):
        """
        Initialize the MedusaModel adapter.
        
        Args:
            base_model: The underlying model to add Medusa heads to
            hidden_size: Hidden size of the model
            vocab_size: Vocabulary size of the model
            medusa_num_heads: Number of Medusa heads to add
            medusa_num_layers: Number of layers in each Medusa head
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        
        # Create Medusa heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size, medusa_num_layers)
            for _ in range(medusa_num_heads)
        ])
        
        # Tree attention state
        self.medusa_mode = False
        self.medusa_mask = None
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs
    ):
        """
        Forward pass of the MedusaModel.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Cached key values for faster inference
            output_orig: Whether to output the original logits
            position_ids: Position IDs
            medusa_forward: Whether to use Medusa forward pass
            **kwargs: Additional arguments to pass to the base model
            
        Returns:
            Tuple of (medusa_logits, outputs, logits) if output_orig is True,
            otherwise just medusa_logits
        """
        # Only set medusa_mode and mask if medusa_forward is True
        if medusa_forward:
            self.medusa_mode = True
            
        # Forward pass through the base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **kwargs
        )
        
        # Get the last hidden states
        last_hidden_state = outputs.last_hidden_state
        
        # Apply Medusa heads to get logits
        medusa_logits = [head(last_hidden_state) for head in self.medusa_heads]
        
        # Original model logits
        logits = outputs.logits
        
        # Reset medusa_mode after forward pass
        self.medusa_mode = False
        
        if output_orig:
            return medusa_logits, outputs, logits
        else:
            return medusa_logits
    
    def get_tokenizer(self):
        """Get the tokenizer from the base model."""
        return getattr(self.base_model, "tokenizer", None)
    
    def generate(self, *args, **kwargs):
        """Delegate generate to base model."""
        return self.base_model.generate(*args, **kwargs)
    
    @classmethod
    def from_pretrained(cls, model, medusa_num_heads=4, medusa_num_layers=1):
        """
        Create a MedusaModel from a pretrained model.
        
        Args:
            model: Pretrained model
            medusa_num_heads: Number of Medusa heads
            medusa_num_layers: Number of layers in each Medusa head
            
        Returns:
            MedusaModel instance
        """
        # Get model configuration
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Model does not have a config attribute")
            
        # Get hidden size and vocab size
        hidden_size = getattr(config, "hidden_size", None)
        vocab_size = getattr(config, "vocab_size", None)
        
        if hidden_size is None or vocab_size is None:
            raise ValueError("Model config must have hidden_size and vocab_size")
            
        # Create MedusaModel
        return cls(
            model,
            hidden_size,
            vocab_size,
            medusa_num_heads,
            medusa_num_layers
        )
        
    def save_pretrained(self, save_directory):
        """
        Save the MedusaModel to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(save_directory)
        
        # Save Medusa heads
        torch.save(
            self.medusa_heads.state_dict(),
            os.path.join(save_directory, "medusa_heads.pt")
        )
        
        # Save Medusa configuration
        with open(os.path.join(save_directory, "medusa_config.json"), "w") as f:
            json.dump({
                "medusa_num_heads": self.medusa_num_heads,
                "medusa_num_layers": self.medusa_num_layers,
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size
            }, f)
    
    @classmethod
    def load_medusa_heads(cls, model, save_directory):
        """
        Load Medusa heads from a directory.
        
        Args:
            model: MedusaModel instance
            save_directory: Directory where the model is saved
            
        Returns:
            Updated MedusaModel instance
        """
        # Load Medusa heads
        medusa_heads_path = os.path.join(save_directory, "medusa_heads.pt")
        if os.path.exists(medusa_heads_path):
            model.medusa_heads.load_state_dict(
                torch.load(medusa_heads_path, map_location=model.base_model.device)
            )
        
        return model 