#!/usr/bin/env python
"""
Script to convert a regular language model to a Medusa model by adding Medusa heads.

Usage:
  python scripts/convert_to_medusa.py --model path/to/model --output path/to/output --heads 4 --layers 1
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import OrderedDict
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a model to a Medusa model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model to convert")
    parser.add_argument("--output", type=str, required=True, help="Path to save the Medusa model")
    parser.add_argument("--heads", type=int, default=4, help="Number of Medusa heads")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers in each Medusa head")
    return parser.parse_args()

class ResBlock(torch.nn.Module):
    """
    A residual block for Medusa heads.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

def initialize_medusa_head(hidden_size, vocab_size, num_layers=1):
    """
    Initialize a Medusa head with given parameters.
    
    Args:
        hidden_size: Hidden size of the model
        vocab_size: Vocabulary size of the model
        num_layers: Number of layers in the Medusa head
        
    Returns:
        An OrderedDict with the initialized parameters
    """
    state_dict = OrderedDict()
    
    # Create layers for the Medusa head
    for i in range(num_layers):
        # Initialize ResBlock weights
        state_dict[f"layers.{i*2}.linear.weight"] = torch.zeros((hidden_size, hidden_size))
        state_dict[f"layers.{i*2}.linear.bias"] = torch.zeros(hidden_size)
    
    # Initialize the final linear layer
    state_dict[f"layers.{num_layers*2}.weight"] = torch.zeros((vocab_size, hidden_size))
    
    return state_dict

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize Medusa heads
    medusa_heads = []
    for i in range(args.heads):
        head_state_dict = initialize_medusa_head(
            config.hidden_size, 
            config.vocab_size, 
            args.layers
        )
        medusa_heads.append(head_state_dict)
    
    # Combine Medusa heads into a single state dict
    medusa_heads_state_dict = OrderedDict()
    for i, head in enumerate(medusa_heads):
        for k, v in head.items():
            medusa_heads_state_dict[f"{i}.{k}"] = v
    
    # Save the base model
    print(f"Saving base model to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    # Save Medusa heads
    print(f"Saving Medusa heads to {args.output}/medusa_heads.pt")
    torch.save(medusa_heads_state_dict, os.path.join(args.output, "medusa_heads.pt"))
    
    # Save Medusa configuration
    medusa_config = {
        "medusa_num_heads": args.heads,
        "medusa_num_layers": args.layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "base_model_name_or_path": args.model
    }
    
    with open(os.path.join(args.output, "medusa_config.json"), "w") as f:
        json.dump(medusa_config, f, indent=2)
    
    # Update model config to indicate this is a Medusa model
    config.is_medusa = True
    config.medusa_num_heads = args.heads
    config.medusa_num_layers = args.layers
    
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"Successfully created Medusa model with {args.heads} heads and {args.layers} layers per head")
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main() 