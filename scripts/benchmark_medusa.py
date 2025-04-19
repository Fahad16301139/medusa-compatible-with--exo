#!/usr/bin/env python
"""
Script to benchmark Medusa decoding vs. regular autoregressive decoding.

Usage:
  python scripts/benchmark_medusa.py --model medusa-model-path --prompt "Your prompt here" --max-tokens 100
"""

import os
import sys
import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from exo.inference.medusa_model import MedusaModel
from exo.inference.medusa_decoder import MedusaDecoder

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Medusa decoding")
    parser.add_argument("--model", type=str, required=True, help="Path to the Medusa model")
    parser.add_argument("--prompt", type=str, default="What is the meaning of life?", help="Prompt to use for benchmarking")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling parameter")
    parser.add_argument("--repeats", type=int, default=3, help="Number of times to repeat the benchmark")
    parser.add_argument("--heads", type=int, default=4, help="Number of Medusa heads to use")
    parser.add_argument("--tree-size", type=int, default=5, help="Maximum tree size for Medusa decoding")
    parser.add_argument("--candidates", type=int, default=5, help="Maximum number of candidates to consider")
    return parser.parse_args()

def generate_autoregressive(model, tokenizer, prompt, max_tokens, temperature, top_p):
    """Generate tokens using regular autoregressive decoding."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0
        )
    end_time = time.time()
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    generation_time = end_time - start_time
    tokens_per_second = generated_tokens / generation_time
    
    return {
        "output_text": output_text,
        "generated_tokens": generated_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second
    }

def generate_with_medusa(model, tokenizer, prompt, max_tokens, temperature, top_p, medusa_config):
    """Generate tokens using Medusa decoding."""
    # Wrap the model with MedusaModel adapter
    medusa_model = MedusaModel.from_pretrained(
        model,
        medusa_num_heads=medusa_config["heads"],
        medusa_num_layers=1
    )
    medusa_model.to(model.device)
    
    # Create MedusaDecoder
    medusa_decoder = MedusaDecoder(
        model=medusa_model,
        tokenizer=tokenizer,
        medusa_heads=medusa_config["heads"],
        tree_size=medusa_config["tree_size"],
        max_candidates=medusa_config["candidates"]
    )
    
    # Generate tokens
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # For now, we're using a placeholder implementation that doesn't do true Medusa decoding
    # This will be replaced with the actual implementation once it's complete
    start_time = time.time()
    with torch.no_grad():
        # This would be medusa_decoder.generate() in the future
        output_ids = medusa_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0
        )
    end_time = time.time()
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    generation_time = end_time - start_time
    tokens_per_second = generated_tokens / generation_time
    
    return {
        "output_text": output_text,
        "generated_tokens": generated_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second
    }

def main():
    args = parse_args()
    
    print(f"Loading model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")
    
    medusa_config = {
        "heads": args.heads,
        "tree_size": args.tree_size,
        "candidates": args.candidates
    }
    
    # Run benchmarks
    autoregressive_times = []
    medusa_times = []
    
    for i in range(args.repeats):
        print(f"\nRun {i+1}/{args.repeats}")
        
        print("Running autoregressive decoding...")
        auto_result = generate_autoregressive(
            model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_p
        )
        autoregressive_times.append(auto_result["tokens_per_second"])
        
        print(f"Autoregressive: {auto_result['tokens_per_second']:.2f} tokens/sec")
        
        print("\nRunning Medusa decoding...")
        medusa_result = generate_with_medusa(
            model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_p, medusa_config
        )
        medusa_times.append(medusa_result["tokens_per_second"])
        
        print(f"Medusa: {medusa_result['tokens_per_second']:.2f} tokens/sec")
    
    # Print results
    print("\n--- Results ---")
    print(f"Prompt: '{args.prompt}'")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Medusa heads: {args.heads}")
    print(f"Medusa tree size: {args.tree_size}")
    print(f"Medusa candidates: {args.candidates}")
    
    avg_auto = np.mean(autoregressive_times)
    avg_medusa = np.mean(medusa_times)
    speedup = avg_medusa / avg_auto
    
    print(f"\nAverage autoregressive: {avg_auto:.2f} tokens/sec")
    print(f"Average Medusa: {avg_medusa:.2f} tokens/sec")
    print(f"Speedup: {speedup:.2f}x")
    
    print("\nNote: This benchmark does not yet implement full Medusa decoding.")
    print("In an actual implementation, Medusa should be significantly faster.")

if __name__ == "__main__":
    main() 