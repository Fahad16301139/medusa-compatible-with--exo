#!/usr/bin/env python3
import time
import subprocess
import os
import argparse

def run_command(cmd, description):
    """Run a command and time it"""
    print(f"Running {description}...")
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Get output while showing progress
    while process.poll() is None:
        print(".", end="", flush=True)
        time.sleep(0.5)
    
    stdout, stderr = process.communicate()
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nCompleted in {elapsed:.2f} seconds")
    return elapsed, stdout, stderr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Medusa vs standard inference")
    parser.add_argument("--prompt", type=str, default="Write a 5 paragraph essay about artificial intelligence.", 
                        help="Prompt to use for generation")
    parser.add_argument("--tokens", type=int, default=200, 
                        help="Number of tokens to generate")
    parser.add_argument("--model", type=str, default="medusa-v1.0-vicuna-7b-v1.5", 
                        help="Model to use")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Benchmarking standard vs Medusa inference")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Tokens: {args.tokens}")
    print("=" * 80)
    print()

    # Set up base command
    base_cmd = f"exo run {args.model} --inference-engine torch --prompt \"{args.prompt}\" --max-generate-tokens {args.tokens}"
    
    # Run standard inference (no Medusa)
    standard_cmd = base_cmd
    standard_time, std_out, std_err = run_command(standard_cmd, "standard inference")
    
    print("\nWaiting 5 seconds before next run...\n")
    time.sleep(5)
    
    # Run with Medusa
    medusa_cmd = f"{base_cmd} --medusa-enable"
    medusa_time, medusa_out, medusa_err = run_command(medusa_cmd, "Medusa inference")
    
    # Calculate speedup
    speedup = standard_time / medusa_time if medusa_time > 0 else 0
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Standard inference time: {standard_time:.2f} seconds")
    print(f"Medusa inference time:   {medusa_time:.2f} seconds")
    print(f"Speedup:                 {speedup:.2f}x")
    
    if speedup >= 1.5:
        print("\n✅ Medusa is significantly faster than standard inference")
    elif speedup >= 1.0:
        print("\n✅ Medusa is faster than standard inference")
    else:
        print("\n❌ Medusa is not faster in this test - check configuration") 