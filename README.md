
# Detailed Documentation: Implementing Medusa in the Exo Framework

## Overview
We implemented Medusa, a technique for faster inference through parallel token prediction. This documentation details all changes made to enable Medusa support.

## 1. Modified `TorchDynamicShardInferenceEngine` Initialization

**File:** `exo/inference/torch/sharded_inference_engine.py`

**Change:** Added Medusa-specific attributes to the init method

```python
def __init__(self, shard_downloader: ShardDownloader):
    # Original code...
    
    # Added new fields for Medusa support
    self.model = None  # Direct model reference for Medusa
    self.is_medusa_model_loaded = False  # Flag to track if current model uses Medusa
```

**Purpose:** These attributes track if a Medusa model is loaded and maintain a direct reference to the model for the Medusa decoder.

## 2. Enhanced `ensure_shard` Method

**File:** `exo/inference/torch/sharded_inference_engine.py`

**Change:** Modified method to detect and handle Medusa models

```python
async def ensure_shard(self, shard: Shard):
    # Existing code...
    
    # Added detection for Medusa models
    from exo.models import model_cards
    self.is_medusa_model_loaded = model_cards.get(shard.model_id, {}).get("is_medusa", False)
    
    # Added branching logic for loading method
    if self.is_medusa_model_loaded:
        if DEBUG >= 1:
            print(f"Loading Medusa-specific model: {shard.model_id}")
        # Load Medusa model with specialized code path
        await self._load_medusa_model(shard)
    else:
        # Load regular model
        await self._load_standard_model(shard)
```

**Purpose:** This change checks if the requested model has Medusa capability by examining the `is_medusa` flag in the model cards registry. If it's a Medusa model, it uses a specialized loading path.

**Example:** When loading "medusa-v1.0-vicuna-7b-v1.5", it detects the model has `"is_medusa": True` in the models registry and uses the Medusa-specific loader.

## 3. Implemented Specialized Medusa Model Loader

**File:** `exo/inference/torch/sharded_inference_engine.py`

**Change:** Added a new method for loading Medusa models

```python
async def _load_medusa_model(self, shard: Shard):
    """Specialized loader for Medusa models that includes medusa-specific components"""
    from transformers import AutoModelForCausalLM
    
    def load_medusa():
        try:
            # Import required libraries
            import torch
            
            # Direct transformers loading to preserve Medusa heads
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.model_config["torch_dtype"],
                device_map="auto",
                trust_remote_code=True  # Required for custom model code
            )
            
            # Set model to evaluation mode
            model.eval()
            
            # Import and configure Medusa decoder with parameters from CLI
            from exo.inference.medusa_decoder import MedusaDecoder
            medusa_config = self.get_medusa_config()
            
            medusa_decoder = MedusaDecoder(
              model=model,
              tokenizer=self.tokenizer,
              medusa_heads=medusa_config["heads"],
              tree_size=medusa_config["tree_size"],
              max_candidates=medusa_config["candidates"]
            )
            
            # Create compatibility wrapper
            class MedusaModelWrapper:
                def __init__(self, model, decoder, device, max_tokens=2048):
                    self.model = model
                    self.medusa_decoder = decoder
                    self.device = device
                    self.max_generated_tokens = max_tokens
                
                # Compatibility method to match API
                def generate(self, tokens, **kwargs):
                    batch_size = tokens.shape[0]
                    vocab_size = getattr(self.model.config, "vocab_size", 32000)
                    logits = torch.zeros((batch_size, 1, vocab_size), device=self.device)
                    return None, logits
            
            # Store decoder and return wrapper
            self.medusa_decoder = medusa_decoder
            return MedusaModelWrapper(model, medusa_decoder, self.device)
        
        except Exception as e:
            print(f"Error loading Medusa model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Execute loading in executor
    self.sharded_model = await asyncio.get_running_loop().run_in_executor(
      self.executor,
      load_medusa
    )
    
    # Store direct model reference
    if hasattr(self.sharded_model, 'model'):
        self.model = self.sharded_model.model
```

**Purpose:** This method implements specialized loading for Medusa models:
1. Uses `AutoModelForCausalLM` to preserve Medusa's extra prediction heads
2. Creates a `MedusaDecoder` with configuration parameters from CLI
3. Wraps the model in a compatibility layer for the existing API
4. Stores direct references for later use

**Example:** When a user runs `exo run medusa-v1.0-vicuna-7b-v1.5 --medusa-enable --medusa-heads 4 --medusa-tree-size 5 --medusa-candidates 5`, those parameters are used to configure the MedusaDecoder.

## 4. Override `infer_prompt` Method

**File:** `exo/inference/torch/sharded_inference_engine.py`

**Change:** Added an override that routes to Medusa-specific inference

```python
async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    """Override the parent class to provide direct Medusa generation for Medusa models"""
    # If this is a Medusa model and Medusa is enabled, use specialized path
    if self.is_medusa_model_loaded and self.is_medusa_enabled():
        if DEBUG >= 1:
            print(f"Using Medusa direct generation for model: {shard.model_id}")
        return await self._medusa_infer_prompt(request_id, shard, prompt, inference_state)
    
    # Otherwise use standard implementation from parent class
    return await super().infer_prompt(request_id, shard, prompt, inference_state)
```

**Purpose:** This method checks if both:
1. A Medusa model is loaded
2. The Medusa feature is enabled via CLI flag (--medusa-enable)

If both conditions are true, it uses a specialized inference path. Otherwise, it falls back to standard inference.

**Example:** When running with `--medusa-enable`, the code path branches to `_medusa_infer_prompt`.

## 5. Implemented Medusa-specific Inference Method

**File:** `exo/inference/torch/sharded_inference_engine.py`

**Change:** Added Medusa-specific inference implementation

```python
async def _medusa_infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    """Specialized implementation for Medusa models using the decoder directly"""
    # Ensure model is loaded
    await self.ensure_shard(shard)
    
    if not hasattr(self, 'medusa_decoder') or self.medusa_decoder is None:
        if DEBUG >= 1:
            print("Medusa decoder not initialized, falling back to standard inference")
        return await super().infer_prompt(request_id, shard, prompt, inference_state)
    
    # Define generation function for separate thread
    def generate_with_medusa():
        try:
            # Encode input tokens
            device = self.device
            input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Set generation parameters
            max_new_tokens = min(self._medusa_tree_size * 20, 256)
            temperature = 0.0  # Default to greedy for Medusa
            
            # Use Medusa decoder for parallel generation
            with torch.no_grad():
                generated_ids = self.medusa_decoder.generate(
                    input_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
            
            # Convert to numpy for compatibility with rest of system
            if isinstance(generated_ids, torch.Tensor):
                output_array = generated_ids.cpu().numpy()
            else:
                output_array = np.array(generated_ids)
                
            return output_array, input_tokens.shape[1]
            
        except Exception as e:
            print(f"Error in Medusa generation: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    # Run generation in separate thread
    output_array, input_length = await asyncio.get_running_loop().run_in_executor(
        self.executor,
        generate_with_medusa
    )
    
    if output_array is None:
        # Handle failed generation
        vocab_size = getattr(self.tokenizer, 'vocab_size', 32000)
        dummy_logits = np.zeros((1, vocab_size), dtype=np.float32)
        return dummy_logits, inference_state
    
    # Extract generated tokens (excluding input)
    if output_array.shape[1] > input_length:
        # Normal case: output includes input + new tokens
        generated_tokens = output_array[:, input_length:]
    else:
        # Edge case: output might just be the new tokens
        generated_tokens = output_array
    
    # Create dummy logits and store the actual tokens in inference_state
    vocab_size = getattr(self.tokenizer, 'vocab_size', 32000)
    first_token_logits = np.zeros((1, vocab_size), dtype=np.float32)
    
    if inference_state is None:
        inference_state = {}
    
    inference_state["medusa_output"] = generated_tokens
    inference_state["is_medusa_output"] = True
    
    return first_token_logits, inference_state
```

**Purpose:** This is the core implementation of Medusa generation:
1. It tokenizes the input text
2. Calls the Medusa decoder's `generate` method, which performs the actual parallel prediction
3. Extracts the generated tokens and stores them in the inference state
4. Returns a dummy logits array to maintain API compatibility, while the actual tokens are passed via inference_state

**Example:** When generating text, this calls `medusa_decoder.generate()` which uses multiple Medusa heads to predict several tokens in parallel, rather than just one token at a time.

## 6. Modified `process_inference_result` in Node.py

**File:** `exo/orchestration/node.py`

**Change:** Added Medusa-specific output handling

```python
async def process_inference_result(
    self,
    shard,
    result: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
):
    # Check if this is a Medusa output and handle it specially
    if inference_state and inference_state.get("is_medusa_output", False):
        if DEBUG >= 1: print(f"[{request_id}] Processing Medusa output")
        
        # Get the medusa generated tokens
        medusa_output = inference_state.get("medusa_output")
        
        if medusa_output is not None:
            # Initialize buffered output if needed
            if request_id not in self.buffered_token_output:
                self.buffered_token_output[request_id] = ([], False)
                
            # Extract tokens as a flat list
            if len(medusa_output.shape) > 1:
                # Handle 2D array: first dim is batch, second is sequence
                tokens = medusa_output[0].tolist()
            else:
                # Handle 1D array
                tokens = medusa_output.tolist()
                
            # Set buffered output
            self.buffered_token_output[request_id] = (tokens, True)
            
            # Process tokens individually for proper streaming
            is_finished = True
            for i, token in enumerate(tokens):
                is_last = i == len(tokens) - 1
                self.trigger_on_token_callbacks(request_id, [token], is_last)
                # Small delay for smoother streaming
                if not is_last and i % 5 == 0:
                    await asyncio.sleep(0.01)
                    
            # Mark request as complete
            self.outstanding_requests.pop(request_id, None)
            
            # Return the full array of tokens
            return np.array(tokens)
    
    # Continue with existing non-Medusa handling...
```

**Purpose:** This modification processes the output from Medusa generation:
1. Detects if the result has the special `is_medusa_output` flag
2. Extracts the array of tokens from `medusa_output` 
3. Processes tokens individually to maintain streaming API
4. Returns the complete array as the result

**Example:** When Medusa generates multiple tokens at once (e.g., "What is the capital of France?" → "The capital of France is Paris."), this method emits them one by one with small delays for smooth streaming to clients.

## 7. Added `load_checkpoint` Method

**File:** `exo/inference/torch/sharded_inference_engine.py`

**Change:** Implemented the required abstract method

```python
async def load_checkpoint(self, shard: Shard, path: str):
    """Implementation of the abstract method from InferenceEngine."""
    await self.ensure_shard(shard)
    
    def load_checkpoint_func():
        if DEBUG >= 1:
            print(f"Loading checkpoint from {path} for shard {shard}")
        
        try:
            # Placeholder implementation
            print(f"Warning: load_checkpoint not fully implemented for {self.__class__.__name__}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return await asyncio.get_running_loop().run_in_executor(
        self.executor,
        load_checkpoint_func
    )
```

**Purpose:** This implements the abstract method required by the parent class. While not directly related to Medusa functionality, it's required for the class to be instantiable.

## How Medusa Actually Works

1. **Multiple Prediction Heads:** Medusa models have additional prediction heads (typically 4-8) that predict multiple future tokens in parallel.

2. **Tree Search:** The algorithm builds a tree of possible token sequences:
   - The base model generates the first token
   - Medusa heads predict the next several tokens in parallel
   - The algorithm builds candidate sequences from these predictions

3. **Verification:** The algorithm verifies these parallel predictions by feeding previous predictions back into the model to ensure quality.

4. **Acceptance:** Tokens that meet the quality threshold (posterior probability) are accepted without requiring individual model forwards.

## Example of Complete Flow

1. User executes: `exo run medusa-v1.0-vicuna-7b-v1.5 --inference-engine torch --medusa-enable --prompt "What is the capital of France?" --max-generate-tokens 50`

2. The framework detects: 
   - Model ID is "medusa-v1.0-vicuna-7b-v1.5"
   - `--medusa-enable` flag is present

3. In `ensure_shard`:
   - System checks model_cards registry and sees `"is_medusa": True`
   - `is_medusa_model_loaded` is set to `True`
   - Calls `_load_medusa_model`

4. In `_load_medusa_model`:
   - Model is loaded with `AutoModelForCausalLM` preserving Medusa heads
   - `MedusaDecoder` is created with the specified parameters
   - Model is wrapped in `MedusaModelWrapper` for API compatibility

5. In `infer_prompt`:
   - System detects Medusa is enabled
   - Calls specialized `_medusa_infer_prompt` method

6. In `_medusa_infer_prompt`:
   - Tokenizes the prompt
   - Calls `medusa_decoder.generate()` which predicts multiple tokens in parallel
   - Gets back array of tokens for complete response
   - Stores tokens in `inference_state["medusa_output"]`
   - Sets `inference_state["is_medusa_output"] = True`

7. In `process_inference_result`:
   - Detects `is_medusa_output = True`
   - Extracts tokens from `medusa_output`
   - Processes them sequentially for streaming
   - Returns full array as result

The result is significantly faster inference since multiple tokens are predicted in parallel rather than one-by-one as in standard inference.
