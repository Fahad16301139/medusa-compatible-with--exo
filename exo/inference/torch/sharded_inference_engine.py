"""
TorchDynamicShardInferenceEngine
Sharded inference engine using PyTorch based torchtune models
"""

import os
import functools
from concurrent.futures import ThreadPoolExecutor
import asyncio
import uuid
import re
from typing import Optional

import numpy as np
import torch
import torchtune.generation as ttg
from transformers import AutoTokenizer, AutoModelForCausalLM

from exo.inference.inference_engine import InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.inference.shard import Shard
from exo.inference.tokenizers import _resolve_tokenizer
from exo.helpers import DEBUG
from exo.inference.torch.models.llm_utils import (
  load_model_config,
  load_model_weights_torchtune,
  ShardInferenceState
)

from exo.inference.torch.models.general_mha import ShardedGeneralModel

# from torchtune generate recipe
# https://github.com/pytorch/torchtune/blob/main/recipes/configs/generation.yaml#L40
TEMP = 0.6
TOP_K = 35

class TorchDynamicShardInferenceEngine(InferenceEngine):
  """
  Pytorch based inferece engine for sharded models
  """
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.sharded_model = None
    self.request_id = None
    self.executor = ThreadPoolExecutor(max_workers=1)
    self.uuid = str(uuid.uuid4())
    self.model_path = None
    self.model_config = None
    self.state = None
    self.oom_cnt = 0
    
    # This will hold the actual model reference for direct access by Medusa
    self.model = None
    
    # Flag to indicate if this is a Medusa model
    self.is_medusa_model_loaded = False
    
    # cache settings
    self.use_cache = bool(os.getenv("TORCH_USE_CACHE", "True").lower() == "true")
    self.cache_setup = False

    # device settings
    if os.environ.get("TORCH_DEVICE"):
      self.device = torch.device(os.environ["TORCH_DEVICE"])
    elif torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

    # rng setup for sampling
    self.rng = torch.Generator(device=self.device)
    self.rng.manual_seed(1234)

  def setup_cache(self, batch_size: int=1, total_response_length: int=1024):
    # setup cache
    # this is needed for a primary node that gets the initial encoding
    if not self.sharded_model.model.caches_are_enabled() and self.use_cache:
      with self.device:
        self.sharded_model.model.setup_caches(
          batch_size,
          self.model_config["torch_dtype"],
          decoder_max_seq_len=total_response_length
        )
      
      self.cache_setup = True


  def clear_model(self):
    """
    Clear out model and shard
    A way to avoid OOM issues
    
    All prompts are stored in VRAM
    while inference engine is up and using the same
    model class instance, this will clear it for each prompt.

    OOM issue might occur in longer chats/contexts depending on your machine.
    """
    if self.sharded_model.model.caches_are_enabled():
      self.sharded_model.model.reset_caches()
    
    del self.sharded_model
    self.sharded_model = None
    
    if self.device == torch.device("cuda"):
      torch.cuda.empty_cache()
    
    self.shard = None
    self.state = None

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    if DEBUG >= 4:
      print("encode called")
      print(f"shard: {shard}")
      print(f"prompt: {prompt}")

    await self.ensure_shard(shard)

    def encode_wrapper() -> np.ndarray:
      """
      Encode the tensors from prompt along with the
      initial input_pos and mask
      """
      tokens = self.tokenizer.encode(
        prompt,
        return_tensors="pt"
      )

      # move to proper device, default is CPU
      if tokens.device != self.device:
        tokens = tokens.to(device=self.device)
      
      if DEBUG >= 4:
        print("encoded_wrapper called")
        print(f"tokens: {tokens}")

      # if going past max, just take from max onward
      if len(tokens) > self.sharded_model.max_generated_tokens:
        max_gen_tokens = self.sharded_model.max_generated_tokens
        tokens = tokens[-max_gen_tokens:]

      self.state.tokens = tokens

      bsz, tklng = tokens.size()
      total_response_length = tklng + self.sharded_model.max_generated_tokens

      self.setup_cache(bsz, total_response_length)
      
      # setup max sequence length
      if not self.sharded_model.model.caches_are_enabled():
        max_seq_len = total_response_length
      else:
        max_seq_len = self.sharded_model.model.decoder_max_cache_seq_len

      # set pad_id
      if hasattr(self.tokenizer, "pad_id"):
        pad_id = self.tokenizer.pad_id
      elif hasattr(self.tokenizer, "pad_token_id"):
        print(f"pad_token_id: {self.tokenizer.pad_token_id}")
        if self.tokenizer.pad_token_id is not None:
          pad_id = self.tokenizer.pad_token_id
        else:
          pad_id = 0
      else:
        pad_id = 0
      
      padding_masks = tokens != pad_id
      if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
          padding_masks,
          (0, self.sharded_model.max_generated_tokens),
          value=True,
        )

        self.state.mask = ttg.get_causal_mask_from_padding_mask(padding_masks, target_seq_len=max_seq_len)

        self.state.input_pos = ttg.get_position_ids_from_padding_mask(padding_masks)
      else:
        self.state.mask = torch.tril(torch.ones(
          total_response_length,
          max_seq_len,
          dtype=torch.bool,
          device=self.device,
        )).unsqueeze(0)

        self.state.input_pos = torch.arange(0, total_response_length, device=self.device).unsqueeze(0)

      return tokens

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(encode_wrapper),
    )

  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    if DEBUG >= 4:
      print("decode called")
      print(f"shard: {shard}")
      print(f"tokens: {tokens}")

    await self.ensure_shard(shard)

    return await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(self.tokenizer.decode, tokens.tolist()),
    )

  async def sample(self, x: np.ndarray, temp=TEMP, top_k=TOP_K) -> np.ndarray:
    if DEBUG >= 4:
      print("sample called")
      print(f"x: {x}")
      print(f"temp: {temp}")
      print(f"top_k: {top_k}")
      print(self.device)

    logits = torch.tensor(x).to(self.device)

    def sample_wrapper():
      q = torch.empty((logits.size(0), self.sharded_model.model.tok_embeddings.num_embeddings), device=logits.device).exponential_(1, generator=self.rng)

      tokens = ttg.sample(logits.clone(), temperature=temp, top_k=top_k, q=q.to(self.device))
      
      if DEBUG >= 4:
        print(f"tokens: {tokens}")

      return tokens.numpy(force=True)

    return await asyncio.get_running_loop().run_in_executor(self.executor, functools.partial(sample_wrapper))

  async def infer_tensor(
    self,
    request_id: str,
    shard: Shard,
    input_data: np.ndarray,
    inference_state: Optional[dict] = None
  ) -> tuple[np.ndarray, Optional[dict]]:

    await self.ensure_shard(shard)

    # ensure shard
    if DEBUG >= 4:
      print("infer_tensor called")
      print(f"shard: {shard}")
      print(f"input_data: {input_data}")

    if inference_state.get("tokens") is not None:
      self.state.from_dict(inference_state)

    self.request_id = request_id if not self.request_id else self.request_id

    hidden_state = None
    input_tensor = None
    if input_data.ndim == 3:
      hidden_state = torch.tensor(input_data).to(
        device=self.device,
        dtype=self.model_config["torch_dtype"]
      )
    elif input_data.ndim == 2:
      input_tensor = torch.tensor(input_data).to(
        device=self.device
      )

    if self.use_cache and not self.cache_setup:
      if input_tensor is not None:
        bsz, tklng = input_tensor.size()
        self.setup_cache(
          bsz,
          tklng + self.sharded_model.max_generated_tokens
        )
      else:
        bsz, tklng = self.state.tokens.size()
        self.setup_cache(
          bsz,
          tklng + self.sharded_model.max_generated_tokens
        )

    def infer_wrapper():
      if DEBUG >= 4:
        print(f"infer_wrapper called [{self.oom_cnt} OOM]")
        print(f"self.state.tokens: {self.state.tokens}")
        print(f"hidden_state: {hidden_state}")

      model_cache = self.sharded_model.model.caches_are_enabled()

      if self.state.tokens is not None:
        if input_data.ndim == 2 and input_tensor.size(-1) == 1:
          self.state.tokens = torch.cat([
            self.state.tokens.to(self.device),
            input_tensor.clone()
          ], dim=-1).to(self.device)
      else:
        self.state.tokens = input_tensor.clone()

      try:
        in_tokens = self.state.tokens.clone().to(
          device=self.device
        )

        in_input_pos = self.state.input_pos.clone().to(
          device=self.device
        )

        in_mask = self.state.mask.clone().to(
          device=self.device
        )

        if hidden_state is not None:
          model_hs, model_logits = self.sharded_model.generate(
            tokens=in_tokens,
            hidden_state=hidden_state,
            input_pos=in_input_pos,
            mask=in_mask,
            curr_pos=self.state.curr_pos
          )
        else:
          if not model_cache:
            model_hs, model_logits = self.sharded_model.generate(
              tokens=in_tokens,
              input_pos=in_input_pos,
              mask=in_mask,
              curr_pos=self.state.curr_pos
            )
          else:
            model_hs, model_logits = self.sharded_model.generate(
              tokens=input_tensor,
              input_pos=in_input_pos,
              mask=in_mask,
              curr_pos=self.state.curr_pos
            )
      except torch.cuda.OutOfMemoryError:
        print(f"OOM on cuda, clearing model and stopping")
        self.oom_cnt += 1
        self.clear_model()
        return
      except Exception as err:
        print(f"infer_tensor err\n{err}")
        raise

      if model_hs is not None:
        # numpy current no support for bf16
        if model_hs.dtype == torch.bfloat16:
          model_hs = model_hs.float()

        if DEBUG >= 4:
          print("sending hidden states")
          print(f"model_hs: {model_hs.size()}")
          print(f"state.tokens: {self.state.tokens}")
          print(f"state.input_pos: {self.state.input_pos.size()}")
          print(f"state.mask: {self.state.mask.size()}")
        
        return (
          model_hs.numpy(force=True),
          self.state.to_dict(),
        )
      
      if self.state.curr_pos == 0:
        self.state.curr_pos = self.state.tokens.size(-1)
      else:
        self.state.curr_pos += 1

      # numpy current no support for bf16
      if model_logits.dtype == torch.bfloat16:
        model_logits = model_logits.float()

      return (
        model_logits[:, -1].numpy(force=True),
        self.state.to_dict(),
      )

    return await asyncio.get_running_loop().run_in_executor(self.executor, infer_wrapper)

  async def ensure_shard(self, shard: Shard):
    if DEBUG >= 4:
      print("shard ensured\n")
      print(f"shard: {shard}")
      print(f"class shard: {self.shard}")
      print(f"uuid: {self.uuid}")

    # Skip if already loaded
    if self.shard == shard:
      return

    self.shard = shard

    # Check if this is a Medusa model
    from exo.models import model_cards
    self.is_medusa_model_loaded = model_cards.get(shard.model_id, {}).get("is_medusa", False)
    
    # Using CPU to store inference state
    self.state = ShardInferenceState()

    # download model safetensors and shard
    self.model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
    self.model_config = load_model_config(self.model_path/"config.json")

    # self.tokenizer = await _resolve_tokenizer(model_path)
    self.tokenizer = await _resolve_tokenizer(self.model_path)

    # Different loading process for Medusa models vs standard models
    if self.is_medusa_model_loaded:
      if DEBUG >= 1:
        print(f"Loading Medusa-specific model: {shard.model_id}")
      # Load Medusa model with specialized code path
      await self._load_medusa_model(shard)
    else:
      # Load regular model
      await self._load_standard_model(shard)
    
  async def _load_medusa_model(self, shard: Shard):
    """Specialized loader for Medusa models that includes medusa-specific components"""
    def load_medusa():
      try:
        if DEBUG >= 1:
          print(f"Loading Medusa model from {self.model_path}")
        
        # Import medusa-specific libraries
        import torch
        
        # Direct transformers loading bypassing ShardedGeneralModel
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.model_config["torch_dtype"],
            device_map="auto",  # Let transformers decide optimal device mapping
            trust_remote_code=True  # Important for Medusa's custom code
        )
        
        # Set model to evaluation mode
        model.eval()
        
        if DEBUG >= 1:
          print(f"Medusa model successfully loaded")
          print(f"Model type: {type(model).__name__}")
        
        # Create a simple wrapper model that provides the same API as ShardedGeneralModel
        # but is specifically designed for Medusa models
        from exo.inference.medusa_decoder import MedusaDecoder
        
        # Extract Medusa configuration from our settings
        medusa_config = self.get_medusa_config()
        
        # Create Medusa decoder which will actually be used for generation
        medusa_decoder = MedusaDecoder(
          model=model,
          tokenizer=self.tokenizer,
          medusa_heads=medusa_config["heads"],
          tree_size=medusa_config["tree_size"],
          max_candidates=medusa_config["candidates"]
        )
        
        if DEBUG >= 1:
          print(f"Medusa decoder initialized with:")
          print(f"  - {medusa_config['heads']} heads")
          print(f"  - Tree size: {medusa_config['tree_size']}")
          print(f"  - Max candidates: {medusa_config['candidates']}")
        
        # Create a ShardedGeneralModel-like wrapper for compatibility with rest of code
        class MedusaModelWrapper:
          def __init__(self, model, decoder, device, max_tokens=2048):
            self.model = model
            self.medusa_decoder = decoder
            self.device = device
            self.max_generated_tokens = max_tokens
          
          def generate(self, tokens, **kwargs):
            """Compatibility method to match ShardedGeneralModel API"""
            # Return placeholder tensors since actual generation happens in infer_prompt
            batch_size = tokens.shape[0]
            vocab_size = getattr(self.model.config, "vocab_size", 32000)
            
            # Create dummy tensors that maintain the API contract
            logits = torch.zeros((batch_size, 1, vocab_size), device=self.device)
            return None, logits
        
        # Create and return the wrapper
        self.medusa_decoder = medusa_decoder
        return MedusaModelWrapper(model, medusa_decoder, self.device)
      
      except Exception as e:
        print(f"Error loading Medusa model: {e}")
        import traceback
        traceback.print_exc()
        # Return placeholder to avoid None errors
        return PlaceholderShardedModel(self.device)
    
    # Execute loading in executor
    self.sharded_model = await asyncio.get_running_loop().run_in_executor(
      self.executor,
      load_medusa
    )
    
    # Set direct reference for other code to use
    if hasattr(self.sharded_model, 'model'):
      self.model = self.sharded_model.model
    
  async def _load_standard_model(self, shard: Shard):
    """Standard model loading for non-Medusa models"""
    def start_model():
      if DEBUG >= 4:
        print("start_model called")

      self.sharded_model = ShardedGeneralModel(
        config=self.model_config,
        shard=self.shard,
        device=self.device,
        dtype=self.model_config["torch_dtype"],
        use_cache=self.use_cache
      )

      load_model_weights_torchtune(
        cache_dir=self.model_path,
        shard=self.shard,
        model=self.sharded_model,
        num_heads=self.model_config["num_heads"],
        num_kv_heads=self.model_config["num_kv_heads"],
        dim=self.model_config["embed_dim"],
        head_dim=self.model_config["head_dim"]
      )
      
      # Set direct model reference
      if self.sharded_model is not None:
        self.model = self.sharded_model.model
    
    await asyncio.get_running_loop().run_in_executor(
      self.executor,
      functools.partial(start_model),
    )
    
  async def load_checkpoint(self, shard: Shard, path: str):
    """Implementation of the abstract method from InferenceEngine.
    Loads a model checkpoint from the specified path."""
    await self.ensure_shard(shard)
    
    def load_checkpoint_func():
      if DEBUG >= 1:
        print(f"Loading checkpoint from {path} for shard {shard}")
      
      try:
        # Actual checkpoint loading logic would go here
        # This is a placeholder implementation to satisfy the abstract method
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
  
  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    """Override the parent class to provide direct Medusa generation for Medusa models"""
    # If this is a Medusa model and Medusa is enabled, use specialized path
    if self.is_medusa_model_loaded and self.is_medusa_enabled():
      if DEBUG >= 1:
        print(f"Using Medusa direct generation for model: {shard.model_id}")
      return await self._medusa_infer_prompt(request_id, shard, prompt, inference_state)
    
    # Otherwise use standard implementation from parent class
    return await super().infer_prompt(request_id, shard, prompt, inference_state)
    
  async def _medusa_infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    """Specialized implementation for Medusa models that uses the Medusa decoder directly"""
    # Ensure model is loaded
    await self.ensure_shard(shard)
    
    if not hasattr(self, 'medusa_decoder') or self.medusa_decoder is None:
      if DEBUG >= 1:
        print("Medusa decoder not initialized, falling back to standard inference")
      return await super().infer_prompt(request_id, shard, prompt, inference_state)
      
    # Define the generation function to run in a separate thread
    def generate_with_medusa():
      try:
        # Encode input
        device = self.device
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        if DEBUG >= 1:
          print(f"Generating with Medusa, input shape: {input_tokens.shape}")
          
        # Set generation parameters
        max_new_tokens = min(self._medusa_tree_size * 20, 256)  # Reasonable default
        temperature = 0.0  # Default to greedy for Medusa
        
        # Use Medusa decoder for generation
        with torch.no_grad():
          generated_ids = self.medusa_decoder.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature
          )
          
        if DEBUG >= 1:
          print(f"Medusa generation complete, output shape: {generated_ids.shape}")
        
        # Convert to numpy for compatibility with the rest of the system
        if isinstance(generated_ids, torch.Tensor):
          output_array = generated_ids.cpu().numpy()
        else:
          # If not a tensor, try to convert it
          output_array = np.array(generated_ids)
          
        return output_array, input_tokens.shape[1]
        
      except Exception as e:
        print(f"Error in Medusa generation: {e}")
        import traceback
        traceback.print_exc()
        return None, 0
      
    # Run generation in a separate thread to avoid blocking
    output_array, input_length = await asyncio.get_running_loop().run_in_executor(
      self.executor,
      generate_with_medusa
    )
    
    if output_array is None:
      # If generation failed, return dummy logits
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
      
    # Create a dummy logits array from the first token
    # The real tokens will be processed by the Node class after this
    vocab_size = getattr(self.tokenizer, 'vocab_size', 32000)
    first_token_logits = np.zeros((1, vocab_size), dtype=np.float32)
    
    # Store the complete generated sequence in inference_state so Node can access it
    if inference_state is None:
      inference_state = {}
    
    inference_state["medusa_output"] = generated_tokens
    inference_state["is_medusa_output"] = True
    
    return first_token_logits, inference_state
