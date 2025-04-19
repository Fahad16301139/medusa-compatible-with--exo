import os
import numpy as np
import json
import asyncio
import uuid
import time
import traceback
from typing import List, Dict, Optional, Tuple, Union, Set
from exo.networking import Discovery, PeerHandle, Server
from exo.inference.inference_engine import InferenceEngine, Shard
from exo.topology.topology import Topology
from exo.topology.device_capabilities import device_capabilities, UNKNOWN_DEVICE_CAPABILITIES
from exo.topology.partitioning_strategy import Partition, PartitioningStrategy, map_partitions_to_shards
from exo import DEBUG
from exo.helpers import AsyncCallbackSystem
from exo.viz.topology_viz import TopologyViz
from exo.download.download_progress import RepoProgressEvent
from exo.inference.inference_engine import get_inference_engine, InferenceEngine
from exo.download.shard_download import ShardDownloader
from exo.models import model_cards
import concurrent.futures
import torch


class Node:
  def __init__(
    self,
    _id: str,
    server: Server,
    inference_engine: InferenceEngine,
    discovery: Discovery,
    shard_downloader: ShardDownloader,
    partitioning_strategy: PartitioningStrategy = None,
    max_generate_tokens: int = 1024,
    default_sample_temperature: float = 0.0,
    topology_viz: Optional[TopologyViz] = None,
  ):
    self.id = _id
    self.inference_engine = inference_engine
    self.server = server
    self.discovery = discovery
    self.shard_downloader = shard_downloader
    self.partitioning_strategy = partitioning_strategy
    self.peers: List[PeerHandle] = {}
    self.topology: Topology = Topology()
    self.device_capabilities = UNKNOWN_DEVICE_CAPABILITIES
    self.buffered_token_output: Dict[str, Tuple[List[int], bool]] = {}
    self.buffered_logits: Dict[str, List[np.ndarray]] = {}
    self.buffered_inputs: Dict[str, List[np.ndarray]] = {}
    self.buffered_partials: Dict[str, List[np.ndarray]] = {}
    self.checkpoints: Dict[str, Dict[str, int]] = {}

    self.max_generate_tokens = max_generate_tokens
    self.topology_viz = topology_viz
    self.default_sample_temperature = default_sample_temperature
    self._on_token = AsyncCallbackSystem[str, Tuple[str, List[int], bool]]()
    self._on_opaque_status = AsyncCallbackSystem[str, Tuple[str, str]]()
    self._on_opaque_status.register("node_status").on_next(self.on_node_status)
    self.node_download_progress: Dict[str, RepoProgressEvent] = {}
    self.topology_inference_engines_pool: List[List[str]] = []
    self.outstanding_requests = {}

  async def start(self, wait_for_peers: int = 0) -> None:
    self.device_capabilities = await device_capabilities()
    await self.server.start()
    await self.discovery.start()
    await self.update_peers(wait_for_peers)
    await self.collect_topology(set())
    if DEBUG >= 2: print(f"Collected topology: {self.topology}")
    asyncio.create_task(self.periodic_topology_collection(2.0))

  async def stop(self) -> None:
    await self.discovery.stop()
    await self.server.stop()

  def on_node_status(self, request_id, opaque_status):
    try:
      status_data = json.loads(opaque_status)
      status_type = status_data.get("type", "")
      if status_type == "supported_inference_engines":
        node_id = status_data.get("node_id")
        engines = status_data.get("engines", [])
        self.topology_inference_engines_pool.append(engines)
      elif status_type == "node_status":
        if status_data.get("status", "").startswith("start_"):
          self.current_topology.active_node_id = status_data.get("node_id")
        elif status_data.get("status", "").startswith("end_"):
          if status_data.get("node_id") == self.current_topology.active_node_id:
            self.current_topology.active_node_id = None

      download_progress = None
      if status_type == "download_progress":
        if DEBUG >= 8: print(f"Download progress from {status_data.get('node_id')}: {status_data.get('progress')}")
        download_progress = RepoProgressEvent.from_dict(status_data.get('progress'))
        self.node_download_progress[status_data.get('node_id')] = download_progress

      if self.topology_viz:
        self.topology_viz.update_visualization(self.topology, self.partitioning_strategy.partition(self.topology), self.id, self.node_download_progress)
    except Exception as e:
      if DEBUG >= 1: print(f"Error on_node_status: {e}")
      if DEBUG >= 1: traceback.print_exc()

  def get_supported_inference_engines(self):
    supported_engine_names = []
    if self.inference_engine.__class__.__name__ == 'MLXDynamicShardInferenceEngine':
      supported_engine_names.append('mlx')
      supported_engine_names.append('tinygrad')
    else:
      supported_engine_names.append(os.environ.get("EXO_INFER_ENGINE", 'tinygrad'))

    return supported_engine_names

  async def broadcast_supported_engines(self, supported_engines_names: List[str]):
    status_message = json.dumps({"type": "supported_inference_engines", "node_id": self.id, "engines": supported_engines_names})
    await self.broadcast_opaque_status("", status_message)

  def get_topology_inference_engines(self) -> List[List[str]]:
    return self.topology_inference_engines_pool
  
  token_count = 0
  first_token_time = 0
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
          
        if DEBUG >= 1: print(f"[{request_id}] Processing {len(tokens)} Medusa-generated tokens")
          
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
        
    # Standard non-Medusa processing
    if shard.model_id != 'stable-diffusion-2-1-base':
      if request_id not in self.buffered_token_output:
        self.buffered_token_output[request_id] = ([], False)
      is_finished = len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
      if shard.is_last_layer() and not is_finished:
        token = await self.inference_engine.sample(result, temp=self.default_sample_temperature)
        await self.inference_engine.ensure_shard(shard)
        self.buffered_token_output[request_id][0].append(token.item())
        is_finished = token.item() == self.inference_engine.tokenizer.eos_token_id or is_finished or len(self.buffered_token_output[request_id][0]) >= self.max_generate_tokens
        if DEBUG >= 2: print(f"[{request_id}] result size: {result.size}, is finished: {is_finished}, buffered tokens: {len(self.buffered_token_output[request_id][0])}")
        forward = token.reshape(1, -1)
        intermediate_result = [self.buffered_token_output[request_id][0][-1]]
      else:
        forward = result
    else:
      await self.inference_engine.ensure_shard(shard)
      is_finished = inference_state.get("is_finished", False)
      intermediate_result, inference_state = self.handle_stable_diffusion(inference_state, result)
      forward = result
    if shard.is_last_layer():
      self.trigger_on_token_callbacks(request_id, intermediate_result, is_finished)
      asyncio.create_task(self.broadcast_result(request_id, intermediate_result, is_finished))

    if is_finished:
      if shard.model_id != 'stable-diffusion-2-1-base':
        self.buffered_token_output[request_id] = (self.buffered_token_output[request_id][0], True)
      self.outstanding_requests.pop(request_id)
    else:
      self.outstanding_requests[request_id] = "waiting"
      asyncio.create_task(self.forward_tensor(shard, forward, request_id, self.get_partition_index(offset=1), inference_state))

    return np.array(self.buffered_token_output[request_id][0]) if shard.model_id != 'stable-diffusion-2-1-base' else intermediate_result

  async def process_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = {},
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "start_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_prompt(base_shard, prompt, request_id, inference_state)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": "end_process_prompt",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "prompt": prompt,
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      )
    )
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=} {elapsed_time_ns=}")

  async def _process_prompt(self, base_shard: Shard, prompt: str, request_id: Optional[str] = None, inference_state: Optional[dict] = None) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 2: print(f"[{request_id}] process prompt: {base_shard=} {shard=} {prompt=}")

    if not shard.is_first_layer():
      if DEBUG >= 2: print(f"[{request_id}] forwarding to next shard: {base_shard=} {shard=} {prompt=}")
      self.outstanding_requests[request_id] = "waiting"
      resp = await self.forward_prompt(shard, prompt, request_id, 0, inference_state)
      return None
    else:
      self.outstanding_requests[request_id] = "processing"
      
      # Check if this is a Medusa model and if Medusa decoding is enabled
      is_medusa_model = model_cards.get(shard.model_id, {}).get("is_medusa", False)
      
      if is_medusa_model and self.inference_engine.is_medusa_enabled():
        if DEBUG >= 1: print(f"[{request_id}] Using Medusa decoding for {shard.model_id}")
        
        # Import and initialize the Medusa decoder
        try:
          from exo.inference.medusa_decoder import MedusaDecoder
          
          # Get Medusa configuration from the inference engine
          medusa_config = self.inference_engine.get_medusa_config()
          
          # First ensure the model is loaded in the inference engine
          await self.inference_engine.ensure_shard(shard)
          
          # Get the appropriate model reference based on inference engine type
          model = None
          tokenizer = self.inference_engine.tokenizer
          
          # Check for sharded_model in TorchDynamicShardInferenceEngine
          if hasattr(self.inference_engine, 'sharded_model') and self.inference_engine.sharded_model is not None:
            model = self.inference_engine.sharded_model.model
            if DEBUG >= 1: print(f"[{request_id}] Found model in sharded_model.model")
          # Direct model access as fallback
          elif hasattr(self.inference_engine, 'model'):
            model = self.inference_engine.model
            if DEBUG >= 1: print(f"[{request_id}] Found direct model reference")
          else:
            if DEBUG >= 1: print(f"[{request_id}] No model reference found, falling back to regular inference")
            
          # Create a MedusaDecoder instance if we have a model
          if model is not None:
            medusa_decoder = MedusaDecoder(
              model=model,
              tokenizer=tokenizer,
              medusa_heads=medusa_config["heads"],
              tree_size=medusa_config["tree_size"],
              max_candidates=medusa_config["candidates"],
              posterior_threshold=0.09,  # Default value
              posterior_alpha=0.3  # Default value
            )
            if DEBUG >= 1: print(f"[{request_id}] Successfully initialized MedusaDecoder")
            
            # Actually use Medusa for generation
            try:
              # Encode the input prompt to get initial tokens
              input_tokens = tokenizer.encode(prompt, return_tensors="pt")
              
              # Determine the device to use
              device = None
              if hasattr(model, 'device'):
                device = model.device
              elif hasattr(self.inference_engine, 'device'):
                device = self.inference_engine.device
              else:
                # Default to CUDA if available, otherwise CPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              
              # Move input tokens to the appropriate device
              input_tokens = input_tokens.to(device)
              
              # Generate with Medusa
              # This should be run in a thread executor since it's a CPU-bound operation
              def generate_with_medusa():
                with torch.no_grad():
                  if DEBUG >= 1: print(f"[{request_id}] Starting Medusa generation with shape {input_tokens.shape}")
                  
                  # Get output from the Medusa decoder
                  output_tokens = medusa_decoder.generate(
                      input_tokens,
                      max_new_tokens=min(self.max_generate_tokens, 256),  # Start with reasonable limit
                      temperature=self.default_sample_temperature
                  )
                  if DEBUG >= 1: print(f"[{request_id}] Medusa generation completed with shape {output_tokens.shape if output_tokens is not None else None}")
                  return output_tokens
              
              # Execute Medusa generation in a separate thread to avoid blocking
              loop = asyncio.get_running_loop()
              if hasattr(self.inference_engine, 'executor'):
                  output_tokens = await loop.run_in_executor(
                      self.inference_engine.executor, 
                      generate_with_medusa
                  )
              else:
                  # Create a temporary executor if none exists
                  with concurrent.futures.ThreadPoolExecutor() as executor:
                      output_tokens = await loop.run_in_executor(
                          executor, 
                          generate_with_medusa
                      )
              
              # Process the Medusa output
              if DEBUG >= 1: print(f"[{request_id}] Medusa generation successful")
              
              # If we have tokens successfully generated with Medusa
              if output_tokens is not None:
                if DEBUG >= 1: print(f"[{request_id}] Processing output with shape: {output_tokens.shape}")
                
                # Extract the generated tokens (skip the input prompt)
                input_length = input_tokens.shape[1]
                # Make sure we handle both cases - where output includes input or just new tokens
                if output_tokens.shape[1] > input_length:
                  generated_tokens = output_tokens[0, input_length:].cpu().tolist()
                else:
                  # Just use all tokens as sometimes models return only new tokens
                  generated_tokens = output_tokens[0].cpu().tolist()
                
                if DEBUG >= 1: print(f"[{request_id}] Generated {len(generated_tokens)} new tokens")
                
                # Create synthetic results that match the expected format
                # Setup buffered tokens for the result processing logic
                if request_id not in self.buffered_token_output:
                  self.buffered_token_output[request_id] = ([], False)
                
                # Add the generated tokens to the buffer
                self.buffered_token_output[request_id] = (generated_tokens, True)
                
                # Process and return all tokens at once - trigger callbacks one by one
                for i, token in enumerate(generated_tokens):
                  is_last = i == len(generated_tokens) - 1
                  self.trigger_on_token_callbacks(request_id, [token], is_last)
                  # Small delay to allow processing between tokens
                  if not is_last and i % 10 == 0:  # Every 10 tokens
                    await asyncio.sleep(0.01)  # Small delay
                  
                # Mark as finished  
                self.outstanding_requests.pop(request_id, None)
                
                # Return a placeholder result (logits with appropriate vocab size)
                vocab_size = 32000  # Default for most models
                if hasattr(tokenizer, 'vocab_size'):
                  vocab_size = tokenizer.vocab_size
                elif hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                  vocab_size = model.config.vocab_size
                
                placeholder_result = np.zeros((1, vocab_size), dtype=np.float32)
                return placeholder_result
              else:
                if DEBUG >= 1: print(f"[{request_id}] Medusa generation returned None, falling back to regular inference")
                
            except Exception as e:
              if DEBUG >= 1:
                print(f"[{request_id}] Error during Medusa generation: {e}")
                traceback.print_exc()
              # Fall back to regular inference on any error
          
        except Exception as e:
          if DEBUG >= 1: 
            print(f"[{request_id}] Error initializing Medusa: {e}")
            traceback.print_exc()
        
        # If we reach here, either Medusa init failed or generation failed
        if DEBUG >= 1: print(f"[{request_id}] Falling back to regular inference")
        result, inference_state = await self.inference_engine.infer_prompt(request_id, shard, prompt, inference_state)
      else:
        # Regular non-Medusa inference
        result, inference_state = await self.inference_engine.infer_prompt(request_id, shard, prompt, inference_state)
      
      ret = await self.process_inference_result(shard, result, request_id, inference_state)
      return result

  async def enqueue_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    request_id: Optional[str] = None,
    train: bool = False,
  ):
    shard = self.get_current_shard(base_shard)
    if shard.is_first_layer():
      loss = await self.process_example(shard, example, target, length, train, request_id)
      return loss
    else:
      if request_id is None:
        request_id = str(uuid.uuid4())
      self.outstanding_requests[request_id] = "waiting"
      loss = await self.forward_example(shard, example, target, length, train, request_id, 0)
    return loss

  async def coordinate_save(
    self,
    base_shard: Shard,
    iteration: int,
    destination: str,
  ):
    shard = self.get_current_shard(base_shard)
    model = shard.model_id
    sid = shard.__hash__()
    path = f"{destination}/{model}/{sid}-{iteration}.safetensors"
    self.outstanding_requests[f"{sid}::{iteration}"] = "Checking"
    if model not in self.checkpoints:
      self.checkpoints[model] = {}
    if sid not in self.checkpoints[model]:
      self.checkpoints[model][sid] = []
    if len(self.checkpoints[model][sid]) < 1 or self.checkpoints[model][sid][-1] < iteration:
      print(f"Saving checkpoint to {path}")
      self.outstanding_requests[f"{sid}::{iteration}"] = "Saving"
      import os
      os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
      await self.inference_engine.save_checkpoint(shard, path)
      self.checkpoints[model][sid] = sorted(self.checkpoints[model][sid] + [iteration])
    self.outstanding_requests.pop(f"{sid}::{iteration}")

  async def process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ):
    shard = self.get_current_shard(base_shard)
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"start_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "example_size": example.size,
          "example_shape": example.shape,
          "request_id": request_id,
        }),
      )
    )
    start_time = time.perf_counter_ns()
    resp = await self._process_example(shard, example, target, length, train, request_id)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    asyncio.create_task(
      self.broadcast_opaque_status(
        request_id,
        json.dumps({
          "type": "node_status",
          "node_id": self.id,
          "status": f"end_{'train' if train else 'eval'}_example",
          "base_shard": base_shard.to_dict(),
          "shard": shard.to_dict(),
          "request_id": request_id,
          "elapsed_time_ns": elapsed_time_ns,
        }),
      )
    )
    return resp

  async def _process_example(
    self,
    base_shard: Shard,
    example: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool = False,
    request_id: Optional[str] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)
    if DEBUG >= 1: print(f"[{request_id}] process_example: {example.shape=}")
    try:
      target = target.astype(int)
      if train:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "training"
          loss, grad = await self.inference_engine.train(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step, _ = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss, backgrad = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset=1))
          self.outstanding_requests[request_id] = "training"
          partial_loss, grad = await self.inference_engine.train(request_id, shard, example, backgrad, length, loss="back_gradient")
        self.outstanding_requests.pop(request_id)
        if shard.is_first_layer():
          return loss
        else:
          return loss, grad
      else:
        if shard.is_last_layer():
          self.outstanding_requests[request_id] = "evaluating"
          loss = await self.inference_engine.evaluate(request_id, shard, example, target, length)
        else:
          self.outstanding_requests[request_id] = "preprocessing"
          step, _ = await self.inference_engine.infer_tensor(request_id, shard, example)
          self.outstanding_requests[request_id] = "waiting"
          loss = await self.forward_example(shard, step, target, length, train, request_id, self.get_partition_index(offset=1))
        self.outstanding_requests.pop(request_id)
        return loss
    except Exception as e:
      self.outstanding_requests.pop(request_id)
      print(f"Error processing example for shard {shard}: {e}")
      traceback.print_exc()
      return None

  async def process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ) -> Optional[np.ndarray]:
    shard = self.get_current_shard(base_shard)
    start_time = time.perf_counter_ns()
    resp = await self._process_tensor(shard, tensor, request_id, inference_state)
    end_time = time.perf_counter_ns()
    elapsed_time_ns = end_time - start_time
    if DEBUG >= 2: print(f"[{request_id}] process_tensor: {base_shard=} {shard=} {tensor.size=} {tensor.shape=} {elapsed_time_ns=}")

  async def _process_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: Optional[str] = None,
    inference_state: Optional[dict] = None,
  ) -> Optional[np.ndarray]:
    if request_id is None:
      request_id = str(uuid.uuid4())
    shard = self.get_current_shard(base_shard)

    try:
      self.outstanding_requests[request_id] = "processing"
      result, inference_state = await self.inference_engine.infer_tensor(request_id, shard, tensor, inference_state)
      ret = await self.process_inference_result(shard, result, request_id, inference_state)
      return ret
    except Exception as e:
      self.outstanding_requests.pop(request_id)
      print(f"Error processing tensor for shard {shard}: {e}")
      traceback.print_exc()
  
  async def forward_example(
    self,
    base_shard: Shard,
    step: np.ndarray,
    target: np.ndarray,
    length: np.ndarray,
    train: bool,
    request_id: str,
    target_index: int,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    target_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"computed target from: {base_shard} {target_index}, {self.topology}. target shard: {target_shard}")
    target_peer = next((p for p in self.peers if p.id() == target_id), None)
    if not target_peer:
      raise ValueError(f"peer for {target_index} not found")
    if DEBUG >= 1: print(f"sending example to {target_peer.id()}: {step} => {target} ({length})")
    resp = await target_peer.send_example(target_shard, step, target, length, request_id=request_id, train=train)
    return resp

  async def forward_prompt(
    self,
    base_shard: Shard,
    prompt: str,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. next shard: {next_shard}")
    if target_id == self.id:
      await self.process_prompt(next_shard, prompt, request_id, inference_state)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending prompt to {target_peer.id()}: {prompt}")
      await target_peer.send_prompt(next_shard, prompt, request_id=request_id, inference_state=inference_state)

  async def forward_tensor(
    self,
    base_shard: Shard,
    tensor: np.ndarray,
    request_id: str,
    target_index: int,
    inference_state: Optional[dict] = None,
  ) -> None:
    if DEBUG >= 1: print(f"target partition index: {target_index}")
    target_id = self.partitioning_strategy.partition(self.topology)[target_index].node_id
    next_shard = self.get_current_shard(base_shard, target_index)
    if DEBUG >= 2: print(f"Computed target from: {base_shard} {target_index}, {self.topology}. target shard: {next_shard}")
    if target_id == self.id:
      await self.process_tensor(next_shard, tensor, request_id, inference_state)
    else:
      target_peer = next((p for p in self.peers if p.id() == target_id), None)
      if not target_peer:
        raise ValueError(f"Peer for {target_index} not found")
      if DEBUG >= 1: print(f"Sending tensor to {target_peer.id()}: {tensor}")
      await target_peer.send_tensor(next_shard, tensor, request_id=request_id, inference_state=inference_state)

  def get_partition_index(self, offset: int = 0):
    if not self.partitioning_strategy:
      if DEBUG >= 1: print("No partitioning strategy found. Skipping forward.")
      return None
    partitions = self.partitioning_strategy.partition(self.topology)
    current_partition_index = next((i for i, p in enumerate(partitions) if p.node_id == self.id), None)
    if current_partition_index is None:
      raise ValueError(f"No current partition found for node: {self.id}")
    return (current_partition_index+offset) % len(partitions)

  def get_current_shard(self, base_shard: Shard, index: Optional[int] = None) -> Shard:
    if index is None:
      index = self.get_partition_index()
    partitions = self.partitioning_strategy.partition(self.topology)
    shards = map_partitions_to_shards(partitions, base_shard.n_layers, base_shard.model_id)
    return shards[index]

  async def update_peers(self, wait_for_peers: int = 0) -> bool:
    next_peers = await self.discovery.discover_peers(wait_for_peers)
    current_peer_ids = {peer.id() for peer in self.peers}
    next_peer_ids = {peer.id() for peer in next_peers}
    peers_added = [peer for peer in next_peers if peer.id() not in current_peer_ids]
    peers_removed = [peer for peer in self.peers if peer.id() not in next_peer_ids]
    peers_updated = [peer for peer in next_peers if peer.id() in current_peer_ids and any(p.addr() != peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_unchanged = [peer for peer in next_peers if peer.id() in current_peer_ids and all(p.addr() == peer.addr() for p in self.peers if p.id() == peer.id())]
    peers_to_disconnect = [peer for peer in peers_removed if await peer.is_connected()]
    peers_to_connect = [peer for peer in peers_added + peers_updated + peers_unchanged if not await peer.is_connected()]

    def _pretty(peers: List[PeerHandle]) -> List[str]:
      return [f"{peer.id()}@{peer.addr()}" for peer in peers]

    if DEBUG >= 2:
      print(f"update_peers: added={peers_added} removed={peers_removed} updated={peers_updated} unchanged={peers_unchanged} to_disconnect={peers_to_disconnect} to_connect={peers_to_connect}")

    async def disconnect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.disconnect(), timeout)
        return True
      except Exception as e:
        print(f"Error disconnecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    async def connect_with_timeout(peer, timeout=5):
      try:
        await asyncio.wait_for(peer.connect(), timeout)
        return True
      except Exception as e:
        print(f"Error connecting peer {peer.id()}@{peer.addr()}: {e}")
        traceback.print_exc()
        return False

    disconnect_results = await asyncio.gather(*(disconnect_with_timeout(peer) for peer in peers_to_disconnect), return_exceptions=True)
    connect_results = await asyncio.gather(*(connect_with_timeout(peer) for peer in peers_to_connect), return_exceptions=True)

    successful_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is True]
    failed_disconnects = [peer for peer, result in zip(peers_to_disconnect, disconnect_results) if result is False]
    successful_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is True]
    failed_connects = [peer for peer, result in zip(peers_to_connect, connect_results) if result is False]
    if DEBUG >= 1:
      if successful_disconnects: print(f"Successfully disconnected peers: {_pretty(successful_disconnects)}")
      if failed_disconnects: print(f"Failed to disconnect peers: {_pretty(failed_disconnects)}")
      if successful_connects: print(f"Successfully connected peers: {_pretty(successful_connects)}")
      if failed_connects: print(f"Failed to connect peers: {_pretty(failed_connects)}")

    self.peers = next_peers
    return len(peers_added) > 0 or len(peers_removed) > 0 or len(peers_updated) > 0

  async def select_best_inference_engine(self):
    if self.inference_engine.__class__.__name__ == 'DummyInferenceEngine': return
    supported_engines = self.get_supported_inference_engines()
    await self.broadcast_supported_engines(supported_engines)
    if len(self.get_topology_inference_engines()):
      self.inference_engine = get_inference_engine(supported_engines[0], self.shard_downloader)

  async def periodic_topology_collection(self, interval: int):
    while True:
      await asyncio.sleep(interval)
      try:
        did_peers_change = await self.update_peers()
        if DEBUG >= 2: print(f"{did_peers_change=}")
        await self.collect_topology(set())
        if did_peers_change:
          await self.select_best_inference_engine()
      except Exception as e:
        print(f"Error collecting topology: {e}")
        traceback.print_exc()

  async def collect_topology(self, visited: set[str], max_depth: int = 4) -> Topology:
    next_topology = Topology()
    next_topology.update_node(self.id, self.device_capabilities)

    if DEBUG >= 2: print(f"Collecting topology {max_depth=} {visited=}")

    prev_visited = visited.copy()
    visited.add(self.id)
    visited.update(p.id() for p in self.peers)

    for peer in self.peers:
      next_topology.update_node(peer.id(), peer.device_capabilities())
      next_topology.add_edge(self.id, peer.id(), peer.description())

      if peer.id() in prev_visited:
        continue

      if max_depth <= 0:
        if DEBUG >= 2: print("Max depth reached. Skipping...")
        continue

      try:
        other_topology = await asyncio.wait_for(peer.collect_topology(visited, max_depth=max_depth - 1), timeout=5.0)
        if DEBUG >= 2: print(f"Collected topology from: {peer.id()}: {other_topology}")
        next_topology.merge(peer.id(), other_topology)
      except Exception as e:
        print(f"Error collecting topology from {peer.id()}: {e}")
        traceback.print_exc()

    next_topology.active_node_id = self.topology.active_node_id
    self.topology = next_topology
    if self.topology_viz:
      self.topology_viz.update_visualization(self.topology, self.partitioning_strategy.partition(self.topology), self.id)
    return self.topology

  @property
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    return self._on_token

  @property
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    return self._on_opaque_status

  def trigger_on_token_callbacks(self, request_id: str, tokens: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Triggering all on_token callbacks with {request_id=} {tokens=} {is_finished=}")
    self.on_token.trigger_all(request_id, tokens, is_finished)

  async def broadcast_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    if DEBUG >= 2: print(f"Broadcasting result: {request_id=} {result=} {is_finished=}")
    async def send_result_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_result(request_id, result, is_finished), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout broadcasting result to {peer.id()}")
      except Exception as e:
        print(f"Error broadcasting result to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_result_to_peer(peer) for peer in self.peers], return_exceptions=True)

  async def broadcast_opaque_status(self, request_id: str, status: str) -> None:
    if DEBUG >= 8: print(f"Broadcasting opaque status: {request_id=} {status=}")

    async def send_status_to_peer(peer):
      try:
        await asyncio.wait_for(peer.send_opaque_status(request_id, status), timeout=15.0)
      except asyncio.TimeoutError:
        print(f"Timeout sending opaque status to {peer.id()}")
      except Exception as e:
        print(f"Error sending opaque status to {peer.id()}: {e}")
        traceback.print_exc()

    await asyncio.gather(*[send_status_to_peer(peer) for peer in self.peers], return_exceptions=True)
    # in the case of opaque status, we also want to receive our own opaque statuses
    self.on_opaque_status.trigger_all(request_id, status)

  @property
  def current_topology(self) -> Topology:
    return self.topology

  def handle_stable_diffusion(self, inference_state, result):
    if inference_state['is_step_finished']:
      inference_state['step'] += 1
    progress = [inference_state['step'], inference_state['total_steps']]
    intermediate_result = result
    if progress[0] == progress[1]:
      intermediate_result = result
    return intermediate_result, inference_state
