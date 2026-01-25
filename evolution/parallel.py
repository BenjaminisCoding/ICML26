
import multiprocessing
import time
import queue
import traceback
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

# Constants
STOP_SIGNAL = "STOP"

@dataclass
class LLMRequest:
    method: str  # 'generate_initial_model', 'evolve_model', 'crossover_model'
    args: tuple
    kwargs: dict
    correlation_id: str

@dataclass
class LLMResponse:
    correlation_id: str
    result: Any
    error: Optional[str] = None

class ProxyLLMClient:
    """
    Acts as a drop-in replacement for LLMClient in worker processes.
    Instead of calling API, it puts requests in a queue and waits for response.
    """
    def __init__(self, request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue, worker_id: str):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id

    def _call_proxy(self, method_name, *args, **kwargs):
        import uuid
        corr_id = str(uuid.uuid4())
        req = LLMRequest(method=method_name, args=args, kwargs=kwargs, correlation_id=corr_id)
        
        # Send Request as (worker_id, request)
        self.request_queue.put((self.worker_id, req))
        
        # Block until OUR response arrives
        # Note: In a simple implementation, we might consume other's responses if sharing one queue?
        # NO: multiprocessing Queue is shared. We need a way to filter.
        # Ideally, each worker has its own response queue or we just loop until we find ours (inefficient).
        # BETTER: Worker implementations usually have a dedicated pipe or we use a Manager.
        # BUT: For simplicity in this specific "Manager-Worker" setup, 
        # let's assume the Manager puts the response in a dedicated pipe for this worker? 
        # OR: We can just have one response queue and put back items that aren't ours? (Terrible race conditions).
        
        # REVISION: To avoid complex routing, let's use a SyncManager dict for responses? 
        # OR: Pass a specific response queue for THIS worker when creating the Proxy?
        # Yes, that's better. 
        # Wait, the `response_queue` passed to __init__ should be unique to this worker process if possible.
        # However, `multiprocessing.Queue` is expensive.
        
        # SIMPLIFICATION: 
        # Since we are inside a blocking call `evolve_island`, the worker does nothing else.
        # So we can just wait on `self.response_queue` assuming it is dedicated to THIS worker.
        # The Manager needs to know WHICH queue to send the response to.
        # So we include `response_queue` in the request? No, can't pickle Queue easily in some contexts.
        # Standard Pattern: Manager -> Worker via Pipe.
        
        # Let's use the provided queues. If `response_queue` is shared, we have a problem.
        # We will assume `response_queue` is DEDICATED to this worker instance.
        
        response = self.response_queue.get()
        if response.correlation_id != corr_id:
            # This shouldn't happen if queue is dedicated
            print(f"CRITICAL WARNING: Received mismatched correlation ID in worker. Got {response.correlation_id}, expected {corr_id}")
            # If shared queue, we'd need to put it back, but that's messy.
            # We will ensure unique queues in implementation.
        
        if response.error:
            print(f"ProxyLLMClient Error from Manager: {response.error}")
            return None
        return response.result

    def generate_initial_model(self, num_vars):
        return self._call_proxy('generate_initial_model', num_vars)

    def evolve_model(self, parent_code, instruction=None):
        return self._call_proxy('evolve_model', parent_code, instruction=instruction)

    def crossover_model(self, parent_codes_list, instruction=None):
        return self._call_proxy('crossover_model', parent_codes_list, instruction=instruction)
    
    # We also need these to satisfy interface, though mostly unused in workers
    def set_problem_description(self, problem_description):
        # This acts globally on the real client? 
        # The real client is in the Manager. We might want to send a signal?
        # For now, we assume Manager matches Worker state.
        pass

class RequestManager:
    """
    Runs in the main process.
    Consumes requests from all workers.
    Executes them using the REAL LLMClient.
    Sends responses back to the specific worker's queue.
    """
    def __init__(self, client_config, max_concurrent=5):
        self.client_config = client_config
        self.client = None # Will be initialized in process_requests
        self.max_concurrent = max_concurrent
        self.active_futures = []
        # We need to know which worker sent the request to send it back.
        # We can implement a Registry: worker_id -> response_queue.
        # Request needs to include worker_id.
    
    def process_requests(self, request_queue: multiprocessing.Queue, worker_queues: Dict[str, multiprocessing.Queue], stop_event):
        """
        Main loop for the Manager Thread/Process.
        """
        from concurrent.futures import ThreadPoolExecutor
        from evolution.llm_interface import LLMClient
        
        self.client = LLMClient(self.client_config)
        
        executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        
        while not stop_event.is_set():
            try:
                # Non-blocking get to check stop_event frequently
                try:
                    # (worker_id, request)
                    item = request_queue.get(timeout=0.1) 
                except queue.Empty:
                    continue
                
                worker_id, req = item
                
                if req == STOP_SIGNAL:
                    continue
                
                # Submit to executor
                future = executor.submit(self._execute_llm, req)
                
                # Callback to send response
                # We need to capture worker_id and corr_id
                def done_callback(fut, w_id=worker_id, c_id=req.correlation_id):
                    try:
                        result, error = fut.result()
                        resp = LLMResponse(correlation_id=c_id, result=result, error=error)
                        worker_queues[w_id].put(resp)
                    except Exception as e:
                        print(f"Manager Callback Error: {e}")
                
                future.add_done_callback(done_callback)
                
            except Exception as e:
                print(f"RequestManager Error: {e}")
                traceback.print_exc()

    def _execute_llm(self, req: LLMRequest):
        try:
            method = getattr(self.client, req.method)
            result = method(*req.args, **req.kwargs)
            return result, None
        except Exception as e:
            return None, str(e)


def worker_evolution(worker_id, islands, request_queue, response_queue, result_queue, data_pkg, obs_indices, time_points, config, stop_event, commands_queue):
    """
    The main loop for a worker process.
    It holds a subset of islands.
    It waits for commands (e.g., "Step") from the main process.
    """
    import os
    # Re-seed random to avoid all workers doing same thing?
    # Numpy random seed is usually process-safe if fork, but good practice.
    import numpy as np
    import random
    np.random.seed(os.getpid() + int(time.time()))
    random.seed(os.getpid() + int(time.time()))
    
    # Create Proxy Client
    proxy_client = ProxyLLMClient(request_queue, response_queue, worker_id)
    
    # We need a local EvolutionEngine-like structure or just use the logic?
    # To reuse code, we can instantiate a mini-EvolutionEngine with the proxy.
    # But EvolutionEngine expects the full list of islands? 
    # We can give it just OUR islands.
    
    from evolution.algorithm import EvolutionEngine
    # Engine requires llm_client
    
    # We need to reconstruct the engine params.
    # Ideally, we refactor algorithm.py to separate 'IslandEvolver' from 'EvolutionCoordinator'.
    # But to stick to plan "Update EvolutionEngine":
    
    engine = EvolutionEngine(
        islands=islands,
        llm_client=proxy_client,
        data=data_pkg,
        obs_indices=obs_indices,
        time_points=time_points,
        n_iterations=1000 # controlled by commands
    )
    # Manually inject config if needed (Engine loads it, but maybe we passed dynamic config?)
    # Config is passed to __init__ if needed, but we already set it.
    # Manually inject config if needed (Engine loads it, but maybe we passed dynamic config?)
    # Config is passed to __init__ if needed, but we already set it.
    engine.config = config 
    
    # Re-initialize encoder with the injected config
    from evolution.encoding import ExpressionEncoder
    features = engine.config.get('analysis', {}).get('features', [])
    engine.encoder = ExpressionEncoder(features)
    
    # Recompile any individuals that were stripped during pickling
    print(f"Worker {worker_id}: Recompiling models...")
    for island in engine.islands:
        count = 0
        for ind in island.population:
            if not ind.is_compiled:
                ind.compile()
                count += 1
        # Also recompile best candidate cache if Island keeps it? No it computes on fly.
    print(f"Worker {worker_id} ready with {len(islands)} islands.")
    
    while not stop_event.is_set():
        try:
            cmd = commands_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        if cmd == "STOP":
            break
        
        if cmd == "STEP":
            engine.log_data = [] # Clear previous logs
            
            stats = []
            for island in engine.islands:
                engine.evolve_island(island)
            
            engine.generation += 1
            
            # Collect Stats
            for i, island in enumerate(engine.islands):
                best = island.get_best()
                best_fit = best.fitness if best else float('inf')
                stats.append({
                    'island_idx': i, # Note: this idx is relative to chunk, probably need global ID? 
                    # But Main process knows assignment. 
                    # Actually, we can just send back island.id if needed, but current analysis uses idx.
                    # Let's send back list of stats.
                    'num_vars': island.num_vars,
                    'best_fitness': best_fit,
                    'generation': engine.generation - 1 # Refers to the one just done
                })
            
            # Send back results
            # We send deeply copied data to avoid pickling issues with complex objects if any?
            # Basic dicts are fine.
            result_queue.put(("STEP_DONE", engine.log_data, stats))
    
    print(f"Worker {worker_id} stopping.")
