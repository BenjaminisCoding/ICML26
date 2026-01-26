
import multiprocessing
import time
import copy
import queue
import traceback
import random
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
        
        start_wait = time.time()
        while True:
            try:
                 # Check if we exceeded total timeout
                 if time.time() - start_wait > 120:
                     print(f"CRITICAL: Worker {self.worker_id} timed out waiting for {corr_id}")
                     return None
                     
                 response = self.response_queue.get(timeout=5.0) 
                 
                 if response.correlation_id == corr_id:
                     # Match!
                     break
                 else:
                     # Mismatch - likely a late response from a previous timed-out request.
                     # Log debug but continue waiting for OUR response.
                     # print(f"DEBUG: Worker {self.worker_id} skipping late/mismatched ID {response.correlation_id} (wanted {corr_id})")
                     continue
            
            except queue.Empty:
                 continue
                 
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
        
    def batch_evolve_model(self, parent_code_list: List[tuple], instruction=None):
        """"
        Batch version of evolve_model.
        Args:
            parent_code_list: List of (code, parent_id) tuples or just codes.
            instruction: Common instruction.
            
        Returns:
            List of result codes (or None if failed).
            Order might be scrambled? No, we need to map back.
            Actually, sending corr_ids allows mapping.
        """
        import uuid
        
        # 1. Send all requests
        pending_ids = []
        for code in parent_code_list:
            if isinstance(code, tuple): code = code[0] # Handle tuple args if passed
            
            corr_id = str(uuid.uuid4())
            req = LLMRequest(method='evolve_model', args=(code,), kwargs={'instruction': instruction}, correlation_id=corr_id)
            self.request_queue.put((self.worker_id, req))
            pending_ids.append(corr_id)
            
        # 2. Collections
        results = {}
        pending_set = set(pending_ids)
        
        start_wait = time.time()
        while pending_set:
            try:
                # Timed wait to detect stalls
                response = self.response_queue.get(timeout=5.0) 
            except queue.Empty:
                elapsed = time.time() - start_wait
                if elapsed > 120: # 2 minutes
                     print(f"CRITICAL: Worker {self.worker_id} waiting > 120s for Batch LLM. Pending: {len(pending_set)}. Aborting batch.")
                     break # Exit the wait loop
                continue
                
            if response.correlation_id in pending_set:
                pending_set.remove(response.correlation_id)
                results[response.correlation_id] = response.result if not response.error else None
            else:
                # Received response for something else?
                pass
                
        # 3. Return in order (Fill missing with None)
        ordered_res = []
        for cid in pending_ids:
            if cid in results:
                ordered_res.append(results[cid])
            else:
                ordered_res.append(None) # Missing/Timed out
        return ordered_res
                
        return ordered_res
    
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
            # engine.log_data is NOT cleared here. It accumulates from Extinction/Injection events.
            
            stats = []
            for island in engine.islands:
                engine.evolve_island(island)
            
            engine.generation += 1
            
            # Collect Stats & State for Sync
            island_states = []
            
            for i, island in enumerate(engine.islands):
                best = island.get_best()
                best_fit = best.fitness if best else float('inf')
                
                # Stats for analysis
                stats.append({
                    'island_idx': i, 
                    'num_vars': island.num_vars,
                    'best_fitness': best_fit,
                    'generation': engine.generation - 1
                })
                
                # State for Manager (Sync)
                state = {
                    'id': island.id,
                    'stagnation_counter': island.stagnation_counter,
                    'best_individual': best # will be pickled (uncompiled)
                }
                island_states.append(state)
            
            # Send back results
            # Result: ("STEP_DONE", logs, stats, island_states)
            # Use a copy or slice, then clear
            current_logs = list(engine.log_data)
            engine.log_data = [] # Clear AFTER capturing
            result_queue.put(("STEP_DONE", current_logs, stats, island_states))
            
        elif isinstance(cmd, tuple):
             # Complex Commands
             op = cmd[0]
             
             if op == "INJECT_INDIVIDUAL":
                 # ("INJECT_INDIVIDUAL", island_id, individual)
                 target_id = cmd[1]
                 ind = cmd[2]
                 
                 # Find island
                 for island in engine.islands:
                     if island.id == target_id:
                         # Ensure ind is compiled if needed? 
                         # Usually it comes from another worker where it was compiled.
                         # But unpickling strips model.
                         ind.compile() 
                         island.add_individual(ind)
                         print(f"Worker {worker_id}: Injected individual into {target_id}")
                         break
                         
             elif op == "TRIGGER_EXTINCTION":
                 # ("TRIGGER_EXTINCTION", island_id, elites_list)
                 target_id = cmd[1]
                 elites = cmd[2]
                 
                 for island in engine.islands:
                     if island.id == target_id:
                         print(f"Worker {worker_id}: Executing Extinction on {target_id}")
                         
                         # 1. Clear population
                         island.clear_population()
                         
                         # 2. Repopulate using provided Elites
                         n_repop = engine.config.get('evolution_strategy', {}).get('max_capacity_island', {}).get('value', 10)
                         instruction = engine.config.get('llm_instructions', {}).get('extinction_mutation', {}).get('value', "Mutate creatively.")
                         
                         # Ensure elites are compiled
                         clean_elites = []
                         for e in elites:
                             if not e.is_compiled: e.compile() # Just in case
                             if e.is_compiled: clean_elites.append(e)
                             
                         if not clean_elites:
                             print(f"Worker {worker_id}: Warning - No valid elites for extinction. Skipping repopulation.")
                             continue
                             
                         prevent_infinite_loop = 0
                         needed = n_repop - len(island.population)
                        
                         if needed > 0:
                             print(f"Worker {worker_id}: Batch requesting {needed} individuals...")
                             # Prepare batch args
                             parents_selected = [random.choice(clean_elites) for _ in range(needed)]
                             parent_codes = [p.code for p in parents_selected]
                             
                             # Batch Call
                             child_codes = engine.llm_client.batch_evolve_model(parent_codes, instruction)
                             
                             # Process results
                             for i, c_code in enumerate(child_codes):
                                 if c_code:
                                     pid = parents_selected[i].id
                                     from evolution.models import Individual
                                     child = Individual(c_code, parents=[pid], creation_op="extinction", generation=engine.generation)
                                     child.metadata['instruction'] = instruction
                                     child.metadata['extinction_event'] = True
                                     
                                     if child.compile():
                                         f = child.evaluate(engine.data['observed_data'], engine.obs_indices, engine.time_points)
                                         if f < float('inf'):
                                             child.metadata['fingerprint'] = engine.encoder.encode(child)
                                             island.add_individual(child)
                                             
                                             # LOGGING
                                             log_entry = {
                                                 "event": "reproduction",
                                                 "generation": engine.generation,
                                                 "op": "extinction",
                                                 "island": island.id,
                                                 "child_id": child.id,
                                                 "parent_ids": [pid],
                                                 "fitness": f,
                                                 "code": c_code,
                                                 "metadata": child.metadata
                                             }
                                             engine.log_data.append(log_entry)
                        
                         # Cleanup (Serial fill if needed)
                         while len(island.population) < n_repop and prevent_infinite_loop < 20:
                             parent = random.choice(clean_elites)
                             child_code = engine.llm_client.evolve_model(parent.code, instruction)
                             # ... serial logic ...
                             if child_code:
                                  from evolution.models import Individual
                                  child = Individual(child_code, parents=[parent.id], creation_op="extinction", generation=engine.generation)
                                  child.metadata['instruction'] = instruction
                                  child.metadata['extinction_event'] = True
                                  if child.compile():
                                      f = child.evaluate(engine.data['observed_data'], engine.obs_indices, engine.time_points)
                                      if f < float('inf'):
                                          child.metadata['fingerprint'] = engine.encoder.encode(child)
                                          island.add_individual(child)

                                          # LOGGING
                                          log_entry = {
                                              "event": "reproduction",
                                              "generation": engine.generation,
                                              "op": "extinction",
                                              "island": island.id,
                                              "child_id": child.id,
                                              "parent_ids": [parent.id],
                                              "fitness": f,
                                              "code": child_code,
                                              "metadata": child.metadata
                                          }
                                          engine.log_data.append(log_entry)
                             prevent_infinite_loop += 1
                             
                         island.stagnation_counter = 0 # Verify reset
                         print(f"Worker {worker_id}: Extinction complete. Pop size: {len(island.population)}")
                         break
    
    print(f"Worker {worker_id} stopping.")
