
import random
import sys
import uuid
import copy
import numpy as np
import multiprocessing
import traceback

import json
import os

class EvolutionEngine:
    def __init__(self, islands, llm_client, data, obs_indices, time_points, n_iterations=None, n_cores=1):
        """
        islands: List of Island objects
        llm_client: LLMClient instance
        data: Dict containing problem description etc.
        n_cores: Number of cores to use (1 = Serial, >1 = Parallel)
        """
        self.islands = islands
        self.llm_client = llm_client
        self.data = data
        self.obs_indices = obs_indices
        self.time_points = time_points
        self.n_cores = n_cores
        
        # Load Config
        try:
            with open("evolution/config.json", "r") as f:
                self.config = json.load(f)
            print("EvolutionEngine loaded evolution/config.json")
        except FileNotFoundError:
            print("EvolutionEngine could not find evolution/config.json. Using defaults.")
            self.config = {}

        evo_strat = self.config.get('evolution_strategy', {})
        
        # Hyperparameters from config
        self.mutation_rate = evo_strat.get('mutation_rate', {}).get('value', 0.7)
        self.crossover_rate = evo_strat.get('crossover_rate', {}).get('value', 0.3)
        self.temperature_selec_ind = evo_strat.get('selection_temperature', {}).get('value', 10.0)
        self.N_parent_crossover = evo_strat.get('N_parent_crossover', {}).get('value', 2)
        self.max_capacity_island = evo_strat.get('max_capacity_island', {}).get('value', 50)
        
        #set max capacity for islands
        self._set_max_capacity_island()
        
        self.iter_migration = int(evo_strat.get('iter_migration', {}).get('value', 5))
        self.n_local_stagnation = int(evo_strat.get('n_local_stagnation', {}).get('value', 20))

        # Stopping criterion
        if n_iterations is None:
             self.n_iterations = int(evo_strat.get('n_iterations_stagnation', {}).get('value', 20))
        else:
             self.n_iterations = n_iterations

        self.current_iter = 0 # updated if no improvement observed, set to 0 else
        self.generation = 0 # first generation are the islands argument
        self.stop_flag = False
        self.current_best_fitness = self.best_fitness() #useful for stopping criterion if no improvement observed
        
        # Logging
        self.log_data = []

        # Parallel State
        self.workers = []
        self.manager_process = None
        self.request_queue = None
        self.response_queues = {}
        self.result_queues = {} # [NEW] For worker -> main communication
        self.command_queues = []
        self.island_to_worker = {} # [NEW] Map island_id -> worker_index (for command routing)
        self.stop_event = None
        self.parallel_started = False

        # Encoding
        from evolution.encoding import ExpressionEncoder
        features = self.config.get('analysis', {}).get('features', [])
        self.encoder = ExpressionEncoder(features)

        # Auto-save configuration
        # Only the MainProcess should manage log files and Run IDs
        import time
        import multiprocessing
        
        self.run_id = None
        self.history_dir = "history"
        self.autosave_path = None
        self.unique_log_path = None
        
        if multiprocessing.current_process().name == "MainProcess":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp}_{uuid.uuid4().hex[:4]}"
            
            os.makedirs(self.history_dir, exist_ok=True)
            
            # Paths
            self.autosave_path = os.path.join(self.history_dir, "autosave_latest.json")
            self.autosave_config_path = os.path.join(self.history_dir, "autosave_config.json")
            
            self.unique_log_path = os.path.join(self.history_dir, f"{self.run_id}_logs.json")
            self.unique_config_path = os.path.join(self.history_dir, f"{self.run_id}_config.json")
            
            # Save initial config to both locations
            self.save_config(self.autosave_config_path)
            self.save_config(self.unique_config_path)
            
            print(f"Evolution Run ID: {self.run_id}")
            print(f"Logs will be saved to: {self.unique_log_path}")


        self._log_initial_islands() # Log initial population
        self._setup_neighbors()

    def _setup_neighbors(self):
        """
        Sets up cyclic neighbors for islands of the same dimension.
        """
        # Group by dimension
        islands_by_dim = {}
        for island in self.islands:
            d = island.num_vars
            if d not in islands_by_dim:
                islands_by_dim[d] = []
            islands_by_dim[d].append(island)
            
        # Link neighbors
        for d, group in islands_by_dim.items():
            n = len(group)
            if n < 2: continue
            # Sort by ID to ensure deterministic ring
            group.sort(key=lambda x: x.id)
            for i in range(n):
                # i -> (i+1)%n
                current = group[i]
                nxt = group[(i+1)%n]
                current.register_neighbor(nxt.id)
                print(f"Island {current.id} ({d} vars) -> Neighbor {nxt.id}")

    def perform_migration(self):
        """
        Migrate best individual to neighbor.
        Clone best -> Neighbor.add_individual.
        """
        print("PERFORMING MIGRATION")
        # Gather bests first to avoid chain reaction in same step (synchronous migration)
        # Or sequential? Synchronous is fairer.
        migrations = [] # (target_island_id, individual_clone)
        
        # In Serial mode, self.islands is up to date.
        # In Parallel, we must have synced before calling this.
        
        island_map = {isl.id: isl for isl in self.islands}
        
        for island in self.islands:
            best = island.get_best()
            if not best: continue
            
            for neighbor_id in island.neighbors:
                if neighbor_id in island_map and neighbor_id != island.id:
                    # Clone best
                    clone = copy.deepcopy(best)
                    clone.id = str(uuid.uuid4()) # Unique ID
                    clone.parents = [best.id] # Lineage
                    clone.origin = "migration"
                    clone.metadata['migration_source'] = island.id
                    migrations.append((neighbor_id, clone))
                    
        # Apply Migrations
        if self.n_cores > 1:
            # Parallel: Send INJECT commands
            for tid, ind in migrations:
                # Find worker for target island
                wid_idx = self.island_to_worker.get(tid)
                if wid_idx is not None:
                    # Send command ("INJECT_INDIVIDUAL", island_id, individual)
                    self.command_queues[wid_idx].put(("INJECT_INDIVIDUAL", tid, ind))
                    
                    # Log event locally since Main tracks logs
                    log_entry = {
                        "event": "migration",
                        "generation": self.generation,
                        "op": "migration",
                        "island": tid,
                        "child_id": ind.id,
                        "parent_ids": ind.parents,
                        "fitness": ind.fitness,
                        "code": ind.code,
                        "metadata": ind.metadata
                    }
                    self.log_data.append(log_entry)
        else:
            # Serial: Apply directly
            for tid, ind in migrations:
                target = island_map[tid]
                target.add_individual(ind)
                # Log
                log_entry = {
                        "event": "migration",
                        "generation": self.generation,
                        "op": "migration",
                        "island": tid,
                        "child_id": ind.id,
                        "parent_ids": ind.parents,
                        "fitness": ind.fitness,
                        "code": ind.code,
                        "metadata": ind.metadata
                }
                self.log_data.append(log_entry)

    def perform_extinction(self):
        """
        Check stagnation and wipe islands if needed.
        Repopulate with mutants of elites from other islands.
        """
        # print("CHECKING EXTINCTION") 
        # Don't spam print every gen if we call it often.
        
        island_map = {isl.id: isl for isl in self.islands}
        import uuid # Ensure uuid is imported if used here (though simpler to use top level)
        
        # Group islands for repopulation pool
        islands_by_dim = {}
        for island in self.islands:
            d = island.num_vars
            if d not in islands_by_dim: islands_by_dim[d] = []
            islands_by_dim[d].append(island)
            
        for island in self.islands:
            # Check stagnation trigger
            # User defined threshold: n_local_stagnation
            if island.stagnation_counter > self.n_local_stagnation:
                # 1. Check if we have help (Elites from others)
                group = islands_by_dim.get(island.num_vars, [])
                sources = [isl for isl in group if isl.id != island.id]
                
                elites = []
                for src in sources:
                    b = src.get_best()
                    if b: elites.append(b)
                
                if not elites:
                    print(f"Extinction check for Island {island.id}: Stagnated but no elites found to rescue. Aborting extinction.")
                    # Optional: Reset counter slightly to avoid checking every single step? 
                    # Or keep it high so it triggers as soon as an elite appears?
                    # Let's keep it.
                    continue

                # 2. SAFE TO KILL. We have a backup plan.
                print(f"Extinction Triggered for Island {island.id} (Stagnation: {island.stagnation_counter}). Rebooting population...")
                
                if self.n_cores > 1:
                    # Parallel: Send TRIGGER_EXTINCTION with Elites Payload
                    wid_idx = self.island_to_worker.get(island.id)
                    if wid_idx is not None:
                        # Payload: List of Elite Individuals
                        # We send copies to avoid modification issues
                        payload_elites = [copy.deepcopy(e) for e in elites]
                        self.command_queues[wid_idx].put(("TRIGGER_EXTINCTION", island.id, payload_elites))
                        
                        # We should verify reset locally? 
                        # Update local state immediately to avoid re-triggering next step before sync
                        island.stagnation_counter = 0 
                else:
                    # Serial: Execute Locally
                    island.clear_population()
                    
                    # 3. Repopulate
                    n_repop = self.config.get('evolution_strategy', {}).get('max_capacity_island', {}).get('value', 10)
                    instruction = self.config.get('llm_instructions', {}).get('extinction_mutation', {}).get('value', "Mutate creatively.")
                    
                    prevent_infinite_loop = 0
                    while len(island.population) < n_repop and prevent_infinite_loop < 100:
                        parent = random.choice(elites)
                        # Call LLM
                        child_code = self.llm_client.evolve_model(parent.code, instruction)
                        if child_code:
                             from evolution.models import Individual
                             child = Individual(child_code, parents=[parent.id], creation_op="extinction", generation=self.generation)
                             child.metadata['instruction'] = instruction
                             child.metadata['extinction_event'] = True
                             
                             if child.compile():
                                 f = child.evaluate(self.data['observed_data'], self.obs_indices, self.time_points)
                                 if f < float('inf'):
                                     child.metadata['fingerprint'] = self.encoder.encode(child)
                                     island.add_individual(child)
                        prevent_infinite_loop += 1
                    if prevent_infinite_loop >= 100:
                        print(f"Infinite loop detected for island {island.id} when performing extinction mutation.")
                    
                    # Reset counter is done in clear_population, already done.
                    island.stagnation_counter = 0

    def save_logs(self, filepath):
        """Saves the evolutionary log to a JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        print(f"Evolution logs saved to {filepath}")

    def save_config(self, filepath):
        """Saves the current configuration to a JSON file."""
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {filepath}")
        except Exception as e:
            print(f"Failed to save config to {filepath}: {e}")
            
    def best_fitness(self):
        """
        Returns the best fitness across all islands.
        """
        best_fitness = float('inf')
        for island in self.islands:
            candidate = island.get_best()
            if candidate:
                 best_fitness = min(best_fitness, candidate.fitness)
        return best_fitness

    def _set_max_capacity_island(self):
        for island in self.islands:
            island.capacity = self.max_capacity_island

    def _log_initial_islands(self):
        """Log initial population if starting"""
        if self.generation == 0 and not self.log_data:
            print("Logging initial population...")
            for i, island in enumerate(self.islands):
                for ind in island.population:
                    log_entry = {
                        "event": "reproduction", # Marked as reproduction to fit schema, but op is 'init'
                        "generation": 0,
                        "op": "init",
                        "island": island.id,
                        "child_id": ind.id,
                        "parent_ids": [],
                        "fitness": ind.fitness,
                        "code": ind.code,
                        "metadata": ind.metadata if hasattr(ind, 'metadata') else {}
                    }
                    self.log_data.append(log_entry)
        self.generation += 1 

    
    def select_parents(self, island, k=1):
        """
        Selects k parents from an island using Softmax (Boltzmann) selection.
        Implements two strategies:
        1. Rejection Sampling: For low k/N ratios. Sample, check duplicate, retry.
           Fallbacks to Renormalization if retries exceed limit.
        2. Renormalization Sampling: For high k/N ratios. Mask selected, re-normalize.
        """
        population = island.population
        N = len(population)
        if N == 0:
            return []
        
        # Calculate weights for all individuals (Assuming finite fitness)
        fitnesses = np.array([ind.fitness for ind in population])
        
        # Stability shift using LOG fitness (handles orders of magnitude differences)
        # Fitness is MSE, so we use log(fitness) to scale appropriately
        # If fitness is 0, clip to epsilon
        safe_fitness = np.maximum(fitnesses, 1e-12)
        log_fitness = np.log(safe_fitness)
        
        min_log_fit = np.min(log_fitness)
        std_log_fit = np.std(log_fitness)
        
        # Adaptive Temperature:
        # Scale T by the standard deviation of log-fitness.
        # This ensures that exponents are roughly normalized (z-scores), 
        # preventing argmax (if spread is huge) or uniform (if spread is tiny).
        # We assume self.temperature_selec_ind is a scaling factor (default around 1.0).
        effective_T = self.temperature_selec_ind * (std_log_fit if std_log_fit > 1e-6 else 1.0)
        
        shifted = log_fitness - min_log_fit
        
        # Use log-fitness for probability: P ~ exp(-log_fitness / T)
        weights = np.exp(-shifted / effective_T)
        
        if np.sum(weights) == 0:
             probs = np.ones(N) / N
        else:
             probs = weights / np.sum(weights)

        # Handle k > N case: Must sample with replacement
        if k > N:
            indices = np.random.choice(N, size=k, replace=True, p=probs)
            return [population[i] for i in indices]

        # --- k <= N: Distinct Selection Logic ---

        def get_probs(w):
            s = np.sum(w)
            if s == 0: return np.ones(len(w)) / len(w)
            return w / s

        selected_indices = []
        
        # Decide Strategy based on ratio
        # Threshold 0.5 as requested
        ratio = k / N
        use_renormalization = ratio > 0.5
        
        # Strategy 1: Rejection Sampling
        if not use_renormalization:
            # Reuse probs calculated above
            current_probs = probs
            consecutive_fails = 0
            max_fails = 10
            
            while len(selected_indices) < k:
                idx = np.random.choice(N, p=current_probs)
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
                
                # Switch to renormalization if failing too often
                if consecutive_fails >= max_fails:
                    use_renormalization = True
                    break
        
        # Strategy 2: Renormalization (Sequential Masking)
        if use_renormalization:
            # We need mutatable weights
            current_weights = weights.copy()
            # Mask already selected (if switched from strategy 1)
            for idx in selected_indices:
                current_weights[idx] = 0.0
            
            while len(selected_indices) < k:
                # Re-normalize on remaining
                # If all remaining are 0 weight (unlikely unless only bad ones left), Uniform on remaining
                if np.sum(current_weights) == 0:
                     # Identify remaining candidates uniformally
                     remaining = [i for i in range(N) if i not in selected_indices]
                     if not remaining: break
                     idx = np.random.choice(remaining)
                else:
                     current_probs = get_probs(current_weights)
                     idx = np.random.choice(N, p=current_probs)
                
                # Just in case logic
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    current_weights[idx] = 0.0
                
        return [population[i] for i in selected_indices]

    def evolve_island(self, island):
        """
        Performs one evolutionary step on a single island.
        """
        if len(island.population) == 0:
            # Force stagnation increment to trigger extinction (Reboot) on dead islands.
            # This allows recovery if repopulation failed previously due to API errors.
            island.stagnation_counter += 1
            return

        child = None
        child_code = None
        fitness = float('inf')
        
        op_type = "none"
        parent_ids = []
        metadata = {}

        # Check stopping criterion
        if self.current_iter < self.n_iterations:
            # Crossover or Mutation
            # If we don't have enough for crossover, forced mutation
            do_crossover = (random.random() < self.crossover_rate) and (len(island.population) >= self.N_parent_crossover)
            
            if do_crossover:
                parents = self.select_parents(island, k=self.N_parent_crossover)
                if len(parents) == self.N_parent_crossover:
                    parent_codes = [p.code for p in parents]
                    parent_ids = [p.id for p in parents]
                    op_type = "crossover"
                    
                    # Get instruction from config
                    instruction = self.config.get('llm_instructions', {}).get('crossover', {}).get('value')
                    child_code = self.llm_client.crossover_model(parent_codes, instruction)
                    
                    if child_code:
                        from evolution.models import Individual
                        child = Individual(child_code, parents=parent_ids, creation_op=op_type, generation=self.generation)
                        child.metadata['instruction'] = instruction
                        
                        if child.compile(): 
                            fitness = child.evaluate(self.data['observed_data'], self.obs_indices, self.time_points)
                            if fitness < float('inf'):
                                child.metadata['fingerprint'] = self.encoder.encode(child)
                                island.add_individual(child)
            else:
                # Mutation
                parents = self.select_parents(island, k=1)
                if parents:
                    parent = parents[0]
                    parent_ids = [parent.id]
                    op_type = "mutation"
                    
                    # Get instruction from config
                    instruction = self.config.get('llm_instructions', {}).get('mutation', {}).get('value')
                    child_code = self.llm_client.evolve_model(parent.code, instruction)
                    
                    if child_code:
                        from evolution.models import Individual
                        child = Individual(child_code, parents=parent_ids, creation_op=op_type, generation=self.generation)
                        child.metadata['instruction'] = instruction
                        
                        if child.compile():
                            fitness = child.evaluate(self.data['observed_data'], self.obs_indices, self.time_points)
                            if fitness < float('inf'):
                                child.metadata['fingerprint'] = self.encoder.encode(child)
                                island.add_individual(child)
        else:
             # Stagnation reached, raising flag
             self.stop_flag = True
             print("    Stagnation limit reached. Raising stop flag.")
             return

        # check improvement (Global)
        # User logic used self.current_best_fitness which is global.
        # If the new child is better than global best -> reset counter
        
        # We need to re-verify if child was valid
        success = (child is not None) and (child.is_compiled) and (fitness < float('inf'))
        
        if success:
            # LOGGING
            log_entry = {
                "event": "reproduction",
                "generation": self.generation,
                "op": op_type,
                "island": island.id,
                "child_id": child.id,
                "parent_ids": parent_ids,
                "fitness": fitness,
                "code": child_code,
                "metadata": child.metadata
            }
            self.log_data.append(log_entry)
        
        if success and fitness < self.current_best_fitness:
            self.current_iter = 0
            self.current_best_fitness = fitness
            print(f"    Improvement found! New Best: {fitness:.4f}")
        else:
            self.current_iter += 1

        # Check local island stagnation
        island.update_stagnation()

    def start_parallel_workers(self):
        if self.parallel_started:
            return

        import multiprocessing
        from evolution.parallel import RequestManager, worker_evolution

        print(f"Starting {self.n_cores} Parallel Workers...")
        self.request_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        
        # 1. Start Manager
        # We need a queue for each worker to receive responses
        self.response_queues = {}
        self.result_queues = {}
        worker_ids = [f"w_{i}" for i in range(self.n_cores)]
        
        for wid in worker_ids:
            self.response_queues[wid] = multiprocessing.Queue()
            self.result_queues[wid] = multiprocessing.Queue()

        max_req = self.config.get('azure_api', {}).get('max_concurrent_requests', {}).get('value', 5)
        manager = RequestManager(self.llm_client.config, max_concurrent=max_req)
        
        self.manager_process = multiprocessing.Process(
            target=manager.process_requests,
            args=(self.request_queue, self.response_queues, self.stop_event)
        )
        self.manager_process.start()

        # 2. Split Islands, assign to workers
        # Simple Round Robin distribution
        island_chunks = [[] for _ in range(self.n_cores)]
        self.island_to_worker = {} # Reset map
        
        for i, island in enumerate(self.islands):
            worker_idx = i % self.n_cores
            island_chunks[worker_idx].append(island)
            self.island_to_worker[island.id] = worker_idx

        # 3. Start Workers
        self.command_queues = []
        for i, wid in enumerate(worker_ids):
            cmd_q = multiprocessing.Queue()
            self.command_queues.append(cmd_q)
            
            p = multiprocessing.Process(
                target=worker_evolution,
                args=(
                    wid,
                    island_chunks[i],
                    self.request_queue,
                    self.response_queues[wid],
                    self.result_queues[wid],
                    self.data, 
                    self.obs_indices,
                    self.time_points,
                    self.config,
                    self.stop_event,
                    cmd_q
                )
            )
            p.start()
            self.workers.append(p)
        
        self.parallel_started = True
        print("Parallel system ready.")

    def stop_parallel_workers(self):
        if not self.parallel_started:
            return
            
        print("Stopping Parallel Workers...")
        self.stop_event.set()
        
        # Send STOP to command queues just in case
        for q in self.command_queues:
            q.put("STOP")
        
        for p in self.workers:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
        
        if self.manager_process:
            self.manager_process.join(timeout=1)
            if self.manager_process.is_alive():
                self.manager_process.terminate()
                
        self.parallel_started = False
        self.workers = []

    def run_generation(self):
        # Log initial population if starting
        # if self.generation == 0 and not self.log_data:
        #     print("Logging initial population...")
        #     for i, island in enumerate(self.islands):
        #         for ind in island.population:
        #             log_entry = {
        #                 "event": "reproduction", # Marked as reproduction to fit schema, but op is 'init'
        #                 "generation": 0,
        #                 "op": "init",
        #                 "island": island.id,
        #                 "child_id": ind.id,
        #                 "parent_ids": [],
        #                 "fitness": ind.fitness,
        #                 "code": ind.code,
        #                 "metadata": ind.metadata if hasattr(ind, 'metadata') else {}
        #             }
        #             self.log_data.append(log_entry)

        print(f"--- Generation {self.generation} (Stagnation: {self.current_iter}/{self.n_iterations}) ---")
        
        if not self.islands:
            return []

        # --- Advanced Evolution Triggers ---
        # Scaling: In serial, 1 gen = 1 island step. In parallel, 1 gen = all islands step.
        scale_factor = len(self.islands) if self.n_cores == 1 else 1
        
        mig_interval = self.iter_migration * scale_factor
        
        if self.generation > 0:
            # Extinction Check: Constant monitoring
            self.perform_extinction()
            
            # Migration Check: Periodic
            if self.generation % mig_interval == 0:
                self.perform_migration()

        if self.n_cores > 1:
            # --- PARALLEL EXECUTION ---
            if not self.parallel_started:
                self.start_parallel_workers()
            
            # Send STEP command
            print("Broadcasting STEP to workers...")
            for q in self.command_queues:
                q.put("STEP")
            
            # Wait for completion
            # Each worker sends a "STEP_DONE" 
            # But wait, where do they send it? response_queue is for LLM responses.
            # We can reuse response_queue if we are careful, or use a separate synchronization method.
            # In `worker_evolution` I wrote: `response_queue.put("STEP_DONE")`
            # But `RequestManager` also puts things there.
            # This is a RACE CONDITION if manager is active.
            # However, during STEP, workers are blocked waiting for LLM?
            # No. Worker puts request, waits for response.
            # Only when `evolve_island` returns does it send `STEP_DONE`.
            # So `STEP_DONE` arrives AFTER LLM calls are finished for that step.
            # BUT: Are we sure? `evolve_island` is synchronous in worker.
            # So yes, it's safeish.
            # We just need to distinguish `LLMResponse` (dataclass) from "STEP_DONE" (str).
            
            # Wait for completion and collect results
            stats = []
            
            # Wait for completion and collect results
            stats = []
            
            for wid, q in self.result_queues.items():
                while True:
                    msg = q.get()
                    # Check for Tuple ("STEP_DONE", logs, worker_stats, island_states)
                    if isinstance(msg, tuple) and msg[0] == "STEP_DONE":
                        # Updated unpack
                        if len(msg) == 4:
                            _, w_logs, w_stats, w_states = msg
                            
                            # Sync State
                            # Update local proxies
                            # w_states is list of {id, stagnation, best_ind}
                            for state in w_states:
                                isl_id = state['id']
                                # Find local island
                                # Use dict for speed if needed, but list is short
                                for loc_isl in self.islands:
                                    if loc_isl.id == isl_id:
                                        loc_isl.stagnation_counter = state['stagnation_counter']
                                        # Update best? 
                                        # Since we don't have full population, we can't fully sync.
                                        # But we can inject the best individual into our local proxy 
                                        # so that 'get_best()' returns it?
                                        # Or just rely on w_stats for best fitness?
                                        # Migration logic calls 'island.get_best()'.
                                        # So YES, we must insert the best individual.
                                        best_ind = state['best_individual']
                                        if best_ind:
                                            # Add to population? Or just set as best?
                                            # If we add, population grows indefinitely in Main.
                                            # We should probably clear Main population and set only Best?
                                            # For Proxy purposes, keeping 1 (Best) is enough.
                                            loc_isl.population = [best_ind] 
                                        break
                                        
                        else:
                            # Backward compat / Fallback
                            _, w_logs, w_stats = msg
                        
                        self.log_data.extend(w_logs)
                        stats.extend(w_stats)
                        break
                    # Should not receive anything else on result_queue
                    print(f"WARNING: Unexpected message in result queue: {msg}")
            
            # Update global best fitness for tracking
            current_best = float('inf')
            for s in stats:
                if s['best_fitness'] < current_best:
                    current_best = s['best_fitness']
            
            if current_best < self.current_best_fitness:
                self.current_iter = 0
                self.current_best_fitness = current_best
                print(f"    Global Improvement! New Best: {current_best:.4f}")
            else:
                self.current_iter += 1
            self.generation += 1 
            return stats
        
        else:
            # --- SERIAL EXECUTION ---
            island_idx = random.randint(0, len(self.islands) - 1) #uniform random selection of an island
            target_island = self.islands[island_idx]
            
            print(f"Evolving Island {island_idx} (Vars: {target_island.num_vars})...")
            self.evolve_island(target_island)
            target_island.update_stagnation()
        
        # Log stats (For parallel, we are not updating local islands objects yet!)
        # CRITICAL: Since workers have their own copies of islands (proceses fork/spawn), 
        # self.islands in Main Process IS NOT UPDATED.
        # We need to retrieve the stats or the islands.
        # Ideally, we should sync stats at least.
        # But if we don't sync islands, `run_generation` loop becomes meaningless for Main Process 
        # because it doesn't see improvements?
        # `self.current_best_fitness` is checked locally.
        
        # SOLUTION: Workers must send back at least the best fitness info.
        # Better: Workers send back the `log_entry` generated?
        # In `evolve_island`, we append to `self.log_data`.
        # In Parallel, that `self.log_data` is local to worker.
        # We need to gather it.
        
        # Refinement needed for Parallel State Sync.
        # For now, let's assume workers evolve partially detached. 
        # But we need stopping criteria!
        # Let's assume workers send back (step_log_entries, step_best_fitness).
        # We need to update `worker_evolution` in `parallel.py` to send this.
        # And `run_generation` here needs to unpack it.
        # Autosave logs to prevent data loss on crash
        if self.autosave_path:
            # Save to 'latest'
            self.save_logs(self.autosave_path)
            # Save to unique history file
            self.save_logs(self.unique_log_path)
            
            # Also ensure config is up to date (in case of dynamic changes, though rare)
            if hasattr(self, 'autosave_config_path'):
                 self.save_config(self.autosave_config_path)
                 self.save_config(self.unique_config_path)
            
        pass # Placeholder comment, proceeding with simple implementation for now.
        self.generation += 1 
        # Log stats
        stats = []
        for i, island in enumerate(self.islands):
            best = island.get_best()
            best_fit = best.fitness if best else float('inf')
            stats.append({
                'island_idx': i,
                'num_vars': island.num_vars,
                'best_fitness': best_fit,
                'generation': self.generation
            })
            
        return stats

def _init_island_worker(args):
    """
    Worker function for parallel island initialization.
    Args:
        args (tuple): (island_config, n_pop, config, data, obs_indices, time_points, problem_description)
    """
    try:
        island_idx, island_config, n_pop, config, data, obs_indices, time_points, problem_description = args
        num_vars = island_config['num_vars']
        
        # Imports inside worker to ensure clean state if needed, though top-level imports work too
        from evolution.models import Island, Individual
        from evolution.llm_interface import LLMClient
        
        # Instantiate LLM Client (local to this process)
        llm = LLMClient(config)
        if problem_description:
            llm.set_problem_description(problem_description)
            
        island = Island(num_vars=num_vars, capacity=n_pop)
        attempts = 0
        max_attempts = n_pop * 5 # Generous retry limit
        
        print(f"Worker for Island {island_idx} (Dim {num_vars}) started.")
        
        while len(island.population) < n_pop and attempts < max_attempts:
            attempts += 1
            try:
                code = llm.generate_initial_model(island.num_vars)
                if code:
                    ind = Individual(code, creation_op="init", generation=0)
                    if ind.compile():
                        fit = ind.evaluate(data, obs_indices, time_points)
                        if fit < float('inf'):
                            # ind.model = None # Ensure it's stripped before return (handled by __getstate__ anyway)
                            island.add_individual(ind)
            except Exception as e:
                # print(f"Error in worker {island_idx}: {e}")
                pass
                
        print(f"Worker for Island {island_idx} finished. Pop: {len(island.population)}")
        return island
        
    except Exception as e:
        print(f"CRITICAL WORKER FAIL: {e}")
        traceback.print_exc()
        return None

def parallel_create_islands(n_pop, island_structure, config, data, obs_indices, time_points, problem_description, n_cores=4):
    """
    Parallel version of create_initial_islands.
    
    Args:
        n_pop (int): Population size per island.
        island_structure (list): List of dicts, e.g. [{'num_vars': 2}, {'num_vars': 3}]
        config (dict): Configuration dictionary.
        data (tensor): Observed data.
        obs_indices (list): Observed indices.
        time_points (tensor): Time points.
        problem_description (str): Text description for LLM.
        n_cores (int): Number of parallel workers.
        
    Returns:
        list[Island]: Fully populated Island objects.
    """
    print(f"Initializing {len(island_structure)} islands using {n_cores} cores...")
    
    tasks = []
    for i, isl_conf in enumerate(island_structure):
        # Prepare args
        # Note: We pass copies of data/config. Multiprocessing handles serialization.
        task_args = (i, isl_conf, n_pop, config, data, obs_indices, time_points, problem_description)
        tasks.append(task_args)
        
    results = []
    with multiprocessing.Pool(processes=n_cores) as pool:
        # chunksize=1 is fine as tasks are heavy
        results = pool.map(_init_island_worker, tasks)
        
    # Filter out failures
    islands = [res for res in results if res is not None]
    
    if len(islands) < len(island_structure):
        print(f"WARNING: Only {len(islands)}/{len(island_structure)} islands successfully initialized.")
        
    return islands
