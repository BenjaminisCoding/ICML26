
import torch
import uuid
import torch.nn as nn
import numpy as np
import traceback

from optimization_utils import run_homotopy_optimization

# Helper for integration (Reuse simple Euler for speed in evolution)
def integrate_euler(model, x0, t_points):
    trajectory = [x0]
    x = x0
    dt = t_points[1] - t_points[0]
    
    for i in range(len(t_points) - 1):
        dx = model(x) # model.forward(x)
        x = x + dx * dt
        trajectory.append(x)
    
    return torch.stack(trajectory)

import uuid

class Individual:
    def __init__(self, code, model=None, parents=None, creation_op="init", generation=0):
        self.id = str(uuid.uuid4())
        self.code = code
        self.model = model
        self.fitness = float('inf')
        self.is_compiled = model is not None
        self.loss_history = []
        
        # Lineage info
        self.parents = parents if parents else [] # List of parent IDs
        self.creation_op = creation_op # "init", "mutation", "crossover"
        self.generation = generation
        self.metadata = {} # For logging extra info like specific instruction used
        self.origin = creation_op # "migration", "extinction", etc. can be passed here or set later

    def __getstate__(self):
        """
        Custom pickling state.
        We cannot pickle dynamically defined model classes/instances safely without dill.
        So we drop 'model' and force re-compilation on unpickle.
        """
        state = self.__dict__.copy()
        state['model'] = None
        state['is_compiled'] = False # Will need re-compilation
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # We don't auto-compile here to avoid slow imports during unpickle.
        # Worker must call compile() explicitly.

    def compile(self):
        """
        Compiles the Python code into a runnable PyTorch model.
        Assumes the code defines a class named 'ProposedModel'.
        """
        try:
            # Create a safe local dictionary to execute code
            local_scope = {}
            exec(self.code, globals(), local_scope)
            
            if 'ProposedModel' not in local_scope:
                raise ValueError("Class 'ProposedModel' not found in code.")
            
            ModelClass = local_scope['ProposedModel']
            self.model = ModelClass()
            self.is_compiled = True
            return True
        except Exception as e:
            print(f"Compilation Failed: {e}")
            # print(self.code) # Debug
            self.is_compiled = False
            return False

    def evaluate(self, observed_data, obs_indices, time_points):
        """
        Computes fitness using Homotopy Optimization.
        Optimizes parameters and returns the final loss.
        """
        if not self.is_compiled:
            return float('inf')
            
        try:
            # Filter observed data for dimensions present in the model
            valid_mask = [i for i, idx in enumerate(obs_indices) if idx < self.model.num_vars]
            
            if len(valid_mask) == 0:
                 return float('inf')
                 
            mapped_obs_indices = [obs_indices[i] for i in valid_mask]
            mapped_observed_data = observed_data[:, valid_mask]

            # Run Homotopy Optimization
            # Using specific schedule for EA speed
            fast_schedule = [0.01, 1.0, 100.0] 
            steps_fast = 200 # 600 total steps vs 6000
            
            X_est, loss_hist, traj_hist = run_homotopy_optimization(
                self.model, 
                mapped_observed_data, 
                mapped_obs_indices, 
                time_points,
                tau_schedule=fast_schedule,
                steps_per_tau=steps_fast,
                verbose=False,
                system_dim=self.model.num_vars
            )
            
            final_loss = loss_hist[-1]
            
            if np.isnan(final_loss) or np.isinf(final_loss):
                 self.fitness = float('inf')
                 return float('inf')

            self.fitness = final_loss
            self.loss_history = loss_hist # Store history if needed
            return self.fitness
            
        except Exception as e:
            print(f"Evaluation Failed (Homotopy): {e}")
            self.fitness = float('inf')
            return float('inf')

class Island:
    def __init__(self, num_vars, capacity=10):
        self.num_vars = num_vars
        self.capacity = capacity
        self.population = []
        self.id = str(uuid.uuid4())
        self.neighbors = [] # List of neighbor Island IDs
        self.stagnation_counter = 0 # Generations since last best fitness improvement
        self.last_best_fitness = float('inf')

    def add_individual(self, individual):
        """
        Adds an individual to the population.
        """
        # Robustness check
        if not np.isfinite(individual.fitness):
            return

        self.population.append(individual)
        
        # Trim to capacity by removing the worst individual (highest fitness)
        while len(self.population) > self.capacity:
            worst = max(self.population, key=lambda ind: ind.fitness)
            if worst.fitness < individual.fitness:
                 # If the new guy is worse than the worst, actually we should maybe not add him?
                 # But standard algorithm usually adds children then kills worst.
                 # If we already added him, and he is the worst, he gets removed.
                 pass
            self.population.remove(worst)
            
    def register_neighbor(self, island_id):
        if island_id not in self.neighbors and island_id != self.id:
            self.neighbors.append(island_id)
            
    def update_stagnation(self):
        current_best = self.get_best()
        if not current_best:
            return
            
        if current_best.fitness < self.last_best_fitness:
            self.last_best_fitness = current_best.fitness
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
    def clear_population(self):
        self.population = []
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')

    def get_best(self):
        if not self.population:
            return None
        return min(self.population, key=lambda ind: ind.fitness)
