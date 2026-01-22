
import torch
import torch.nn as nn
import numpy as np
import traceback

# Helper for integration (Reuse simple Euler for speed in evolution)
def integrate_euler(model, x0, t_points):
    trajectory = [x0]
    x = x0
    dt = t_points[1] - t_points[0]
    
    for i in range(len(t_points) - 1):
        dx = model(t_points[i], x) # model.forward(t, x)
        x = x + dx * dt
        trajectory.append(x)
    
    return torch.stack(trajectory)

class Individual:
    def __init__(self, code, model=None):
        self.code = code
        self.model = model
        self.fitness = float('inf')
        self.is_compiled = model is not None

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
                raise ValueError("Class 'ProposedModel' not found in code code.")
            
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
        Computes fitness based on integration error (MSE).
        Does NOT optimize parameters.
        """
        if not self.is_compiled:
            return float('inf')
            
        try:
            # 1. Construct Initial State
            # We need a strategy to set x0. 
            # Strategy: Use observed data for obs indices, 0.5/default for hidden.
            x0 = torch.zeros(self.model.num_vars)
            
            # Map observed indices assuming strict ordering for now (Simplification)
            # In a real scenario, the LLM should tell us the mapping, but for block 2 
            # we assume the first K variables correspond to the K active obs indices if possible
            # or we pass explicit mapping.
            
            # Better Strategy: Just use the same heuristic as before
            # If model has 4 vars and we observe 2 vars.
            # We assume model variables 0 and 2 match observed data 0 and 1? No.
            # We need the LLM to respect an observable map.
            # For this Block, let's assume the LLM generates models where 
            # the FIRST len(obs_indices) variables correspond to the observations.
            # (Or we fix x0 to a standard guess).
            
            # Let's try to map strictly by index for now:
            # valid indices < num_vars get mapped.
            for i, obs_idx in enumerate(obs_indices):
                if obs_idx < self.model.num_vars:
                    x0[obs_idx] = observed_data[0, i] # Taking the i-th column of obs data
            
            # Fill hidden
            x0[x0 == 0] = 0.1 # avoidance of zero
            
            # 2. Integrate
            X_pred = integrate_euler(self.model, x0, time_points)
            
            # 3. Compute Loss on known indices
            # We need to extract the columns from X_pred that correspond to obs_indices
            valid_obs_indices = [idx for idx in obs_indices if idx < self.model.num_vars]
            
            if len(valid_obs_indices) == 0:
                 # Model doesn't have enough variables to match observations?
                 return float('inf')

            # We compare X_pred[:, valid_indices] vs observed_data
            # Note: observed_data shape is (T, len(obs_indices))
            # We need to match the columns correctly.
            mse = 0
            for i, obs_idx in enumerate(obs_indices):
                if obs_idx in valid_obs_indices:
                    mse += torch.mean((X_pred[:, obs_idx] - observed_data[:, i])**2)
            
            self.fitness = mse.item()
            return self.fitness
            
        except Exception as e:
            print(f"Evaluation Failed: {e}")
            self.fitness = float('inf')
            return float('inf')

class Island:
    def __init__(self, num_vars, capacity=10):
        self.num_vars = num_vars
        self.capacity = capacity
        self.population = []

    def add_individual(self, individual):
        """
        Adds an individual to the population.
        """
        self.population.append(individual)
        # Sort by fitness (lowest is best)
        self.population.sort(key=lambda ind: ind.fitness)
        
        # Trim to capacity
        if len(self.population) > self.capacity:
            self.population = self.population[:self.capacity]

    def get_best(self):
        if not self.population:
            return None
        return self.population[0]
