
import random
import sys
import copy
import numpy as np

import json
import os

class EvolutionEngine:
    def __init__(self, islands, llm_client, data, obs_indices, time_points, n_iterations=None):
        """
        islands: List of Island objects
        llm_client: LLMClient instance
        data: Dict containing problem description etc.
        """
        self.islands = islands
        self.llm_client = llm_client
        self.data = data
        self.obs_indices = obs_indices
        self.time_points = time_points
        
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
        
        # Stopping criterion
        if n_iterations is None:
             self.n_iterations = int(evo_strat.get('n_iterations_stagnation', {}).get('value', 20))
        else:
             self.n_iterations = n_iterations

        self.current_iter = 0 # updated if no improvement observed, set to 0 else
        self.generation = 0
        self.stop_flag = False
        self.current_best_fitness = self.best_fitness() #useful for stopping criterion if no improvement observed

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
        
        # Stability shift
        min_fit = np.min(fitnesses)
        shifted = fitnesses - min_fit
        weights = np.exp(-shifted / self.temperature_selec_ind)
        
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
            return

        child = None
        child_code = None
        fitness = float('inf')

        # Check stopping criterion
        if self.current_iter < self.n_iterations:
            # Crossover or Mutation
            # If we don't have enough for crossover, forced mutation
            do_crossover = (random.random() < self.crossover_rate) and (len(island.population) >= self.N_parent_crossover)
            
            if do_crossover:
                parents = self.select_parents(island, k=self.N_parent_crossover)
                if len(parents) == self.N_parent_crossover:
                    parent_codes = [p.code for p in parents]
                    child_code = self.llm_client.crossover_model(parent_codes)
                    if child_code:
                        from evolution.models import Individual
                        child = Individual(child_code)
                        if child.compile(): 
                            fitness = child.evaluate(self.data['observed_data'], self.obs_indices, self.time_points)
                            if fitness < float('inf'):
                                island.add_individual(child)
            else:
                # Mutation
                parents = self.select_parents(island, k=1)
                if parents:
                    parent = parents[0]
                    child_code = self.llm_client.evolve_model(parent.code, instruction = None)
                    if child_code:
                        from evolution.models import Individual
                        child = Individual(child_code)
                        if child.compile():
                            fitness = child.evaluate(self.data['observed_data'], self.obs_indices, self.time_points)
                            if fitness < float('inf'):
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
        success = (child is not None) and (child.is_compiled)
        
        if success and fitness < self.current_best_fitness:
            self.current_iter = 0
            self.current_best_fitness = fitness
            print(f"    Improvement found! New Best: {fitness:.4f}")
        else:
            self.current_iter += 1

    def run_generation(self):
        print(f"--- Generation {self.generation} (Stagnation: {self.current_iter}/{self.n_iterations}) ---")
        
        # User requested: "use a random uniform variable to chose an island to evolve"
        if not self.islands:
            return []
            
        island_idx = random.randint(0, len(self.islands) - 1) #uniform random selection of an island
        target_island = self.islands[island_idx]
        
        print(f"Evolving Island {island_idx} (Vars: {target_island.num_vars})...")
        self.evolve_island(target_island)
        
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
            
        self.generation += 1
        return stats
