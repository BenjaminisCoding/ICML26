
import sys
import os
import torch
sys.path.append(os.getcwd())

from evolution.llm_interface import LLMClient
from evolution.models import Individual, Island

def test_initialization():
    print("--- Testing EA Initialization ---")
    
    # 1. Setup LLM
    llm = LLMClient()
    
    # 2. Define Problem
    problem_desc = "A 2-compartment pharmacokinetic model (Central, Peripheral). Drug enters Central, distributes to Peripheral, and is cleared from Central."
    
    # 3. Create Island (Dimension = 3 vars: Central, Peripheral, maybe latent absorption?)
    island = Island(num_vars=3)
    
    print("Requesting model from LLM...")
    code = llm.generate_initial_model(problem_desc, num_vars=island.num_vars)
    
    if code:
        print("\n--- Generated Code ---")
        print(code)
        print("----------------------")
        
        # 4. Create Individual
        ind = Individual(code)
        
        # 5. Compile
        success = ind.compile()
        if success:
            print("Compilation SUCCESS.")
            print(f"Model Parameters: {list(ind.model.parameters())}")
            
            # 6. Test Evaluation
            print("Testing Evaluation...")
            # Dummy Data
            time_points = torch.linspace(0, 10, 20)
            obs_indices = [0] 
            observed_data = torch.randn(20, 1) # 1 observed var
            
            fit = ind.evaluate(observed_data, obs_indices, time_points)
            print(f"Computed Fitness: {fit}")
        else:
            print("Compilation FAILED.")
    else:
        print("LLM returned None.")

if __name__ == "__main__":
    test_initialization()
