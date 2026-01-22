
import os
import re
from openai import AzureOpenAI

# Configuration (Reused from solve/method.py)
API_KEY = "36PxHP6vflMUsFh6M9oCrVVmLW645VZtz18DKjQYNShtRevvTB76JQQJ99BIACL93NaXJ3w3AAABACOGyfOR"
API_VERSION = "2024-10-21"
AZURE_ENDPOINT = "https://vdslabazuremloai-ae.openai.azure.com/"
DEPLOYMENT_NAME = 'gpt-4o-benjamin'

class LLMClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=API_KEY,
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
        self.deployment_name = DEPLOYMENT_NAME
        
        self.system_prompt = """
You are an expert AI scientist specializing in discovering ordinary differential equations (ODEs) from partial data.
Your task is to propose the right-hand side of ODE systems to explain observed time-series data.

**Crucial Constraints:**
1. You will be provided with a **CODE SKELETON**.
2. You must **NEVER** change the structure of the skeleton.
3. You must **ONLY** fill in the parts marked with `###`.
4. Do not change class names, method signatures, or import statements.
5. The system may have latent (hidden) variables that you need to account for.
6. Return **ONLY** the python code. No markdown formatting like ```python, no explanations.

**The Skeleton Format:**
The skeleton defines a `ProposedModel` class. You will need to fill in:
- The parameters in `__init__`.
- The dynamics in `forward`.

The user might specify the number of variables (dimension) for the system.
"""

    def _get_skeleton(self, num_vars):
        """
        Returns the code skeleton for a model with `num_vars` variables.
        """
        skeleton = f"""
import torch
import torch.nn as nn

class ProposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_vars = {num_vars}
        
        # Define learnable parameters here
        self.params = nn.ParameterDict({{
            ###
        }})
        
    def forward(self, t, state):
        # state is a list/tensor of size {num_vars}
        # Unpack state variables if needed (e.g., x1, x2, ...)
        ###
        
        # access params via self.params
        ###
        
        # Compute derivatives dx/dt
        # Must return stack of {num_vars} derivatives
        ###
        return torch.stack([###])
"""
        return skeleton.strip()

    def generate_initial_model(self, problem_description, num_vars):
        """
        Asks LLM to populate the skeleton for a specific dimension.
        """
        skeleton = self._get_skeleton(num_vars)
        
        user_prompt = f"""
Problem Description: {problem_description}
System Dimension: {num_vars} variables.

Please fill in the following skeleton to propose a candidate ODE system.
Replace the `###` markers with valid Python code.

SKELETON:
{skeleton}
"""
        
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0 # High exploration
            )
            return resp.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return None # Handle error upstream

    def evolve_model(self, parent_code, instruction):
        """
        Asks LLM to modify an existing model based on instruction (Mutation).
        """
        user_prompt = f"""
Here is an existing ODE model:
{parent_code}

Task: {instruction}
Modify the code to satisfy the task. Keep the class structure identical.
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def crossover_model(self, parent1_code, parent2_code, instruction="Combine the best features of both models."):
        """
        Asks LLM to combine two models.
        """
        user_prompt = f"""
Here are two ODE models:

--- Model 1 ---
{parent1_code}

--- Model 2 ---
{parent2_code}

Task: {instruction}
Create a new child model that merges their structural ideas. 
Keep the class structure identical to the skeleton.
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return None
