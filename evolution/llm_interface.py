
import os
import re
from openai import AzureOpenAI

# Configuration (Reused from solve/method.py)
API_KEY = "36PxHP6vflMUsFh6M9oCrVVmLW645VZtz18DKjQYNShtRevvTB76JQQJ99BIACL93NaXJ3w3AAABACOGyfOR"
API_VERSION = "2024-10-21"
AZURE_ENDPOINT = "https://vdslabazuremloai-ae.openai.azure.com/"
DEPLOYMENT_NAME = 'gpt-4o-benjamin'

class LLMClient:
    def __init__(self, config=None):
        self.config = config or {}
        self.deployment_name = DEPLOYMENT_NAME
        self.client = AzureOpenAI(
            api_key=API_KEY,
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
        self.instructions = self.config.get('llm_instructions', '')
        self.system_prompt = """
You are an expert AI scientist specializing in discovering ordinary differential equations (ODEs) from partial data. 
Your task is to propose the right-hand side of ODE systems to explain observed time-series data.
You will be given the descritpion of the problem and of the observed variables. 
You will be ask to generate initial proposals for the ODE system, 
and will be asked to perform mutation operation or crossover operation on existing models, like in genetic algorithms. 

**Crucial Constraints:**
1. You will be provided with a **CODE SKELETON**.
2. You must **NEVER** change the structure of the skeleton.
3. You must **ONLY** fill in the parts marked with `###`.
4. Do not change class names, method signatures, or import statements.
5. As you only observe partial data, the system may have hidden (latent) variables that you need to account for.
6. Return **ONLY** the python code. No markdown formatting like ```python, no explanations.

**The Skeleton Format:**
The skeleton defines a `ProposedModel` class. You will need to fill in:
- The parameters in `__init__`.
- The dynamics in `forward`.

The user might specify the number of variables (dimension) for the system.
"""
        self.system_prompt += f"\n\n**Further Instructions:**\n{self.instructions}\n"

    def set_problem_description(self, problem_description):
        """
        Appends the specific problem description to the system prompt.
        This provides context for all subsequent API calls.
        """
        self.system_prompt += f"\n\n**Problem Description & Context:**\n{problem_description}\n"
        print("Problem description added to System Prompt.")

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
        
    def forward(self, state):
        # state is a tensor of size (..., {num_vars})
        # Unpack state variables using slicing:
        # x1 = state[..., 0]
        # x2 = state[..., 1]
        ###
        
        # access params via self.params
        ###
        
        # Compute derivatives dx/dt
        # Must return stack of {num_vars} derivatives
        ###
        return torch.stack([###], dim = -1)
"""
        return skeleton.strip()

    def generate_initial_model(self, num_vars):
        """
        Asks LLM to populate the skeleton for a specific dimension.
        """
        skeleton = self._get_skeleton(num_vars)
        
        # Get instruction from config or default
        instruction = self.config.get('llm_instructions', {}).get('initial_generation', {}).get(
            'value', 
            "Focus on simple, mechanistic models that could explain the observed dynamics, possibly introducing latent variables if the system seems under-specified."
        )

        user_prompt = f"""
System Dimension: {num_vars} variables.
Task: {instruction}

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
                temperature=self.config.get('azure_api', {}).get('temperature_initial', {}).get('value', 1.0)
            )
            return resp.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return None # Handle error upstream

    def evolve_model(self, parent_code, instruction=None):
        """
        Asks LLM to modify an existing model based on instruction (Mutation).
        """
        if instruction is None:
            instruction = self.config.get('llm_instructions', {}).get('mutation', {}).get(
                'value', 
                "Optimise the parameters and improve the dynamics to fit the data."
            )

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
                temperature=self.config.get('azure_api', {}).get('temperature_mutation', {}).get('value', 0.8)
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return None

    def crossover_model(self, parent_codes_list, instruction=None):
        """
        Asks LLM to combine multiple models.
        """
        if instruction is None:
            instruction = self.config.get('llm_instructions', {}).get('crossover', {}).get(
                'value', 
                "Combine the best features of both models."
            )

        # Build prompt with flexible number of models
        models_text = ""
        for i, code in enumerate(parent_codes_list):
            models_text += f"\n--- Model {i+1} ---\n{code}\n"

        user_prompt = f"""
Here are {len(parent_codes_list)} ODE models:

{models_text}

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
                temperature=self.config.get('azure_api', {}).get('temperature_crossover', {}).get('value', 0.8)
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return None
