import numpy as np
import importlib.util
import sys
from scipy.integrate import odeint # This import seems unused if torchdiffeq is used
import os
import random # This import seems unused
import torch
import json
import inspect

import textwrap
import re
from openai import AzureOpenAI

from multiprocessing import Pool, cpu_count # Pool and cpu_count are imported but not used, consider removing
from functools import partial # Unused import
from tqdm import tqdm # Import tqdm for progress bars, used only in _train_single_model currently
from torchdiffeq import odeint, odeint_adjoint

# Conditional import for `train_model` based on execution context
if __name__ == "__main__":
    from utils import train_model
else:
    from solve.utils import train_model

# Azure OpenAI API configuration
api_key = "36PxHP6vflMUsFh6M9oCrVVmLW645VZtz18DKjQYNShtRevvTB76JQQJ99BIACL93NaXJ3w3AAABACOGyfOR" # Consider loading from environment variables or a secure config for production
api_version = "2024-10-21"
azure_endpoint = "https://vdslabazuremloai-ae.openai.azure.com/"
deployment_name = 'gpt-4o-benjamin' # LLM deployment name

# Constants for model generation and simulation flags
NUM_PROCESSES = 1 # Currently set to 1, implies sequential training. If parallel processing is desired, this needs to be integrated with `multiprocessing.Pool`
NUM_MODELS_TO_GENERATE = 2
UNSIMULABLE_FLAG = -1 # A flag to denote models that could not be simulated

def call_llm_api(messages):
    """
    Calls the Azure OpenAI LLM API with a given list of messages.
    Includes a fallback mock response in case of API errors for development/testing.

    Args:
        messages (list): A list of message dictionaries for the LLM conversation.

    Returns:
        str: The content of the LLM's response, which should be Python code.
    """
    print("\n--- LLM API Call (Real) ---")
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

        resp = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=1,
            # max_tokens=500, # Optionally set a max_tokens for model generation
            timeout=60 # Increased timeout for potentially longer generations
        )

        llm_response_content = resp.choices[0].message.content
        print("Real LLM generated a model proposal.")
        return llm_response_content
    except Exception as e:
        print(f"Error during LLM API call: {e}")
        print("Returning a mock response due to API error.")
        # Fallback to a mock response for continued testing if API fails
        mock_llm_response = """
import torch
import torch.nn as nn

class ProposedModel_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'sigma': nn.Parameter(torch.tensor(5.0, dtype=torch.float32)),
            'rho':   nn.Parameter(torch.tensor(20.0, dtype=torch.float32)),
            'beta':  nn.Parameter(torch.tensor(1, dtype=torch.float32))
        })
        self.variables_order = ['x', 'y', 'z']
        self.observable_map = {0: 0, 1: 1}

    def forward(self, t, state):
        x, y, z = state
        sigma = self.params['sigma']
        rho    = self.params['rho']
        beta  = self.params['beta']

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return torch.stack([dx, dy, dz])

---MODEL_DELIMITER---

import torch
import torch.nn as nn

class ProposedModel_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'sigma': nn.Parameter(torch.tensor(5.0, dtype=torch.float32)),
            'rho':   nn.Parameter(torch.tensor(20.0, dtype=torch.float32)),
            'beta':  nn.Parameter(torch.tensor(1, dtype=torch.float32))
        })
        self.variables_order = ['x', 'y', 'z']
        self.observable_map = {0: 0, 1: 1}

    def forward(self, t, state):
        x, y, z = state
        sigma = self.params['sigma']
        rho   = self.params['rho']
        beta  = self.params['beta']

        dx = sigma * (y - x) + x * z
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return torch.stack([dx, dy, dz])
        """
        return mock_llm_response

def _train_single_model(model_tuple, time_points, observed_data, config):
    """
    Helper function to train a single ODE model.
    Designed to be called by `__optimize__`, potentially in a multiprocessing context.

    Args:
        model_tuple (tuple): A tuple containing (model_index, model_instance).
        time_points (torch.Tensor): Time points for ODE integration.
        observed_data (torch.Tensor): The observed experimental data.
        config (dict): Configuration parameters for the training process.

    Returns:
        dict: A dictionary containing training results (loss, parameters, simulability flag).
    """
    model_idx, model_instance = model_tuple
    print(f"Starting training for Model {model_idx}...")

    t = torch.tensor(time_points, dtype=torch.float32).detach().clone()
    y_obs_tensor = torch.tensor(observed_data, dtype=torch.float32).detach().clone()

    # Call the `train_model` utility function
    losses, final_params, flag = train_model(
            ode_func=model_instance,
            t=t,
            y_obs=y_obs_tensor,
            config=config,
            observable_map=model_instance.observable_map # Pass observable map to train_model
        )

    final_loss = losses[-1] if losses else float('inf')
    print(f"Finished training for Model {model_idx}. Final Loss: {final_loss:.6f}")
    return {
        'model_idx': model_idx,
        'model_instance': model_instance,
        'final_loss': final_loss,
        'final_params': final_params,
        'losses_history': losses,
        'simulable': flag
    }

class Method():
    """
    Implements a scientific discovery method using an LLM to propose and refine ODE models.
    The method iterates between LLM-based model generation, optimization against data,
    and providing performance feedback to the LLM for subsequent refinements.
    """
    def __init__(self, data, iter_max, instructions, num_models_to_generate=NUM_MODELS_TO_GENERATE, noise_level=None):
        """
        Initializes the Method with data, iteration limits, LLM instructions, and configuration.

        Args:
            data (dict): Dictionary containing problem description, observed variables, time points, and observed data.
            iter_max (int): Maximum number of iterations for the discovery process.
            instructions (str): Specific instructions for the LLM regarding model generation.
            num_models_to_generate (int): Number of candidate models the LLM should propose per iteration.
            noise_level (float, optional): Level of noise in the data, used for logging file naming. Defaults to None.
        """
        self.iter = 0
        self.iter_max = iter_max
        self.data = data
        self.conversation_history = [] # Stores the full conversation with the LLM
        self.current_best_model = None
        self.current_best_score = float('inf')
        self.num_models_to_generate = num_models_to_generate
        self.instructions = instructions
        self.log_file_path = f"optimization_log_{noise_level}.txt" if noise_level is not None else "optimization_log.txt"

        self.initial_losses = {} # To monitor initial losses of models per iteration
        self.final_losses = {} # To monitor final losses of models per iteration
        self._setup_context() # Set up the LLM's system prompt and few-shot examples

    def __gen_nnum_models__(self):
        """
        Dynamically updates the number of models the LLM is requested to generate
        based on a simple heuristic tied to the current iteration.
        """
        if self.iter == 2 and self.num_models_to_generate >= 4:
            self.num_models_to_generate -= 1
        if self.iter == 5 and self.num_models_to_generate >= 3:
            self.num_models_to_generate -= 1

    def _setup_context(self):
        """
        Configures the LLM's system prompt and provides few-shot examples
        to guide its model generation behavior. This is done once at initialization.
        """
        system_prompt = """
You are an expert AI scientist specialized in discovering partially-observed dynamical systems.
Your task is to propose candidate mechanistic models in the form of Python classes.
Each proposed model must be compatible with the `torchdiffeq` package for simulation and parameter estimation.
Specifically, each model must include:
1.  A class named `ProposedModel` (you can append a number to it, e.g., `ProposedModel_1`) that inherits from `torch.nn.Module`.
2.  An `__init__` method that defines learnable parameters as `torch.nn.Parameter` objects stored inside a dictionary `self.params`.
3.  A `variables_order` list in `__init__` specifying the order of state variables (e.g., `['x', 'y', 'z']`).
4.  `_set_z0_observed(self, z0_observed)` method: This method will be called by the training pipeline to provide the initial conditions for the *observed* variables. Store `z0_observed` (a `torch.Tensor`) as `self._observed_z0_values`. This should be a plain tensor, not a learnable parameter.
5.  `_get_initial_state(self)` method: This method will construct the full initial state vector `z0` by combining the observed initial conditions (from `self._observed_z0_values`) and the learnable initial conditions for *hidden* variables (from `self.hidden_initial_state_params`).
    *   In `__init__`, identify observed and hidden variable indices:
        *   `self.observed_system_indices`: A sorted list of indices from `self.observable_map.keys()`. These are the indices in the model's full state vector that correspond to observed variables.
        *   `self.hidden_system_indices`: A sorted list of indices in the model's full state vector that are *not* observed.
    *   In `__init__`, define `self.hidden_initial_state_params` as an `nn.Parameter` holding initial guesses for the hidden variables. Its size must match `len(self.hidden_system_indices)`.
    *   The `_get_initial_state` method should correctly map `self._observed_z0_values` to the `self.observed_system_indices` positions in `full_z0` and `self.hidden_initial_state_params` to the `self.hidden_system_indices` positions.
6.  A `forward(self, t, state)` method that computes the derivatives `[dx/dt, dy/dt, dz/dt, ...]`. This matches the signature required by `torchdiffeq.odeint`.
7.  All parameters in the `forward` method must be accessed from the `params` dictionary (or `self.params`).
8.  The model should be a system of ordinary differential equations (ODEs).
9.  Aim for interpretable models. You will be given feedback on your propositions so you can further refine them. For each model you propose, an optimization will be done on the parameters of the ode system. You will then be given the final value of the parameters for which the loss was the best, and also the value of the loss to help you decide how to further improve the different models.
10. Consider both observed and unobserved (latent) variables in your proposed dynamics.
11. *   **Purpose:** The input data `y_obs` will always have its observed variables ordered by their original indices (e.g., `y_obs[:, 0]` is the first observed variable, `y_obs[:, 1]` is the second). Your `observable_map` must tell the system which component of your model's internal state (`state[idx]`) corresponds to each observed data column.
*   **Format:** `self.observable_map = {observed_data_index: model_state_index}`.
*   **Example:**
    *   If your `self.variables_order` is `['x', 'y', 'z']` (meaning `x` is `state[0]`, `y` is `state[1]`, `z` is `state[2]`).
    *   And your input `y_obs` data has its first column (index `0`) representing `x`, and its second column (index `1`) representing `z`.
    *   Then `self.observable_map` should be `{0: 0, 1: 2}`. (Because observed data index `0` maps to `state[0]` which is `x`, and observed data index `1` maps to `state[2]` which is `z`).
*   **Important:** The keys of `self.observable_map` must be the integer indices of the observed data columns (e.g., `0, 1, 2`). The values must be the integer indices of the corresponding state variables *in your `self.variables_order` list* (e.g., if `'x'` is `self.variables_order[0]`, then `0` would be the value).
12. When providing multiple models, separate each complete `ProposedModel` class block with the string `---MODEL_DELIMITER---` on its own line.
13. Do NOT include any explanatory text or conversational remarks in your response, only Python code.
        """

        self.conversation_history.append({"role": "system", "content": system_prompt})
        self.conversation_history.append({"role": "system", "content": self.instructions})
        # Few-shot examples demonstrating the expected output format and complexity
        few_shot_input_1 = """
# Problem Description:
# A simple system exhibiting exponential decay.
#
# Observed Variables and their Descriptions:
# 0: x: Quantity exhibiting decay.
#
# Propose 2 Python classes `ProposedModel` for the governing ODEs.
# Include initial parameter guesses in `self.params` and specify `self.variables_order`.
        """
        few_shot_output_1 = """
import torch
import torch.nn as nn

class ProposedModel_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'k': nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        })
        self.variables_order = ['x1', 'x2', 'x3'] # Changed to x1, x2, x3
        self.num_variables = len(self.variables_order)
        self.observable_map = {0: 0} # Only 'x1' is observed (at index 0 in variables_order)

        self.observed_system_indices = sorted(list(self.observable_map.values())) # Indices in the model's state vector that are observed
        self.hidden_system_indices = sorted(list(set(range(self.num_variables)) - set(self.observed_system_indices))) # Indices in the model's state vector that are hidden
        self.hidden_initial_state_params = nn.Parameter(torch.tensor([0.0, 5.0], dtype=torch.float32)) # Initial guesses for 'x2' and 'x3'

        self._observed_z0_values = None # Placeholder for observed initial conditions

    def _set_z0_observed(self, z0_observed):
        Sets the initial observed variable values.
        self._observed_z0_values = z0_observed.clone().detach()

    def _get_initial_state(self):
        Constructs the full initial state vector combining observed and hidden initial conditions.
        full_z0 = torch.empty(self.num_variables, dtype=torch.float32)
        # Map observed values to their respective positions in the full state vector
        for i, obs_data_idx in enumerate(sorted(self.observable_map.keys())):
            model_state_idx = self.observable_map[obs_data_idx]
            full_z0[model_state_idx] = self._observed_z0_values[i] # Use i as index for z0_observed

        # Map hidden initial state parameters to their respective positions
        for i, hidden_sys_idx in enumerate(self.hidden_system_indices):
            full_z0[hidden_sys_idx] = self.hidden_initial_state_params[i]
        return full_z0

    def forward(self, t, state):
        x1 = state[self.variables_order.index('x1')]
        x2 = state[self.variables_order.index('x2')] # Hidden variable
        x3 = state[self.variables_order.index('x3')] # Hidden variable
        k = self.params['k']
        dx1dt = -k * x1 + x2 # x1 depends on x2
        dx2dt = x1 - x2 + x3 # Simple dynamics for hidden variables
        dx3dt = -x3 + k # Another hidden variable
        return torch.stack([dx1dt, dx2dt, dx3dt])

---MODEL_DELIMITER---

import torch
import torch.nn as nn

class ProposedModel_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'k': nn.Parameter(torch.tensor(0.5, dtype=torch.float32)),
            'A': nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        })
        self.variables_order = ['x1', 'x2', 'x3', 'x4'] # Changed to x1, x2, x3, x4
        self.num_variables = len(self.variables_order)
        self.observable_map = {0: 0} # Only 'x1' is observed

        self.observed_system_indices = sorted(list(self.observable_map.values()))
        self.hidden_system_indices = sorted(list(set(range(self.num_variables)) - set(self.observed_system_indices)))
        self.hidden_initial_state_params = nn.Parameter(torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)) # Initial guesses for 'x2', 'x3', and 'x4'

        self._observed_z0_values = None

    def _set_z0_observed(self, z0_observed):
        Sets the initial observed variable values.
        self._observed_z0_values = z0_observed.clone().detach()

    def _get_initial_state(self):
        Constructs the full initial state vector combining observed and hidden initial conditions.
        full_z0 = torch.empty(self.num_variables, dtype=torch.float32)
        # Map observed values to their respective positions in the full state vector
        for i, obs_data_idx in enumerate(sorted(self.observable_map.keys())):
            model_state_idx = self.observable_map[obs_data_idx]
            full_z0[model_state_idx] = self._observed_z0_values[i]

        # Map hidden initial state parameters to their respective positions
        for i, hidden_sys_idx in enumerate(self.hidden_system_indices):
            full_z0[hidden_sys_idx] = self.hidden_initial_state_params[i]
        return full_z0

    def forward(self, t, state):
        x1 = state[self.variables_order.index('x1')]
        x2 = state[self.variables_order.index('x2')]
        x3 = state[self.variables_order.index('x3')]
        x4 = state[self.variables_order.index('x4')]
        k = self.params['k']
        A = self.params['A']
        dx1dt = -k * x1 + A + x2 # Simple linear decay with a constant input, plus influence from x2
        dx2dt = k * x1 - A * x2 + x4 # Dynamics for hidden x2
        dx3dt = A - k * x3 + x2 # Dynamics for hidden x3
        dx4dt = k * x3 - A * x4 # Dynamics for hidden x4
        return torch.stack([dx1dt, dx2dt, dx3dt, dx4dt])
        """
        self.conversation_history.append({"role": "user", "content": few_shot_input_1})
        self.conversation_history.append({"role": "assistant", "content": few_shot_output_1})

        print("LLM context and few-shot examples loaded.")

    def _split_llm_response_into_models(self, llm_response_str):
        """
        Splits the LLM's response string into individual model code blocks
        based on the ---MODEL_DELIMITER--- and removes Markdown code fences.

        Args:
            llm_response_str (str): The raw string response from the LLM.

        Returns:
            list: A list of cleaned Python code strings, each representing a model.
        """
        # First, split by the delimiter
        model_code_blocks_raw = [
            block.strip() for block in re.split(r'---MODEL_DELIMITER---', llm_response_str) if block.strip()
        ]

        cleaned_model_code_blocks = []
        for block in model_code_blocks_raw:
            # Remove Markdown code fences (e.g., ```python or ```) at the start and end of the block
            cleaned_block = re.sub(r'^\s*```(?:\w+)?\s*\n', '', block, count=1) # Remove leading ```
            cleaned_block = re.sub(r'\n\s*```\s*$', '', cleaned_block, count=1) # Remove trailing ```

            cleaned_model_code_blocks.append(cleaned_block.strip()) # Ensure overall block is stripped

        return cleaned_model_code_blocks

    def _parse_and_load_model_code(self, model_code_str_list):
        """
        Parses a list of Python code strings received from the LLM, dynamically loads them
        as modules, and instantiates the `ProposedModel` class from each module.

        Args:
            model_code_str_list (list): A list of Python code strings, each defining a model.

        Returns:
            list: A list of instantiated `torch.nn.Module` model objects.
        """
        model_instances = []
        for i, model_code_str in enumerate(model_code_str_list):
            try:
                # Dedent the code string to remove leading whitespace and ensure proper parsing
                dedented_code_str = textwrap.dedent(model_code_str).strip()
                if not dedented_code_str: # Skip empty blocks
                    continue

                # Create a temporary file to write the code for dynamic loading
                # Ensures a unique module name for each model to prevent conflicts
                model_module_name = f"proposed_model_iter{self.iter}_idx{i}"
                temp_file_path = os.path.join(os.getcwd(), 'generated_code', f"{model_module_name}.py")

                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True) # Ensure the 'generated_code' directory exists

                with open(temp_file_path, "w") as f:
                    f.write(dedented_code_str)

                # Dynamically load the module
                spec = importlib.util.spec_from_file_location(model_module_name, temp_file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[model_module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f'Error while loading the module for block {i}: {e}')
                    continue # Skip to the next model if loading fails

                # Instantiate the ProposedModel class from the loaded module
                class_name_match = re.search(r'class\s+(ProposedModel_\d+|ProposedModel)\s*\(nn\.Module\):', dedented_code_str)
                if class_name_match:
                    class_name = class_name_match.group(1)
                    model_instance = getattr(module, class_name)()
                    model_instances.append(model_instance)
                else:
                    print(f"Warning: Could not find 'ProposedModel' or 'ProposedModel_X' class name in block {i}. Skipping.")

                # Consider removing the temporary file for cleanliness after loading
                # os.remove(temp_file_path)

            except Exception as e:
                print(f"Error processing model code block {i}: {e}")
                print(f"Code received for block {i}:\n{model_code_str}")
        return model_instances

    def _generate_initial_model_proposal(self):
        """
        Generates the first set of model proposals from the LLM based on the initial problem description.

        Returns:
            list: A list of instantiated model objects.
        """
        problem_desc = self.data['problem_description']
        observable_variables_desc = self.data['observable_variable_descriptions']

        # Format observable variables for the prompt to the LLM
        formatted_obs_vars = ""
        for i, desc in enumerate(observable_variables_desc):
            formatted_obs_vars += f"{i}: {desc}\n"

        user_prompt = f"""
        # Problem Description:
        # {problem_desc}
        #
        # Observed Variables and their Descriptions:
        # {formatted_obs_vars.strip()}
        #
        # You have observed time-series data for the variables listed above.
        # Propose {self.num_models_to_generate} Python classes for the governing ODEs.
        # Include initial parameter guesses in `self.params` and specify `self.variables_order` for each.
        # Focus on simple, mechanistic models that could explain the observed dynamics, possibly introducing latent variables if the system seems under-specified.
        # Remember to separate each model with `---MODEL_DELIMITER---`.
        """
        self.conversation_history.append({"role": "user", "content": user_prompt})
        llm_response_code = call_llm_api(self.conversation_history)
        self.conversation_history.append({"role": "assistant", "content": llm_response_code})

        # Split the raw LLM response into individual model code blocks
        model_code_blocks = self._split_llm_response_into_models(llm_response_code)

        # Parse and load each model code block into Python objects
        return self._parse_and_load_model_code(model_code_blocks)

    def __start__(self):
        """
        Initiates the LLM-driven model generation process.
        """
        return self._generate_initial_model_proposal()

    def __optimize__(self, L_models):
        """
        Orchestrates the optimization (training) of a list of ODE models in parallel.
        It loads optimization configuration, trains each model, and collects results.

        Args:
            L_models (list): A list of instantiated ODE model objects.

        Returns:
            list: A list of dictionaries, each containing detailed optimization
                  results for a model (final loss, parameters, etc.).
        """
        print("\n--- Starting Optimization ---")
        config_file_path = os.path.join(os.path.dirname(__file__), 'config_optim.json')
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Optimization configuration file not found at {config_file_path}. Please create it.")

        with open(config_file_path, 'r') as f:
            config = json.load(f)

        print(f"Optimization config loaded: {config}")

        # Extract data needed for training
        time_points = self.data['time_points']
        observed_data = self.data['observed_data']

        models_with_indices = list(enumerate(L_models)) # Pair models with their original indices for tracking
        results = []
        # Currently, NUM_PROCESSES is 1, so this loop effectively runs sequentially.
        # To enable true multiprocessing, replace this with `multiprocessing.Pool`.
        for elem in models_with_indices:
            result = _train_single_model(elem, time_points, observed_data, config)
            results.append(result)

        optimized_models_info = []

        # Store initial and final losses for plotting and monitoring
        self.initial_losses[self.iter] = []
        self.final_losses[self.iter] = []
        for result in results:
                optimized_models_info.append({
                    'model_idx': result['model_idx'],
                    'model_instance': result['model_instance'],
                    'final_loss': result['final_loss'],
                    'final_params': result['final_params'], # Parameters dictionary
                    'losses_history': result['losses_history'],
                    'simulable': result['simulable']
                })
                # Append initial loss (first element of losses_history) if available
                if result['losses_history']:
                    self.initial_losses[self.iter].append(result['losses_history'][0])
                else:
                    self.initial_losses[self.iter].append(float('inf')) # Indicate no valid initial loss
                self.final_losses[self.iter].append(result['final_loss'])

        print("\n--- Optimization Complete ---")
        return optimized_models_info

    def __feedback__(self, optimized_models_info):
        """
        Generates structured feedback for the LLM based on the optimization results
        of the proposed models. This feedback guides the LLM in refining future model proposals.
        It also logs detailed results to a file.

        Args:
            optimized_models_info (list): List of dictionaries containing optimization results for each model.
        """
        print("\n--- Generating Feedback for LLM ---")
        feedback_message = "Here is the performance feedback for the recently proposed models:\n\n"

        # Open log file to append results for the current iteration
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"\n--- Iteration {self.iter} ---\n")

            if not optimized_models_info:
                feedback_message += "No models were successfully optimized in the last iteration. Please propose new and diverse models.\n"
                log_file.write(f"  Iteration {self.iter}: No models optimized.\n")
                self.conversation_history.append({"role": "user", "content": feedback_message})
                return

            # Separate models into simulable and unsimulable categories
            simulable_models = [m for m in optimized_models_info if m['simulable'] is True]
            unsimulable_models = [m for m in optimized_models_info if m['simulable'] is False]

            if unsimulable_models:
                feedback_message += "**The following models could NOT be simulated by the ODE solver and were skipped for optimization. Please review their structure, especially the `forward` method and initial parameter values, to ensure they are mathematically well-defined and numerically stable:**\n"
                for model_info in unsimulable_models:
                    model_instance = model_info['model_instance']
                    model_idx = model_info['model_idx']
                    try: # Attempt to dynamically get the class name for clearer feedback
                        class_name_match = re.search(r'class\s+(\w+)\s*\(nn\.Module\):', textwrap.dedent(inspect.getsource(model_instance.__class__)))
                        model_class_name = class_name_match.group(1) if class_name_match else f"ProposedModel_Unsimulable_{model_idx}"
                    except (TypeError, OSError): # Handle cases where source code might not be inspectable
                        model_class_name = f"ProposedModel_Unsimulable_idx{model_idx}"

                    feedback_message += f"- Model {model_class_name} (Index {model_idx})\n"
                    log_file.write(f"  Model (Index: {model_idx}, Class: {model_class_name}): Status=UNSIMULABLE\n")

                feedback_message += "\n"

            if not simulable_models:
                feedback_message += "All proposed models were unsimulable or failed optimization. Please propose new and diverse, *simulable* models.\n"
                log_file.write(f"  Iteration {self.iter}: All models were unsimulable.\n")
                self.conversation_history.append({"role": "user", "content": feedback_message})
                return

            # Track the best model within the current iteration for LLM summary
            current_iteration_best_loss = float('inf')
            current_iteration_best_model_class_name = "N/A"
            current_iteration_best_model_idx = "N/A"


            for i, model_info in enumerate(simulable_models):
                model_instance = model_info['model_instance']
                model_idx = model_info['model_idx']
                final_loss = model_info['final_loss']
                # Extract and format final parameters for logging and feedback
                final_params = {name: f"{param.cpu().item():.6f}" for name, param in model_info['final_params'].items()}

                # Dynamically get class name for detailed feedback
                try:
                    class_name_match = re.search(r'class\s+(\w+)\s*\(nn\.Module\):', textwrap.dedent(inspect.getsource(model_instance.__class__)))
                    model_class_name = class_name_match.group(1) if class_name_match else f"ProposedModel_{model_idx}"
                except (TypeError, OSError):
                    model_class_name = f"ProposedModel_idx{model_idx}"

                # Update best model for the *current iteration* (for LLM's summary)
                if final_loss < current_iteration_best_loss:
                    current_iteration_best_loss = final_loss
                    current_iteration_best_model_class_name = model_class_name
                    current_iteration_best_model_idx = model_idx

                # Update the *overall best model* found across all iterations
                if final_loss < self.current_best_score:
                    self.current_best_model = model_instance
                    self.current_best_score = final_loss

                # Log detailed results for the current model
                log_file.write(
                    f"  Model (Index: {model_idx}, Class: {model_class_name}): "
                    f"Loss={final_loss:.6f}, Params={json.dumps(final_params)}, "
                    f"VarsOrder={model_instance.variables_order}, ObsMap={model_instance.observable_map}\n"
                )

                # Add model-specific feedback to the message for the LLM
                feedback_message += f"--- Model {model_class_name} (Index {model_idx}) ---\n"
                feedback_message += f"  Final Loss: {final_loss:.6f}\n"
                feedback_message += f"  Optimized Parameters: {final_params}\n"
                feedback_message += f"  Variables Order: {model_instance.variables_order}\n"
                feedback_message += f"  Observable Map: {model_instance.observable_map}\n"
                feedback_message += "  Comment: Model optimized and results reported.\n" # A generic comment for simulable models
                feedback_message += "\n"

            # Summary feedback for the LLM after processing all models in the batch
            feedback_message += f"**Summary for this iteration:**\n"
            if current_iteration_best_loss != float('inf'):
                feedback_message += f"The best performing model in this batch was {current_iteration_best_model_class_name} (Index {current_iteration_best_model_idx}) with a loss of {current_iteration_best_loss:.6f}.\n"

            if self.current_best_score == current_iteration_best_loss:
                feedback_message += f"**Overall Best Model Updated!** The new overall best model is {current_iteration_best_model_class_name} (Index {current_iteration_best_model_idx}) with an improved loss of {self.current_best_score:.6f}.\n"
            else:
                feedback_message += f"The overall best model remains with a loss of {self.current_best_score:.6f}.\n"

            feedback_message += "Based on this feedback, please propose a new set of improved or diverse model structures. Aim for models that balance predictive accuracy with mechanistic interpretability.\n"

            log_file.write(f"  Overall Best Loss (to date): {self.current_best_score:.6f}\n")
            log_file.write("-" * 50 + "\n") # Separator for log file

        self.conversation_history.append({"role": "user", "content": feedback_message})
        print("Feedback generated and added to conversation history.")
        # print(feedback_message) # Print the feedback for debugging/monitoring

    def run(self):
        """
        Executes the main loop of the scientific discovery method.
        It iteratively generates models, optimizes them, provides feedback to the LLM,
        and continues until the maximum number of iterations is reached.
        """
        self.iter = 0 # Initialize iteration counter at the start of the run

        # Initialize the log file with general run information
        with open(self.log_file_path, 'w') as log_file:
            log_file.write(f"--- Optimization Run Started ---\n")
            log_file.write(f"Problem: {self.data['problem_description']}\n")
            log_file.write(f"Observable Variables: {', '.join(self.data['observable_variable_descriptions'])}\n")
            log_file.write(f"Max Iterations: {self.iter_max}\n")
            log_file.write(f"Initial Models per Iteration: {self.num_models_to_generate}\n")
            log_file.write("--- Detailed iteration results below ---\n")

        # Initial model generation phase
        print(f"\n--- Iteration {self.iter}: Initial Model Generation ---")
        L_models = self.__start__()

        # Main iterative loop for model refinement
        for i in range(self.iter_max):
            print(f"\n--- Iteration {self.iter}: Optimizing Models ---")
            optimized_models_info = self.__optimize__(L_models)

            print(f"\n--- Iteration {self.iter}: Providing Feedback to LLM ---")
            self.__feedback__(optimized_models_info)

            # Increment iteration counter
            self.iter += 1
            self.__gen_nnum_models__() # Update the number of models to generate for the next LLM call

            # Generate new models if not the last iteration
            if i < self.iter_max - 1:
                print(f"\n--- Iteration {self.iter}: LLM Generating New Proposals ---")
                # Prepare context for the next LLM generation request
                problem_desc = self.data['problem_description']
                observable_variables_desc = self.data['observable_variable_descriptions']
                formatted_obs_vars = ""
                for idx, desc in enumerate(observable_variables_desc):
                    formatted_obs_vars += f"{idx}: {desc}\n"

                # User prompt for the LLM to generate new models based on feedback
                user_prompt_for_new_generation = f"""
                Based on the comprehensive feedback provided, please propose {self.num_models_to_generate} new or refined Python classes for the governing ODEs. Focus on improving performance, addressing any issues (like unsimulability), and exploring diverse mechanistic interpretations.
                Remember to separate each model with `---MODEL_DELIMITER---`.
                """
                self.conversation_history.append({"role": "user", "content": user_prompt_for_new_generation})