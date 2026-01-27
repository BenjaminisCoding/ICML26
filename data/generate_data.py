"""
This file defines a base Dataset class and specific subclasses for generating
time-series data from various dynamical systems using `torchdiffeq`.
It supports Lorenz, Warfarin TMDD, and Macro RBC models.
"""

import numpy as np # Used for linspace initially, but later replaced by torch.linspace
from torchdiffeq import odeint, odeint_adjoint # ODE integrators
import torch

# Conditional import for `prepare_data_for_method` based on execution context
if __name__ == "__main__":
    from utils import prepare_data_for_method
else:
    from data.utils import prepare_data_for_method

class Dataset:
    """
    Base class for dynamical system datasets.
    Provides a common interface for defining system dynamics and generating time-series data.
    Subclasses must implement `dynamics` and `generate` methods.
    """
    def __init__(self, generation_parameters=None):
        """
        Initializes the base dataset with common attributes.

        Args:
            generation_parameters (dict, optional): Dictionary of parameters
                                                   specific to the dynamics generation. Defaults to None.
        """
        self.variables = None  # Stores the generated time-series data (torch.Tensor)
        self.description = "Base dataset for a dynamical system." # General description of the system
        self.variables_description = [] # List of text descriptions for each state variable
        self.generation_parameters = generation_parameters if generation_parameters is not None else {}
        self.time_points = None # Array of time points (torch.Tensor)

    def dynamics(self, t, state):
        """
        Abstract method that describes the dynamics of the system (e.g., dx/dt = f(x, t)).
        This method must be implemented by subclasses.

        Args:
            t (float): Current time.
            state (torch.Tensor): Current state variables.

        Returns:
            torch.Tensor: Derivatives of the state variables (dx/dt, dy/dt, ...).
        """
        raise NotImplementedError("Subclasses must implement the 'dynamics' method.")


    def generate(self):
        """
        Abstract method to generate the time-series data by integrating the ODEs.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the 'generate' method.")

    def get_full_data(self):
        """
        Returns the full generated dataset (all variables).

        Returns:
            torch.Tensor: The generated time-series data.
        """
        return self.variables

    def get_description(self):
        """
        Returns the text description of the problem.

        Returns:
            str: The problem description.
        """
        return self.description

    def get_variables_description(self):
        """
        Returns a list of text descriptions for each variable.

        Returns:
            list: A list of strings, where each string describes a variable.
        """
        return self.variables_description

    def get_time_points(self):
        """
        Returns the time points at which the data was generated.

        Returns:
            torch.Tensor: Array of time points.
        """
        return self.time_points

class WarfarinTMDDDataset(Dataset):
    """
    Dataset class for the Warfarin TMDD (Target-Mediated Drug Disposition) system.
    This model describes pharmacokinetics where a drug (Warfarin) interacts reversibly
    with a specific biological target.
    """
    def __init__(self, generation_parameters=None):
        """
        Initializes the Warfarin TMDD system with default or provided parameters.

        Args:
            generation_parameters (dict, optional): Dictionary containing pharmacokinetic
                                                   parameters (CL, V1, Q, V2, kon, koff, kdeg, ksyn),
                                                   'initial_state', 't_start', 't_end', 'num_points'.
                                                   Defaults to None.
        """
        super().__init__(generation_parameters)
        self.description = "This system models the pharmacokinetics of Warfarin, an anticoagulant, within a biological system. It is characterized by Target-Mediated Drug Disposition (TMDD), where Warfarin interacts reversibly with a specific biological target. The model involves the drug's distribution between various compartments, elimination processes, and the dynamics of both free and target-bound Warfarin, as well as the free target itself. The exact number and nature of pharmacokinetic compartments may need to be inferred."
        self.variables_description = [
            "x1: Represents the concentration or amount of free Warfarin in a primary, central pharmacokinetic compartment (e.g., blood plasma).",
            "x2: Represents the concentration or amount of free Warfarin in a secondary pharmacokinetic compartment (e.g., peripheral tissues), which exchanges with the central compartment.",
            "x3: Represents the concentration or amount of Warfarin that is bound to its specific biological target.",
            "x4: Represents the concentration or amount of the free (unoccupied) biological target to which Warfarin can bind."
        ]

        # Default Warfarin TMDD parameters; can be overridden
        self.CL = self.generation_parameters.get('CL', 0.1)  # Clearance rate
        self.V1 = self.generation_parameters.get('V1', 1.0)  # Volume of central compartment
        self.Q = self.generation_parameters.get('Q', 0.5)   # Intercompartmental clearance
        self.V2 = self.generation_parameters.get('V2', 2.0)  # Volume of peripheral compartment
        self.kon = self.generation_parameters.get('kon', 0.01) # Association rate constant (drug-target binding)
        self.koff = self.generation_parameters.get('koff', 0.005) # Dissociation rate constant (drug-target unbinding)
        self.kdeg = self.generation_parameters.get('kdeg', 0.001) # Degradation rate constant (for bound drug and target)
        self.ksyn = self.generation_parameters.get('ksyn', 0.002) # Synthesis rate constant for the biological target

        self.initial_state = torch.tensor(self.generation_parameters.get('initial_state', [10.0, 0.0, 0.0, 5.0]), dtype=torch.float32) # Initial conditions
        self.t_start = self.generation_parameters.get('t_start', 0)
        self.t_end = self.generation_parameters.get('t_end', 200) # Longer simulation for drug dynamics
        self.num_points = self.generation_parameters.get('num_points', 2000)

    def dynamics(self, t, state):
        """
        Warfarin TMDD system dynamics implementation.

        Args:
            t (torch.Tensor): Current time.
            state (torch.Tensor): Current state variables [x1, x2, x3, x4].

        Returns:
            torch.Tensor: Derivatives [dx1/dt, dx2/dt, dx3/dt, dx4/dt].
        """
        x1, x2, x3, x4 = state

        # Dynamics for free drug in central compartment (x1)
        dx1dt = -self.CL/self.V1 * x1 - self.Q/self.V1 * x1 + self.Q/self.V2 * x2 \
                - self.kon * x1 * x4 + self.koff * x3
        # Dynamics for free drug in peripheral compartment (x2)
        dx2dt = self.Q/self.V1 * x1 - self.Q/self.V2 * x2
        # Dynamics for target-bound drug (x3)
        dx3dt = self.kon * x1 * x4 - self.koff * x3 - self.kdeg * x3
        # Dynamics for free target (x4)
        dx4dt = -self.kon * x1 * x4 + self.koff * x3 + self.ksyn - self.kdeg * x4

        return torch.stack([dx1dt, dx2dt, dx3dt, dx4dt])

    def generate(self):
        """
        Generates the Warfarin TMDD time-series data by integrating the ODEs.
        """
        self.time_points = torch.linspace(self.t_start, self.t_end, self.num_points, dtype=torch.float32)
        self.variables = odeint(self.dynamics, self.initial_state, self.time_points)
        return self.variables


class CancerDataset(Dataset):
    """
    Dataset class for the Cancer Growth and Resistance model.
    Models tumor growth, drug effect, and resistance evolution.
    """
    def __init__(self, generation_parameters=None):
        super().__init__(generation_parameters)
        self.description = """
        This system models tumor growth under drug treatment, and we observe the concentration of the drug and the tumor size.
        Note that for this system, we apply a regular treatment D that we will add to your discovered equation on the concentration.
        To this end, ensure the first equation in your proposal is always the one describing the evolution of the concentration of the drug.
        """
        self.variables_description = [
            "x1: Drug Concentration (C)",
            "x2: Resistance Level (R)",
            "x3: Tumor Size (E)"
        ]
        
        # Parameters
        self.k_el = self.generation_parameters.get('k_el', 0.5)
        self.alpha = self.generation_parameters.get('alpha', 0.2)
        self.beta = self.generation_parameters.get('beta', 0.1)
        self.r_growth = self.generation_parameters.get('r_growth', 0.8)
        self.K = self.generation_parameters.get('K', 10.0)
        self.E_max = self.generation_parameters.get('E_max', 1.0)
        self.EC50 = self.generation_parameters.get('EC50', 0.5)
        self.D = self.generation_parameters.get('D', 0.5) # Treatment dose Magnitude
        self.treatment_interval = self.generation_parameters.get('treatment_interval', 2.0) # Cycle length
        self.treatment_duration = self.generation_parameters.get('treatment_duration', 1.0) # Duration of dose within cycle
        self.treatment_start_time = self.generation_parameters.get('treatment_start_time', 0.0)
        
        self.initial_state = torch.tensor(self.generation_parameters.get('initial_state', [1.0, 0.0, 1.0]), dtype=torch.float32)
        self.t_start = self.generation_parameters.get('t_start', 0)
        self.t_end = self.generation_parameters.get('t_end', 50)
        self.num_points = self.generation_parameters.get('num_points', 200)

    def dynamics(self, t, state):
        C, R, E = state
        
        # Swish
        # x / (1 + exp(-x)) -> x * sigmoid(x)
        def swish(x):
            return x * torch.sigmoid(x)
            
        # Pulsed Treatment Logic
        time_since_start = t - self.treatment_start_time
        if time_since_start >= 0:
            cycle_pos = time_since_start % self.treatment_interval
            if cycle_pos < self.treatment_duration:
                effective_D = self.D
            else:
                effective_D = 0.0
        else:
            effective_D = 0.0

        dCdt = -self.k_el * C + effective_D
        
        induction = self.alpha * swish(C)
        decay = self.beta * torch.tanh(R**3)
        dRdt = induction - decay
        
        growth_term = self.r_growth * E * (1 - E / self.K)
        denominator = self.EC50 + C * (1 + R)
        killing_rate = (self.E_max * C) / denominator
        dEdt = growth_term - (killing_rate * E)
        
        return torch.stack([dCdt, dRdt, dEdt])

    def generate(self):
        self.time_points = torch.linspace(self.t_start, self.t_end, self.num_points, dtype=torch.float32)
        self.variables = odeint(self.dynamics, self.initial_state, self.time_points)
        return self.variables


class AlienDataset(Dataset):
    """
    Dataset class for the Alien Binding system.
    Features a specific non-linear 'crowding' binding interaction: binding ~ C * R * exp(-alpha * C).
    """
    def __init__(self, generation_parameters=None):
        super().__init__(generation_parameters)
        self.description = "A pharmacodynamic system with a non-linear 'Alien' binding mechanism where binding efficiency decreases with drug crowding (C * R * exp(-alpha * C)). Variables: C (Free Drug), R (Free Receptor), RC (Drug-Receptor Complex)."
        self.variables_description = [
            "x1: Free Drug Concentration (C)",
            "x2: Free Receptor Concentration (R)",
            "x3: Drug-Receptor Complex (RC)"
        ]
        
        # Parameters
        self.k_el = self.generation_parameters.get('k_el', 0.1)
        self.k_syn = self.generation_parameters.get('k_syn', 2.0)
        self.k_deg = self.generation_parameters.get('k_deg', 0.5)
        self.k_int = self.generation_parameters.get('k_int', 0.3)
        self.alpha = self.generation_parameters.get('alpha', 0.2)
        self.k_bind_base = self.generation_parameters.get('k_bind_base', 1.5)
        
        self.initial_state = torch.tensor(self.generation_parameters.get('initial_state', [10.0, 1.0, 0.0]), dtype=torch.float32)
        self.t_start = self.generation_parameters.get('t_start', 0)
        self.t_end = self.generation_parameters.get('t_end', 50)
        self.num_points = self.generation_parameters.get('num_points', 200)

    def dynamics(self, t, state):
        C, R, RC = state
        
        # Alien binding rate
        binding_flux = self.k_bind_base * C * R * torch.exp(-self.alpha * C)
        # binding_flux = self.k_bind_base * C * R
        # Standard dissociation
        dissociation_flux = 0.1 * RC
        
        dCdt = -self.k_el * C - binding_flux + dissociation_flux
        dRdt = self.k_syn - self.k_deg * R - binding_flux + dissociation_flux
        dRCdt = binding_flux - dissociation_flux - self.k_int * RC
        
        return torch.stack([dCdt, dRdt, dRCdt])

    def generate(self):
        self.time_points = torch.linspace(self.t_start, self.t_end, self.num_points, dtype=torch.float32)
        self.variables = odeint(self.dynamics, self.initial_state, self.time_points)
        return self.variables


class PreyPredatorDataset(Dataset):
    """
    Dataset for Prey-Predator system with Fear-Driven Infection.
    Variables: N (Total Prey), I (Infected Prey), P (Predators).
    """
    def __init__(self, generation_parameters=None):
        super().__init__(generation_parameters)
        #self.description = "Prey-Predator system with disease. Predators eat prey (preferentially infected ones?) and induce fear which increases infection rate. Variables: N (Total Prey), I (Infected Prey), P (Predators)."
        # self.description = "prey-Predator system"
        self.dexription = """ 
            You are observing a population of prey and the predators that eat them. The environment can affect the dynamics of their populations.
        """
        self.variables_description = [
            "x1: Total Prey Population (N)",
            "x2: Infected Prey Population (I)",
            "x3: Predator Population (P)"
        ]
        
        # Parameters
        self.r = self.generation_parameters.get('r', 0.5)      # Prey reproduction
        self.K = self.generation_parameters.get('K', 100.0)    # Carrying capacity
        self.a = self.generation_parameters.get('a', 0.02)     # Predation efficiency
        self.mu = self.generation_parameters.get('mu', 0.05)   # Disease death rate
        self.beta = self.generation_parameters.get('beta', 0.02) # Infection rate
        self.alpha = self.generation_parameters.get('alpha', 0.1) # FEAR FACTOR
        self.e = self.generation_parameters.get('e', 0.5)      # Predator conversion
        self.d = self.generation_parameters.get('d', 0.1)      # Predator death
        
        self.initial_state = torch.tensor(self.generation_parameters.get('initial_state', [100.0, 1.0, 10.0]), dtype=torch.float32)
        self.t_start = self.generation_parameters.get('t_start', 0)
        self.t_end = self.generation_parameters.get('t_end', 200)
        self.num_points = self.generation_parameters.get('num_points', 2000)

    def dynamics(self, t, state):
        N, I, P = state
        
        # Logic check: I cannot exceed N
        # In ODE integration, we use soft constraints or rely on dynamics.
        # But if the user code had `if I > N: I = N`, we can mimic that or just let it flow.
        # Strict clamp inside ODE might be unstable for gradients, but ok for generation.
        # N = torch.clamp(N, min=1e-6) # Avoid division by zero if N=0
        
        S = N - I
        
        # dNdt = r*S*(1 - N/K) - (a*N*P) - (mu*I)
        # Note: Deaths from predation (a*N*P) and disease (mu*I) remove from N.
        term_birth = self.r * S * (1 - N / self.K)
        term_predation = self.a * N * P
        term_disease_death = self.mu * I
        
        dNdt = term_birth - term_predation - term_disease_death
        
        # dIdt = (beta * S * I * (1 + alpha * P)) - (a * I * P) - (mu * I)
        # Infection spreads, potentially boosted by fear (alpha*P)
        term_infection = self.beta * S * I * (1 + self.alpha * P)
        term_predation_I = self.a * I * P # Predators eat Infected too (assumed proportional or same rate 'a' on 'I'?)
        # User code: dIdt = ... - (a * I * P) - ...
        # Wait, if a*N*P is total predation, and I is part of N.
        # If predation is random, rate on I is a*I*P? Yes.
        
        dIdt = term_infection - term_predation_I - term_disease_death
        
        # dPdt = (e * a * N * P) - (d * P)
        dPdt = (self.e * term_predation) - (self.d * P)
        
        return torch.stack([dNdt, dIdt, dPdt])

    def generate(self):
        self.time_points = torch.linspace(self.t_start, self.t_end, self.num_points, dtype=torch.float32)
        self.variables = odeint(self.dynamics, self.initial_state, self.time_points)
        return self.variables


class PreyPredatorHibernationDataset(Dataset):
    """
    Dataset class for the Prey-Predator Hibernation system.
    Logic: Predators only wake up when N > H (Holling Type III activation).
    Variables: N (Prey), P (Predator).
    """
    def __init__(self, generation_parameters=None):
        super().__init__(generation_parameters)
        self.description = "Measurements of the population of a prey in an environment."
        self.variables_description = [
            "x1: Prey Population (N)",
            "x2: Predator Population (P)"
        ]
        
        # Parameters
        self.r = self.generation_parameters.get('r', 1.0)
        self.K = self.generation_parameters.get('K', 2000.0)
        self.a = self.generation_parameters.get('a', 10.0)
        self.e = self.generation_parameters.get('e', 0.1)
        self.d = self.generation_parameters.get('d', 0.8)
        self.H = self.generation_parameters.get('H', 50.0)
        
        self.initial_state = torch.tensor(self.generation_parameters.get('initial_state', [5.0, 15.0]), dtype=torch.float32)
        self.t_start = self.generation_parameters.get('t_start', 0)
        self.t_end = self.generation_parameters.get('t_end', 300)
        self.num_points = self.generation_parameters.get('num_points', 3000)

    def dynamics(self, t, state):
        N, P = state
        
        # Activation (Holling Type III)
        activation = (N**2) / (self.H**2 + N**2)
        
        # dNdt = r*N*(1 - N/K) - (a * activation * P)
        dNdt = self.r * N * (1 - N / self.K) - (self.a * activation * P)
        
        # dPdt = (e * a * activation * P) - (d * P)
        dPdt = (self.e * self.a * activation * P) - (self.d * P)
        
        return torch.stack([dNdt, dPdt])

    def generate(self):
        self.time_points = torch.linspace(self.t_start, self.t_end, self.num_points, dtype=torch.float32)
        self.variables = odeint(self.dynamics, self.initial_state, self.time_points)
        return self.variables

# --- Example Usage (for testing and visualization) ---
if __name__ == '__main__':
    print("--- Testing Lorenz Dataset Generation ---")

    
    # --- Conditional Testing for other datasets ---
    # Set these flags to True to test their respective dataset generation and preparation
    warfarin_test = True

    if warfarin_test:
        print("\n--- Testing Warfarin TMDD Dataset Generation ---")
        # Example parameters for Warfarin TMDD system
        warfarin_params = {
            'CL': 0.1, 'V1': 1.0, 'Q': 0.5, 'V2': 2.0,
            'kon': 0.01, 'koff': 0.005, 'kdeg': 0.001, 'ksyn': 0.002,
            'initial_state': [10.0, 0.0, 0.0, 5.0],
            't_start': 0,
            't_end': 50,
            'num_points': 200
        }
        warfarin_generator = WarfarinTMDDDataset(generation_parameters=warfarin_params)
        full_warfarin_data = warfarin_generator.generate()

        print(f"Full Warfarin TMDD data shape: {full_warfarin_data.shape}")
        print(f"Warfarin TMDD system description: {warfarin_generator.get_description()}")
        print(f"Warfarin TMDD variables description: {warfarin_generator.get_variables_description()}")
        print(f"First 5 time points: {warfarin_generator.get_time_points()[:5]}")
        print(f"First 5 rows of full data:\n{full_warfarin_data[:5]}")

        print("\n--- Preparing Partially Observed Warfarin TMDD Data ---")

        # Scenario: Only 'x1' (central compartment) and 'x3' (target-bound drug) observed, with noise
        prepared_warfarin_data = prepare_data_for_method(
            warfarin_generator,
            observable_indices=[0, 2], # Observe x1 and x3
            noise_level=0.1,       # Add 10% additive Gaussian noise
            sparsity_ratio=0,      # No sparsity
            noise_type='additive'
        )

        observed_data_wf = prepared_warfarin_data['observed_data']
        time_points_wf = prepared_warfarin_data['time_points']
        obs_var_desc_wf = prepared_warfarin_data['observable_variable_descriptions']
        prob_desc_wf = prepared_warfarin_data['problem_description']
        gt_data_wf = prepared_warfarin_data['full_system_data']

        print(f"Observed Warfarin data shape: {observed_data_wf.shape}")
        print(f"Observable Warfarin variables description: {obs_var_desc_wf}")
        print(f"First 5 rows of observed Warfarin data:\n{observed_data_wf[:5]}")

        # Plotting Warfarin TMDD observed data
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(time_points_wf.numpy(), gt_data_wf[:, 0].numpy(), 'b-', label='Ground Truth x1 (Central)', alpha=0.5)
        observed_x1 = observed_data_wf[~torch.isnan(observed_data_wf[:,0]), 0]
        observed_time_x1 = time_points_wf[~torch.isnan(observed_data_wf[:,0])]
        plt.scatter(observed_time_x1.numpy(), observed_x1.numpy(), s=5, c='r', label='Observed x1 (with noise)')
        plt.title('Warfarin TMDD System: Partially Observed x1 (Central Compartment)')
        plt.xlabel('Time')
        plt.ylabel('x1 Value')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_points_wf.numpy(), gt_data_wf[:, 2].numpy(), 'g-', label='Ground Truth x3 (Target-Bound)', alpha=0.5)
        # Note: observed_data_wf has only 2 columns, so index 1 refers to the second observed variable, which is x3 here
        observed_x3 = observed_data_wf[~torch.isnan(observed_data_wf[:,1]), 1]
        observed_time_x3 = time_points_wf[~torch.isnan(observed_data_wf[:,1])]
        plt.scatter(observed_time_x3.numpy(), observed_x3.numpy(), s=5, c='m', label='Observed x3 (with noise)')
        plt.title('Warfarin TMDD System: Partially Observed x3 (Target-Bound Drug)')
        plt.xlabel('Time')
        plt.ylabel('x3 Value')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        print("Plot generated for Warfarin TMDD observed data.")

    