import numpy as np
import torch

def prepare_data_for_method(dataset_obj, observable_indices, noise_level=0.0, sparsity_ratio=0.0, noise_type="additive"):
    """
    Takes a generated Dataset object and processes it for the ADUD (Automated Discovery of Unknown Dynamics) method
    by applying partial observability, noise, and optional sparsity.

    Args:
        dataset_obj (Dataset): An instance of a Dataset subclass that has already generated its data.
        observable_indices (list): A list of integer indices indicating which variables from the full system are observable.
                                   e.g., [0] for only the first variable, [0, 2] for first and third.
        noise_level (float): The magnitude of noise to add. Interpretation depends on `noise_type`.
                             For 'additive', it's std dev. for 'multiplicative' and 'log-normal', it's a factor.
        sparsity_ratio (float): The fraction of observed data points to randomly drop (set to NaN).
        noise_type (str): Type of noise to apply. Can be 'additive', 'multiplicative', or 'log-normal'.

    Returns:
        dict: A dictionary containing:
            'observed_data': A torch.Tensor of the partially observed and processed data.
            'time_points': The time points corresponding to the full dataset (torch.Tensor).
            'observable_variable_descriptions': List of descriptions for the observed variables.
            'problem_description': The overall description of the dynamical system problem.
            'full_system_data': The complete, unadulterated data from the generator (for ground truth comparison).
    """
    if dataset_obj.variables is None:
        raise ValueError("Dataset object has not generated its variables yet. Call .generate() first.")

    full_data = dataset_obj.get_full_data()
    time_points = dataset_obj.get_time_points()
    full_variable_descriptions = dataset_obj.get_variables_description()

    # Apply partial observability: select only the specified observable variables
    if not observable_indices:
        raise ValueError("At least one variable must be observable.")

    observed_data = full_data[:, observable_indices]
    observed_variable_descriptions = [full_variable_descriptions[i] for i in observable_indices]

    # Add noise based on the specified type and level
    if noise_level > 0:
        if noise_type == 'additive':
            # Additive noise scaled by the mean of the observed data
            # This ensures noise magnitude is relative to the data's scale
            data_scale = torch.mean(observed_data)
            # noise_std_dev = noise_level * data_scale # This line might be redundant if noise_level is already scaled
            noise = torch.normal(0, noise_level, observed_data.shape) # noise_level is directly used as std dev
            observed_data = observed_data + noise
        elif noise_type == "multiplicative":
            # Multiplicative noise: observed_data * (1 + noise_factor)
            # noise_factor is drawn from a Gaussian distribution with mean 0 and std_dev = noise_level
            multiplicative_factor = torch.normal(0, noise_level, observed_data.shape)
            observed_data = observed_data * (1 + multiplicative_factor)
        elif noise_type == "log-normal":
            # Log-normal noise: exp(noise) where noise is Gaussian, then multiplied by data
            noise = torch.normal(0, noise_level, observed_data.shape)
            multiplicative_noise_factor = torch.exp(noise) # Changed np.exp to torch.exp for consistency
            observed_data = observed_data * multiplicative_noise_factor
        else:
            raise ValueError("noise_type must be 'additive', 'multiplicative', or 'log-normal'.")

    # Introduce sparsity by randomly setting a fraction of data points to NaN
    if sparsity_ratio > 0:
        total_points = observed_data.shape[0] * observed_data.shape[1]
        num_to_drop = int(total_points * sparsity_ratio)

        # Generate random indices to drop, ensuring no replacement for unique NaN positions
        flat_indices = np.random.choice(total_points, num_to_drop, replace=False)
        row_indices, col_indices = np.unravel_index(flat_indices, observed_data.shape)

        # Set selected data points to NaN (Not a Number)
        observed_data[row_indices, col_indices] = torch.nan # Changed np.nan to torch.nan for consistency

    return {
        'observed_data': observed_data,
        'time_points': time_points,
        'observable_variable_descriptions': observed_variable_descriptions,
        'problem_description': dataset_obj.get_description(),
        'full_system_data': full_data # Keep full data for ground truth comparison
    }