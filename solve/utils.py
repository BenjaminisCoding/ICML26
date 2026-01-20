from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt # Imported but not used in this file
from tqdm import tqdm # Duplicate import, remove one

# ODE solver with adjoint support
from torchdiffeq import odeint, odeint_adjoint
from multiprocessing import Process, Queue # Imported but not used directly in current train_model
import threading # Used for simulation timeout
import copy # To copy model parameters, implicitly handled by .clone() for state_dict

# Define a reasonable timeout for the initial simulation test (e.g., 30 seconds)
SIMULATION_TEST_TIMEOUT = 30 # seconds

def read_config(config):
    """
    Reads and parses training configuration parameters from a dictionary.

    Args:
        config (dict): A dictionary containing training configuration settings.

    Returns:
        tuple: A tuple containing:
            - solver_func (callable): The ODE solver function (odeint or odeint_adjoint).
            - lr (float): Learning rate.
            - n_epochs (int): Number of training epochs.
            - normalize (bool): Whether to normalize observed data.
            - es_patience (int): Early stopping patience.
            - es_min_delta (float): Minimum change to qualify as an improvement for early stopping.
            - lr_scheduler_enabled (bool): Whether learning rate scheduler is enabled.
            - lr_scheduler_mode (str): Mode for LR scheduler ('min' or 'max').
            - lr_scheduler_factor (float): Factor by which LR is reduced.
            - lr_scheduler_patience (int): LR scheduler patience.
            - lr_scheduler_min_lr (float): Minimum learning rate.
    """
    # Determine the ODE solver function based on config
    if config['solver'] == 'odeint_adjoint':
        solver_func = odeint_adjoint
    elif config['solver'] == 'odeint':
        solver_func = odeint
    else:
        raise ValueError(
        f"The 'solver' parameter in the config file should be either 'odeint_adjoint' or 'odeint', "
        f"but received '{config['solver']}'."
        )

    lr, n_epochs = config['lr'], config['n_epochs']

    # Early stopping parameters
    es_patience = config['early_stopping']['patience']
    es_min_delta = config['early_stopping']['min_delta']

    # Learning Rate scheduler parameters (with defaults)
    lr_scheduler_config = config.get('lr_scheduler', {})
    lr_scheduler_enabled = lr_scheduler_config.get('enabled', False)
    lr_scheduler_mode = lr_scheduler_config.get('mode', 'min')
    lr_scheduler_factor = lr_scheduler_config.get('factor', 0.5)
    lr_scheduler_patience = lr_scheduler_config.get('patience', 10)
    lr_scheduler_min_lr = lr_scheduler_config.get('min_lr', 1e-6)

    normalize = config['normalize']

    return solver_func, lr, n_epochs, normalize, \
           es_patience, es_min_delta, \
           lr_scheduler_enabled, lr_scheduler_mode, lr_scheduler_factor, lr_scheduler_patience, lr_scheduler_min_lr

def test_ode_system_threaded(ode_func, z0, t, method="dopri5", max_time=SIMULATION_TEST_TIMEOUT):
    """
    Tests if an ODE system can be simulated for a given initial state and time points
    within a specified timeout. Runs the simulation in a separate thread.

    Args:
        ode_func (torch.nn.Module): The ODE function (model) to simulate.
        z0 (torch.Tensor): Initial state for the simulation.
        t (torch.Tensor): Time points for the simulation.
        method (str, optional): Integration method for `odeint`. Defaults to "dopri5".
        max_time (float, optional): Maximum time in seconds to wait for the simulation.

    Returns:
        tuple: (bool, Union[torch.Tensor, str]):
            - True and the simulated trajectory if successful.
            - False and an error message if simulation fails or times out.
    """
    result = {"ok": False, "traj": None, "msg": None}

    def worker():
        """Worker function to run ODE integration in a separate thread."""
        try:
            with torch.no_grad(): # Disable gradient tracking for this initial test
                traj = odeint(ode_func, z0.cpu(), t.cpu(), method=method)
            result["ok"] = True
            result["traj"] = traj
        except Exception as e:
            result["ok"] = False
            result["msg"] = str(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=max_time) # Wait for the thread to complete or timeout

    if thread.is_alive():
        return False, f"Integration exceeded {max_time} seconds"
    if result["ok"]:
        return True, result["traj"]
    else:
        return False, result.get("msg", "Unknown error during simulation")

def train_model(ode_func, t, y_obs, config, observable_map):
    """
    Trains an ODE model using observed time-series data.
    Includes initial simulability testing, early stopping, and an optional learning rate scheduler.

    Args:
        ode_func (torch.nn.Module): An instance of an ODE model with trainable parameters.
        t (torch.Tensor): Time points for simulation.
        y_obs (torch.Tensor): Observed data, typically a 2D tensor (time_points, num_observed_variables).
        config (dict): Dictionary containing training configuration (epochs, lr, optimizer, early_stopping, lr_scheduler).
        observable_map (dict): Dictionary mapping observed data column indices to model state indices.

    Returns:
        tuple: (list, dict, bool):
            - losses (list): List of loss values recorded per epoch.
            - final_params (dict): A dictionary of the model's parameters (name: tensor) at the best observed loss.
            - simulable (bool): True if the model was simulable throughout training and completed successfully, False otherwise.
    """

    # Set initial observed conditions for the model
    # y_obs[0] represents the initial values for the observed variables
    ode_func._set_z0_observed(y_obs[0])
    z0_initial_guess = ode_func._get_initial_state()

    # Test initial simulability of the model before starting training
    flag_initial_simulable, test_result = test_ode_system_threaded(
        ode_func, z0_initial_guess, t, method="dopri5", max_time=SIMULATION_TEST_TIMEOUT
    )

    if not flag_initial_simulable:
        print(f"Initial simulation failed: {test_result}. Model cannot be trained.")
        return [], None, False # Return empty losses, no params, and False for simulable

    # Read configuration parameters
    solver_func, lr, n_epochs, normalize, \
    es_patience, es_min_delta, \
    lr_scheduler_enabled, lr_scheduler_mode, lr_scheduler_factor, lr_scheduler_patience, lr_scheduler_min_lr = read_config(config)

    # Initialize optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(ode_func.parameters(), lr=lr)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(ode_func.parameters(), lr=lr)
    else:
        raise ValueError(
            f"The 'optimizer' parameter in the config file should either be 'Adam' or 'SGD', "
            f"but received '{config['optimizer']}'"
        )

    # Initialize learning rate scheduler if enabled
    scheduler = None
    if lr_scheduler_enabled:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=lr_scheduler_mode,
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min_lr
            )

    pbar = tqdm(range(n_epochs), desc=f"Training Model", leave=False)

    losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    best_ode_func_params = None # To store the model's parameters corresponding to the best loss

    # Deep copy the initial parameters as the current best.
    # This ensures that even if training fails immediately, we have a valid initial state.
    best_ode_func_params = {name: param.data.clone() for name, param in ode_func.named_parameters()}

    # Calculate normalization statistics for observed data if normalization is enabled
    y_obs_means = y_obs.mean(dim=0, keepdim=True)
    y_obs_stds = y_obs.std(dim=0, keepdim=True)
    y_obs_stds[y_obs_stds == 0] = 1e-8 # Prevent division by zero for constant observed variables
    normalized_y_obs = (y_obs - y_obs_means) / y_obs_stds if normalize else y_obs


    for epoch in pbar:
        optimizer.zero_grad()

        # Get the current initial state, which may include learnable hidden initial conditions
        z0 = ode_func._get_initial_state()

        # Re-test for simulability during training. Parameters can drift to unstable regions.
        current_sim_flag, current_sim_result = test_ode_system_threaded(
            ode_func, z0, t, method='dopri5', max_time=SIMULATION_TEST_TIMEOUT
        )
        if not current_sim_flag:
            print(f"\nSimulation failed during epoch {epoch}: {current_sim_result}. Stopping training for this model.")
            # If the model becomes unsimulable, revert to the best known parameters
            if best_ode_func_params:
                for name, param in ode_func.named_parameters():
                    param.data = best_ode_func_params[name]
                return losses, ode_func.params, False # Mark as not fully simulable throughout training
            else:
                return losses, None, False # If no best params were ever recorded (e.g., failed on first sim)

        # Integrate the ODE system
        z_pred_full = solver_func(ode_func, z0, t, method='dopri5')

        # Select only the observable variables from the full predicted trajectory
        # The order of indices ensures correspondence with y_obs
        z_indices_in_model_state = [observable_map[i] for i in sorted(observable_map.keys())]
        z_pred_observable = z_pred_full[:, z_indices_in_model_state]

        # Calculate loss (Mean Squared Error)
        if normalize:
            normalized_z_pred = (z_pred_observable - y_obs_means) / y_obs_stds
            loss = torch.mean((normalized_z_pred - normalized_y_obs)**2)
        else:
            loss = torch.mean((z_pred_observable - y_obs)**2)

        loss.backward() # Backpropagation
        optimizer.step() # Update model parameters
        current_loss = loss.item()
        losses.append(current_loss)

        # Learning Rate Scheduler step
        if scheduler:
            scheduler.step(current_loss)

        pbar.set_postfix({'loss': f'{current_loss:.6f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})

        # Early Stopping logic
        if current_loss < best_loss - es_min_delta:
            best_loss = current_loss
            epochs_no_improve = 0
            # Save a deep copy of the current best parameters
            best_ode_func_params = {name: param.data.clone() for name, param in ode_func.named_parameters()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement for {es_patience} epochs.")
                break

    # After the training loop, load the best parameters (if any were saved) back into the model instance
    if best_ode_func_params:
        for name, param in ode_func.named_parameters():
            param.data = best_ode_func_params[name]

    return losses, ode_func.params, True # Return losses, the best parameters found, and True for successful completion