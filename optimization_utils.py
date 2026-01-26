
import torch
import torch.nn as nn
import torch.optim as optim

class WarfarinModel_true(nn.Module):
    def __init__(self, init_params=None):
        """
        Args:
            init_params (dict): Initial parameter values.
        """
        super().__init__()
        
        # Initialize parameters
        # If init_params is None, we use some defaults (though usually we pass them)
        defaults = {
            'CL': 0.1, 'V1': 1.0, 'Q': 0.5, 'V2': 2.0,
            'kon': 0.01, 'koff': 0.005, 'kdeg': 0.001, 'ksyn': 0.002
        }
        params = init_params if init_params is not None else defaults

        self.CL = nn.Parameter(torch.tensor(params['CL']))
        self.V1 = nn.Parameter(torch.tensor(params['V1']))
        self.Q  = nn.Parameter(torch.tensor(params['Q']))
        self.V2 = nn.Parameter(torch.tensor(params['V2']))
        self.kon = nn.Parameter(torch.tensor(params['kon']))
        self.koff = nn.Parameter(torch.tensor(params['koff']))
        self.kdeg = nn.Parameter(torch.tensor(params['kdeg']))
        self.ksyn = nn.Parameter(torch.tensor(params['ksyn']))

    def forward(self, state):
        x1 = state[..., 0]
        x2 = state[..., 1]
        x3 = state[..., 2]
        x4 = state[..., 3]

        dx1dt = -self.CL/self.V1 * x1 - self.Q/self.V1 * x1 + self.Q/self.V2 * x2 - self.kon * x1 * x4 + self.koff * x3
        dx2dt = self.Q/self.V1 * x1 - self.Q/self.V2 * x2
        dx3dt = self.kon * x1 * x4 - self.koff * x3 - self.kdeg * x3
        dx4dt = -self.kon * x1 * x4 + self.koff * x3 + self.ksyn - self.kdeg * x4

        return torch.stack([dx1dt, dx2dt, dx3dt, dx4dt], dim=-1)



class WarfarinModel_miss(nn.Module):
    def __init__(self, init_params=None):
        """
        Args:
            init_params (dict): Initial parameter values.
        """
        super().__init__()
        
        # Initialize parameters
        # If init_params is None, we use some defaults (though usually we pass them)
        defaults = {
            'CL': 0.1, 'V1': 1.0, 'Q': 0.5, 'V2': 2.0,
            'kon': 0.01, 'koff': 0.005, 'kdeg': 0.001, 'ksyn': 0.002
        }
        params = init_params if init_params is not None else defaults

        self.CL = nn.Parameter(torch.tensor(params['CL']))
        self.V1 = nn.Parameter(torch.tensor(params['V1']))
        self.Q  = nn.Parameter(torch.tensor(params['Q']))
        self.V2 = nn.Parameter(torch.tensor(params['V2']))
        self.kon = nn.Parameter(torch.tensor(params['kon']))
        #self.koff = nn.Parameter(torch.tensor(params['koff']))
        self.kdeg = nn.Parameter(torch.tensor(params['kdeg']))
        self.ksyn = nn.Parameter(torch.tensor(params['ksyn']))

    def forward(self, state):
        x1 = state[..., 0]
        x2 = state[..., 1]
        x3 = state[..., 2]
        x4 = state[..., 3]

        dx1dt = -self.CL/self.V1 * x1 - self.Q/self.V1 * x1 + self.Q/self.V2 * x2 - self.kon * x1 * x4 #+ self.koff * x3
        dx2dt = self.Q/self.V1 * x1 - self.Q/self.V2 * x2
        dx3dt = self.kon * x1 * x4 - self.kdeg * x3 #- self.koff * x3
        dx4dt = -self.kon * x1 * x4 + self.ksyn - self.kdeg * x4 #+ self.koff * x3

        return torch.stack([dx1dt, dx2dt, dx3dt, dx4dt], dim=-1)



def integrate_euler(model, x0, t_points):
    trajectory = [x0]
    x = x0
    dt = t_points[1] - t_points[0]
    
    for i in range(len(t_points) - 1):
        dx = model(x)
        x = x + dx * dt
        trajectory.append(x)
    
    return torch.stack(trajectory)

def run_homotopy_optimization(model, observed_data, obs_indices, time_points, 
                              tau_schedule=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                              steps_per_tau=1000,
                              verbose=True,
                              system_dim=4,
                              optimize_params=True): # New Argument
    """
    Runs PODS Homotopy Optimization.
    Optimizes both parameters AND trajectory X_est.
    Returns: X_est, loss_history, trajectory_history
    """
    dt = time_points[1] - time_points[0]
    T, D = observed_data.shape[0], system_dim
    
    # 1. Initialize Trajectory Estimate (X_est)
    X_init = torch.zeros(T, D)
    X_init[:, obs_indices] = observed_data
    
    # Defaults for hidden
    all_dims = set(range(D))
    hidden_dims = list(all_dims - set(obs_indices))
    # Simple defaults to avoid 0
    if 2 in hidden_dims: X_init[:, 2] = 0.5
    if 3 in hidden_dims: X_init[:, 3] = 4.0
    if 1 in hidden_dims: X_init[:, 1] = 0.5
    
    X_est = nn.Parameter(X_init.clone())
    
    # Optimizer
    param_groups = [{'params': [X_est], 'lr': 0.05}]
    
    if optimize_params:
        param_groups.append({'params': model.parameters(), 'lr': 0.001})
        if verbose: print("Optimization Mode: Trajectory + Parameters")
    else:
        if verbose: print("Optimization Mode: Trajectory ONLY (Fixed Params)")
    
    optimizer = optim.Adam(param_groups, lr=0.01)
    
    loss_history = []
    trajectory_history = {} # Key: tau, Value: snapshot of X_est
    
    if verbose: print(f"--- Starting Homotopy Optimization ---")

    for tau in tau_schedule:
        if verbose: print(f"Tau: {tau}")
        for step in range(steps_per_tau):
            optimizer.zero_grad()
            
            # Obs Loss
            loss_obs = torch.mean((X_est[:, obs_indices] - observed_data) ** 2)
            
            # Dyn Loss
            X_curr = X_est[:-1]
            X_next_pred = X_curr + model(X_curr) * dt
            loss_dyn = torch.mean((X_est[1:] - X_next_pred) ** 2)
            
            loss = loss_obs + tau * loss_dyn
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        
        # Save snapshot
        trajectory_history[tau] = X_est.detach().clone().numpy()

    return X_est.detach(), loss_history, trajectory_history


def run_shooting_optimization(model, observed_data, obs_indices, time_points,
                              steps=2000, verbose=True):
    """
    Runs Shooting Optimization.
    Optimizes ONLY parameters, assumes fixed IC guess.
    """
    # 1. Fixed Initial Condition Guess
    x0_guess = torch.zeros(4)
    x0_guess[obs_indices] = observed_data[0]
    # Guess for hidden
    if 1 not in obs_indices: x0_guess[1] = 0.5
    if 2 not in obs_indices: x0_guess[2] = 0.5
    if 3 not in obs_indices: x0_guess[3] = 4.0
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_history = []
    
    if verbose: print(f"--- Starting Shooting Optimization ---")

    for step in range(steps):
        optimizer.zero_grad()
        
        # Integrate
        X_pred = integrate_euler(model, x0_guess, time_points)
        
        # Loss
        loss = torch.mean((X_pred[:, obs_indices] - observed_data) ** 2)
        
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    return integrate_euler(model, x0_guess, time_points).detach(), loss_history

def calc_param_error(model, true_params):
    total_err = 0
    count = 0
    for name, param in model.named_parameters():
        if name in true_params:
            true_val = true_params[name]
            est_val = param.item()
            err = abs(est_val - true_val) / true_val * 100
            total_err += err
            count += 1
    return total_err / count if count > 0 else 0.0

def run_shooting_free_x0(model, observed_data, obs_indices, time_points,
                         steps=2000, verbose=True):
    """
    Runs Shooting Optimization with FREE Initial Condition.
    Optimizes parameters AND x0_est.
    """
    # 1. Initial Condition Guess (Learnable)
    x0_guess = torch.zeros(4)
    x0_guess[obs_indices] = observed_data[0]
    # Guess for hidden
    if 1 not in obs_indices: x0_guess[1] = 0.5
    if 2 not in obs_indices: x0_guess[2] = 0.5
    if 3 not in obs_indices: x0_guess[3] = 4.0
    
    # Make x0 a parameter
    x0_est = nn.Parameter(x0_guess.clone())
    
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.01},
        {'params': [x0_est], 'lr': 0.05} 
    ], lr=0.01)
    
    loss_history = []
    
    if verbose: print(f"--- Starting Shooting (Free x0) Optimization ---")

    for step in range(steps):
        optimizer.zero_grad()
        
        # Integrate from ESTIMATED x0
        X_pred = integrate_euler(model, x0_est, time_points)
        
        # Loss
        loss = torch.mean((X_pred[:, obs_indices] - observed_data) ** 2)
        
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    return integrate_euler(model, x0_est, time_points).detach(), loss_history
