from data.generate_data import WarfarinTMDDDataset
from data.utils import prepare_data_for_method
from solve.method import Method
import os
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Set the maximum number of iterations for the LLM-driven discovery process
    iter_max = 3

    # Define the path to the dataset configuration file
    config_file_path = os.path.join(os.path.dirname(__file__), 'data', 'config_WarfarinTMDD.json')
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found at {config_file_path}. Please create it.")

    # Load dataset configuration
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    # Initialize and generate the dataset (e.g., Warfarin TMDD)
    data_generator = WarfarinTMDDDataset(config)
    data_generator.generate()

    # Prepare the generated data for the `Method`
    # This includes specifying observable variables, noise level, and noise type.
    # For example, observing the first two variables (indices 0 and 1) with 2% multiplicative noise.
    data_prep = prepare_data_for_method(data_generator, [0, 1], noise_level=0.02, noise_type='multiplicative')

    # Define specific instructions for the LLM to guide its model generation.
    # This emphasizes the need for latent variables and encourages imaginative dynamics.
    instructions = '''
You only observe two variables from this dataset. However, you know that there are variables you don't have access to.
Really important: Only propose models with 3 state variables or more, so the attribute self.variables_order should always have at least 3 variables. For the functions, be imaginative and reason to find potential form.
Explore various structures.
'''

    # Initialize the `Method` with the prepared data, max iterations, and LLM instructions.
    # The noise level is passed to ensure the log file name reflects the experiment conditions.
    method = Method(data_prep, iter_max, instructions, noise_level=0.02)
    method.run() # Execute the LLM-driven scientific discovery process

    '''
    Code to plot the best initial and final losses over iterations.
    This visualization helps in understanding the convergence and improvement of models
    generated and optimized by the LLM over successive iterations.
    '''
    initial_losses_plot = [] # Stores the initial loss of the best model for each iteration
    final_losses_plot = []   # Stores the final (optimized) loss of the best model for each iteration
    best_overall_loss = np.inf # Tracks the minimum loss found across all iterations

    # Iterate through the recorded losses from each `method` iteration
    for i in range(method.iter):
        if i in method.final_losses and method.final_losses[i]:
            # Find the best final loss in the current iteration
            current_iter_best_loss = np.min(method.final_losses[i])
            # Find the index of that best loss to get its corresponding initial loss
            arg_min_loss = np.argmin(method.final_losses[i])

            # Update the overall best loss if the current iteration's best is lower
            if current_iter_best_loss <= best_overall_loss:
                best_overall_loss = current_iter_best_loss
                final_losses_plot.append(current_iter_best_loss)
                initial_losses_plot.append(method.initial_losses[i][arg_min_loss])
            else:
                # If no improvement, append the previous best losses to show stagnation or lack of new best
                final_losses_plot.append(final_losses_plot[-1] if final_losses_plot else float('inf'))
                initial_losses_plot.append(initial_losses_plot[-1] if initial_losses_plot else float('inf'))
        else:
            # Handle cases where an iteration might not have any valid losses (e.g., all models unsimulable)
            final_losses_plot.append(final_losses_plot[-1] if final_losses_plot else float('inf'))
            initial_losses_plot.append(initial_losses_plot[-1] if initial_losses_plot else float('inf'))


    # Plotting configuration
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean, modern style for plots

    fig, ax = plt.subplots(figsize=(12, 7)) # Create a figure and axes for fine-grained control

    # Plot the best final loss achieved per iteration
    ax.plot(final_losses_plot, marker='o', linestyle='-', color='dodgerblue', label='Best Final Loss per Iteration')
    # Plot the initial loss corresponding to the best final model of each iteration
    ax.plot(initial_losses_plot, marker='x', linestyle='--', color='crimson', alpha=0.7, label='Initial Loss of Best Model per Iteration') # Changed marker for distinction
    ax.set_yscale('log') # Use a logarithmic scale for the y-axis, common for loss plots
    ax.set_title('Optimization Progress: Best Initial vs. Final Loss Across Iterations', fontsize=18, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.legend(fontsize=10) # Display legend to differentiate lines
    ax.tick_params(axis='both', which='major', labelsize=10) # Adjust tick parameter font sizes

    plt.tight_layout() # Adjust plot to prevent labels from overlapping
    plt.show() # Display the plot