'''
File to run the method. I think there will be two folders, one data and one for method or solve.
Should be a nice architecture to write them like this. 
'''

from data.generate_data import LorenzDataset, WarfarinTMDDDataset
from data.utils import prepare_data_for_method
from solve.method import Method
import os
import json
import shutil 
import datetime # Import datetime for unique folder names

if __name__ == '__main__':

    '''
    might have a section where either I change a file and choose variables from it, or I precise 
    using --arg=5 to decide which thing I want to run

    '''
    iter_max = 9
    # config_file_path = os.path.join(os.path.dirname(__file__), 'data', 'config_Lorenz.json')
    # if not os.path.exists(config_file_path):
    #     raise FileNotFoundError(f"Config file not found at {config_file_path}. Please create it.")
    
    # with open(config_file_path, 'r') as f:
    #     config = json.load(f)

    # data = LorenzDataset(config)

    config_file_path = os.path.join(os.path.dirname(__file__), 'data', 'config_WarfarinTMDD_2.json')
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found at {config_file_path}. Please create it.")
    
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    instructions = '''
    You only observe two variables from this dataset. However, you know that there are variables you don't have access to. 
    Really important: Only propose models with 3 state variables or more, so the attribute self.variables_order should always have at least 3 variables. The original dataset was constructed with the idea
    of compartments interacting between each other. Moreover, especially at for your first proposals explore different type of structures, be imaginative. Do not constraint yourself at only proposing simple function form such as monomial. 
    Explore various structures.
    '''

    data = WarfarinTMDDDataset(config)
    noise_levels = [0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    experiences_base_dir = os.path.join(os.getcwd(), "Experiences")
    for noise_level in noise_levels:

        data.generate()
        data_prep = prepare_data_for_method(data, [0,1], noise_level=.05)
        
        method = Method(data_prep, iter_max, instructions, noise_level = noise_level)
        method.run()

        # --- Archiving the generated_code folder ---
        source_code_folder = "generated_code"
        
        # Create a unique name for the destination folder including noise level and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        destination_folder_name = f"noise_{str(noise_level).replace('.', '')}_{timestamp}" # Remove '.' from noise for folder name
        destination_path = os.path.join(experiences_base_dir, destination_folder_name)
        
        if os.path.exists(source_code_folder):
            print(f"Archiving '{source_code_folder}' to '{destination_path}'...")
            # Copy the entire directory
            shutil.copytree(source_code_folder, destination_path)
            print("Archiving complete.")
            
            # Optional: Clear the generated_code folder for the next iteration if desired
            # This can prevent old files from previous runs interfering or being copied unnecessarily
            # shutil.rmtree(source_code_folder)
            # os.makedirs(source_code_folder, exist_ok=True) # Recreate empty folder
            
        else:
            print(f"Warning: '{source_code_folder}' not found. No code to archive for noise level {noise_level}.")

    print("\n--- All experiments complete ---")

    # founded_models = method.__start__()
    # opt = method.__optimize__(founded_models)
    # method.__feedback__(opt)
    # a = 5

    # dataset = generate_data(parameters_data) 
    # 
    # parameters_data = (name_dataset, properties of generation = number of points, possible_noise, parameters 
    # related to the generative method)
    # return: just the generated dataset with all its variable. Also, it must return a text description of
    # the problem we are trying to solve and a text description of the variable.
    #
    # data = prepare_data(dataset)  function that take the dataset generated and return the text information associated
    # as it will be useful for the method, and only give a small number of the variable of the dataset, specifically
    # the variable we will consider as observable in our study. 

    # big black-box solve function that encompass all the solve method 

    # discovered_dyn = solve(data, hyperparameters) 

    # data is the partially-obsvered dataset. Hyperparameters are just the parameters of the method. We
    # have yet to define which one there will be. 