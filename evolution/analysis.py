
import json
import matplotlib.pyplot as plt
import difflib
import pandas as pd
import seaborn as sns

def load_history(filepath):
    """Loads evolution history from a JSON log file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_fitness_trajectories(history_data, title="Fitness Evolution"):
    """
    Plots the best fitness over generations for each island/run.
    Handles data from both the traditional 'stats' list and the new detailed 'log_data'.
    """
    # Check data format
    if not history_data:
        print("No data to plot.")
        return

    # If it's a list of events (new log format)
    if isinstance(history_data[0], dict) and 'event' in history_data[0]:
        # We need to reconstruct the "best fitness per generation"
        # This is a bit tricky since we log *every* creation.
        # Let's aggregate by generation.
        df = pd.DataFrame(history_data)
        # Filter for reproduction events
        df = df[df['event'] == 'reproduction']
        
        if df.empty:
            print("No reproduction events found.")
            return

        # We want the cumulative minimum fitness up to generation G
        # Actually, global best fitness is usually tracked. 
        # But let's look at the "accepted" children fitness.
        
        # Calculate rolling min fitness
        df = df.sort_values('generation')
        df['best_so_far'] = df['fitness'].cummin()
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='generation', y='best_so_far', marker='o', label='Best Fitness')
        plt.title(title)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()
        
    else:
        # Assume it's the old 'stats' format: list of dicts with 'best_fitness'
        df = pd.DataFrame(history_data)
        if 'best_fitness' not in df.columns:
             print("Unknown data format.")
             return
             
        plt.figure(figsize=(10, 6))
        if 'num_vars' in df.columns:
             sns.lineplot(data=df, x='generation', y='best_fitness', hue='num_vars', marker='o', palette='viridis')
        else:
             sns.lineplot(data=df, x='generation', y='best_fitness', marker='o')
             
        plt.title(title)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

def get_lineage(log_data, individual_id):
    """
    Reconstructs the lineage of a specific individual ID from the log.
    Returns a list of event dicts ordered from ancestor to descendant.
    """
    # Create lookup map
    lookup = {entry['child_id']: entry for entry in log_data if 'child_id' in entry}
    
    lineage = []
    current_id = individual_id
    
    while current_id in lookup:
        entry = lookup[current_id]
        lineage.append(entry)
        
        # Move to first parent (primary ancestor)
        # For crossover, we trace the first parent for simplicity in a linear view
        if entry['parent_ids']:
            current_id = entry['parent_ids'][0]
        else:
            break
            
    return lineage[::-1] # Reverse to chronological order

def print_diff(code_a, code_b, title="Diff"):
    """Prints a colored diff between two code strings."""
    a_lines = code_a.splitlines()
    b_lines = code_b.splitlines()
    
    diff = difflib.unified_diff(a_lines, b_lines, lineterm='', n=2) # n=context lines
    
    print(f"--- {title} ---")
    for line in diff:
        if line.startswith('+'):
            print(f"\033[92m{line}\033[0m") # Green
        elif line.startswith('-'):
            print(f"\033[91m{line}\033[0m") # Red
        elif line.startswith('@'):
            print(f"\033[96m{line}\033[0m") # Cyan
        else:
            print(line)
    print("-" * 20)

def extract_model_summary(code):
    """
    Extracts the key parts of the model code: parameters and forward logic.
    """
    lines = code.splitlines()
    summary = []
    
    in_params = False
    in_forward = False
    
    for line in lines:
        stripped = line.strip()
        
        # Capture Parameter Definitions
        if "self.params = nn.ParameterDict" in stripped:
            in_params = True
            summary.append("--- Parameters ---")
            continue
        
        if in_params:
            if stripped.startswith("})"):
                in_params = False
            else:
                # Clean up parameter lines for display
                if ":" in stripped and "nn.Parameter" in stripped:
                    # Extract 'name': value
                    parts = stripped.split(":")
                    name = parts[0].strip().replace("'", "")
                    # Extract tensor value roughly
                    val_part = parts[1].split("torch.tensor(")[-1].split(")")[0]
                    comment = ""
                    if "#" in stripped:
                        comment = " # " + stripped.split("#")[1].strip()
                    summary.append(f"  {name}: {val_part}{comment}")

        # Capture Forward Logic (Derivatives)
        if "def forward" in stripped:
            in_forward = True
            summary.append("\n--- ODE System ---")
            continue
        
        if stripped.startswith("#"):
             continue
            
        if in_forward:
            if stripped.startswith("return"):
                in_forward = False
            elif "=" in stripped and ("dx" in stripped or "dt" in stripped or "*" in stripped):
                 # Heuristic to find equation lines
                 summary.append(f"  {stripped}")

    if not summary:
        return "Could not extract summary. Full code available in logs."
        
    return "\n".join(summary)

def print_evolution_trace(log_data, individual_id=None):
    """
    Prints a readable trace of the evolution leading to a specific individual.
    If individual_id is None, it traces the global best individual.
    """
    if not log_data:
        print("No log data.")
        return

    # Create full lookup for parents
    log_lookup = {entry['child_id']: entry for entry in log_data if 'child_id' in entry}

    if individual_id is None:
        # Find best
        best_entry = min(log_data, key=lambda x: x.get('fitness', float('inf')))
        individual_id = best_entry.get('child_id')
        print(f"Tracing Best Individual found: {individual_id[:8]} (Fitness: {best_entry['fitness']:.4f})")
        
    lineage = get_lineage(log_data, individual_id)
    
    if not lineage:
        print(f"Individual {individual_id} not found in logs.")
        return

    print(f"\n=== Evolution Trace for {individual_id[:8]}... ===")
    print(f"History Length: {len(lineage)} recorded generations")
    
    for i, step in enumerate(lineage):
        child_id = step.get('child_id', '???')
        gen = step.get('generation', '?')
        op = step.get('op', 'init')
        fit = step.get('fitness', float('inf'))
        parents = step.get('parent_ids', [])
        
        print(f"\n[Gen {gen}] Step {i+1}: {op.upper()} -> Child {child_id[:8]}")
        print(f"Fitness: {fit:.6f}")
        
        # Show specific Instruction
        metadata = step.get('metadata', {})
        if 'instruction' in metadata and metadata['instruction']:
             print(f"Instruction: {metadata['instruction']}")

        # Show Parents details if available
        if parents:
            print(f"\n  --- Parent(s) used: {[p[:8] for p in parents]} ---")
            for pid in parents:
                if pid in log_lookup:
                    print(f"  [Parent {pid[:8]}] Summary:")
                    summary = extract_model_summary(log_lookup[pid]['code'])
                    # Indent parent summary
                    print('\n'.join("    " + line for line in summary.splitlines()))
                else:
                    print(f"  [Parent {pid[:8]}] (Data not available in log)")
        
        print("-" * 30)
        print(f"  [Child {child_id[:8]}] Resulting Code Summary:")
        current_code = step['code']
        print(extract_model_summary(current_code))
        print("="*40)

def plot_lineage_graph(log_data, title="Evolution Lineage Graph", island_id=None):
    """
    Plots a 2D graph of the evolution:
    X-axis: Generation
    Y-axis: Fitness (Log Scale)
    island_id: If specified, only shows individuals from that island.

    Nodes are individuals. Edges connect parents to children.
    """
    if not log_data:
        print("No log data to plot.")
        return

    # Filter valid reproduction events
    # We need a quick lookup for (gen, fitness) of every individual to draw lines
    
    entries = {} # id -> entry
    for entry in log_data:
        if 'child_id' in entry:
            if island_id is not None and entry.get('island') != island_id:
                continue
            entries[entry['child_id']] = entry

    # Lists for plotting
    generations = []
    fitnesses = []
    colors = [] # Color by Op
    
    op_color_map = {
        'init': 'gray',
        'mutation': 'blue',
        'crossover': 'red'
    }
    
    for entry in entries.values():
        gen = entry.get('generation', 0)
        fit = entry.get('fitness', float('inf'))
        op = entry.get('op', 'init')
        
        if fit == float('inf') or fit > 1e6: # Skip failed/huge fitness for clarity
            continue
            
        generations.append(gen)
        fitnesses.append(fit)
        colors.append(op_color_map.get(op, 'black'))

    plt.figure(figsize=(12, 8))
    
    # Draw Lines first (so points are on top)
    for child_id, entry in entries.items():
        child_gen = entry.get('generation', 0)
        child_fit = entry.get('fitness', float('inf'))
        
        if child_fit == float('inf') or child_fit > 1e6:
            continue
            
        parent_ids = entry.get('parent_ids', [])
        
        for pid in parent_ids:
            if pid in entries:
                parent = entries[pid]
                parent_gen = parent.get('generation', 0)
                parent_fit = parent.get('fitness', float('inf'))
                
                if parent_fit == float('inf') or parent_fit > 1e6:
                    continue
                
                # Draw line
                # color based on child's op
                line_color = op_color_map.get(entry.get('op'), 'gray')
                alpha = 0.5
                plt.plot([parent_gen, child_gen], [parent_fit, child_fit], color=line_color, alpha=alpha, linewidth=0.8)

    # Draw Points
    plt.scatter(generations, fitnesses, c=colors, alpha=0.8, s=30, edgecolors='w')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Init'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Mutation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Crossover')
    ]
    plt.legend(handles=legend_elements)
    
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (MSE)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()
