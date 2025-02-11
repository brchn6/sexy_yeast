#!/home/labs/pilpel/barc/.conda/envs/sexy_yeast_env/bin/python
import numpy as np
import uuid
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap
from collections import defaultdict
import logging as log
import os
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from enum import Enum
from itertools import combinations, product
import random

#######################################################################
# Helper functions
#######################################################################
def compute_fit_slow(sigma, his, Jijs, F_off=0.0):
    """
    Compute the fitness of the genome configuration sigma using full slow computation.

    Parameters:
    sigma (np.ndarray): The genome configuration (vector of -1 or 1).
    his (np.ndarray): The vector of site-specific contributions to fitness.
    Jijs (np.ndarray): The interaction matrix between genome sites.
    F_off (float): The fitness offset, defaults to 0.

    Returns:
    float: The fitness value for the configuration sigma.
    Divide by 2 because every term appears twice in symmetric case.
    """
    return sigma @ (his + 0.5 * Jijs @ sigma) - F_off

def init_J(N, beta, rho, random_state=None,):
    """
    Initialize the coupling matrix for the Sherrington-Kirkpatrick model with sparsity.

    Parameters
    ----------
    N : int
        The number of spins.
    beta : float, optional
        Inverse temperature parameter.
    rho : float
        Fraction of non-zero elements in the coupling matrix (0 < rho ≤ 1).
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        The symmetric coupling matrix Jij with sparsity controlled by rho.
    """
    if not (0 < rho <= 1):
        raise ValueError("rho must be between 0 (exclusive) and 1 (inclusive).")
    rng = np.random.default_rng(random_state)
    sig_J = np.sqrt(beta / (N * rho))  # Adjusted standard deviation for sparsity
    # Initialize an empty upper triangular matrix (excluding diagonal)
    J_upper = np.zeros((N, N))
    # Total number of upper triangular elements excluding diagonal
    total_elements = N * (N - 1) // 2
    # Number of non-zero elements based on rho
    num_nonzero = int(np.floor(rho * total_elements))
    if num_nonzero == 0 and rho > 0:
        num_nonzero = 1  # Ensure at least one non-zero element if rho > 0
    # Get the indices for the upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(N, k=1)
    # Randomly select indices to assign non-zero Gaussian values
    selected_indices = rng.choice(total_elements, size=num_nonzero, replace=False)
    # Map the selected flat indices to row and column indices
    rows = triu_indices[0][selected_indices]
    cols = triu_indices[1][selected_indices]
    # Assign Gaussian-distributed values to the selected positions
    J_upper[rows, cols] = rng.normal(loc=0.0, scale=sig_J, size=num_nonzero)
    # Symmetrize the matrix to make Jij symmetric
    Jij = J_upper + J_upper.T

    return Jij

def init_h(N, beta, random_state=None, ):
    #
    """
    Initialize the external fields for the Sherrington-Kirkpatrick model.

    Parameters
    ----------
    N : int
        The number of spins.
    beta : float
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        The external fields.
    """
    rng = np.random.default_rng(random_state)
    sig_h = np.sqrt(1 - beta)
    return rng.normal(0.0, sig_h, N)

def calc_F_off(sigma_init, his, Jijs):
    """
    Calculate the fitness offset for the given configuration.

    Parameters
    ----------
    sigma_init : numpy.ndarray
        The initial spin configuration.
    his : numpy.ndarray
        The local fitness fields.
    Jijs : numpy.ndarray
        The coupling matrix.

    Returns
    -------
    float
        The fitness offset.
    """
    return compute_fit_slow(sigma_init, his, Jijs) - 1

def init_log(Resu_path):
    """Initialize logging"""
    # init logging
    log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    # add file handler
    path=os.path.join(Resu_path, 'sexy_yeast.log')
    fh = log.FileHandler(path, mode='w')
    fh.setLevel(log.INFO)
    log.getLogger().addHandler(fh)

    return log

def calculate_genomic_distance(genome1, genome2):
    """
    Calculate Hamming distance between two genomes.
    
    Parameters:
    -----------
    genome1, genome2 : numpy.ndarray
        Binary genome arrays (-1 or 1)
        
    Returns:
    --------
    float
        Normalized Hamming distance (0 to 1)
    """
    if len(genome1) != len(genome2):
        raise ValueError("Genomes must be the same length")
    
    # Calculate Hamming distance (number of positions where genomes differ)
    differences = np.sum(genome1 != genome2)
    
    # Normalize by genome length
    return differences / len(genome1)

def assign_mating_types(organisms):
    """Randomly assign mating types A or alpha to organisms."""
    typed_organisms = []
    for org in organisms:
        mating_type = random.choice([MatingType.A, MatingType.ALPHA])
        typed_organisms.append(OrganismWithMatingType(org, mating_type))
    return typed_organisms

def calculate_mating_statistics(diploid_offspring):
    """
    Calculate statistics about the mating outcomes.
    
    Parameters:
    -----------
    diploid_offspring : dict
        Dictionary containing diploid organisms by fitness model
        
    Returns:
    --------
    dict : Dictionary containing mating statistics
    """
    stats = {}
    for model, organisms in diploid_offspring.items():
        model_stats = {
            'total_offspring': len(organisms),
            'avg_offspring_fitness': np.mean([org.fitness for org in organisms]),
            'max_offspring_fitness': np.max([org.fitness for org in organisms]),
            'min_offspring_fitness': np.min([org.fitness for org in organisms]),
            'avg_parent_fitness': np.mean([org.avg_parent_fitness for org in organisms]),
            'fitness_improvement': np.mean([org.fitness - org.avg_parent_fitness for org in organisms])
        }
        stats[model] = model_stats
    return stats

#######################################################################
# Classes
#######################################################################
class Environment:
    def __init__(self, genome_size, beta=0.5, rho=0.25):
        self.genome_size = genome_size
        self.beta = beta
        self.rho = rho
        
        # Initialize fitness landscape
        self.h = self._init_h()
        self.J = self._init_J()
    
    def _init_h(self):
        """Initialize external fields using SK model."""
        return init_h(self.genome_size, self.beta)
    
    def _init_J(self):
        """Initialize coupling matrix with sparsity"""
        return init_J(self.genome_size, self.beta, self.rho)
            
        sig_J = np.sqrt(self.beta / (self.genome_size * self.rho))
        
        # Create sparse coupling matrix
        J = np.zeros((self.genome_size, self.genome_size))
        # Get upper triangular indices
        upper_indices = np.triu_indices(self.genome_size, k=1)
        
        # Number of non-zero elements
        total_elements = len(upper_indices[0])
        num_nonzero = int(np.floor(self.rho * total_elements))
        
        # Randomly select indices for non-zero elements
        selected_idx = np.random.choice(total_elements, size=num_nonzero, replace=False)
        rows = upper_indices[0][selected_idx]
        cols = upper_indices[1][selected_idx]
        
        # Fill selected positions with random values
        J[rows, cols] = np.random.normal(0.0, sig_J, num_nonzero)
        # Make matrix symmetric
        J = J + J.T
        
        return J
    
    def calculate_fitness(self, genome):
        """Calculate fitness for a given genome"""
        # Calculate energy (negative fitness)
        energy = compute_fit_slow(genome, self.h, self.J , F_off=0.0)
        # Convert energy to fitness (higher is better)
        return 1.0 / (1.0 + np.exp(-energy))
    
    def calculate_mutation_effects(self, genome):
        """Calculate the effect of each possible mutation"""
        effects = []
        for i in range(len(genome)):
            mutated = genome.copy()
            mutated[i] *= -1
            effects.append(self.calculate_fitness(mutated))
        return np.array(effects)

class Organism:
    def __init__(self, environment, genome=None, generation=0, parent_id=None):
        self.id = str(uuid.uuid4())
        self.environment = environment
        
        # Initialize or copy genome
        if genome is None:
            self.genome = np.random.choice([-1, 1], environment.genome_size)
        else:
            self.genome = genome.copy()
            
        self.generation = generation
        self.parent_id = parent_id
        self.fitness = self.calculate_fitness()
    
    def calculate_fitness(self):
        """Calculate organism's fitness using the environment"""
        return self.environment.calculate_fitness(self.genome)
    
    def mutate(self):
        """Randomly mutate the genome"""
        mutation_rate = 1/len(self.genome)
        mutation_mask = np.random.random(len(self.genome)) < mutation_rate
        self.genome[mutation_mask] *= -1
        self.fitness = self.calculate_fitness()
    
    def reproduce(self):
        """Create two offspring with the same genome"""
        child1 = Organism(self.environment, 
                         genome=self.genome,
                         generation=self.generation + 1, 
                         parent_id=self.id)
        
        child2 = Organism(self.environment,
                         genome=self.genome,
                         generation=self.generation + 1, 
                         parent_id=self.id)
        
        return child1, child2

class MatingStrategy(Enum):
    ONE_TO_ONE = "one_to_one"
    ALL_VS_ALL = "all_vs_all"
    MATING_TYPES = "mating_types"

class MatingType(Enum):
    A = "A"
    ALPHA = "alpha"

class DiploidOrganism:
    def __init__(self, parent1, parent2, fitness_model="dominant", mating_type=None):
        if len(parent1.genome) != len(parent2.genome):
            raise ValueError("Parent genomes must have the same length.")
        
        self.allele1 = parent1.genome.copy()
        self.allele2 = parent2.genome.copy()
        self.fitness_model = fitness_model
        self.environment = parent1.environment
        self.id = str(uuid.uuid4())
        self.parent1_id = parent1.id
        self.parent2_id = parent2.id
        self.mating_type = mating_type
        
        # *** Store individual parent fitness values ***
        self.parent1_fitness = parent1.fitness
        self.parent2_fitness = parent2.fitness
        
        # Compute average parent fitness (used elsewhere) but not for the heatmap
        self.avg_parent_fitness = (parent1.fitness + parent2.fitness) / 2
        
        self.fitness = self.calculate_fitness()

    def _get_effective_genome(self):
        """Calculate effective genome based on inheritance model."""
        if self.fitness_model == "dominant":
            return np.where((self.allele1 == -1) | (self.allele2 == -1), -1, 1)
        elif self.fitness_model == "recessive":
            return np.where((self.allele1 == -1) & (self.allele2 == -1), -1, 1)
        elif self.fitness_model == "codominant":
            return np.where(self.allele1 == self.allele2, self.allele1, 1)
        else:
            raise ValueError(f"Unknown fitness model: {self.fitness_model}")

    def calculate_fitness(self):
        """Calculate fitness using the effective genome."""
        effective_genome = self._get_effective_genome()
        energy = compute_fit_slow(
        effective_genome,
        self.environment.h,
        self.environment.J,
        F_off=0.0)
        return 1.0 / (1.0 + np.exp(-energy))

class OrganismWithMatingType:
    def __init__(self, organism, mating_type):
        self.organism = organism
        self.mating_type = mating_type

#######################################################################
# plotting functions
#######################################################################
def plot_detailed_fitness(individual_fitness, generation_stats, Resu_path):
    """Create detailed fitness visualizations"""
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Individual fitness trajectories
    for org_id, fitness_data in individual_fitness.items():
        generations, fitness = zip(*fitness_data)
        ax1.plot(generations, fitness, '-', alpha=0.3, linewidth=1)
    
    # Add statistical overlays
    generations = [stat['generation'] for stat in generation_stats]
    avg_fitness = [stat['avg_fitness'] for stat in generation_stats]
    max_fitness = [stat['max_fitness'] for stat in generation_stats]
    min_fitness = [stat['min_fitness'] for stat in generation_stats]
    
    ax1.plot(generations, avg_fitness, 'k-', linewidth=2, label='Average')
    ax1.plot(generations, max_fitness, 'g-', linewidth=2, label='Maximum')
    ax1.plot(generations, min_fitness, 'r-', linewidth=2, label='Minimum')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Individual Fitness Trajectories')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Fitness distribution violin plot
    fitness_distributions = []
    for gen in range(1, max(generations) + 1):
        gen_fitness = [f for org_id, fit_data in individual_fitness.items() 
                      for g, f in fit_data if g == gen]
        fitness_distributions.append(gen_fitness)
    
    ax2.violinplot(fitness_distributions, positions=range(1, len(fitness_distributions) + 1))
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Distribution')
    ax2.set_title('Fitness Distribution per Generation')
    ax2.grid(True)
    
    plt.tight_layout()
    # /home/labs/pilpel/barc/sexy_yeast/Resu save into Resu from the dir __file__
    plt.savefig(os.path.join(Resu_path, 'fitness_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_relationship_tree(organisms, Resu_path):
    """Create a visual representation of the evolutionary tree"""
    G = nx.DiGraph()
    
    # Add nodes and edges with fitness information
    for org in organisms:
        G.add_node(org.id, generation=org.generation, fitness=org.fitness)
        if org.parent_id:
            G.add_edge(org.parent_id, org.id)
    
    # Position nodes using generational layout
    pos = {}
    for org in organisms:
        # Use fitness to determine vertical position
        pos[org.id] = (org.generation, org.fitness)
    
    # Create figure with specific layout for colorbar
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Color nodes based on fitness
    fitness_values = [G.nodes[node]['fitness'] for node in G.nodes()]
    
    # Create a normalization for the colormap
    norm = plt.Normalize(vmin=min(fitness_values), vmax=max(fitness_values))
    
    # Draw the network nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100,
                           node_color=fitness_values,
                           cmap=plt.cm.viridis)
    
    # Draw the network edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, edge_color='gray')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Fitness')
    
    plt.title("Evolutionary Relationship Tree")
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, 'relationship_tree.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_parent_offspring_fitness(diploid_dict, Resu_path):
    """
    Create a figure with three subplots (one per diploid model) where the x-axis shows 
    the offspring fitness and the y-axis shows the mean parent fitness.
    
    Parameters:
      diploid_dict : dict
          Dictionary with keys as fitness model names (e.g., "dominant", "recessive", "codominant")
          and values as lists of DiploidOrganism instances.
      Resu_path : str
          Directory path where the resulting figure will be saved.
    """
    # Define the order, colors, and markers for the three models
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "blue", "recessive": "green", "codominant": "purple"}
    
    # Create a pandas DataFrame
    df = pd.DataFrame({
        'Model': [],
        'Offspring Fitness': [],
        'Mean Parent Fitness': []
    })
    
    for model in models:
        diploids = diploid_dict.get(model, [])
        if not diploids:
            continue
        
        offspring_fitness = [org.fitness for org in diploids]
        parent_fitness = [org.avg_parent_fitness for org in diploids]
        
        df = pd.concat([df, pd.DataFrame({
            'Model': [model]*len(offspring_fitness),
            'Offspring Fitness': offspring_fitness,
            'Mean Parent Fitness': parent_fitness
        })])
    
    # Use seaborn
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    for model, ax in zip(models, axes):
        df_model = df[df['Model'] == model]
        sns.regplot(x='Offspring Fitness', y='Mean Parent Fitness', data=df_model, ax=ax, 
                    scatter_kws={'alpha': 0.7, 'color': colors.get(model, "black")})
        
        all_values = df_model['Offspring Fitness'].tolist() + df_model['Mean Parent Fitness'].tolist()
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='x = y')
        
        ax.set_title(f"{model.capitalize()} Model")
        ax.set_xlabel("Offspring Fitness")
        ax.set_ylabel("Mean Parent Fitness")
        ax.grid(True)
        ax.legend()
    
    plt.suptitle("Parent vs Offspring Fitness Comparison\n(Mean Parent Fitness on Y-Axis)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(Resu_path, 'parent_offspring_fitness_subplots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parent_offspring_heatmap(diploid_offspring_dict, Resu_path):
    """
    Create heatmap plots comparing each parent's fitness (Parent 1 and Parent 2)
    with the offspring fitness for each fitness model.
    
    Parameters:
        diploid_offspring_dict (dict): Keys are fitness model names ("dominant", "recessive", "codominant")
            and values are lists of DiploidOrganism instances.
        Resu_path (str): The directory path where the heatmap images will be saved.
    """
    # List of models (modify if your simulation uses a different set)
    models = ["dominant", "recessive", "codominant"]
    
    # Initialize an empty figure with three subplots (one per model)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Loop over each model in the dictionary
    for i, model in enumerate(models):
        offspring_list = diploid_offspring_dict.get(model, [])
        if not offspring_list:
            continue  # Skip if no data for this model
        
        # Extract the fitness values for Parent 1, Parent 2, and the Offspring.
        parent1_fit = [d.parent1_fitness for d in offspring_list]
        parent2_fit = [d.parent2_fitness for d in offspring_list]
        offspring_fit = [d.fitness for d in offspring_list]
        
        # (Optional) If you want to create separate heatmaps for each parent:
        # h1, xedges1, yedges1 = np.histogram2d(parent1_fit, offspring_fit, bins=20)
        # h2, xedges2, yedges2 = np.histogram2d(parent2_fit, offspring_fit, bins=20)
        # You can plot these separately if desired.
        
        # Now, to create one combined heatmap:
        combined_parent_fit = np.concatenate((parent1_fit, parent2_fit))
        # Duplicate offspring_fit so that each parent's fitness has a corresponding offspring value.
        combined_offspring_fit = np.concatenate((offspring_fit, offspring_fit))
        
        h3, xedges3, yedges3 = np.histogram2d(combined_parent_fit, combined_offspring_fit, bins=20)
        im3 = axs[i].imshow(
            h3.T,
            origin='lower',
            aspect='auto',
            extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]],
            cmap='viridis'
        )
        axs[i].set_xlabel('Parent Fitness')
        axs[i].set_ylabel('Offspring Fitness')
        axs[i].set_title(f"{model.capitalize()} Model: Both Parents vs Offspring")
        fig.colorbar(im3, ax=axs[i], label='Count')
    
    plt.tight_layout()
    output_path = os.path.join(Resu_path, 'parent_offspring_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parent_fitness_vs_offspring_distance(diploid_dict, Resu_path):
    """
    Create a scatter plot comparing average parent fitness to genomic distance between offspring.
    
    Parameters:
    -----------
    diploid_dict : dict
        Dictionary with fitness models as keys and lists of DiploidOrganism instances as values
    Resu_path : str
        Path to save the resulting plot
        
    Returns:
    --------
    None
    """
    # Create figure with subplots for each model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors and models
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "blue", "recessive": "green", "codominant": "purple"}
    
    # Process each model
    for model, ax in zip(models, axes):
        diploids = diploid_dict.get(model, [])
        if not diploids:
            continue
            
        # Calculate genomic distances and parent fitnesses
        distances = []
        parent_fitnesses = []
        
        # Compare each pair of diploid organisms
        for i, org1 in enumerate(diploids):
            for j, org2 in enumerate(diploids):
                if i < j:  # Only compare each pair once
                    # Calculate genomic distance using effective genomes
                    distance = calculate_genomic_distance(
                        org1._get_effective_genome(),
                        org2._get_effective_genome()
                    )
                    
                    # Calculate average parent fitness for both organisms
                    avg_parent_fitness = (org1.avg_parent_fitness + org2.avg_parent_fitness) / 2
                    
                    distances.append(distance)
                    parent_fitnesses.append(avg_parent_fitness)
        
        # Create scatter plot
        ax.scatter(distances, parent_fitnesses, alpha=0.5, c=colors[model])
        
        # Add trend line
        if distances:
            z = np.polyfit(distances, parent_fitnesses, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(distances), max(distances), 100)
            ax.plot(x_trend, p(x_trend), '--', color='red')
        
        # Customize plot
        ax.set_title(f"{model.capitalize()} Model")
        ax.set_xlabel("Genomic Distance Between Offspring")
        ax.set_ylabel("Average Parent Fitness")
        ax.grid(True)
    
    plt.suptitle("Parent Fitness vs Offspring Genomic Distance")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    output_path = os.path.join(Resu_path, 'parent_fitness_vs_offspring_distance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

#######################################################################
# Simulation functions
#######################################################################
def run_simulation(num_generations, environment):
    # Create initial population
    initial_organism = Organism(environment)
    population = [initial_organism]
    all_organisms = [initial_organism]
    
    # Track fitness per individual
    individual_fitness = defaultdict(list)
    generation_stats = []
    
    # Run for specified number of generations
    for gen in range(num_generations):
        next_generation = []
        gen_fitness = []
        
        # Each organism reproduces and mutates
        for org in population:
            # Reproduce
            child1, child2 = org.reproduce()
            
            # Mutate offspring
            child1.mutate()
            child2.mutate()
            
            # Track individual fitness
            individual_fitness[child1.id].append((gen+1, child1.fitness))
            individual_fitness[child2.id].append((gen+1, child2.fitness))
            
            gen_fitness.extend([child1.fitness, child2.fitness])
            next_generation.extend([child1, child2])
            all_organisms.extend([child1, child2])
        
        # Update current population
        population = next_generation
        
        # Store generation statistics
        stats = {
            'generation': gen + 1,
            'population_size': len(population),
            'avg_fitness': np.mean(gen_fitness),
            'max_fitness': np.max(gen_fitness),
            'min_fitness': np.min(gen_fitness),
            'std_fitness': np.std(gen_fitness)
        }
        generation_stats.append(stats)
        
        log.info(f"Generation {gen+1}: Pop size = {stats['population_size']}, "
              f"Avg fitness = {stats['avg_fitness']:.4f}, "
              f"Max fitness = {stats['max_fitness']:.4f}, "
              f"Min fitness = {stats['min_fitness']:.4f}")

    
    return all_organisms, individual_fitness, generation_stats

def mate_last_generation(last_generation, mating_strategy=MatingStrategy.ONE_TO_ONE, fitness_models=["dominant", "recessive", "codominant"]):
    """
    Enhanced mating function supporting multiple mating strategies.
    
    Parameters:
    -----------
    last_generation : list
        List of haploid organisms from the last generation
    mating_strategy : MatingStrategy
        The mating strategy to use
    fitness_models : list
        List of fitness models to use for creating diploid offspring
        
    Returns:
    --------
    dict : Dictionary with fitness models as keys and lists of diploid organisms as values
    """
    diploid_offspring = defaultdict(list)
    
    if mating_strategy == MatingStrategy.ONE_TO_ONE:
        # Ensure even number of parents
        if len(last_generation) % 2 != 0:
            last_generation = last_generation[:-1]
            
        for model in fitness_models:
            for i in range(0, len(last_generation), 2):
                parent1 = last_generation[i]
                parent2 = last_generation[i + 1]
                offspring = DiploidOrganism(parent1, parent2, fitness_model=model)
                diploid_offspring[model].append(offspring)
                
    elif mating_strategy == MatingStrategy.ALL_VS_ALL:
        for model in fitness_models:
            # Generate all possible pairs using combinations
            for parent1, parent2 in combinations(last_generation, 2):
                offspring = DiploidOrganism(parent1, parent2, fitness_model=model)
                diploid_offspring[model].append(offspring)
                
    elif mating_strategy == MatingStrategy.MATING_TYPES:
        # Assign mating types to organisms
        typed_organisms = assign_mating_types(last_generation)
        
        # Separate organisms by mating type
        type_a = [org for org in typed_organisms if org.mating_type == MatingType.A]
        type_alpha = [org for org in typed_organisms if org.mating_type == MatingType.ALPHA]
        
        for model in fitness_models:
            # Mate all type A with all type alpha
            for a_org, alpha_org in product(type_a, type_alpha):
                offspring = DiploidOrganism(
                    a_org.organism, 
                    alpha_org.organism, 
                    fitness_model=model,
                    mating_type=random.choice([MatingType.A, MatingType.ALPHA])
                )
                diploid_offspring[model].append(offspring)
    
    else:
        raise ValueError(f"Unknown mating strategy: {mating_strategy}")
    
    # Log results
    for model in fitness_models:
        log.info(f"Created {len(diploid_offspring[model])} diploid organisms using {model} model "
                f"with {mating_strategy.value} strategy")
    
    return diploid_offspring

#######################################################################
# Main function
#######################################################################
def main():
    aparser = ap.ArgumentParser(description="Run the sexy yeast simulation with detailed fitness tracking")
    aparser.add_argument("--generations", type=int, default=5, help="Number of generations to simulate")
    aparser.add_argument("--genome_size", type=int, default=100, help="Size of the genome")
    aparser.add_argument("--beta", type=float, default=0.5, help="Beta parameter")
    aparser.add_argument("--rho", type=float, default=0.25, help="Rho parameter")
    aparser.add_argument("--mating_strategy ", type=str, default="one_to_one", help=["one_to_one", "all_vs_all", "mating_types"]) #["one_to_one", "all_vs_all", "mating_types"],
    aparser.add_argument("--output_dir", type=str, default="Resu", help="Output directory for results")


    args = aparser.parse_args()

    # val Resu directory one back from the dir __file__
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output_dir)):
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output_dir))

    Resu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output_dir)
    
    # init logging
    log = init_log(Resu_path)
    
    # Create environment
    environment = Environment(genome_size=args.genome_size, beta=args.beta, rho=args.rho)
    log.info(f"Environment created with genome size = {args.genome_size}, beta = {args.beta}, rho = {args.rho}")

    # Run simulation
    all_organisms, individual_fitness, generation_stats = run_simulation(args.generations, environment)

    # Create visualizations
    log.info("Creating visualizations")
    plot_detailed_fitness(individual_fitness, generation_stats , Resu_path)
    plot_relationship_tree(all_organisms, Resu_path)

    # Mate last generation to create diploid organisms
    log.info(f"Mating last generation to create diploid organisms the mating strategy is {MatingStrategy.MATING_TYPES}")
    last_generation = [org for org in all_organisms if org.generation == args.generations]
    # Mate last generation to create diploid organisms for each model
    diploid_offspring_dict = mate_last_generation(
        last_generation, 
        mating_strategy=MatingStrategy.MATING_TYPES,
        fitness_models=["dominant", "recessive", "codominant"]
    )

    mating_stats = calculate_mating_statistics(diploid_offspring_dict)
    
    # Plot parent-offspring fitness comparison
    log.info("Creating parent-offspring fitness comparison plot")
    plot_parent_offspring_fitness(diploid_offspring_dict, Resu_path)

    #plot parent fitness vs offspring genomic distance
    log.info("Creating parent fitness vs offspring genomic distance plot")
    plot_parent_fitness_vs_offspring_distance(diploid_offspring_dict, Resu_path)
    
    # plot heatmap parent-offspring fitness comparison
    log.info("Creating parent-offspring fitness heatmap plot")
    plot_parent_offspring_heatmap(diploid_offspring_dict, Resu_path)
   
   
   # Log the final results
    log.info("Simulation complete")

if __name__ == "__main__":
    main()
    

"""
example usage: 
bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 10 --genome_size 100 --beta 0.5 --rho 0.25 --mating_strategy all_vs_all --output_dir /home/labs/pilpel/barc/sexy_yeast/10gen
"""

