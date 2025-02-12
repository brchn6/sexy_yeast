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
import sys
import time
import psutil
import gc 
from tqdm import tqdm 

#######################################################################
# SetUUp functions
#######################################################################

def _get_lsf_job_details() -> list[str]:
    """
    Retrieves environment variables for LSF job details, if available.
    """
    lsf_job_id = os.environ.get('LSB_JOBID')    # Job ID
    lsf_queue = os.environ.get('LSB_QUEUE')     # Queue name
    lsf_host = os.environ.get('LSB_HOSTS')      # Hosts allocated
    lsf_job_name = os.environ.get('LSB_JOBNAME')  # Job name
    lsf_command = os.environ.get('LSB_CMD')     # Command used to submit job

    details = [
        f"LSF Job ID: {lsf_job_id}",
        f"LSF Queue: {lsf_queue}",
        f"LSF Hosts: {lsf_host}",
        f"LSF Job Name: {lsf_job_name}",
        f"LSF Command: {lsf_command}"
    ]
    return details

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

def monitor_memory():
    """Log current memory usage."""
    mem = psutil.virtual_memory().used / (1024 ** 3)
    log.info(f"Current memory usage: {mem:.2f} GB")

def calculate_regression_stats(x, y):
    """
    Calculate regression statistics including R², slope, and intercept.
    
    Parameters:
    -----------
    x : np.array
        Independent variable
    y : np.array
        Dependent variable
        
    Returns:
    --------
    dict
        Dictionary containing R², slope, intercept, and other statistics
    """
    from scipy import stats
    import numpy as np
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    # Quadratic regression
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)
    y_mean = np.mean(y)
    r_squared_quad = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2))
    
    return {
        'linear_r_squared': r_squared,
        'linear_slope': slope,
        'linear_intercept': intercept,
        'linear_p_value': p_value,
        'quadratic_r_squared': r_squared_quad,
        'quadratic_coeffs': coeffs.tolist()
    }

def summarize_simulation_stats(generation_stats, individual_fitness, diploid_dict):
    """
    Create a comprehensive summary of simulation statistics.
    
    Parameters:
    -----------
    generation_stats : list
        List of dictionaries containing generation statistics
    individual_fitness : dict
        Dictionary of individual fitness trajectories
    diploid_dict : dict
        Dictionary containing diploid offspring data
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    # Overall fitness statistics
    final_gen = max(stat['generation'] for stat in generation_stats)
    final_stats = next(stat for stat in generation_stats if stat['generation'] == final_gen)
    
    # Calculate fitness improvement
    initial_fitness = generation_stats[0]['avg_fitness']
    final_fitness = final_stats['avg_fitness']
    fitness_improvement = ((final_fitness - initial_fitness) / initial_fitness) * 100
    
    # Diploid statistics by model
    diploid_stats = {}
    for model, organisms in diploid_dict.items():
        if organisms:
            parent_fitness = np.array([org.avg_parent_fitness for org in organisms])
            offspring_fitness = np.array([org.fitness for org in organisms])
            
            # Calculate regression statistics
            reg_stats = calculate_regression_stats(parent_fitness, offspring_fitness)
            
            diploid_stats[model] = {
                'count': len(organisms),
                'avg_parent_fitness': np.mean(parent_fitness),
                'avg_offspring_fitness': np.mean(offspring_fitness),
                'fitness_improvement': np.mean(offspring_fitness - parent_fitness),
                'regression_stats': reg_stats
            }
    
    return {
        'generations': final_gen,
        'final_population': final_stats['population_size'],
        'initial_fitness': initial_fitness,
        'final_fitness': final_fitness,
        'fitness_improvement_percent': fitness_improvement,
        'fitness_std_final': final_stats['std_fitness'],
        'diploid_stats': diploid_stats
    }

def log_simulation_summary(log, summary_stats):
    """
    Log comprehensive simulation summary statistics.
    
    Parameters:
    -----------
    log : logging.Logger
        Logger instance
    summary_stats : dict
        Dictionary containing summary statistics
    """
    log.info("\n=== SIMULATION SUMMARY ===")
    log.info(f"Total Generations: {summary_stats['generations']}")
    log.info(f"Final Population Size: {summary_stats['final_population']}")
    log.info(f"\nFitness Statistics:")
    log.info(f"  Initial Average Fitness: {summary_stats['initial_fitness']:.4f}")
    log.info(f"  Final Average Fitness: {summary_stats['final_fitness']:.4f}")
    log.info(f"  Overall Fitness Improvement: {summary_stats['fitness_improvement_percent']:.2f}%")
    log.info(f"  Final Fitness Standard Deviation: {summary_stats['fitness_std_final']:.4f}")
    
    log.info("\nDiploid Analysis by Model:")
    for model, stats in summary_stats['diploid_stats'].items():
        log.info(f"\n{model.upper()} Model Statistics:")
        log.info(f"  Number of Organisms: {stats['count']}")
        log.info(f"  Average Parent Fitness: {stats['avg_parent_fitness']:.4f}")
        log.info(f"  Average Offspring Fitness: {stats['avg_offspring_fitness']:.4f}")
        log.info(f"  Average Fitness Improvement: {stats['fitness_improvement']:.4f}")
        
        reg_stats = stats['regression_stats']
        log.info(f"  Regression Statistics:")
        log.info(f"    Linear R²: {reg_stats['linear_r_squared']:.4f}")
        log.info(f"    Linear Slope: {reg_stats['linear_slope']:.4f}")
        log.info(f"    Quadratic R²: {reg_stats['quadratic_r_squared']:.4f}")
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
    def __init__(self, environment, genome=None, generation=0, parent_id=None, mutation_rate=None):
        self.id = str(uuid.uuid4())
        self.environment = environment
        
        # Initialize or copy genome
        if genome is None:
            self.genome = np.random.choice([-1, 1], environment.genome_size)
        else:
            self.genome = genome.copy()
            
        self.generation = generation
        self.parent_id = parent_id
        # Allow mutation rate to be specified or use default
        self.mutation_rate = mutation_rate if mutation_rate is not None else 1.0/environment.genome_size
        self.fitness = self.calculate_fitness()
    
    def calculate_fitness(self):
        """Calculate organism's fitness using the environment"""
        return self.environment.calculate_fitness(self.genome)
    
    def mutate(self):
        """
        Randomly mutate the genome with a per-locus mutation rate.
        
        The mutation rate is applied independently to each position in the genome,
        so on average, mutation_rate * genome_length sites will mutate.
        """
        # Generate random numbers for each position in genome
        mutation_sites = np.random.random(len(self.genome)) < self.mutation_rate
        
        # Flip the selected sites (-1 to 1 or 1 to -1)
        self.genome[mutation_sites] *= -1
        
        # Update fitness after mutation
        self.fitness = self.calculate_fitness()
    
    def reproduce(self):
        """Create two offspring with the same genome"""
        child1 = Organism(self.environment, 
                         genome=self.genome,
                         generation=self.generation + 1, 
                         parent_id=self.id,
                         mutation_rate=self.mutation_rate)  # Pass mutation rate to children
        
        child2 = Organism(self.environment,
                         genome=self.genome,
                         generation=self.generation + 1, 
                         parent_id=self.id,
                         mutation_rate=self.mutation_rate)  # Pass mutation rate to children
        
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
        """
        Calculate effective genome based on inheritance model.
        
        For codominant model:
        - If alleles are the same (1,1) or (-1,-1): use that value
        - If alleles are different (1,-1) or (-1,1): use 0.5 * (allele1 + allele2)
        - If environment prefers 1 and alleles are (1,1): returns 1
        """
        if self.fitness_model == "dominant":
            return np.where((self.allele1 == -1) | (self.allele2 == -1), -1, 1)
        elif self.fitness_model == "recessive":
            return np.where((self.allele1 == -1) & (self.allele2 == -1), -1, 1)
        elif self.fitness_model == "codominant":
            # Create a mask for mixed alleles (1,-1 or -1,1)
            mixed_alleles = self.allele1 != self.allele2
            
            # For mixed alleles, calculate the average (will give 0.5 * (1 + -1) = 0)
            # For same alleles, use either allele (they're the same)
            effective = np.where(
                mixed_alleles,
                0.5 * (self.allele1 + self.allele2),  # Mixed case: average of alleles
                self.allele1  # Same alleles case: use either allele
            )
            
            # Special case: if environment prefers 1 and both alleles are 1
            both_positive = (self.allele1 == 1) & (self.allele2 == 1)
            effective = np.where(both_positive, 1, effective)
            
            return effective
        else:
            raise ValueError(f"Unknown fitness model: {self.fitness_model}")

    def calculate_fitness(self):
        """Calculate fitness using the effective genome."""
        effective_genome = self._get_effective_genome()
        energy = compute_fit_slow(
            effective_genome,
            self.environment.h,
            self.environment.J,
            F_off=0.0
        )
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

def plot_parent_offspring_fitness(diploid_dict, Resu_path , mating_strategy):
    """
    Create a figure with three subplots (one per diploid model) where the x-axis shows 
    the mean parent fitness and the y-axis shows the offspring fitness. Includes both linear 
    and quadratic regression lines, KDE plots, and a reference line (x = y).

    Parameters:
    -----------
    diploid_dict : dict
        Dictionary with keys as fitness model names (e.g., "dominant", "recessive", "codominant")
        and values as lists of DiploidOrganism instances.
    Resu_path : str
        Directory path where the resulting figure will be saved.
    """
    # Define the order, colors, and markers for the three models
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "blue", "recessive": "green", "codominant": "purple"}

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)


    for idx, model in enumerate(tqdm(models, desc=f"Processing Models for {mating_strategy} Strategy")):
        diploids = diploid_dict.get(model, [])
        if not diploids:
            axs[idx].set_title(f"{model.capitalize()} Model (No Data)")
            continue  # Skip if no data for this model

        offspring_fitness = np.array([org.fitness for org in diploids])
        parent_fitness = np.array([org.avg_parent_fitness for org in diploids])

        if len(offspring_fitness) > 10 and len(parent_fitness) > 10:
            # Scatter plot
            sns.scatterplot(
                x=parent_fitness, 
                y=offspring_fitness, 
                ax=axs[idx], 
                alpha=0.7, 
                color=colors[model], 
                label="Data"
            )

            # Linear regression line
            sns.regplot(
                x=parent_fitness, 
                y=offspring_fitness, 
                ax=axs[idx], 
                scatter=False, 
                line_kws={'color': 'red', 'label': 'Linear Fit'}
            )

            # Quadratic (second-order polynomial) regression line
            quadratic_fit = np.poly1d(np.polyfit(parent_fitness, offspring_fitness, 2))
            x_vals = np.linspace(parent_fitness.min(), parent_fitness.max(), 500)
            axs[idx].plot(x_vals, quadratic_fit(x_vals), color='orange', linestyle='--', label='Quadratic Fit')

            # KDE plot
            try:
                if len(offspring_fitness) > 10 and len(parent_fitness) > 10:  # Ensure enough data points
                    if np.std(parent_fitness) > 1e-6 and np.std(offspring_fitness) > 1e-6:  # Ensure variability
                        sns.kdeplot(
                            x=parent_fitness, 
                            y=offspring_fitness, 
                            ax=axs[idx], 
                            cmap="Blues", 
                            fill=True, 
                            alpha=0.3, 
                            label="KDE"
                        )
            except Exception as e:
                print(f"Warning: KDE plot failed for model {model} due to {e}")
            # Add reference line (x = y)
            min_val = min(parent_fitness.min(), offspring_fitness.min())
            max_val = max(parent_fitness.max(), offspring_fitness.max())
            axs[idx].plot([min_val, max_val], [min_val, max_val], 'k--', label='x = y')
        
        # Customize subplot
        axs[idx].set_title(f"{model.capitalize()} Model - {len(diploids)} \n mating strategy: {mating_strategy}")
        axs[idx].set_xlabel("Mean Parent Fitness")
        axs[idx].set_ylabel("Offspring Fitness")
        axs[idx].grid(True)
        axs[idx].legend()

    # Add a global title
    plt.suptitle("Parent vs Offspring Fitness Comparison\n(Mean Parent Fitness on X-Axis)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    output_path = os.path.join(Resu_path, 'parent_offspring_fitness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parent_fitness_vs_offspring_distance(diploid_dict, Resu_path, mating_strategy):
    """
    Optimized version of the plotting function with improved performance.
    
    Key improvements:
    1. Vectorized distance calculations
    2. Pre-allocation of arrays
    3. Optimized pair generation
    4. Cached property calculations
    5. Reduced redundant computations
    6. Optimized memory usage
    """


    models = ["dominant", "recessive", "codominant"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    def calculate_distances_batch(genomes):
        """Vectorized distance calculation for a batch of genomes"""
        return np.array([
            calculate_genomic_distance(g1, g2) 
            for g1, g2 in combinations(genomes, 2)
        ])

    def get_fitness_pairs(organisms):
        """Efficient extraction of fitness pairs"""
        return np.array([
            (org1.avg_parent_fitness + org2.avg_parent_fitness) / 2 
            for org1, org2 in combinations(organisms, 2)
        ])

    for idx, model in enumerate(models):
        diploids = diploid_dict.get(model, [])
        if not diploids:
            axs[idx].set_title(f"{model.capitalize()} Model (No Data)")
            continue

        # Pre-compute effective genomes to avoid repeated calls
        effective_genomes = [org._get_effective_genome() for org in diploids]
        
        # Vectorized calculations
        distances = calculate_distances_batch(effective_genomes)
        parent_fitnesses = get_fitness_pairs(diploids)

        if len(distances) > 0:
            # Create mask for valid data points
            valid_mask = ~np.isnan(distances) & ~np.isnan(parent_fitnesses)
            distances = distances[valid_mask]
            parent_fitnesses = parent_fitnesses[valid_mask]

            # Plotting with optimized parameters
            ax = axs[idx]
            
            # Scatter plot with reduced alpha for large datasets
            alpha = min(0.7, 5000 / len(distances))
            ax.scatter(distances, parent_fitnesses, alpha=alpha, color="blue", label="Data", s=10)

            # Efficient regression calculations
            if len(distances) > 1:
                # Linear regression
                z = np.polyfit(distances, parent_fitnesses, 1)
                x_range = np.linspace(distances.min(), distances.max(), 100)
                ax.plot(x_range, np.poly1d(z)(x_range), color='red', label='Linear Fit')

                # Quadratic regression
                z2 = np.polyfit(distances, parent_fitnesses, 2)
                ax.plot(x_range, np.poly1d(z2)(x_range), '--', color='orange', label='Quadratic Fit')

                # KDE plot with optimized parameters
                try:
                    if (np.std(distances) > 1e-6 and 
                        np.std(parent_fitnesses) > 1e-6 and 
                        len(distances) >= 50):  # Only plot KDE if enough points
                        sns.kdeplot(
                            x=distances,
                            y=parent_fitnesses,
                            ax=ax,
                            levels=5,  # Reduced number of levels
                            cmap='Blues',
                            fill=True,
                            alpha=0.3,
                            label="KDE",
                            bw_adjust=1.5  # Increased bandwidth for smoother plot
                        )
                except Exception as e:
                    print(f"Warning: KDE plot failed for {model} model: {e}")

            # Set plot properties
            ax.set_title(f"{model.capitalize()} Model")
            ax.set_xlabel('Genomic Distance Between Offspring')
            ax.set_ylabel('Average Parent Fitness')
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.suptitle(f"Genomic Distance vs Average Parent Fitness Comparison ({mating_strategy} Strategy)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save with optimized parameters
    output_path = os.path.join(Resu_path, f'parent_fitness_vs_offspring_distance_{mating_strategy}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parent_offspring_heatmap(diploid_offspring_dict, Resu_path):
    """
    Create heatmap plots comparing Parent 1 fitness (x-axis) and Parent 2 fitness (y-axis),
    with the color representing the average offspring fitness for each fitness model.

    Parameters:
    -----------
    diploid_offspring_dict (dict): Keys are fitness model names ("dominant", "recessive", "codominant"),
        and values are lists of DiploidOrganism instances.
    Resu_path (str): The directory path where the heatmap images will be saved.

    Returns:
    --------
    None
    """

    models = ["dominant", "recessive", "codominant"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    for idx, model in enumerate(tqdm(models, desc="Processing Models")):
        offspring_list = diploid_offspring_dict.get(model, [])
        if not offspring_list:
            axs[idx].set_title(f"{model.capitalize()} Model (No Data)")
            continue

        # Extract Parent 1 fitness, Parent 2 fitness, and Offspring fitness
        parent1_fit = np.array([d.parent1_fitness for d in offspring_list])
        parent2_fit = np.array([d.parent2_fitness for d in offspring_list])
        offspring_fit = np.array([d.fitness for d in offspring_list])

        # Create a 2D histogram using Parent 1 and Parent 2 fitness
        h, xedges, yedges = np.histogram2d(
            parent1_fit, parent2_fit, bins=20, weights=offspring_fit, density=False
        )
        # Calculate the count of pairs in each bin (to get the average offspring fitness per bin)
        counts, _, _ = np.histogram2d(parent1_fit, parent2_fit, bins=20)
        average_offspring_fitness = np.divide(h, counts, out=np.zeros_like(h), where=counts > 0)

        # Plot the heatmap
        im = axs[idx].imshow(
            average_offspring_fitness.T,
            origin='lower',
            aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap='viridis'
        )
        axs[idx].set_xlabel('Parent 1 Fitness')
        axs[idx].set_ylabel('Parent 2 Fitness')
        axs[idx].set_title(f"{model.capitalize()} Model")
        fig.colorbar(im, ax=axs[idx], label='Avg Offspring Fitness')

    plt.tight_layout()
    output_path = os.path.join(Resu_path, 'parent_offspring_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

#######################################################################
# Simulation functions
#######################################################################
def run_simulation(num_generations, environment, max_population_size=10000):
    """Run the evolutionary simulation with garbage collection and memory optimization."""
    initial_organism = Organism(environment)
    population = [initial_organism]
    all_organisms = [initial_organism]
    individual_fitness = defaultdict(list)
    generation_stats = []

    for gen in tqdm(range(num_generations), desc="Running Generations"):
        next_generation = []
        gen_fitness = []

        for org in population:
            child1, child2 = org.reproduce()
            child1.mutate()
            child2.mutate()
            individual_fitness[child1.id].append((gen + 1, child1.fitness))
            individual_fitness[child2.id].append((gen + 1, child2.fitness))
            gen_fitness.extend([child1.fitness, child2.fitness])
            next_generation.extend([child1, child2])
            all_organisms.extend([child1, child2])

        # Limit population size
        if len(next_generation) > max_population_size:
            next_generation = sorted(next_generation, key=lambda x: x.fitness, reverse=True)[:max_population_size]

        population = next_generation
        stats = {
            'generation': gen + 1,
            'population_size': len(population),
            'avg_fitness': np.mean(gen_fitness),
            'max_fitness': np.max(gen_fitness),
            'min_fitness': np.min(gen_fitness),
            'std_fitness': np.std(gen_fitness)
        }
        generation_stats.append(stats)
        log.info(f"Generation {gen+1}: Pop size = {stats['population_size']}, Avg fitness = {stats['avg_fitness']:.4f}")

        # Garbage collection and memory monitoring
        gc.collect()
        monitor_memory()

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
    aparser.add_argument("--mating_strategy", type=str, default="one_to_one", help=["one_to_one", "all_vs_all", "mating_types"]) #["one_to_one", "all_vs_all", "mating_types"],
    aparser.add_argument("--output_dir", type=str, default="Resu", help="Output directory for results")


    args = aparser.parse_args()

    # val Resu directory one back from the dir __file__
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output_dir)):
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output_dir))

    Resu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output_dir)
    
    # init logging
    log = init_log(Resu_path)
    start_time = time.time()

    # Initialize logging
    log.info(
        f"Starting {__file__} with arguments: {args}\n"
        f"Command line: {' '.join(sys.argv)}\n"
        f"Output directory: {Resu_path}\n"
        f"CPU count: {os.cpu_count()}\n"
        f"Total memory (GB): {psutil.virtual_memory().total/1024**3:.2f}\n"
        f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    log.info(f"LSF job details: {', '.join(_get_lsf_job_details())}")
    
    # Create environment
    environment = Environment(genome_size=args.genome_size, beta=args.beta, rho=args.rho)
    log.info(f"Environment created with genome size = {args.genome_size}, beta = {args.beta}, rho = {args.rho}")

    # Run simulation
    all_organisms, individual_fitness, generation_stats = run_simulation(args.generations, environment)
    gc.collect()  # Final garbage collection

    # Create visualizations
    log.info("Creating visualizations")
    plot_detailed_fitness(individual_fitness, generation_stats , Resu_path)
    plot_relationship_tree(all_organisms, Resu_path)

    # Mate last generation to create diploid organisms
    log.info(f"Mating last generation to create diploid organisms using {MatingStrategy(args.mating_strategy)} strategy")
    last_generation = [org for org in all_organisms if org.generation == args.generations]
    # Mate last generation to create diploid organisms for each model
    diploid_offspring_dict = mate_last_generation(
        last_generation, 
        mating_strategy=MatingStrategy(args.mating_strategy),   
        fitness_models=["dominant", "recessive", "codominant"]
    )

    mating_stats = calculate_mating_statistics(diploid_offspring_dict)
    
    # Plot parent-offspring fitness comparison
    log.info("Creating parent-offspring fitness comparison plot")
    plot_parent_offspring_fitness(diploid_offspring_dict, Resu_path , args.mating_strategy)

    #plot parent fitness vs offspring genomic distance
    log.info("Creating parent fitness vs offspring genomic distance plot")
    plot_parent_fitness_vs_offspring_distance(diploid_offspring_dict, Resu_path , args.mating_strategy)
    
    # plot heatmap parent-offspring fitness comparison
    log.info("Creating parent-offspring fitness heatmap plot")
    plot_parent_offspring_heatmap(diploid_offspring_dict, Resu_path)
   
   
    # create summary statistics

    summary_stats = summarize_simulation_stats(generation_stats, individual_fitness, diploid_offspring_dict)
    log_simulation_summary(log, summary_stats)

    log.info(f"Simulation completed. Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    log.info(f"Memory usage: {psutil.virtual_memory().percent:.2f}%")
    log.info(f"CPU usage: {psutil.cpu_percent(interval=1):.2f}%")
    log.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"LSF job details: {', '.join(_get_lsf_job_details())}")
    log.info("=== END OF SIMULATION ===")

if __name__ == "__main__":
    main()
    

"""
example usage: 
bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 4 --genome_size 128 --beta 0.5 --rho 0.25 --mating_strategy all_vs_all --output_dir /home/labs/pilpel/barc/sexy_yeast/tttt
"""

