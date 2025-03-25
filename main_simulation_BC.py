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

def init_J(N, beta, rho, random_state=None):
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

def calculate_prs(genome):
    """
    Calculate a simple Polygenic Risk Score (PRS) for a genome.
    For this simulation, we use an effect size of 1 for all SNPs.
    
    Parameters:
    -----------
    genome : numpy.ndarray
        Array of -1 or 1 values representing the genome
        
    Returns:
    --------
    float
        The calculated PRS score
    """
    # Convert from -1/1 encoding to 0/1 encoding for SNP risk alleles
    # We'll consider a value of 1 as the "risk allele"
    risk_alleles = (genome + 1) / 2  # Converts -1 to 0 and 1 to 1
    
    # Calculate PRS (sum of risk alleles * effect size)
    # Since effect_size = 1 for all SNPs, this is just the sum of risk alleles
    prs = np.sum(risk_alleles)
    
    return prs

def calculate_diploid_prs(diploid_organism):
    """
    Calculate the Polygenic Risk Score (PRS) for a diploid organism
    based on its effective genome.
    
    Parameters:
    -----------
    diploid_organism : DiploidOrganism
        The diploid organism to calculate PRS for
        
    Returns:
    --------
    float
        The calculated PRS score
    """
    # Get the effective genome based on inheritance model
    effective_genome = diploid_organism._get_effective_genome()
    
    # Calculate PRS using the effective genome
    return calculate_prs(effective_genome)

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
        # return 1.0 / (1.0 + np.exp(-energy))
        return energy
    
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
        # return 1.0 / (1.0 + np.exp(-energy))
        return energy

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

def plot_parent_offspring_fitness(diploid_dict, Resu_path, mating_strategy):
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
    mating_strategy : str
        The mating strategy used (for plot title)
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

        # Customize subplot - Do this for all cases
        axs[idx].set_title(f"{model.capitalize()} Model - {len(diploids)} \n mating strategy: {mating_strategy}")
        axs[idx].set_xlabel("Mean Parent Fitness")
        axs[idx].set_ylabel("Offspring Fitness")
        axs[idx].grid(True)

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

            # Linear regression line and R²
            linear_r = np.corrcoef(parent_fitness, offspring_fitness)[0,1]
            linear_r2 = linear_r**2
            linear_z = np.polyfit(parent_fitness, offspring_fitness, 1)
            linear_model = np.poly1d(linear_z)
            
            # Plot linear fit
            sns.regplot(
                x=parent_fitness, 
                y=offspring_fitness, 
                ax=axs[idx], 
                scatter=False, 
                line_kws={'color': 'red', 'label': f'Linear (R²: {linear_r2:.2f})'}
            )

            # Quadratic (second-order polynomial) regression line and R²
            quad_z = np.polyfit(parent_fitness, offspring_fitness, 2)
            quad_model = np.poly1d(quad_z)
            x_vals = np.linspace(parent_fitness.min(), parent_fitness.max(), 500)
            
            # Calculate quadratic R²
            y_pred = quad_model(parent_fitness)
            y_mean = np.mean(offspring_fitness)
            quad_r2 = 1 - (np.sum((offspring_fitness - y_pred)**2) / 
                          np.sum((offspring_fitness - y_mean)**2))
            
            # Plot quadratic fit
            axs[idx].plot(x_vals, quad_model(x_vals), 
                       color='orange', linestyle='--', 
                       label=f'Quadratic (R²: {quad_r2:.2f})')

            # KDE plot
            try:
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
            
            # Add detailed statistics text
            from scipy import stats
            slope, intercept, _, p_value, std_err = stats.linregress(parent_fitness, offspring_fitness)
            spearman_r, spearman_p = stats.spearmanr(parent_fitness, offspring_fitness)
            
            stats_text = (f"Linear R²: {linear_r2:.3f}\n"
                         f"Slope: {slope:.3f}\n"
                         f"Quad R²: {quad_r2:.3f}\n"
                         f"Pearson r: {linear_r:.3f}\n"
                         f"Spearman ρ: {spearman_r:.3f}")
            
            axs[idx].text(0.05, 0.95, stats_text, transform=axs[idx].transAxes, 
                       fontsize=9, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Only add legend if we have labeled elements
            axs[idx].legend(loc='lower right')
        else:
            # No data message for plots without enough data
            axs[idx].text(0.5, 0.5, f"Insufficient data points ({len(diploids)})\nNeed > 10 for analysis", 
                        ha='center', va='center', transform=axs[idx].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))

    # Add a global title
    plt.suptitle("Parent vs Offspring Fitness Comparison\n(Mean Parent Fitness on X-Axis)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    output_path = os.path.join(Resu_path, f'parent_offspring_fitness_{mating_strategy}.png')
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

def plot_offspring_vs_min_max_parent_fitness(diploid_dict, Resu_path, mating_strategy):
    """
    Create plots comparing offspring fitness to minimum and maximum parent fitness.
    
    Parameters:
    -----------
    diploid_dict : dict
        Dictionary with keys as fitness model names and values as lists of DiploidOrganism instances.
    Resu_path : str
        Directory path where the resulting figures will be saved.
    mating_strategy : str
        The mating strategy used (for plot title)
    """
    # Define the order and colors for the three models
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "blue", "recessive": "green", "codominant": "purple"}
    
    # Plot 1: Offspring vs Max Parent Fitness
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    # Plot 2: Offspring vs Min Parent Fitness
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    for idx, model in enumerate(tqdm(models, desc=f"Creating min-max plots for {mating_strategy}")):
        diploids = diploid_dict.get(model, [])
        if not diploids:
            axs1[idx].set_title(f"{model.capitalize()} Model (No Data)")
            axs2[idx].set_title(f"{model.capitalize()} Model (No Data)")
            continue  # Skip if no data for this model
        
        # Extract data
        offspring_fitness = np.array([org.fitness for org in diploids])
        min_parent_fitness = np.array([min(org.parent1_fitness, org.parent2_fitness) for org in diploids])
        max_parent_fitness = np.array([max(org.parent1_fitness, org.parent2_fitness) for org in diploids])
        
        # Set up basic subplot properties for both plots regardless of data amount
        axs1[idx].set_title(f"{model.capitalize()} Model - {len(diploids)}")
        axs1[idx].set_xlabel("Max Parent Fitness")
        axs1[idx].set_ylabel("Offspring Fitness")
        axs1[idx].grid(True)
        
        axs2[idx].set_title(f"{model.capitalize()} Model - {len(diploids)}")
        axs2[idx].set_xlabel("Min Parent Fitness")
        axs2[idx].set_ylabel("Offspring Fitness")
        axs2[idx].grid(True)
        
        if len(offspring_fitness) > 10:
            # --- PLOT 1: Offspring vs Max Parent Fitness ---
            
            # Scatter plot
            sns.scatterplot(
                x=max_parent_fitness, 
                y=offspring_fitness, 
                ax=axs1[idx], 
                alpha=0.7, 
                color=colors[model], 
                label="Data"
            )
            
            # Linear regression line
            sns.regplot(
                x=max_parent_fitness, 
                y=offspring_fitness, 
                ax=axs1[idx], 
                scatter=False, 
                line_kws={'color': 'red', 'label': 'Linear Fit'}
            )
            
            # Calculate linear R²
            max_parent_r_linear = np.corrcoef(max_parent_fitness, offspring_fitness)[0,1]
            max_parent_r2_linear = max_parent_r_linear**2
            
            # Quadratic regression
            max_z2 = np.polyfit(max_parent_fitness, offspring_fitness, 2)
            max_quad_model = np.poly1d(max_z2)
            max_x_vals = np.linspace(max_parent_fitness.min(), max_parent_fitness.max(), 500)
            
            # Calculate quadratic R²
            max_y_pred = max_quad_model(max_parent_fitness)
            max_y_mean = np.mean(offspring_fitness)
            max_r2_quad = 1 - (np.sum((offspring_fitness - max_y_pred)**2) / 
                             np.sum((offspring_fitness - max_y_mean)**2))
            
            # Plot quadratic fit
            axs1[idx].plot(max_x_vals, max_quad_model(max_x_vals), 
                         color='orange', linestyle='--', 
                         label=f'Quadratic Fit')
            
            # Add statistics text
            max_stats_text = (f"Linear R²: {max_parent_r2_linear:.3f}\n"
                            f"Quadratic R²: {max_r2_quad:.3f}\n"
                            f"Pearson r: {max_parent_r_linear:.3f}")
            
            axs1[idx].text(0.05, 0.95, max_stats_text, transform=axs1[idx].transAxes, 
                         fontsize=9, va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # KDE plot
            try:
                if np.std(max_parent_fitness) > 1e-6 and np.std(offspring_fitness) > 1e-6:
                    sns.kdeplot(
                        x=max_parent_fitness, 
                        y=offspring_fitness, 
                        ax=axs1[idx], 
                        cmap="Blues", 
                        fill=True, 
                        alpha=0.3, 
                        label="KDE"
                    )
            except Exception as e:
                print(f"Warning: KDE plot failed for max parent in model {model} due to {e}")
            
            # Add reference line (x = y)
            min_val = min(max_parent_fitness.min(), offspring_fitness.min())
            max_val = max(max_parent_fitness.max(), offspring_fitness.max())
            axs1[idx].plot([min_val, max_val], [min_val, max_val], 'k--', label='x = y')
            
            # Add legend
            axs1[idx].legend(loc='lower right')
            
            # --- PLOT 2: Offspring vs Min Parent Fitness ---
            
            # Scatter plot
            sns.scatterplot(
                x=min_parent_fitness, 
                y=offspring_fitness, 
                ax=axs2[idx], 
                alpha=0.7, 
                color=colors[model], 
                label="Data"
            )
            
            # Linear regression line
            sns.regplot(
                x=min_parent_fitness, 
                y=offspring_fitness, 
                ax=axs2[idx], 
                scatter=False, 
                line_kws={'color': 'red', 'label': 'Linear Fit'}
            )
            
            # Calculate linear R²
            min_parent_r_linear = np.corrcoef(min_parent_fitness, offspring_fitness)[0,1]
            min_parent_r2_linear = min_parent_r_linear**2
            
            # Quadratic regression
            min_z2 = np.polyfit(min_parent_fitness, offspring_fitness, 2)
            min_quad_model = np.poly1d(min_z2)
            min_x_vals = np.linspace(min_parent_fitness.min(), min_parent_fitness.max(), 500)
            
            # Calculate quadratic R²
            min_y_pred = min_quad_model(min_parent_fitness)
            min_y_mean = np.mean(offspring_fitness)
            min_r2_quad = 1 - (np.sum((offspring_fitness - min_y_pred)**2) / 
                             np.sum((offspring_fitness - min_y_mean)**2))
            
            # Plot quadratic fit
            axs2[idx].plot(min_x_vals, min_quad_model(min_x_vals), 
                         color='orange', linestyle='--', 
                         label=f'Quadratic Fit')
            
            # Add statistics text
            min_stats_text = (f"Linear R²: {min_parent_r2_linear:.3f}\n"
                            f"Quadratic R²: {min_r2_quad:.3f}\n"
                            f"Pearson r: {min_parent_r_linear:.3f}")
            
            axs2[idx].text(0.05, 0.95, min_stats_text, transform=axs2[idx].transAxes, 
                         fontsize=9, va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # KDE plot
            try:
                if np.std(min_parent_fitness) > 1e-6 and np.std(offspring_fitness) > 1e-6:
                    sns.kdeplot(
                        x=min_parent_fitness, 
                        y=offspring_fitness, 
                        ax=axs2[idx], 
                        cmap="Blues", 
                        fill=True, 
                        alpha=0.3, 
                        label="KDE"
                    )
            except Exception as e:
                print(f"Warning: KDE plot failed for min parent in model {model} due to {e}")
            
            # Add reference line (x = y)
            min_val = min(min_parent_fitness.min(), offspring_fitness.min())
            max_val = max(min_parent_fitness.max(), offspring_fitness.max())
            axs2[idx].plot([min_val, max_val], [min_val, max_val], 'k--', label='x = y')
            
            # Add legend
            axs2[idx].legend(loc='lower right')
        else:
            # Add information message for not enough data
            axs1[idx].text(0.5, 0.5, f"Insufficient data points ({len(diploids)})\nNeed > 10 for analysis", 
                         ha='center', va='center', transform=axs1[idx].transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            
            axs2[idx].text(0.5, 0.5, f"Insufficient data points ({len(diploids)})\nNeed > 10 for analysis", 
                         ha='center', va='center', transform=axs2[idx].transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
    
    # Set global titles
    fig1.suptitle(f"Offspring Fitness vs Maximum Parent Fitness\n({mating_strategy} Strategy)")
    fig2.suptitle(f"Offspring Fitness vs Minimum Parent Fitness\n({mating_strategy} Strategy)")
    
    # Adjust layout and save figures
    plt.figure(fig1.number)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(Resu_path, f'offspring_vs_max_parent_fitness_{mating_strategy}.png'), dpi=300, bbox_inches='tight')
    
    plt.figure(fig2.number)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(Resu_path, f'offspring_vs_min_parent_fitness_{mating_strategy}.png'), dpi=300, bbox_inches='tight')
        
    # Close all figures
    plt.close(fig1)
    plt.close(fig2)

def plot_parent_genomic_distance_vs_offspring_fitness(diploid_dict, Resu_path, mating_strategy):
    """
    Create a figure showing the relationship between parent-parent genomic distance (x-axis)
    and offspring fitness (y-axis).
    
    Parameters:
    -----------
    diploid_dict : dict
        Dictionary with keys as fitness model names and values as lists of DiploidOrganism instances.
    Resu_path : str
        Directory path where the resulting figure will be saved.
    mating_strategy : str
        The mating strategy used (for plot title)
    """
    models = ["dominant", "recessive", "codominant"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    for idx, model in enumerate(tqdm(models, desc=f"Creating parent distance plots for {mating_strategy}")):
        diploids = diploid_dict.get(model, [])
        if not diploids:
            axs[idx].set_title(f"{model.capitalize()} Model (No Data)")
            continue  # Skip if no data for this model
        
        # Calculate genomic distances between parents for each offspring
        parent_distances = []
        offspring_fitness = []
        
        # Process in small batches to avoid memory issues
        for i in range(0, len(diploids), 100):
            batch = diploids[i:i+100]
            for org in batch:
                distance = calculate_genomic_distance(org.allele1, org.allele2)
                parent_distances.append(distance)
                offspring_fitness.append(org.fitness)
            gc.collect()  # Force garbage collection after each batch
        
        parent_distances = np.array(parent_distances)
        offspring_fitness = np.array(offspring_fitness)
        
        if len(parent_distances) > 10:
            # Scatter plot
            axs[idx].scatter(
                parent_distances, 
                offspring_fitness, 
                alpha=0.5, 
                color='blue', 
                s=10
            )
            
            # Linear regression
            z = np.polyfit(parent_distances, offspring_fitness, 1)
            x_vals = np.linspace(parent_distances.min(), parent_distances.max(), 100)
            axs[idx].plot(x_vals, np.poly1d(z)(x_vals), 'r-', label=f'Linear (R²: {np.corrcoef(parent_distances, offspring_fitness)[0,1]**2:.2f})')
            
            # Quadratic regression
            z2 = np.polyfit(parent_distances, offspring_fitness, 2)
            axs[idx].plot(x_vals, np.poly1d(z2)(x_vals), 'g--', label='Quadratic Fit')
        
        # Customize subplot
        axs[idx].set_title(f"{model.capitalize()} Model - {len(diploids)}")
        axs[idx].set_xlabel("Genomic Distance Between Parents")
        axs[idx].set_ylabel("Offspring Fitness")
        axs[idx].grid(True)
        axs[idx].legend()
    
    # Add global title
    plt.suptitle(f"Parent Genomic Distance vs Offspring Fitness\n({mating_strategy} Strategy)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(Resu_path, f'parent_distance_vs_offspring_fitness_{mating_strategy}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_offspring_fitness_vs_offspring_prs(diploid_dict, Resu_path, mating_strategy):
    """
    Create plots showing the relationship between offspring's own PRS scores and offspring fitness.
    
    Parameters:
    -----------
    diploid_dict : dict
        Dictionary with keys as fitness model names and values as lists of DiploidOrganism instances.
    Resu_path : str
        Directory path where the resulting figure will be saved.
    mating_strategy : str
        The mating strategy used (for plot title)
    """
    models = ["dominant", "recessive", "codominant"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    
    for idx, model in enumerate(tqdm(models, desc=f"Creating Offspring PRS plots for {mating_strategy}")):
        diploids = diploid_dict.get(model, [])
        if not diploids:
            axs[idx].set_title(f"{model.capitalize()} Model (No Data)")
            continue  # Skip if no data for this model
        
        # Calculate offspring PRS and collect offspring fitness
        offspring_prs = []
        offspring_fitness = []
        parent_avg_prs = []
        
        # Process in small batches to avoid memory issues
        for i in range(0, len(diploids), 100):
            batch = diploids[i:i+100]
            for org in batch:
                # Calculate offspring's own PRS
                org_prs = calculate_diploid_prs(org)
                offspring_prs.append(org_prs)
                
                # Calculate average parent PRS for comparison
                p1_prs = calculate_prs(org.allele1)
                p2_prs = calculate_prs(org.allele2)
                parent_avg_prs.append((p1_prs + p2_prs) / 2)
                
                # Get offspring fitness
                offspring_fitness.append(org.fitness)
            
            gc.collect()  # Force garbage collection after each batch
        
        # Convert to numpy arrays for easier manipulation
        offspring_prs = np.array(offspring_prs)
        offspring_fitness = np.array(offspring_fitness)
        parent_avg_prs = np.array(parent_avg_prs)
        
        if len(offspring_prs) > 10:
            # Scatter plot
            axs[idx].scatter(
                offspring_prs, 
                offspring_fitness, 
                alpha=0.5, 
                color='purple', 
                s=10,
                label="Offspring PRS"
            )
            
            # Linear regression
            z = np.polyfit(offspring_prs, offspring_fitness, 1)
            x_vals = np.linspace(offspring_prs.min(), offspring_prs.max(), 100)
            r_squared = np.corrcoef(offspring_prs, offspring_fitness)[0,1]**2
            axs[idx].plot(x_vals, np.poly1d(z)(x_vals), 'r-', 
                      label=f'Linear (R²: {r_squared:.2f})')
            
            # Quadratic regression
            z2 = np.polyfit(offspring_prs, offspring_fitness, 2)
            axs[idx].plot(x_vals, np.poly1d(z2)(x_vals), 'g--', label='Quadratic Fit')
            
            # Calculate additional statistics
            # Compare offspring PRS vs parent average PRS correlation with fitness
            corr_offspring = np.corrcoef(offspring_prs, offspring_fitness)[0,1]
            corr_parents = np.corrcoef(parent_avg_prs, offspring_fitness)[0,1]
            
            # Customize subplot
            axs[idx].set_title(f"{model.capitalize()} Model - {len(diploids)}")
            axs[idx].set_xlabel("Offspring PRS")
            axs[idx].set_ylabel("Offspring Fitness")
            axs[idx].grid(True)
            axs[idx].legend()
            
            # Add annotations with correlation values
            axs[idx].annotate(
                f"Offspring PRS-Fitness Correlation: {corr_offspring:.3f}\n"
                f"R²: {r_squared:.3f}\n"
                f"Parent Avg PRS-Fitness Correlation: {corr_parents:.3f}", 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top'
            )
    
    # Add global title
    plt.suptitle(f"Offspring PRS vs Offspring Fitness\n({mating_strategy} Strategy)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(Resu_path, f'offspring_prs_vs_offspring_fitness_{mating_strategy}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


#######################################################################
# Simulation functions
#######################################################################
def run_simulation(num_generations, environment, max_population_size=100000):
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
    aparser.add_argument("--mating_strategy", type=str, default="one_to_one", 
                    choices=["one_to_one", "all_vs_all", "mating_types"], 
                    help="Strategy for organism mating")
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
    
    # Plot parent-offspring fitness comparison (PLOT 1)
    log.info("Creating parent-offspring fitness comparison plot")
    plot_parent_offspring_fitness(diploid_offspring_dict, Resu_path, args.mating_strategy)

    # Plot parent genomic distance vs offspring fitness (PLOT 2 - corrected version)
    log.info("Creating parent genomic distance vs offspring fitness plot")
    plot_parent_genomic_distance_vs_offspring_fitness(diploid_offspring_dict, Resu_path, args.mating_strategy)
    
    # Plot offspring fitness vs min/max parent fitness (PLOTS 3, 4, 5)
    log.info("Creating offspring vs min/max parent fitness plots")
    plot_offspring_vs_min_max_parent_fitness(diploid_offspring_dict, Resu_path, args.mating_strategy)
    
    # Plot parent PRS vs offspring fitness (PLOT 6 - new)
    log.info("Creating offspring PRS vs offspring fitness plot")
    plot_offspring_fitness_vs_offspring_prs(diploid_offspring_dict, Resu_path, args.mating_strategy)

    
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
    log.info(f"The log file is located at: {os.path.join(Resu_path, 'sexy_yeast.log')}")
    log.info("=== END OF SIMULATION ===")


if __name__ == "__main__":
    main()
    

"""
example usage: 
bsub -q gsla-cpu -R rusage[mem=42000] /home/labs/pilpel/barc/sexy_yeast/src/main_simulation_BC.py --generations 4 --genome_size 128 --beta 0.5 --rho 0.25 --mating_strategy all_vs_all --output_dir /home/labs/pilpel/barc/sexy_yeast/tttt
"""