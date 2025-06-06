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
import scipy.stats
import scipy
from scipy import stats
import json
from datetime import datetime



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

def init_log(Resu_path, log_level="INFO"):
    """Initialize logging with specified log level"""
    # Set log level
    level = getattr(log, log_level.upper(), log.INFO)
    
    # Initialize logging
    log.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    
    # Add file handler
    path = os.path.join(Resu_path, 'sexy_yeast.log')
    fh = log.FileHandler(path, mode='w')
    fh.setLevel(level)
    log.getLogger().addHandler(fh)

    return log

#######################################################################
# Classes
#######################################################################

class MatingStrategy(Enum):    
    ONE_TO_ONE = "one_to_one"
    ALL_VS_ALL = "all_vs_all"
    MATING_TYPES = "mating_types"

class MatingType(Enum):
    A = "A"
    ALPHA = "alpha"

class AlternativeFitnessMethod(Enum):
    """Enumeration of available fitness calculation methods."""
    SHERRINGTON_KIRKPATRICK = "sherrington_kirkpatrick"  # Original complex method
    SINGLE_POSITION = "single_position"  # Simple method based on a single position
    ADDITIVE = "additive"  # Simple additive method with no interactions

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

        Converts alleles from -1/1 to 0/1:
        - -1 (recessive, a) -> 0
        -  1 (dominant, A)  -> 1

        Inheritance models:
        - Dominant: AA, Aa, aA -> 1 ; aa -> 0
        - Recessive: aa -> 1 ; others -> 0
        - Co-Dominant:
            AA -> 1
            Aa, aA -> 0.5
            aa -> 0
        """

        # Convert alleles from -1/1 to 0/1
        allele1 = (self.allele1 + 1) // 2
        allele2 = (self.allele2 + 1) // 2

        if self.fitness_model == "dominant":
            return np.where((allele1 == 1) | (allele2 == 1), 1.0, 0.0)
        
        elif self.fitness_model == "recessive":
            return np.where((allele1 == 0) & (allele2 == 0), 1.0, 0.0)
        
        elif self.fitness_model == "codominant":
            both_1 = (allele1 == 1) & (allele2 == 1)
            both_0 = (allele1 == 0) & (allele2 == 0)
            mixed = (allele1 != allele2)

            effective = np.where(both_1, 1.0, 0.0)
            effective = np.where(mixed, 0.5, effective)
            return effective

        else:
            raise ValueError(f"Unknown fitness model: {self.fitness_model}")

    def calculate_fitness(self):
        """Calculate fitness using the effective genome and the environment's calculation method."""
        effective_genome = self._get_effective_genome()
        return self.environment.calculate_fitness(effective_genome)

class OrganismWithMatingType:
    def __init__(self, organism, mating_type):
        self.organism = organism
        self.mating_type = mating_type

class Environment:
    """
    Represents an environment with a fitness landscape for simulating evolutionary dynamics.
    Attributes:
        genome_size (int): The size of the genome.
        beta (float): A parameter controlling the ruggedness of the fitness landscape. Default is 0.5.
        rho (float): A parameter controlling the correlation between genome sites in the fitness landscape. Default is 0.25.
        seed (int or None): A seed for the random number generator to ensure reproducibility. Default is None.
        h (numpy.ndarray): The initialized fitness contributions of individual genome sites.
        J (numpy.ndarray): The initialized interaction matrix between genome sites.
    Methods:
        calculate_fitness(genome):
            Calculates the fitness of a given genome based on the fitness landscape.
    """
    def __init__(self, genome_size, beta=0.5, rho=0.25, seed=None, 
                 fitness_method=AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK):
        self.genome_size = genome_size
        self.beta = beta
        self.rho = rho
        self.seed = seed
        self.fitness_method = fitness_method
        
        # Use seeded RNG only for environment initialization
        env_rng = np.random.default_rng(seed)
        
        # Initialize fitness landscape based on the method
        if fitness_method == AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK:
            # Original method - initialize h and J
            self.h = init_h(self.genome_size, self.beta, random_state=env_rng)
            self.J = init_J(self.genome_size, self.beta, self.rho, random_state=env_rng)
            self.alternative_params = None
        else:
            # Alternative method - initialize appropriate parameters
            self.h = None
            self.J = None
            self.alternative_params = init_alternative_fitness(
                self.genome_size, method=fitness_method, random_state=env_rng)
        
    def calculate_fitness(self, genome):
        """Calculate fitness for a genome based on this environment's landscape."""
        if self.fitness_method == AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK:
            # Original complex fitness calculation
            energy = compute_fit_slow(genome, self.h, self.J, F_off=0.0)
            return energy
        else:
            # Alternative simplified fitness calculation
            return calculate_alternative_fitness(genome, self.alternative_params)
    
    def get_fitness_description(self):
        """Return a human-readable description of the fitness calculation method."""
        if self.fitness_method == AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK:
            return "Sherrington-Kirkpatrick model (complex interactions)"
        elif self.fitness_method == AlternativeFitnessMethod.SINGLE_POSITION:
            pos = self.alternative_params["position"]
            return f"Single position model (position {pos} determines fitness)"
        elif self.fitness_method == AlternativeFitnessMethod.ADDITIVE:
            return "Additive model (each position contributes independently)"
        return "Unknown fitness method" 

class Organism:
    """
    Represents an organism in a simulated environment with a genome, fitness, and mutation capabilities.

    Attributes:
        id (str): A unique identifier for the organism.
        environment (Environment): The environment in which the organism exists.
        genome (numpy.ndarray): The genome of the organism, represented as an array of -1 and 1.
        generation (int): The generation number of the organism.
        parent_id (str or None): The ID of the parent organism, if applicable.
        mutation_rate (float): The probability of mutation at each genome site.
        fitness (float): The fitness value of the organism, calculated based on its genome and environment.

    Methods:
        calculate_fitness():
            Calculates and returns the fitness of the organism based on its genome and environment.

        mutate():
            Introduces mutations to the organism's genome based on the mutation rate and updates its fitness.
    """
    def __init__(self, environment, genome=None, generation=0, parent_id=None, 
                mutation_rate=None, genome_seed=None, mutation_seed=None):
        self.id = str(uuid.uuid4())
        self.environment = environment
        
        # Create a new RNG with the provided seed for genome initialization
        genome_rng = np.random.default_rng(genome_seed)
        
        # For mutation RNG, combine the base mutation seed with a unique value for this organism
        # This ensures each organism has its own RNG stream but remains reproducible
        if mutation_seed is not None:
            # Create a value unique to this organism based on generation and a random component
            # Use genome_rng to generate this component for reproducibility
            unique_addition = generation * 1000 + genome_rng.integers(1000)
            organism_mutation_seed = mutation_seed + unique_addition
        else:
            organism_mutation_seed = None
            
        # Use seeded RNG for mutations
        self.rng = np.random.default_rng(organism_mutation_seed)
        
        if genome is None:
            self.genome = genome_rng.choice([-1, 1], environment.genome_size)
        else:
            self.genome = genome.copy()
        self.generation = generation
        self.parent_id = parent_id
        self.mutation_rate = mutation_rate if mutation_rate is not None else 1.0/environment.genome_size
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        return self.environment.calculate_fitness(self.genome)

    def mutate(self):
        # Use the organism's own unseeded RNG for mutations
        mutation_sites = self.rng.random(len(self.genome)) < self.mutation_rate
        self.genome[mutation_sites] *= -1
        self.fitness = self.calculate_fitness()

    def reproduce(self, mutation_seed=None):
        # When creating children, we pass the base mutation seed
        # Each child will generate its unique seed in its __init__ method
        child1 = Organism(self.environment, genome=self.genome,
                        generation=self.generation + 1, parent_id=self.id,
                        mutation_rate=self.mutation_rate, mutation_seed=mutation_seed)
        
        child2 = Organism(self.environment, genome=self.genome,
                        generation=self.generation + 1, parent_id=self.id,
                        mutation_rate=self.mutation_rate, mutation_seed=mutation_seed)
        
        return child1, child2

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
    """
    if not (0 < rho <= 1):
        raise ValueError("rho must be between 0 (exclusive) and 1 (inclusive).")
    
    rng = np.random.default_rng(random_state)
    
    # Handle special case when N=1
    if N == 1:
        return np.zeros((1, 1))  # Return a 1x1 zero matrix
        
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
    if total_elements > 0 and num_nonzero > 0:
        selected_indices = rng.choice(total_elements, size=num_nonzero, replace=False)
        # Map the selected flat indices to row and column indices
        rows = triu_indices[0][selected_indices]
        cols = triu_indices[1][selected_indices]
        # Assign Gaussian-distributed values to the selected positions
        J_upper[rows, cols] = rng.normal(loc=0.0, scale=sig_J, size=num_nonzero)
    
    # Symmetrize the matrix to make Jij symmetric
    Jij = J_upper + J_upper.T

    return Jij

def init_h(N, beta, random_state=None):
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
    
    # return abs differences
    return differences 

def calculate_prs(genome):
    """
    Calculate a simple Polygenic Risk Score (PRS) for a genome.
    Handles both -1/1 and other values properly.
    
    Parameters:
    -----------
    genome : numpy.ndarray
        Array of values representing the genome
        
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
    with proper handling of the codominant model.
    
    Parameters:
    -----------
    diploid_organism : DiploidOrganism
        The diploid organism to calculate PRS for
        
    Returns:
    --------
    float
        The calculated PRS score
    """
    if diploid_organism.fitness_model == "codominant":
        # For codominant model, calculate the PRS directly from alleles
        prs1 = calculate_prs(diploid_organism.allele1)
        prs2 = calculate_prs(diploid_organism.allele2)
        return (prs1 + prs2) / 2
    else:
        # For dominant/recessive models, use the effective genome
        effective_genome = diploid_organism._get_effective_genome()
        
        # Check if effective genome contains zeros (which might happen with codominant model)
        if np.any(np.abs(effective_genome) < 0.5):  # Detect values close to 0
            # Handle these cases specially
            fixed_genome = np.where(np.abs(effective_genome) < 0.5, 0, effective_genome)
            return calculate_prs(fixed_genome)
        else:
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
    """Calculate regression statistics with error handling."""
    from scipy import stats
    import numpy as np
    
    # Check if regression is possible
    if len(x) <= 1 or len(np.unique(x)) <= 1:
        return {
            'linear_r_squared': 0,
            'linear_slope': 0,
            'linear_intercept': 0,
            'linear_p_value': 1,
            'quadratic_r_squared': 0,
            'quadratic_coeffs': [0, 0, 0]
        }
    
    try:
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
    except Exception as e:
        print(f"Warning: Regression analysis failed - {e}")
        return {
            'linear_r_squared': 0,
            'linear_slope': 0,
            'linear_intercept': 0,
            'linear_p_value': 1,
            'quadratic_r_squared': 0,
            'quadratic_coeffs': [0, 0, 0],
            'error': str(e)
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
    
    # Get the last generation's fitness distributions
    # Extract all organisms' fitness values from the final generation
    last_gen_fitness = []
    for org_id, fitness_data in individual_fitness.items():
        for gen, fit in fitness_data:
            if gen == final_gen:
                last_gen_fitness.append(fit)
    
    # Calculate distribution statistics for the last generation
    last_gen_fitness = np.array(last_gen_fitness)
    last_gen_distribution = {
        'count': len(last_gen_fitness),
        'mean': np.mean(last_gen_fitness),
        'median': np.median(last_gen_fitness),
        'std': np.std(last_gen_fitness),
        'min': np.min(last_gen_fitness),
        'max': np.max(last_gen_fitness),
        'percentile_25': np.percentile(last_gen_fitness, 25),
        'percentile_75': np.percentile(last_gen_fitness, 75),
        'skewness': scipy.stats.skew(last_gen_fitness) if len(last_gen_fitness) > 2 else 0,
        'kurtosis': scipy.stats.kurtosis(last_gen_fitness) if len(last_gen_fitness) > 2 else 0
    }
    
    # Add distribution histogram data (10 bins)
    hist, bin_edges = np.histogram(last_gen_fitness, bins=10)
    last_gen_distribution['histogram'] = {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }
    
    # Diploid statistics by model
    diploid_stats = {}
    for model, organisms in diploid_dict.items():
        if organisms:
            parent_fitness = np.array([org.avg_parent_fitness for org in organisms])
            offspring_fitness = np.array([org.fitness for org in organisms])
            
            # Calculate genomic distances between parents
            genomic_distances = np.array([calculate_genomic_distance(org.allele1, org.allele2) for org in organisms])
            
            # Calculate regression statistics for parent fitness vs offspring fitness
            fitness_reg_stats = calculate_regression_stats(parent_fitness, offspring_fitness)
            
            # Calculate regression statistics for genomic distance vs offspring fitness
            distance_reg_stats = calculate_regression_stats(genomic_distances, offspring_fitness)
            
            diploid_stats[model] = {
                'count': len(organisms),
                'avg_parent_fitness': np.mean(parent_fitness),
                'avg_offspring_fitness': np.mean(offspring_fitness),
                'fitness_improvement': np.mean(offspring_fitness - parent_fitness),
                'fitness_regression_stats': fitness_reg_stats,
                # Add genomic distance statistics
                'avg_genomic_distance': np.mean(genomic_distances),
                'min_genomic_distance': np.min(genomic_distances),
                'max_genomic_distance': np.max(genomic_distances),
                'std_genomic_distance': np.std(genomic_distances),
                'distance_regression_stats': distance_reg_stats
            }
    
    return {
        'generations': final_gen,
        'final_population': final_stats['population_size'],
        'initial_fitness': initial_fitness,
        'final_fitness': final_fitness,
        'fitness_improvement_percent': fitness_improvement,
        'fitness_std_final': final_stats['std_fitness'],
        'last_generation_distribution': last_gen_distribution,  # Added distribution data
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
    
    # Log last generation fitness distribution
    last_gen_dist = summary_stats['last_generation_distribution']
    log.info(f"\nLast Generation Fitness Distribution:")
    log.info(f"  Number of Organisms: {last_gen_dist['count']}")
    log.info(f"  Mean Fitness: {last_gen_dist['mean']:.4f}")
    log.info(f"  Median Fitness: {last_gen_dist['median']:.4f}")
    log.info(f"  Standard Deviation: {last_gen_dist['std']:.4f}")
    log.info(f"  Range: {last_gen_dist['min']:.4f} to {last_gen_dist['max']:.4f}")
    log.info(f"  Interquartile Range: {last_gen_dist['percentile_25']:.4f} to {last_gen_dist['percentile_75']:.4f}")
    log.info(f"  Skewness: {last_gen_dist['skewness']:.4f}")
    log.info(f"  Kurtosis: {last_gen_dist['kurtosis']:.4f}")
    
    # Log histogram data in a readable format
    log.info(f"  Fitness Histogram:")
    bin_edges = last_gen_dist['histogram']['bin_edges']
    counts = last_gen_dist['histogram']['counts']
    for i in range(len(counts)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        count = counts[i]
        log.info(f"    {bin_start:.4f} to {bin_end:.4f}: {count} organisms")
    
    log.info("\nDiploid Analysis by Model:")
    for model, stats in summary_stats['diploid_stats'].items():
        log.info(f"\n{model.upper()} Model Statistics:")
        log.info(f"  Number of Organisms: {stats['count']}")
        
        # Fitness statistics
        log.info(f"  Fitness Statistics:")
        log.info(f"    Average Parent Fitness: {stats['avg_parent_fitness']:.4f}")
        log.info(f"    Average Offspring Fitness: {stats['avg_offspring_fitness']:.4f}")
        log.info(f"    Average Fitness Improvement: {stats['fitness_improvement']:.4f}")
        
        # Parent-Offspring Fitness Regression Statistics
        fitness_reg_stats = stats['fitness_regression_stats']
        log.info(f"  Parent-Offspring Fitness Regression:")
        log.info(f"    Linear R²: {fitness_reg_stats['linear_r_squared']:.4f}")
        log.info(f"    Linear Slope: {fitness_reg_stats['linear_slope']:.4f}")
        log.info(f"    Linear P-Value: {fitness_reg_stats['linear_p_value']:.4f}")
        log.info(f"    Quadratic R²: {fitness_reg_stats['quadratic_r_squared']:.4f}")
        
        # Genomic Distance Statistics
        log.info(f"  Genomic Distance Statistics:")
        log.info(f"    Average Genomic Distance: {stats['avg_genomic_distance']:.4f}")
        log.info(f"    Min Genomic Distance: {stats['min_genomic_distance']:.4f}")
        log.info(f"    Max Genomic Distance: {stats['max_genomic_distance']:.4f}")
        log.info(f"    Std Dev of Genomic Distance: {stats['std_genomic_distance']:.4f}")
        
        # Genomic Distance vs Offspring Fitness Regression
        distance_reg_stats = stats['distance_regression_stats']
        log.info(f"  Genomic Distance vs Offspring Fitness Regression:")
        log.info(f"    Linear R²: {distance_reg_stats['linear_r_squared']:.4f}")
        log.info(f"    Linear Slope: {distance_reg_stats['linear_slope']:.4f}")
        log.info(f"    Linear P-Value: {distance_reg_stats['linear_p_value']:.4f}")
        log.info(f"    Quadratic R²: {distance_reg_stats['quadratic_r_squared']:.4f}")

def analyze_model_trends(summary_stats):
    """
    Analyze trends in fitness and genomic distance relationships across different models.
    
    Parameters:
    -----------
    summary_stats : dict
        Dictionary containing summary statistics from summarize_simulation_stats
        
    Returns:
    --------
    dict
        Dictionary containing trend analysis across models
    """
    # Extract models and their stats
    models = list(summary_stats['diploid_stats'].keys())
    
    if len(models) <= 1:
        return {"error": "Need at least two models to compare trends"}
    
    # Initialize trend analysis dictionary
    trend_analysis = {
        "model_comparison": {},
        "overall_trends": {},
        "correlation_analysis": {},
    }
    
    # Extract key metrics for each model
    model_metrics = {}
    for model in models:
        stats = summary_stats['diploid_stats'][model]
        model_metrics[model] = {
            # Fitness metrics
            'avg_parent_fitness': stats['avg_parent_fitness'],
            'avg_offspring_fitness': stats['avg_offspring_fitness'],
            'fitness_improvement': stats['fitness_improvement'],
            
            # Genomic distance metrics
            'avg_genomic_distance': stats['avg_genomic_distance'],
            
            # Regression slopes and significance
            'parent_fitness_slope': stats['fitness_regression_stats']['linear_slope'],
            'parent_fitness_p_value': stats['fitness_regression_stats']['linear_p_value'],
            'parent_fitness_r_squared': stats['fitness_regression_stats']['linear_r_squared'],
            
            'distance_fitness_slope': stats['distance_regression_stats']['linear_slope'],
            'distance_fitness_p_value': stats['distance_regression_stats']['linear_p_value'],
            'distance_fitness_r_squared': stats['distance_regression_stats']['linear_r_squared'],
            
            # Quadratic relationships
            'parent_fitness_quadratic_r2': stats['fitness_regression_stats']['quadratic_r_squared'],
            'distance_fitness_quadratic_r2': stats['distance_regression_stats']['quadratic_r_squared'],
        }
    
    # Compare models pairwise
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            key = f"{model1}_vs_{model2}"
            trend_analysis["model_comparison"][key] = {}
            
            # Compare fitness improvement
            m1_improvement = model_metrics[model1]['fitness_improvement']
            m2_improvement = model_metrics[model2]['fitness_improvement']
            improvement_diff = m1_improvement - m2_improvement
            
            trend_analysis["model_comparison"][key]["fitness_improvement"] = {
                f"{model1}": m1_improvement,
                f"{model2}": m2_improvement,
                "difference": improvement_diff,
                "better_model": model1 if improvement_diff > 0 else model2,
                "percent_difference": (abs(improvement_diff) / min(abs(m1_improvement), abs(m2_improvement))) * 100 if min(abs(m1_improvement), abs(m2_improvement)) > 0 else 0
            }
            
            # Compare parent-offspring fitness relationship
            m1_parent_slope = model_metrics[model1]['parent_fitness_slope']
            m2_parent_slope = model_metrics[model2]['parent_fitness_slope']
            
            trend_analysis["model_comparison"][key]["parent_fitness_relationship"] = {
                f"{model1}_slope": m1_parent_slope,
                f"{model2}_slope": m2_parent_slope,
                "difference": m1_parent_slope - m2_parent_slope,
                "stronger_heritability": model1 if m1_parent_slope > m2_parent_slope else model2,
                f"{model1}_significance": "significant" if model_metrics[model1]['parent_fitness_p_value'] < 0.05 else "not significant",
                f"{model2}_significance": "significant" if model_metrics[model2]['parent_fitness_p_value'] < 0.05 else "not significant",
            }
            
            # Compare genomic distance effect
            m1_distance_slope = model_metrics[model1]['distance_fitness_slope']
            m2_distance_slope = model_metrics[model2]['distance_fitness_slope']
            
            trend_analysis["model_comparison"][key]["genomic_distance_effect"] = {
                f"{model1}_slope": m1_distance_slope,
                f"{model2}_slope": m2_distance_slope,
                "difference": m1_distance_slope - m2_distance_slope,
                "stronger_distance_effect": model1 if abs(m1_distance_slope) > abs(m2_distance_slope) else model2,
                "distance_effect_direction_match": (m1_distance_slope > 0) == (m2_distance_slope > 0),
                f"{model1}_significance": "significant" if model_metrics[model1]['distance_fitness_p_value'] < 0.05 else "not significant",
                f"{model2}_significance": "significant" if model_metrics[model2]['distance_fitness_p_value'] < 0.05 else "not significant",
            }
            
            # Compare linearity vs non-linearity
            m1_parent_linear_vs_quad = model_metrics[model1]['parent_fitness_r_squared'] / model_metrics[model1]['parent_fitness_quadratic_r2'] if model_metrics[model1]['parent_fitness_quadratic_r2'] > 0 else float('inf')
            m2_parent_linear_vs_quad = model_metrics[model2]['parent_fitness_r_squared'] / model_metrics[model2]['parent_fitness_quadratic_r2'] if model_metrics[model2]['parent_fitness_quadratic_r2'] > 0 else float('inf')
            
            trend_analysis["model_comparison"][key]["fitness_relationship_linearity"] = {
                f"{model1}_linear_vs_quadratic_ratio": m1_parent_linear_vs_quad if m1_parent_linear_vs_quad != float('inf') else "infinite",
                f"{model2}_linear_vs_quadratic_ratio": m2_parent_linear_vs_quad if m2_parent_linear_vs_quad != float('inf') else "infinite",
                f"{model1}_more_nonlinear": model_metrics[model1]['parent_fitness_quadratic_r2'] - model_metrics[model1]['parent_fitness_r_squared'],
                f"{model2}_more_nonlinear": model_metrics[model2]['parent_fitness_quadratic_r2'] - model_metrics[model2]['parent_fitness_r_squared'],
                "more_nonlinear_model": model1 if (model_metrics[model1]['parent_fitness_quadratic_r2'] - model_metrics[model1]['parent_fitness_r_squared']) > (model_metrics[model2]['parent_fitness_quadratic_r2'] - model_metrics[model2]['parent_fitness_r_squared']) else model2
            }
    
    # Analyze overall trends across all models
    # 1. Rank models by fitness improvement
    ranked_by_improvement = sorted(models, key=lambda m: model_metrics[m]['fitness_improvement'], reverse=True)
    
    # 2. Rank models by parent-offspring fitness correlation
    ranked_by_heritability = sorted(models, key=lambda m: model_metrics[m]['parent_fitness_r_squared'], reverse=True)
    
    # 3. Categorize genomic distance effect
    positive_distance_effect = [m for m in models if model_metrics[m]['distance_fitness_slope'] > 0]
    negative_distance_effect = [m for m in models if model_metrics[m]['distance_fitness_slope'] < 0]
    significant_distance_effect = [m for m in models if model_metrics[m]['distance_fitness_p_value'] < 0.05]
    
    # 4. Check if higher parent fitness consistently leads to higher offspring fitness
    consistent_parent_effect = all(model_metrics[m]['parent_fitness_slope'] > 0 for m in models)
    
    # Add overall trends
    trend_analysis["overall_trends"] = {
        "ranked_by_fitness_improvement": ranked_by_improvement,
        "best_performing_model": ranked_by_improvement[0] if ranked_by_improvement else None,
        "ranked_by_heritability": ranked_by_heritability,
        "strongest_heritability_model": ranked_by_heritability[0] if ranked_by_heritability else None,
        "models_with_positive_distance_effect": positive_distance_effect,
        "models_with_negative_distance_effect": negative_distance_effect,
        "models_with_significant_distance_effect": significant_distance_effect,
        "consistent_parent_fitness_effect": consistent_parent_effect
    }
    
    # Analyze correlations between metrics across models
    if len(models) >= 3:  # Need at least 3 data points for meaningful correlation
        # Extract arrays for correlation analysis
        parent_fitness_slopes = [model_metrics[m]['parent_fitness_slope'] for m in models]
        distance_slopes = [model_metrics[m]['distance_fitness_slope'] for m in models]
        fitness_improvements = [model_metrics[m]['fitness_improvement'] for m in models]
        avg_genomic_distances = [model_metrics[m]['avg_genomic_distance'] for m in models]
        
        # Calculate correlations
        try:
            import scipy.stats as stats
            
            # Correlation between parent fitness effect and genomic distance effect
            parent_distance_corr, parent_distance_p = stats.pearsonr(parent_fitness_slopes, distance_slopes)
            
            # Correlation between fitness improvement and genomic distance
            improvement_distance_corr, improvement_distance_p = stats.pearsonr(fitness_improvements, avg_genomic_distances)
            
            # Correlation between fitness improvement and parent fitness effect
            improvement_parent_corr, improvement_parent_p = stats.pearsonr(fitness_improvements, parent_fitness_slopes)
            
            trend_analysis["correlation_analysis"] = {
                "parent_fitness_vs_distance_effect": {
                    "correlation": parent_distance_corr,
                    "p_value": parent_distance_p,
                    "significant": parent_distance_p < 0.05,
                    "interpretation": "Models with stronger parent-offspring fitness correlation tend to have " + 
                                     ("stronger" if parent_distance_corr > 0 else "weaker") + 
                                     " genomic distance effects"
                },
                "improvement_vs_genomic_distance": {
                    "correlation": improvement_distance_corr,
                    "p_value": improvement_distance_p,
                    "significant": improvement_distance_p < 0.05,
                    "interpretation": "Models with " + 
                                     ("higher" if improvement_distance_corr > 0 else "lower") + 
                                     " average genomic distances tend to show " +
                                     ("better" if improvement_distance_corr > 0 else "worse") +
                                     " fitness improvement"
                },
                "improvement_vs_parent_fitness_effect": {
                    "correlation": improvement_parent_corr,
                    "p_value": improvement_parent_p,
                    "significant": improvement_parent_p < 0.05,
                    "interpretation": "Models with stronger parent-offspring fitness correlation tend to show " +
                                     ("better" if improvement_parent_corr > 0 else "worse") +
                                     " fitness improvement"
                }
            }
        except (ImportError, ValueError):
            trend_analysis["correlation_analysis"] = {
                "error": "Could not compute correlations - either scipy not available or insufficient data"
            }
    
    return trend_analysis

def log_trend_analysis(log, trend_analysis):
    """
    Log the results of the trend analysis across models.
    
    Parameters:
    -----------
    log : logging.Logger
        Logger instance
    trend_analysis : dict
        Dictionary containing trend analysis from analyze_model_trends
    """
    log.info("\n====== MODEL TREND ANALYSIS ======")
    
    # Check if there was an error in the trend analysis
    if "error" in trend_analysis:
        log.info(f"Trend Analysis Error: {trend_analysis['error']}")
        return
    
    # Log overall trends
    log.info("\n=== OVERALL TRENDS ===")
    overall = trend_analysis["overall_trends"]
    
    log.info(f"Models Ranked by Fitness Improvement:")
    for i, model in enumerate(overall["ranked_by_fitness_improvement"]):
        log.info(f"  {i+1}. {model}")
    
    log.info(f"\nModels Ranked by Parent-Offspring Heritability:")
    for i, model in enumerate(overall["ranked_by_heritability"]):
        log.info(f"  {i+1}. {model}")
    
    log.info(f"\nGenetic Distance Effects:")
    log.info(f"  Models with positive distance effect: {', '.join(overall['models_with_positive_distance_effect']) if overall['models_with_positive_distance_effect'] else 'None'}")
    log.info(f"  Models with negative distance effect: {', '.join(overall['models_with_negative_distance_effect']) if overall['models_with_negative_distance_effect'] else 'None'}")
    log.info(f"  Models with statistically significant distance effect: {', '.join(overall['models_with_significant_distance_effect']) if overall['models_with_significant_distance_effect'] else 'None'}")
    
    log.info(f"\nParent Fitness Effect:")
    log.info(f"  Consistent positive effect across all models: {'Yes' if overall['consistent_parent_fitness_effect'] else 'No'}")
    
    # Log pairwise model comparisons
    log.info("\n=== MODEL COMPARISONS ===")
    for comparison, data in trend_analysis["model_comparison"].items():
        log.info(f"\n{comparison.upper()}:")
        
        # Fitness improvement comparison
        fi_data = data["fitness_improvement"]
        model1, model2 = comparison.split("_vs_")
        
        log.info(f"  Fitness Improvement:")
        log.info(f"    {model1}: {fi_data[model1]:.4f}")
        log.info(f"    {model2}: {fi_data[model2]:.4f}")
        log.info(f"    Difference: {fi_data['difference']:.4f}")
        log.info(f"    Better model: {fi_data['better_model']} (by {fi_data['percent_difference']:.2f}%)")
        
        # Parent-offspring fitness relationship
        pf_data = data["parent_fitness_relationship"]
        log.info(f"  Parent-Offspring Fitness Relationship:")
        log.info(f"    {model1} slope: {pf_data[f'{model1}_slope']:.4f} ({pf_data[f'{model1}_significance']})")
        log.info(f"    {model2} slope: {pf_data[f'{model2}_slope']:.4f} ({pf_data[f'{model2}_significance']})")
        log.info(f"    Stronger heritability: {pf_data['stronger_heritability']}")
        
        # Genomic distance effect
        gd_data = data["genomic_distance_effect"]
        log.info(f"  Genomic Distance Effect:")
        log.info(f"    {model1} slope: {gd_data[f'{model1}_slope']:.4f} ({gd_data[f'{model1}_significance']})")
        log.info(f"    {model2} slope: {gd_data[f'{model2}_slope']:.4f} ({gd_data[f'{model2}_significance']})")
        log.info(f"    Stronger effect: {gd_data['stronger_distance_effect']}")
        log.info(f"    Effect direction match: {'Yes' if gd_data['distance_effect_direction_match'] else 'No'}")
        
        # Linearity vs non-linearity
        ln_data = data["fitness_relationship_linearity"]
        log.info(f"  Fitness Relationship Linearity:")
        log.info(f"    {model1} linear vs quadratic improvement: {ln_data[f'{model1}_more_nonlinear']:.4f}")
        log.info(f"    {model2} linear vs quadratic improvement: {ln_data[f'{model2}_more_nonlinear']:.4f}")
        log.info(f"    More non-linear model: {ln_data['more_nonlinear_model']}")
    
    # Log correlation analysis if available
    if "correlation_analysis" in trend_analysis and "error" not in trend_analysis["correlation_analysis"]:
        log.info("\n=== CORRELATION ANALYSIS ===")
        corr = trend_analysis["correlation_analysis"]
        
        for relationship, data in corr.items():
            log.info(f"\n{relationship.replace('_', ' ').title()}:")
            log.info(f"  Correlation coefficient: {data['correlation']:.4f}")
            log.info(f"  P-value: {data['p_value']:.4f} ({'significant' if data['significant'] else 'not significant'})")
            log.info(f"  Interpretation: {data['interpretation']}")
    
    log.info("\n=== SUMMARY FINDINGS ===")
    
    # Output the best overall model
    best_model = overall["best_performing_model"]
    log.info(f"Best Overall Model: {best_model}")
    
    # Output trends in parent-offspring fitness relationship
    if overall["consistent_parent_fitness_effect"]:
        log.info("Parent-Offspring Fitness: All models show positive relationship between parent and offspring fitness")
    else:
        log.info("Parent-Offspring Fitness: Inconsistent relationship between models")
    
    # Output genomic distance effect trends
    if len(overall["models_with_positive_distance_effect"]) > len(overall["models_with_negative_distance_effect"]):
        log.info("Genomic Distance Effect: Predominantly positive (higher distance -> higher fitness)")
    elif len(overall["models_with_positive_distance_effect"]) < len(overall["models_with_negative_distance_effect"]):
        log.info("Genomic Distance Effect: Predominantly negative (higher distance -> lower fitness)")
    else:
        log.info("Genomic Distance Effect: Mixed effects across models")

def aggregate_simulation_results(all_runs_data, Resu_path):
    """
    Aggregate and analyze results from multiple simulation runs with comprehensive data collection.
    
    Parameters:
    -----------
    all_runs_data : list
        List of dictionaries, each containing data from one simulation run
    Resu_path : str
        Directory path where to save aggregated results
        
    Returns:
    --------
    dict
        Dictionary containing aggregated statistics
    pandas.DataFrame
        DataFrame containing metrics from all runs
    dict
        Comprehensive data structure with all collected data
    """
    # Extract key metrics from each run
    run_metrics = []
    
    # Create comprehensive data store for all raw data
    comprehensive_data = {
        "runs": {},
        "diploid_data": {},
        "regression_stats": {},
        "correlation_data": {},
        "raw_trajectories": {},
        "simulation_parameters": {},
        "prs_data": {},  # Add PRS data storage
        "genomic_data": {},  # Add genomic data storage
        "parent_offspring_pairs": {}  # Add raw parent-offspring pairs
    }
    
    # Try to extract diploid data from the runs if available
    for run_idx, run_data in enumerate(all_runs_data):
        run_id = run_idx + 1
        
        # Get simulation parameters from the first run
        if run_idx == 0 and "parameters" in run_data:
            comprehensive_data["simulation_parameters"] = run_data["parameters"]
        
        # Extract run info for the metrics dataframe
        summary = run_data["summary_stats"]
        run_info = {
            "run_id": run_id,
            "initial_fitness": summary["initial_fitness"],
            "final_fitness": summary["final_fitness"],
            "fitness_improvement_percent": summary["fitness_improvement_percent"],
            "final_population": summary["final_population"],
            "final_fitness_std": summary["fitness_std_final"],
        }
        
        # Store more detailed generation statistics
        if "generation_stats" in run_data:
            comprehensive_data["runs"][run_id] = {
                "generation_stats": run_data["generation_stats"]
            }
        
        # Store fitness distribution data if available
        if "last_generation_distribution" in summary:
            comprehensive_data["runs"][run_id]["fitness_distribution"] = summary["last_generation_distribution"]
        
        # Add diploid model statistics
        comprehensive_data["diploid_data"][run_id] = {}
        comprehensive_data["prs_data"][run_id] = {}
        comprehensive_data["genomic_data"][run_id] = {}
        comprehensive_data["parent_offspring_pairs"][run_id] = {}
        
        # Check if run_data contains the actual diploid_offspring_dict
        if "diploid_offspring_dict" in run_data:
            # If we have the actual diploid organisms, extract PRS and genomic data
            for model, organisms in run_data["diploid_offspring_dict"].items():
                # Store PRS data
                prs_values = []
                genomic_distances = []
                parent_offspring_pairs = []
                
                for org in organisms:
                    try:
                        # Calculate PRS
                        prs = calculate_diploid_prs(org)
                        prs_values.append(prs)
                        
                        # Calculate genomic distance
                        dist = calculate_genomic_distance(org.allele1, org.allele2)
                        genomic_distances.append(dist)
                        
                        # Store parent-offspring fitness pairs
                        parent_offspring_pairs.append({
                            "parent1_fitness": org.parent1_fitness,
                            "parent2_fitness": org.parent2_fitness,
                            "offspring_fitness": org.fitness,
                            "avg_parent_fitness": org.avg_parent_fitness,
                            "genomic_distance": dist,
                            "prs": prs
                        })
                    except Exception as e:
                        print(f"Error processing organism for run {run_id}, model {model}: {e}")
                
                # Store the data
                comprehensive_data["prs_data"][run_id][model] = prs_values
                comprehensive_data["genomic_data"][run_id][model] = genomic_distances
                comprehensive_data["parent_offspring_pairs"][run_id][model] = parent_offspring_pairs
        
        for model in ["dominant", "recessive", "codominant"]:
            if model in summary["diploid_stats"]:
                model_stats = summary["diploid_stats"][model]
                
                # Store parent-offspring regression data
                parent_fitness_stats = model_stats.get("fitness_regression_stats", {})
                distance_fitness_stats = model_stats.get("distance_regression_stats", {})
                
                # Extract key metrics for the summary dataframe
                parent_fitness_slope = parent_fitness_stats.get("linear_slope", 0)
                distance_fitness_slope = distance_fitness_stats.get("linear_slope", 0)
                
                run_info.update({
                    f"{model}_avg_offspring_fitness": model_stats.get("avg_offspring_fitness", 0),
                    f"{model}_fitness_improvement": model_stats.get("fitness_improvement", 0),
                    f"{model}_avg_genomic_distance": model_stats.get("avg_genomic_distance", 0),
                    f"{model}_parent_fitness_r2": parent_fitness_stats.get("linear_r_squared", 0),
                    f"{model}_parent_fitness_slope": parent_fitness_slope,
                    f"{model}_parent_fitness_p_value": parent_fitness_stats.get("linear_p_value", 1),
                    f"{model}_distance_fitness_r2": distance_fitness_stats.get("linear_r_squared", 0),
                    f"{model}_distance_fitness_slope": distance_fitness_slope,
                    f"{model}_distance_fitness_p_value": distance_fitness_stats.get("linear_p_value", 1),
                    f"{model}_parent_fitness_slope_sign": "positive" if parent_fitness_slope > 0 else "negative",
                    f"{model}_distance_fitness_slope_sign": "positive" if distance_fitness_slope > 0 else "negative",
                    # Add quadratic regression statistics
                    f"{model}_parent_fitness_quad_r2": parent_fitness_stats.get("quadratic_r_squared", 0),
                    f"{model}_distance_fitness_quad_r2": distance_fitness_stats.get("quadratic_r_squared", 0),
                })
                
                # Store detailed model statistics for comprehensive analysis
                comprehensive_data["diploid_data"][run_id][model] = model_stats
                
                # Store all regression statistics
                if "regression_stats" not in comprehensive_data:
                    comprehensive_data["regression_stats"] = {}
                if run_id not in comprehensive_data["regression_stats"]:
                    comprehensive_data["regression_stats"][run_id] = {}
                    
                comprehensive_data["regression_stats"][run_id][f"{model}_parent_fitness"] = parent_fitness_stats
                comprehensive_data["regression_stats"][run_id][f"{model}_distance_fitness"] = distance_fitness_stats
        
        run_metrics.append(run_info)
    
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(run_metrics)
    
    # Calculate aggregated statistics
    aggregated_stats = {
        "num_runs": len(all_runs_data),
        "haploid_evolution": {
            "initial_fitness": {
                "mean": metrics_df["initial_fitness"].mean(),
                "std": metrics_df["initial_fitness"].std(),
                "min": metrics_df["initial_fitness"].min(),
                "max": metrics_df["initial_fitness"].max(),
            },
            "final_fitness": {
                "mean": metrics_df["final_fitness"].mean(),
                "std": metrics_df["final_fitness"].std(),
                "min": metrics_df["final_fitness"].min(),
                "max": metrics_df["final_fitness"].max(),
            },
            "fitness_improvement_percent": {
                "mean": metrics_df["fitness_improvement_percent"].mean(),
                "std": metrics_df["fitness_improvement_percent"].std(),
                "min": metrics_df["fitness_improvement_percent"].min(),
                "max": metrics_df["fitness_improvement_percent"].max(),
            }
        },
        "diploid_models": {}
    }
    
    # Add statistics for each diploid model
    for model in ["dominant", "recessive", "codominant"]:
        model_fitness_key = f"{model}_avg_offspring_fitness"
        model_improvement_key = f"{model}_fitness_improvement"
        model_distance_key = f"{model}_avg_genomic_distance"
        parent_slope_key = f"{model}_parent_fitness_slope"
        distance_slope_key = f"{model}_distance_fitness_slope"
        parent_slope_sign_key = f"{model}_parent_fitness_slope_sign"
        distance_slope_sign_key = f"{model}_distance_fitness_slope_sign"
        
        if model_fitness_key in metrics_df.columns:
            # Count positive/negative slopes for each model
            if parent_slope_sign_key in metrics_df.columns:
                parent_slope_positive_count = (metrics_df[parent_slope_sign_key] == "positive").sum()
                parent_slope_negative_count = (metrics_df[parent_slope_sign_key] == "negative").sum()
            else:
                parent_slope_positive_count = parent_slope_negative_count = 0
                
            if distance_slope_sign_key in metrics_df.columns:
                distance_slope_positive_count = (metrics_df[distance_slope_sign_key] == "positive").sum()
                distance_slope_negative_count = (metrics_df[distance_slope_sign_key] == "negative").sum()
            else:
                distance_slope_positive_count = distance_slope_negative_count = 0
            
            aggregated_stats["diploid_models"][model] = {
                "offspring_fitness": {
                    "mean": metrics_df[model_fitness_key].mean(),
                    "std": metrics_df[model_fitness_key].std(),
                    "min": metrics_df[model_fitness_key].min(),
                    "max": metrics_df[model_fitness_key].max(),
                },
                "fitness_improvement": {
                    "mean": metrics_df[model_improvement_key].mean(),
                    "std": metrics_df[model_improvement_key].std(),
                    "min": metrics_df[model_improvement_key].min(),
                    "max": metrics_df[model_improvement_key].max(),
                },
                "genomic_distance": {
                    "mean": metrics_df[model_distance_key].mean(),
                    "std": metrics_df[model_distance_key].std(),
                    "min": metrics_df[model_distance_key].min(),
                    "max": metrics_df[model_distance_key].max(),
                },
                "parent_fitness_slope": {
                    "mean": metrics_df[parent_slope_key].mean() if parent_slope_key in metrics_df.columns else 0,
                    "std": metrics_df[parent_slope_key].std() if parent_slope_key in metrics_df.columns else 0,
                    "positive_count": parent_slope_positive_count,
                    "negative_count": parent_slope_negative_count,
                },
                "distance_fitness_slope": {
                    "mean": metrics_df[distance_slope_key].mean() if distance_slope_key in metrics_df.columns else 0,
                    "std": metrics_df[distance_slope_key].std() if distance_slope_key in metrics_df.columns else 0,
                    "positive_count": distance_slope_positive_count,
                    "negative_count": distance_slope_negative_count,
                },
                "parent_fitness_r2": {
                    "mean": metrics_df[f"{model}_parent_fitness_r2"].mean(),
                    "std": metrics_df[f"{model}_parent_fitness_r2"].std(),
                },
                "distance_fitness_r2": {
                    "mean": metrics_df[f"{model}_distance_fitness_r2"].mean(),
                    "std": metrics_df[f"{model}_distance_fitness_r2"].std(),
                },
                # Add quadratic fit R² statistics
                "parent_fitness_quad_r2": {
                    "mean": metrics_df[f"{model}_parent_fitness_quad_r2"].mean(),
                    "std": metrics_df[f"{model}_parent_fitness_quad_r2"].std(),
                },
                "distance_fitness_quad_r2": {
                    "mean": metrics_df[f"{model}_distance_fitness_quad_r2"].mean(),
                    "std": metrics_df[f"{model}_distance_fitness_quad_r2"].std(),
                }
            }
    
    # Add calculated correlation data between various metrics
    try:
        # Only select numeric columns for correlation
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = metrics_df[numeric_cols].corr()
        comprehensive_data["correlation_data"]["metrics_correlation_matrix"] = correlation_matrix.to_dict()
    except Exception as e:
        print(f"Could not calculate correlation matrix: {e}")
    
    # Save the raw metrics data as JSON
    metrics_file = os.path.join(Resu_path, "run_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(run_metrics, f, indent=2, default=numpy_json_encoder)

    # Save aggregated statistical summary
    aggregated_file = os.path.join(Resu_path, "aggregated_stats.json")
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2, default=numpy_json_encoder)
    
    # Save comprehensive data structure with all raw data
    comprehensive_file = os.path.join(Resu_path, "comprehensive_data.json")
    with open(comprehensive_file, 'w') as f:
        json.dump(comprehensive_data, f, indent=2, default=numpy_json_encoder)
    
    # Save metrics DataFrame to CSV for easy importing to other tools
    metrics_csv = os.path.join(Resu_path, "run_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    
    # Also save as pickle for preserving data types and easier Python loading
    metrics_pickle = os.path.join(Resu_path, "run_metrics.pkl")
    metrics_df.to_pickle(metrics_pickle)
    
    return aggregated_stats, metrics_df, comprehensive_data

def log_aggregated_stats(logger, aggregated_stats):
    """
    Log the aggregated statistics from multiple runs.
    
    Parameters:
    -----------
    logger : logging.Logger
        Logger instance
    aggregated_stats : dict
        Dictionary containing aggregated statistics
    """
    logger.info("\n===== AGGREGATED STATISTICS FROM MULTIPLE RUNS =====")
    logger.info(f"Total number of runs: {aggregated_stats['num_runs']}")
    
    # Log haploid evolution statistics
    haploid = aggregated_stats["haploid_evolution"]
    logger.info("\nHaploid Evolution:")
    logger.info(f"  Initial Fitness: {haploid['initial_fitness']['mean']:.4f} ± {haploid['initial_fitness']['std']:.4f}")
    logger.info(f"  Final Fitness: {haploid['final_fitness']['mean']:.4f} ± {haploid['final_fitness']['std']:.4f}")
    logger.info(f"  Fitness Improvement: {haploid['fitness_improvement_percent']['mean']:.2f}% ± {haploid['fitness_improvement_percent']['std']:.2f}%")
    logger.info(f"  Range: {haploid['fitness_improvement_percent']['min']:.2f}% to {haploid['fitness_improvement_percent']['max']:.2f}%")
    
    # Log diploid model statistics
    logger.info("\nDiploid Models:")
    for model, stats in aggregated_stats["diploid_models"].items():
        logger.info(f"\n{model.upper()}:")
        logger.info(f"  Average Offspring Fitness: {stats['offspring_fitness']['mean']:.4f} ± {stats['offspring_fitness']['std']:.4f}")
        logger.info(f"  Average Fitness Improvement: {stats['fitness_improvement']['mean']:.4f} ± {stats['fitness_improvement']['std']:.4f}")
        logger.info(f"  Average Genomic Distance: {stats['genomic_distance']['mean']:.4f} ± {stats['genomic_distance']['std']:.4f}")
        
        # Log slope statistics
        if 'parent_fitness_slope' in stats:
            parent_slope = stats['parent_fitness_slope']
            logger.info(f"  Parent-Offspring Fitness Slope: {parent_slope['mean']:.4f} ± {parent_slope['std']:.4f}")
            logger.info(f"    Positive slopes: {parent_slope['positive_count']}/{aggregated_stats['num_runs']} runs")
            logger.info(f"    Negative slopes: {parent_slope['negative_count']}/{aggregated_stats['num_runs']} runs")
        
        if 'distance_fitness_slope' in stats:
            distance_slope = stats['distance_fitness_slope']
            logger.info(f"  Genomic Distance-Fitness Slope: {distance_slope['mean']:.4f} ± {distance_slope['std']:.4f}")
            logger.info(f"    Positive slopes: {distance_slope['positive_count']}/{aggregated_stats['num_runs']} runs")
            logger.info(f"    Negative slopes: {distance_slope['negative_count']}/{aggregated_stats['num_runs']} runs")
        
        logger.info(f"  Parent-Offspring Fitness R²: {stats['parent_fitness_r2']['mean']:.4f} ± {stats['parent_fitness_r2']['std']:.4f}")
        logger.info(f"  Distance-Fitness R²: {stats['distance_fitness_r2']['mean']:.4f} ± {stats['distance_fitness_r2']['std']:.4f}")

def numpy_json_encoder(obj):
    """
    Custom function to serialize NumPy data types for JSON.
    
    Parameters:
    -----------
    obj : object
        The object to serialize
        
    Returns:
    --------
    Python native type that can be serialized by json
    """
    import numpy as np
    
    # Handle various numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Add more types if needed
    else:
        # Let the default encoder handle it or throw an error
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

def calculate_alternative_fitness(genome, params):
    """
    Calculate fitness using an alternative method.

    Parameters:
    -----------
    genome : numpy.ndarray
        The genome to calculate fitness for
    params : dict
        Parameters for fitness calculation, as returned by init_alternative_fitness

    Returns:
    --------
    float
        The calculated fitness value
    """
    method = params.get("method", AlternativeFitnessMethod.SINGLE_POSITION)

    if method == AlternativeFitnessMethod.SINGLE_POSITION:
        position = params["position"]
        favorable_value = params["favorable_value"]

        # Make sure the genome is not just haploid but supports recessive models
        # Assume this is effective diploid genome: 1.0 if AA (dominant), or if aa (recessive)

        # Ensure position exists
        if position >= len(genome):
            return 0.0

        # Return high fitness if the effective genome has 1.0 at the position (which recessive model defines only for aa)
        return 1.0 if genome[position] == favorable_value else 0.0
        
    elif method == AlternativeFitnessMethod.ADDITIVE:
        weights = params["weights"]

        # Convert -1/1 to 0/1 encoding
        genome_01 = (genome + 1) / 2
        fitness = np.dot(genome_01, weights)

        # Normalize to a bounded range using sigmoid-like transformation
        normalized_fitness = 0.1 + 0.9 / (1 + np.exp(-fitness))
        return normalized_fitness

    # Default fallback
    return 0.5

def init_alternative_fitness(genome_size, method=AlternativeFitnessMethod.SINGLE_POSITION, random_state=None):
    """
    Initialize parameters for alternative fitness calculations.
    
    Parameters:
    -----------
    genome_size : int
        Size of the genome
    method : AlternativeFitnessMethod
        The alternative fitness method to use
    random_state : int or numpy.random.Generator, optional
        Seed or generator for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing necessary parameters for the alternative fitness calculation
    """
    rng = np.random.default_rng(random_state)
    
    if method == AlternativeFitnessMethod.SINGLE_POSITION:
        # For single position method, select one random position and assign weight
        position = rng.integers(0, genome_size)
        params = {
            "method": method,
            "position": position,
            "favorable_value": 1  # 1 is favorable, -1 is unfavorable
        }
        
    elif method == AlternativeFitnessMethod.ADDITIVE:
        # For additive method, assign random weights to each position
        weights = rng.normal(0, 1, genome_size)
        params = {
            "method": method,
            "weights": weights
        }
    else:
        # Default case (shouldn't occur since we're checking the enum)
        params = {
            "method": AlternativeFitnessMethod.SINGLE_POSITION,
            "position": 0,
            "favorable_value": 1
        }
        
    return params

#######################################################################
# plotting functions
#######################################################################
def plot_detailed_fitness(individual_fitness, generation_stats, Resu_path):
    """Create detailed fitness visualizations with cleaner design"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Individual fitness trajectories with better styling
    for org_id, fitness_data in individual_fitness.items():
        generations, fitness = zip(*fitness_data)
        ax1.plot(generations, fitness, '-', alpha=0.2, linewidth=0.8, color='lightblue')
    
    # Add statistical overlays with clear styling
    generations = [stat['generation'] for stat in generation_stats]
    avg_fitness = [stat['avg_fitness'] for stat in generation_stats]
    max_fitness = [stat['max_fitness'] for stat in generation_stats]
    min_fitness = [stat['min_fitness'] for stat in generation_stats]
    
    ax1.plot(generations, avg_fitness, 'k-', linewidth=3, label='Average', zorder=10)
    ax1.plot(generations, max_fitness, 'g-', linewidth=2, label='Maximum', zorder=9)
    ax1.plot(generations, min_fitness, 'r-', linewidth=2, label='Minimum', zorder=9)
    
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title('Fitness Evolution Over Generations', fontsize=14, pad=20)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Simplified fitness distribution
    # Sample every few generations to avoid overcrowding
    sample_gens = range(1, max(generations) + 1, max(1, len(generations) // 10))
    fitness_data = []
    gen_labels = []

    for gen in sample_gens:
        gen_fitness = [f for org_id, fit_data in individual_fitness.items() 
                    for g, f in fit_data if g == gen]
        fitness_data.append(gen_fitness)
        gen_labels.append(str(gen))

    # Filter out empty fitness lists and their labels
    filtered_data_labels = [(fd, gl) for fd, gl in zip(fitness_data, gen_labels) if fd]
    if filtered_data_labels:
        fitness_data, gen_labels = zip(*filtered_data_labels)
        bp = ax2.boxplot(fitness_data, labels=gen_labels, patch_artist=True)
        # Color the boxes with a nice gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_xlabel('Generation (sampled)', fontsize=12)
    ax2.set_ylabel('Fitness Distribution', fontsize=12)
    ax2.set_title('Fitness Distribution Over Time', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, 'fitness_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_relationship_tree(organisms, Resu_path):
    """Create a simplified evolutionary tree visualization"""
    G = nx.DiGraph()
    
    # Add nodes and edges
    for org in organisms:
        G.add_node(org.id, generation=org.generation, fitness=org.fitness)
        if org.parent_id:
            G.add_edge(org.parent_id, org.id)
    
    # Use a hierarchical layout
    pos = {}
    gen_organisms = defaultdict(list)
    for org in organisms:
        gen_organisms[org.generation].append(org)
    
    # Position nodes by generation and fitness
    for gen, orgs in gen_organisms.items():
        orgs_sorted = sorted(orgs, key=lambda x: x.fitness)
        for i, org in enumerate(orgs_sorted):
            y_offset = (i - len(orgs_sorted)/2) * 0.1
            pos[org.id] = (gen, org.fitness + y_offset)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color nodes based on fitness
    fitness_values = [G.nodes[node]['fitness'] for node in G.nodes()]
    
    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, edge_color='gray', 
                          alpha=0.3, arrowsize=10, width=0.5)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30,
                                  node_color=fitness_values,
                                  cmap=plt.cm.viridis, alpha=0.8)
    
    # Add colorbar
    plt.colorbar(nodes, ax=ax, label='Fitness', shrink=0.8)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title('Evolutionary Relationship Tree', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, 'relationship_tree.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_parent_offspring_fitness(diploid_dict, Resu_path, mating_strategy):
    """Simplified parent-offspring fitness comparison with clear messaging and focused axes"""
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "#2E86AB", "recessive": "#A23B72", "codominant": "#F18F01"}
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, model in enumerate(models):
        ax = axs[idx]
        diploids = diploid_dict.get(model, [])
        
        # Basic setup
        ax.set_title(f'{model.capitalize()} Model\n({len(diploids)} offspring)', 
                    fontsize=12, pad=15)
        ax.set_xlabel('Mean Parent Fitness', fontsize=11)
        ax.set_ylabel('Offspring Fitness', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if not diploids:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7))
            continue
        
        # Extract data
        parent_fitness = np.array([org.avg_parent_fitness for org in diploids])
        offspring_fitness = np.array([org.fitness for org in diploids])
        
        # Set axis limits based on actual data with small padding
        x_min, x_max = parent_fitness.min(), parent_fitness.max()
        y_min, y_max = offspring_fitness.min(), offspring_fitness.max()
        x_padding = (x_max - x_min) * 0.22
        y_padding = (y_max - y_min) * 0.22
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Check for sufficient variation
        if len(np.unique(parent_fitness)) <= 2:
            ax.text(0.5, 0.8, 'Insufficient variation in parent fitness', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
            
            # Still show the data points
            ax.scatter(parent_fitness, offspring_fitness, alpha=0.6, 
                      color=colors[model], s=40)
            
            # Add mean lines
            ax.axhline(np.mean(offspring_fitness), color='red', linestyle='--', alpha=0.7,
                      label=f'Mean offspring: {np.mean(offspring_fitness):.3f}')
            ax.axvline(np.mean(parent_fitness), color='blue', linestyle='--', alpha=0.7,
                      label=f'Mean parent: {np.mean(parent_fitness):.3f}')
        else:
            # Normal case with variation
            ax.scatter(parent_fitness, offspring_fitness, alpha=0.6, 
                      color=colors[model], s=40, edgecolors='white', linewidth=0.5)
            
            # Create x range for plotting fits
            x_range = np.linspace(parent_fitness.min(), parent_fitness.max(), 100)
            
            # Add linear regression line
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, _ = stats.linregress(parent_fitness, offspring_fitness)
                
                y_pred_linear = slope * x_range + intercept
                ax.plot(x_range, y_pred_linear, 'r-', linewidth=2, alpha=0.8,
                       label=f'Linear (R² = {r_value**2:.3f})')
                
                linear_r2 = r_value**2
                
            except Exception as e:
                print(f"Warning: Linear regression failed for {model}: {e}")
                linear_r2 = 0
            
            # Add quadratic (polynomial degree 2) regression line
            try:
                if len(np.unique(parent_fitness)) >= 3:  # Need at least 3 points for quadratic
                    poly_coeffs = np.polyfit(parent_fitness, offspring_fitness, 2)
                    y_pred_quad = np.polyval(poly_coeffs, x_range)
                    
                    # Calculate R² for quadratic fit
                    y_pred_quad_actual = np.polyval(poly_coeffs, parent_fitness)
                    ss_res = np.sum((offspring_fitness - y_pred_quad_actual) ** 2)
                    ss_tot = np.sum((offspring_fitness - np.mean(offspring_fitness)) ** 2)
                    quad_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    ax.plot(x_range, y_pred_quad, 'g--', linewidth=2, alpha=0.8,
                           label=f'Quadratic (R² = {quad_r2:.3f})')
                else:
                    quad_r2 = 0
                    
            except Exception as e:
                print(f"Warning: Quadratic regression failed for {model}: {e}")
                quad_r2 = 0
            
            # Add diagonal reference line (only within data range)
            data_min = min(x_min, y_min)
            data_max = max(x_max, y_max)
            ax.plot([data_min, data_max], [data_min, data_max], 'k--', alpha=0.5,
                   label='No improvement line')
            
            # Add statistics text box
            stats_text = []
            if 'linear_r2' in locals():
                stats_text.extend([
                    f'Linear R² = {linear_r2:.3f}',
                    f'Slope = {slope:.3f}'
                ])
            if 'quad_r2' in locals() and quad_r2 > 0:
                stats_text.append(f'Quad R² = {quad_r2:.3f}')
            
            if stats_text:
                ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
                       fontsize=10, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Legend
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    fig.suptitle(f'Parent vs Offspring Fitness ({mating_strategy} strategy)', 
                fontsize=16, y=1.03)

    # First layout, then save with bounding box
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(Resu_path, f'parent_offspring_fitness_{mating_strategy}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_parent_offspring_heatmap(diploid_offspring_dict, Resu_path):
    """Simplified heatmap with better visual design"""
    models = ["dominant", "recessive", "codominant"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, model in enumerate(models):
        ax = axs[idx]
        offspring_list = diploid_offspring_dict.get(model, [])
        
        if not offspring_list:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(f'{model.capitalize()} Model')
            continue
        
        # Extract data
        parent1_fit = np.array([d.parent1_fitness for d in offspring_list])
        parent2_fit = np.array([d.parent2_fitness for d in offspring_list])
        offspring_fit = np.array([d.fitness for d in offspring_list])
        
        # Create 2D histogram
        bins = min(15, int(np.sqrt(len(offspring_list))))  # Adaptive bin size
        h, xedges, yedges = np.histogram2d(parent1_fit, parent2_fit, 
                                          bins=bins, weights=offspring_fit)
        counts, _, _ = np.histogram2d(parent1_fit, parent2_fit, bins=bins)
        
        # Calculate average fitness per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_fitness = np.divide(h, counts, out=np.zeros_like(h), where=counts>0)
        
        # Plot heatmap
        im = ax.imshow(avg_fitness.T, origin='lower', aspect='auto',
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      cmap='viridis', interpolation='nearest')
        
        ax.set_xlabel('Parent 1 Fitness', fontsize=11)
        ax.set_ylabel('Parent 2 Fitness', fontsize=11)
        ax.set_title(f'{model.capitalize()} Model', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Average Offspring Fitness', fontsize=10)
    
    plt.suptitle('Parent Fitness Combinations vs Offspring Fitness', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, 'parent_offspring_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_offspring_vs_min_max_parent_fitness(diploid_dict, Resu_path, mating_strategy):
    """Simplified min/max parent fitness plots with focused axes and quadratic fits"""
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "#2E86AB", "recessive": "#A23B72", "codominant": "#F18F01"}

    for plot_type in ['max', 'min']:
        fig, axs = plt.subplots(1, 3, figsize=(18, 10))

        for idx, model in enumerate(models):
            ax = axs[idx]
            diploids = diploid_dict.get(model, [])

            ax.set_title(f'{model.capitalize()} Model', fontsize=12, pad=15)
            ax.set_xlabel(f'{"Maximum" if plot_type == "max" else "Minimum"} Parent Fitness', fontsize=11)
            ax.set_ylabel('Offspring Fitness', fontsize=11)
            ax.grid(True, alpha=0.3)

            if not diploids:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14)
                continue

            offspring_fitness = np.array([org.fitness for org in diploids])
            if plot_type == 'max':
                parent_fitness = np.array([max(org.parent1_fitness, org.parent2_fitness) for org in diploids])
            else:
                parent_fitness = np.array([min(org.parent1_fitness, org.parent2_fitness) for org in diploids])

            # X-axis range
            x_min, x_max = parent_fitness.min(), parent_fitness.max()
            x_padding = (x_max - x_min) * 0.05
            ax.set_xlim(x_min - x_padding, x_max + x_padding)

            # Scatter plot
            ax.scatter(parent_fitness, offspring_fitness, alpha=0.6,
                    color=colors[model], s=40, edgecolors='white', linewidth=0.5)

            y_all = list(offspring_fitness)  # collect all y values here

            if len(np.unique(parent_fitness)) > 2:
                x_range = np.linspace(parent_fitness.min(), parent_fitness.max(), 100)

                try:
                    from scipy import stats
                    # Linear fit
                    slope, intercept, r_value, _, _ = stats.linregress(parent_fitness, offspring_fitness)
                    y_pred_linear = slope * x_range + intercept
                    ax.plot(x_range, y_pred_linear, 'r-', linewidth=2, alpha=0.8,
                            label=f'Linear (R² = {r_value**2:.3f})')
                    y_all.extend(y_pred_linear)

                    # Quadratic fit
                    if len(np.unique(parent_fitness)) >= 3:
                        poly_coeffs = np.polyfit(parent_fitness, offspring_fitness, 2)
                        y_pred_quad = np.polyval(poly_coeffs, x_range)
                        y_quad_actual = np.polyval(poly_coeffs, parent_fitness)

                        ss_res = np.sum((offspring_fitness - y_quad_actual) ** 2)
                        ss_tot = np.sum((offspring_fitness - np.mean(offspring_fitness)) ** 2)
                        quad_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                        ax.plot(x_range, y_pred_quad, 'g--', linewidth=2, alpha=0.8,
                                label=f'Quadratic (R² = {quad_r2:.3f})')
                        y_all.extend(y_pred_quad)

                    ax.legend(loc='best', fontsize=9)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Fit error\n{str(e)}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)

            # ✅ Now apply y-limits AFTER collecting all y values
            y_all = np.array(y_all)
            y_min, y_max = y_all.min(), y_all.max()
            y_padding = (y_max - y_min) * 0.25 if y_max > y_min else 0.25  # Increased from 0.15 to 0.25 (50% more padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)


        fig.suptitle(f'{"Maximum" if plot_type == "max" else "Minimum"} Parent Fitness vs Offspring Fitness',
                     fontsize=16, y=1.03)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(Resu_path, f'offspring_vs_{plot_type}_parent_fitness_{mating_strategy}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_parent_genomic_distance_vs_offspring_fitness(diploid_dict, Resu_path, mating_strategy):
    """Simplified genomic distance analysis with focused axes and quadratic fits"""
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "#2E86AB", "recessive": "#A23B72", "codominant": "#F18F01"}
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, model in enumerate(models):
        ax = axs[idx]
        diploids = diploid_dict.get(model, [])
        
        ax.set_title(f'{model.capitalize()} Model', fontsize=12, pad=15)
        ax.set_xlabel('Genomic Distance Between Parents', fontsize=11)
        ax.set_ylabel('Offspring Fitness', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if not diploids:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue
        
        # Calculate genomic distances
        try:
            distances = []
            fitness_vals = []
            
            for org in diploids[:1000]:  # Limit to first 1000 to avoid memory issues
                dist = calculate_genomic_distance(org.allele1, org.allele2)
                distances.append(dist)
                fitness_vals.append(org.fitness)
            
            distances = np.array(distances)
            fitness_vals = np.array(fitness_vals)
            
            # Set axis limits based on actual data
            x_min, x_max = distances.min(), distances.max()
            y_min, y_max = fitness_vals.min(), fitness_vals.max()
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Scatter plot
            ax.scatter(distances, fitness_vals, alpha=0.6, color=colors[model], 
                      s=40, edgecolors='white', linewidth=0.5)
            
            # Add trend lines if sufficient variation
            if len(np.unique(distances)) > 3:
                x_range = np.linspace(distances.min(), distances.max(), 100)
                
                try:
                    from scipy import stats
                    # Linear fit
                    slope, intercept, r_value, p_value, _ = stats.linregress(distances, fitness_vals)
                    y_pred_linear = slope * x_range + intercept
                    ax.plot(x_range, y_pred_linear, 'r-', linewidth=2, alpha=0.8,
                           label=f'Linear (R² = {r_value**2:.3f})')
                    
                    # Quadratic fit
                    poly_coeffs = np.polyfit(distances, fitness_vals, 2)
                    y_pred_quad = np.polyval(poly_coeffs, x_range)
                    
                    # Calculate R² for quadratic
                    y_pred_quad_actual = np.polyval(poly_coeffs, distances)
                    ss_res = np.sum((fitness_vals - y_pred_quad_actual) ** 2)
                    ss_tot = np.sum((fitness_vals - np.mean(fitness_vals)) ** 2)
                    quad_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    ax.plot(x_range, y_pred_quad, 'g--', linewidth=2, alpha=0.8,
                           label=f'Quadratic (R² = {quad_r2:.3f})')
                    
                    # Add stats
                    stats_text = f'Linear: R² = {r_value**2:.3f}, Slope = {slope:.3f}\nQuad: R² = {quad_r2:.3f}'
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           fontsize=10, va='top', ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    ax.legend(loc='best', fontsize=9)
                except:
                    pass
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error calculating distances:\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
    
    plt.suptitle('Parent Genomic Distance vs Offspring Fitness', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, f'parent_distance_vs_offspring_fitness_{mating_strategy}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_offspring_fitness_vs_offspring_prs(diploid_dict, Resu_path, mating_strategy):
    """Simplified PRS analysis with clear interpretation, focused axes, and quadratic fits"""
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "#2E86AB", "recessive": "#A23B72", "codominant": "#F18F01"}
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, model in enumerate(models):
        ax = axs[idx]
        diploids = diploid_dict.get(model, [])
        
        ax.set_title(f'{model.capitalize()} Model', fontsize=12, pad=15)
        ax.set_xlabel('Offspring PRS Score', fontsize=11)
        ax.set_ylabel('Offspring Fitness', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if not diploids:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            continue
        
        try:
            # Calculate PRS scores
            prs_scores = []
            fitness_vals = []
            
            for org in diploids[:1000]:  # Limit for performance
                try:
                    prs = calculate_diploid_prs(org)
                    prs_scores.append(prs)
                    fitness_vals.append(org.fitness)
                except:
                    continue
            
            if not prs_scores:
                ax.text(0.5, 0.5, 'PRS calculation failed', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                continue
            
            prs_scores = np.array(prs_scores)
            fitness_vals = np.array(fitness_vals)
            
            # Set axis limits based on actual data
            x_min, x_max = prs_scores.min(), prs_scores.max()
            y_min, y_max = fitness_vals.min(), fitness_vals.max()
            x_padding = (x_max - x_min) * 0.05
            y_padding = (y_max - y_min) * 0.05
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Scatter plot
            ax.scatter(prs_scores, fitness_vals, alpha=0.6, color=colors[model], 
                      s=40, edgecolors='white', linewidth=0.5)
            
            # Add mean lines
            ax.axhline(np.mean(fitness_vals), color='red', linestyle='--', alpha=0.7,
                      label=f'Mean fitness: {np.mean(fitness_vals):.3f}')
            ax.axvline(np.mean(prs_scores), color='blue', linestyle='--', alpha=0.7,
                      label=f'Mean PRS: {np.mean(prs_scores):.3f}')
            
            # Add correlation and fits if there's variation
            if len(np.unique(prs_scores)) > 3:
                x_range = np.linspace(prs_scores.min(), prs_scores.max(), 100)
                
                try:
                    from scipy import stats
                    # Linear correlation and fit
                    corr, p_value = stats.pearsonr(prs_scores, fitness_vals)
                    slope, intercept, _, _, _ = stats.linregress(prs_scores, fitness_vals)
                    
                    y_pred_linear = slope * x_range + intercept
                    ax.plot(x_range, y_pred_linear, 'r-', linewidth=2, alpha=0.8,
                           label=f'Linear (R² = {corr**2:.3f})')
                    
                    # Quadratic fit
                    poly_coeffs = np.polyfit(prs_scores, fitness_vals, 2)
                    y_pred_quad = np.polyval(poly_coeffs, x_range)
                    
                    # Calculate R² for quadratic
                    y_pred_quad_actual = np.polyval(poly_coeffs, prs_scores)
                    ss_res = np.sum((fitness_vals - y_pred_quad_actual) ** 2)
                    ss_tot = np.sum((fitness_vals - np.mean(fitness_vals)) ** 2)
                    quad_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    ax.plot(x_range, y_pred_quad, 'g--', linewidth=2, alpha=0.8,
                           label=f'Quadratic (R² = {quad_r2:.3f})')
                    
                    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                           transform=ax.transAxes, fontsize=10, va='top', ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                except:
                    pass
            
            ax.legend(loc='lower right', fontsize=9)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
    
    plt.suptitle('Offspring PRS vs Offspring Fitness', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, f'offspring_prs_vs_offspring_fitness_{mating_strategy}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

#######################################################################
# Simulation functions
#######################################################################
def run_simulation_with_initial_fitness(num_generations, environment, initial_fitness=None, max_population_size=100000, log_genomes=True, initial_genome_seed=None, mutation_seed=None):
    """
    Run simulation with an option to specify the initial organism's fitness.
    
    Parameters:
    -----------
    num_generations : int
        Number of generations to simulate
    environment : Environment
        The environment for the simulation
    initial_fitness : float or None
        Target fitness for the initial organism. If None, a random genome is used.
    max_population_size : int
        Maximum population size
    log_genomes : bool
        Whether to log genomes in detail
    initial_genome_seed : int or None
        Seed for the initial genome's random generation
    mutation_seed : int or None
        Seed for the mutations during reproduction
        
    Returns:
    --------
    tuple
        (all_organisms, individual_fitness, generation_stats)
    """
    if initial_fitness is not None:
        # Try to find a genome with the specified fitness
        log.info(f"Searching for an initial genome with fitness close to {initial_fitness}")
        best_organism = None
        best_fitness_diff = float('inf')
        max_attempts = 10000  # Prevent infinite loops
        
        for i in tqdm(range(max_attempts), desc="Searching for initial genome"):
            # Use the same seed + attempt number for reproducibility but different genomes
            test_seed = None if initial_genome_seed is None else initial_genome_seed + i
            test_organism = Organism(environment, genome_seed=test_seed, mutation_seed=mutation_seed)
            fitness_diff = abs(test_organism.fitness - initial_fitness)
            
            if fitness_diff < best_fitness_diff:
                best_fitness_diff = fitness_diff
                best_organism = test_organism
                
                # If we're very close, we can stop
                if fitness_diff < 0.01:
                    log.info(f"Found suitable genome with fitness {best_organism.fitness:.4f} after {i+1} attempts")
                    break
        
        initial_organism = best_organism
        log.info(f"Using initial organism with fitness {initial_organism.fitness:.4f} " 
                f"(target was {initial_fitness:.4f}, difference: {best_fitness_diff:.4f})")
    else:
        # Use random initialization with the provided seed
        initial_organism = Organism(environment, genome_seed=initial_genome_seed, mutation_seed=mutation_seed)
        log.info(f"Created initial organism with random genome (seed: {initial_genome_seed})")
    
    population = [initial_organism]
    all_organisms = [initial_organism]

    log.info(f"Initial organism: ID={initial_organism.id}, "
             f"Genome={initial_organism.genome}, Fit={initial_organism.fitness:.4f}")

    individual_fitness = defaultdict(list)
    individual_fitness[initial_organism.id].append((0, initial_organism.fitness))

    generation_stats = []

    for gen in tqdm(range(num_generations), desc="Running Generations"):
        next_generation = []
        gen_fitness = []
        
        for org in population:
            # Pass the mutation seed when reproducing
            c1, c2 = org.reproduce(mutation_seed=mutation_seed)
            c1.mutate()
            c2.mutate()
            individual_fitness[c1.id].append((gen + 1, c1.fitness))
            individual_fitness[c2.id].append((gen + 1, c2.fitness))
            gen_fitness.extend([c1.fitness, c2.fitness])
            next_generation.extend([c1, c2])
            all_organisms.extend([c1, c2])

        if len(next_generation) > max_population_size:
            next_generation = sorted(next_generation, key=lambda x: (x.fitness, x.id), reverse=True)[:max_population_size]

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
        log.info(f"Generation {gen+1}: Pop size = {stats['population_size']}, "
                 f"Avg fitness = {stats['avg_fitness']:.4f}")

        # If population is small, log each genome/fitness
        if log_genomes and len(population) < 50:
            for org in population:
                log.info(f"   [GEN {gen+1}] ID={org.id}, Genome={org.genome}, Fit={org.fitness:.4f}")

        monitor_memory()
        gc.collect()

    return all_organisms, individual_fitness, generation_stats
    
def mate_last_generation(last_generation, mating_strategy=MatingStrategy.ONE_TO_ONE, fitness_models=["dominant", "recessive", "codominant"], log_every_cross=True ):
    # Sort organisms by ID to ensure consistent ordering
    sorted_organisms = sorted(last_generation, key=lambda x: x.id)
    
    diploid_offspring = defaultdict(list)
    if mating_strategy == MatingStrategy.ONE_TO_ONE:
        if len(sorted_organisms) % 2 != 0:
            sorted_organisms = sorted_organisms[:-1]
        for model in fitness_models:
            for i in range(0, len(sorted_organisms), 2):
                p1 = sorted_organisms[i]
                p2 = sorted_organisms[i+1]
                offspring = DiploidOrganism(p1, p2, fitness_model=model)
                diploid_offspring[model].append(offspring)
                if log_every_cross:
                    dist = calculate_genomic_distance(p1.genome, p2.genome)
                    log.info(f"[MATING: {model}] Parent1({p1.id}) Fit={p1.fitness:.4f}, "
                             f"Parent2({p2.id}) Fit={p2.fitness:.4f}, Dist={dist:.3f}, "
                             f"Offspring({offspring.id}) Fit={offspring.fitness:.4f}")
    elif mating_strategy == MatingStrategy.ALL_VS_ALL:
        for model in fitness_models:
            for p1, p2 in combinations(sorted_organisms, 2):
                offspring = DiploidOrganism(p1, p2, fitness_model=model)
                diploid_offspring[model].append(offspring)
                if log_every_cross:
                    dist = calculate_genomic_distance(p1.genome, p2.genome)
                    log.debug(f"[MATING: {model}] Parent1({p1.id}) Fit={p1.fitness:.4f}, "
                             f"Parent2({p2.id}) Fit={p2.fitness:.4f}, Dist={dist:.3f}, "
                             f"Offspring({offspring.id}) Fit={offspring.fitness:.4f}")
    elif mating_strategy == MatingStrategy.MATING_TYPES:
        typed_organisms = assign_mating_types(sorted_organisms)
        type_a = [org for org in typed_organisms if org.mating_type == MatingType.A]
        type_alpha = [org for org in typed_organisms if org.mating_type == MatingType.ALPHA]
        for model in fitness_models:
            for a_org, alpha_org in product(type_a, type_alpha):
                offspring = DiploidOrganism(a_org.organism, alpha_org.organism, fitness_model=model)
                diploid_offspring[model].append(offspring)
                if log_every_cross:
                    dist = calculate_genomic_distance(a_org.organism.genome, alpha_org.organism.genome)
                    log.debug(f"[MATING: {model}] Parent1({a_org.organism.id}) Fit={a_org.organism.fitness:.4f}, "
                             f"Parent2({alpha_org.organism.id}) Fit={alpha_org.organism.fitness:.4f}, "
                             f"Dist={dist:.3f}, Offspring({offspring.id}) Fit={offspring.fitness:.4f}")
    else:
        raise ValueError(f"Unknown mating strategy: {mating_strategy}")
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
    aparser.add_argument("--mating_strategy", type=str, default="one_to_one", choices=["one_to_one", "all_vs_all", "mating_types"], help="Strategy for organism mating")
    aparser.add_argument("--output_dir", type=str, default="Resu", help="Output directory for results")
    aparser.add_argument("--random_seed_env", type=int, default=None, help=("Seed for the environment-level RNG. If you pass a value, every run will build exactly the same Environment (h, J for Sherrington-Kirkpatrick **or** the chosen locus for single-position fitness, etc.). Leave it unset for a fresh, random environment each run."))
    aparser.add_argument("--log_genomes", action='store_true',help="If set, log every organism's genome each generation.")
    aparser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    aparser.add_argument("--initial_fitness", type=float, default=None, help="Target fitness for the initial organism. If set, the simulation will search for a genome with this approximate fitness.")
    aparser.add_argument("--initial_genome_seed", type=int, default=None, help="Random seed specifically for generating the initial genome")
    aparser.add_argument("--mutation_seed", type=int, default=None, help="Random seed for controlling mutations during reproduction")
    
    # Add argument for number of runs
    aparser.add_argument("--num_runs", type=int, default=1, 
                         help="Number of times to run the simulation with the same parameters")
    aparser.add_argument("--save_individual_runs", action='store_true',
                         help="If set, save data from individual runs. Otherwise, only aggregate data is saved.")
    
    # Add argument for alternative fitness method
    aparser.add_argument("--fitness_method", type=str, default="sherrington_kirkpatrick",
                         choices=["sherrington_kirkpatrick", "single_position", "additive"],
                         help="Method to calculate fitness. 'sherrington_kirkpatrick' is the original complex method, " 
                              "'single_position' is a simple method where only one position determines fitness, "
                              "'additive' is a simple additive model where each position contributes independently.")

    # if not must have arguments are provided, print help
    if len(sys.argv) == 1:
        aparser.print_help(sys.stderr)
        sys.exit(1)
    
    
    args = aparser.parse_args()

    # Convert fitness method string to enum
    fitness_method = AlternativeFitnessMethod(args.fitness_method)

    # Create more informative main results directory with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a descriptive directory name with key parameters
    params_str = f"L_{args.genome_size}_gen{args.generations}_fitness_{args.fitness_method}_runs{args.num_runs}"
    if args.fitness_method == "sherrington_kirkpatrick":
        params_str += f"_beta{args.beta}_rho{args.rho}"
    if args.random_seed_env is not None:
        params_str += f"_random_seed_env{args.random_seed_env}"
    if args.initial_fitness is not None:
        params_str += f"_initial_fitness_seed{args.initial_fitness}"
    if args.initial_genome_seed is not None:
        params_str += f"_initial_genome_seed{args.initial_genome_seed}"
    if args.mutation_seed is not None:
        params_str += f"_mutation_seed{args.mutation_seed}"

    # add the timestamp and parameters to the output directory
    params_str = f"{timestamp}_{params_str}"

    
    main_dir = f"{args.output_dir}_{params_str}"
    
    # Validate Resu directory one back from the dir __file__
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), main_dir)
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    # Initialize main logging
    main_log = init_log(main_path, args.log_level)
    main_start_time = time.time()

    main_log.info(
        f"Starting multi-run simulation with arguments: {args}\n"
        f"Command line: {' '.join(sys.argv)}\n"
        f"Output directory: {main_path}\n"
        f"Number of runs: {args.num_runs}\n"
        f"Fitness method: {args.fitness_method}\n"
        f"CPU count: {os.cpu_count()}\n"
        f"Total memory (GB): {psutil.virtual_memory().total/1024**3:.2f}\n"
        f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Using same random seed for all runs: {args.random_seed_env}\n"
    )
    main_log.info(f"LSF job details: {', '.join(_get_lsf_job_details())}")

    # Store data from all runs
    all_runs_data = []

    # Run the simulation multiple times with the SAME random seed
    for run_idx in range(args.num_runs):
        run_seed = args.random_seed_env  # Use the same seed for all runs
        mutation_seed = args.mutation_seed  # Use the provided mutation seed

        
        # Create run-specific subdirectory
        run_dir = os.path.join(main_path, f"run_{run_idx+1}")
        if args.save_individual_runs:
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            Resu_path = run_dir
        else:
            Resu_path = main_path  # Use main directory if not saving individual runs
        
        main_log.info(f"\n==== Starting Run {run_idx+1}/{args.num_runs} ====")
        main_log.info(f"Random seed: {run_seed}")
        main_log.info(f"Results path: {Resu_path}")
        
        # Initialize run-specific logging
        if args.save_individual_runs:
            log = init_log(Resu_path, args.log_level)
        else:
            log = main_log  # Use main logger if not saving individual runs
            
        run_start_time = time.time()

        log.info(
            f"Starting run {run_idx+1}/{args.num_runs} with arguments: {args}\n"
            f"Random seed: {run_seed}\n"
            f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        
        # Create environment with the selected fitness method
        environment = Environment(
            genome_size=args.genome_size, 
            beta=args.beta, 
            rho=args.rho, 
            seed=run_seed,
            fitness_method=fitness_method
        )
        log.info(f"Environment created with genome size = {args.genome_size}, fitness method = {args.fitness_method}")
        log.info(f"Fitness calculation: {environment.get_fitness_description()}")

        # Log fitness configuration details
        if fitness_method == AlternativeFitnessMethod.SHERRINGTON_KIRKPATRICK and args.genome_size <= 20:
            log.info(f"Environment.h = {environment.h}")
            log.info(f"Environment.J = \n{environment.J}")
        elif fitness_method == AlternativeFitnessMethod.SINGLE_POSITION:
            log.info(f"Key position: {environment.alternative_params['position']}")
            log.info(f"Favorable value: {environment.alternative_params['favorable_value']}")
        elif fitness_method == AlternativeFitnessMethod.ADDITIVE and args.genome_size <= 20:
            log.info(f"Position weights: {environment.alternative_params['weights']}")

        # Run the simulation
        all_organisms, individual_fitness, generation_stats = run_simulation_with_initial_fitness(
            args.generations, 
            environment, 
            initial_fitness=args.initial_fitness,
            max_population_size=100000,
            log_genomes=args.log_genomes,
            initial_genome_seed=args.initial_genome_seed,
            mutation_seed=mutation_seed
        )
        gc.collect()  # Garbage collection


        # Create visualizations only if saving individual runs
        if args.save_individual_runs:
            log.info("Creating visualizations")
            plot_detailed_fitness(individual_fitness, generation_stats, Resu_path)
            
            # Only create relationship tree if population is not too large
            if len(all_organisms) < 10000:  # Arbitrary limit to prevent memory issues
                plot_relationship_tree(all_organisms, Resu_path)
            else:
                log.info(f"Skipping relationship tree plot (too many organisms: {len(all_organisms)})")

        # Mate last generation to create diploid organisms
        log.info(f"Mating last generation using {MatingStrategy(args.mating_strategy)} strategy")
        last_generation = [org for org in all_organisms if org.generation == args.generations]
        diploid_offspring_dict = mate_last_generation(
            last_generation, 
            mating_strategy=MatingStrategy(args.mating_strategy),   
            fitness_models=["dominant", "recessive", "codominant"]
        )

        # Create fitness comparison plots only if saving individual runs
        if args.save_individual_runs:
            log.info("Creating parent-offspring fitness comparison plots")
            plot_parent_offspring_fitness(diploid_offspring_dict, Resu_path, args.mating_strategy)
            plot_parent_genomic_distance_vs_offspring_fitness(diploid_offspring_dict, Resu_path, args.mating_strategy)
            plot_offspring_vs_min_max_parent_fitness(diploid_offspring_dict, Resu_path, args.mating_strategy)
            plot_offspring_fitness_vs_offspring_prs(diploid_offspring_dict, Resu_path, args.mating_strategy)
            plot_parent_offspring_heatmap(diploid_offspring_dict, Resu_path)
        
        # Create summary statistics
        summary_stats = summarize_simulation_stats(generation_stats, individual_fitness, diploid_offspring_dict)
        log_simulation_summary(log, summary_stats)
        
        # Store run data
        run_data = {
            "run_id": run_idx + 1,
            "random_seed_env": run_seed,
            "summary_stats": summary_stats,
            "generation_stats": generation_stats,
            # Add the diploid offspring dictionary for complete raw data
            "diploid_offspring_dict": diploid_offspring_dict 
        }
        all_runs_data.append(run_data)
        
        # Log run completion
        run_time = time.time() - run_start_time
        log.info(f"Run {run_idx+1} completed. Time: {time.strftime('%H:%M:%S', time.gmtime(run_time))}")
        
        # Save memory by clearing large data structures
        del all_organisms, individual_fitness, diploid_offspring_dict
        gc.collect()
    
    # Aggregate and analyze results from all runs
    main_log.info("\n==== Aggregating Results from All Runs ====")
    
    # Use the numpy_json_encoder function when dumping JSON to handle NumPy types
    aggregated_stats, metrics_df , all_runs_data = aggregate_simulation_results(all_runs_data, main_path)
    main_log.info(f"Aggregated results saved to: {os.path.join(main_path, 'aggregated_results.json')}")
    main_log.info(f"Metrics DataFrame saved to: {os.path.join(main_path, 'metrics_df.csv')}")
    main_log.info(f"All runs data saved to: {os.path.join(main_path, 'all_runs_data.json')}")
    # main_log.info(f"Aggregated statistics: {aggregated_stats}")
    # main_log.info(f"Metrics DataFrame: {metrics_df.head()}")
    log_aggregated_stats(main_log, aggregated_stats)
    
    # Create slope analysis plots
    main_log.info("Creating slope analysis plots")
    plot_slope_sign_analysis(metrics_df, main_path)
    
    # Log overall completion
    total_time = time.time() - main_start_time
    main_log.info(f"\nAll {args.num_runs} simulation runs completed.")
    main_log.info(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    main_log.info(f"Average time per run: {time.strftime('%H:%M:%S', time.gmtime(total_time/args.num_runs))}")
    main_log.info(f"Memory usage: {psutil.virtual_memory().percent:.2f}%")
    main_log.info(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    main_log.info(f"The results are saved in: {main_path}")
    main_log.info("=== END OF MULTI-RUN SIMULATION ===")

    
#######################################################################
# Entry point
#######################################################################
if __name__ == "__main__":
    main()
