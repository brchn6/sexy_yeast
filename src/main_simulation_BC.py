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
        
        For codominant model:
        - If alleles are the same (1,1) or (-1,-1): use that value
        - If alleles are different (1,-1) or (-1,1): use 0.5 * (allele1 + allele2)
        - If environment prefers 1 and alleles are (1,1): returns 1
        """
         # Special handling for genome size 1
        if len(self.allele1) == 1:
            if self.fitness_model == "dominant":
                return np.array([-1]) if -1 in [self.allele1[0], self.allele2[0]] else np.array([1])
            elif self.fitness_model == "recessive":
                return np.array([-1]) if self.allele1[0] == -1 and self.allele2[0] == -1 else np.array([1])
            elif self.fitness_model == "codominant":
                if self.allele1[0] == self.allele2[0]:
                    return self.allele1.copy()
                else:
                    return np.array([0])  # Average of -1 and 1
    
        
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
    
    # Normalize by genome length
    return differences / len(genome1)

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
    Aggregate and analyze results from multiple simulation runs.
    
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
    """
    # Extract key metrics from each run
    run_metrics = []
    
    for run_idx, run_data in enumerate(all_runs_data):
        # Extract relevant metrics from each run's summary stats
        summary = run_data["summary_stats"]
        run_info = {
            "run_id": run_idx + 1,
            "initial_fitness": summary["initial_fitness"],
            "final_fitness": summary["final_fitness"],
            "fitness_improvement_percent": summary["fitness_improvement_percent"],
            "final_population": summary["final_population"],
            "final_fitness_std": summary["fitness_std_final"],
        }
        
        # Add diploid model statistics if available
        for model in ["dominant", "recessive", "codominant"]:
            if model in summary["diploid_stats"]:
                model_stats = summary["diploid_stats"][model]
                # Store slope values for parent fitness vs offspring fitness
                parent_fitness_slope = model_stats.get("fitness_regression_stats", {}).get("linear_slope", 0)
                # Store slope values for genomic distance vs offspring fitness
                distance_fitness_slope = model_stats.get("distance_regression_stats", {}).get("linear_slope", 0)
                
                run_info.update({
                    f"{model}_avg_offspring_fitness": model_stats.get("avg_offspring_fitness", 0),
                    f"{model}_fitness_improvement": model_stats.get("fitness_improvement", 0),
                    f"{model}_avg_genomic_distance": model_stats.get("avg_genomic_distance", 0),
                    f"{model}_parent_fitness_r2": model_stats.get("fitness_regression_stats", {}).get("linear_r_squared", 0),
                    f"{model}_parent_fitness_slope": parent_fitness_slope,
                    f"{model}_distance_fitness_r2": model_stats.get("distance_regression_stats", {}).get("linear_r_squared", 0),
                    f"{model}_distance_fitness_slope": distance_fitness_slope,
                    # Track if slopes are positive or negative
                    f"{model}_parent_fitness_slope_sign": "positive" if parent_fitness_slope > 0 else "negative",
                    f"{model}_distance_fitness_slope_sign": "positive" if distance_fitness_slope > 0 else "negative",
                })
        
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
                }
            }
    
    # Save the raw data and aggregated statistics as JSON
    metrics_file = os.path.join(Resu_path, "run_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(run_metrics, f, indent=2, default=numpy_json_encoder)

    aggregated_file = os.path.join(Resu_path, "aggregated_stats.json")
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_stats, f, indent=2, default=numpy_json_encoder)
    
    # Save metrics DataFrame to CSV for easy importing to other tools
    metrics_csv = os.path.join(Resu_path, "run_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    
    return aggregated_stats, metrics_df

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
        
        # Check if the genome has the favorable value at the selected position
        if genome[position] == favorable_value:
            return 1.0  # High fitness
        else:
            return 0.1  # Low fitness
            
    elif method == AlternativeFitnessMethod.ADDITIVE:
        weights = params["weights"]
        
        # Simple dot product of genome and weights
        # Convert -1/1 genome to 0/1 format for easier interpretation
        genome_01 = (genome + 1) / 2
        fitness = np.dot(genome_01, weights)
        
        # Normalize to a reasonable range (0.1 to 1.0)
        normalized_fitness = 0.1 + 0.9 / (1 + np.exp(-fitness))
        
        return normalized_fitness
        
    # Default fallback
    return 0.5

def analyze_fitness_model_impact(all_runs_data, Resu_path):
    """
    Analyze the impact of different fitness calculation methods on parent-offspring fitness relationships.
    
    Parameters:
    -----------
    all_runs_data : list
        List of dictionaries containing data from multiple runs
    Resu_path : str
        Directory path where to save analysis results
    """
    # Extract fitness method from the first run (all runs should use the same method)
    fitness_method = all_runs_data[0].get("fitness_method", "sherrington_kirkpatrick")
    
    # Create a report file with detailed analysis
    report_path = os.path.join(Resu_path, "fitness_model_impact_analysis.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"=== IMPACT ANALYSIS OF FITNESS METHOD: {fitness_method} ===\n\n")
        
        # 1. Analyze slope trends across runs
        f.write("1. SLOPE TREND ANALYSIS\n")
        f.write("------------------------\n")
        
        # Count how many runs have positive/negative parent-fitness slopes for each model
        dominance_models = ["dominant", "recessive", "codominant"]
        models_summary = {}
        
        for model in dominance_models:
            positive_parent_slopes = 0
            negative_parent_slopes = 0
            positive_distance_slopes = 0
            negative_distance_slopes = 0
            total_parent_slope = 0
            total_distance_slope = 0
            
            for run_data in all_runs_data:
                summary = run_data.get("summary_stats", {})
                diploid_stats = summary.get("diploid_stats", {}).get(model, {})
                
                if diploid_stats:
                    # Parent fitness slope
                    parent_slope = diploid_stats.get("fitness_regression_stats", {}).get("linear_slope", 0)
                    if parent_slope > 0:
                        positive_parent_slopes += 1
                    else:
                        negative_parent_slopes += 1
                    total_parent_slope += parent_slope
                    
                    # Distance-fitness slope
                    distance_slope = diploid_stats.get("distance_regression_stats", {}).get("linear_slope", 0)
                    if distance_slope > 0:
                        positive_distance_slopes += 1
                    else:
                        negative_distance_slopes += 1
                    total_distance_slope += distance_slope
            
            # Calculate averages for non-zero cases
            total_runs = len(all_runs_data)
            avg_parent_slope = total_parent_slope / total_runs if total_runs > 0 else 0
            avg_distance_slope = total_distance_slope / total_runs if total_runs > 0 else 0
            
            models_summary[model] = {
                "positive_parent_slopes": positive_parent_slopes,
                "negative_parent_slopes": negative_parent_slopes,
                "positive_distance_slopes": positive_distance_slopes,
                "negative_distance_slopes": negative_distance_slopes,
                "avg_parent_slope": avg_parent_slope,
                "avg_distance_slope": avg_distance_slope
            }
            
            # Write to report
            f.write(f"\n{model.upper()} MODEL:\n")
            f.write(f"  Parent-Offspring Fitness Relationship:\n")
            f.write(f"    Positive slopes: {positive_parent_slopes}/{total_runs} ({positive_parent_slopes/total_runs*100:.1f}%)\n")
            f.write(f"    Negative slopes: {negative_parent_slopes}/{total_runs} ({negative_parent_slopes/total_runs*100:.1f}%)\n")
            f.write(f"    Average slope: {avg_parent_slope:.4f}\n")
            
            f.write(f"  Genomic Distance-Fitness Relationship:\n")
            f.write(f"    Positive slopes: {positive_distance_slopes}/{total_runs} ({positive_distance_slopes/total_runs*100:.1f}%)\n")
            f.write(f"    Negative slopes: {negative_distance_slopes}/{total_runs} ({negative_distance_slopes/total_runs*100:.1f}%)\n")
            f.write(f"    Average slope: {avg_distance_slope:.4f}\n")
            
        # 2. Compare fitness gains across models
        f.write("\n\n2. OFFSPRING FITNESS COMPARISON\n")
        f.write("------------------------------\n")
        
        model_fitness_gains = {}
        
        for model in dominance_models:
            total_fitness_gain = 0
            num_valid_runs = 0
            
            for run_data in all_runs_data:
                summary = run_data.get("summary_stats", {})
                diploid_stats = summary.get("diploid_stats", {}).get(model, {})
                
                if diploid_stats and "fitness_improvement" in diploid_stats:
                    total_fitness_gain += diploid_stats["fitness_improvement"]
                    num_valid_runs += 1
            
            avg_fitness_gain = total_fitness_gain / num_valid_runs if num_valid_runs > 0 else 0
            model_fitness_gains[model] = avg_fitness_gain
            
            f.write(f"\n{model.upper()} MODEL:\n")
            f.write(f"  Average fitness improvement: {avg_fitness_gain:.4f}\n")
        
        # 3. Analyze impact on the expected dominant model negative correlation
        f.write("\n\n3. IMPACT ON DOMINANCE MODEL EXPECTATIONS\n")
        f.write("---------------------------------------\n")
        f.write("\nExpectation: In the dominant model, we expect to see a negative correlation\n")
        f.write("between parent genomic distance and offspring fitness.\n\n")
        
        dominant_data = models_summary.get("dominant", {})
        total = dominant_data.get("positive_distance_slopes", 0) + dominant_data.get("negative_distance_slopes", 0)
        
        if total > 0:
            negative_percentage = dominant_data.get("negative_distance_slopes", 0) / total * 100
            f.write(f"With {fitness_method} fitness method:\n")
            f.write(f"  Negative correlation in dominant model: {negative_percentage:.1f}% of runs\n")
            f.write(f"  Average slope in dominant model: {dominant_data.get('avg_distance_slope', 0):.4f}\n\n")
            
            if negative_percentage > 60:
                f.write("CONCLUSION: The fitness method SUPPORTS the expected negative correlation.\n")
            elif negative_percentage < 40:
                f.write("CONCLUSION: The fitness method CONTRADICTS the expected negative correlation.\n")
            else:
                f.write("CONCLUSION: The fitness method shows MIXED RESULTS regarding the expected pattern.\n")
        
        # 4. Summary and recommendations
        f.write("\n\n4. SUMMARY AND RECOMMENDATIONS\n")
        f.write("-----------------------------\n")
        
        # Determine which model shows the most consistent pattern
        most_consistent_model = ""
        highest_consistency = 0
        
        for model in dominance_models:
            model_data = models_summary.get(model, {})
            pos = model_data.get("positive_distance_slopes", 0)
            neg = model_data.get("negative_distance_slopes", 0)
            
            if pos + neg > 0:
                consistency = max(pos, neg) / (pos + neg)
                if consistency > highest_consistency:
                    highest_consistency = consistency
                    most_consistent_model = model
        
        f.write(f"\nFitness Method: {fitness_method}\n")
        f.write(f"Most consistent model: {most_consistent_model} (consistency: {highest_consistency*100:.1f}%)\n\n")
        
        # General advice based on fitness method
        if fitness_method == "sherrington_kirkpatrick":
            f.write("The Sherrington-Kirkpatrick model creates a complex fitness landscape with\n")
            f.write("many interactions between genome positions. To get more consistent results, try:\n")
            f.write("  - Increasing the genome size\n")
            f.write("  - Running more generations\n")
            f.write("  - Adjusting beta and rho parameters\n")
            f.write("  - Increasing the number of runs\n")
        elif fitness_method == "single_position":
            f.write("The single position model is very simple and might not capture the complexity\n")
            f.write("of realistic fitness landscapes. This simplicity may lead to either very consistent\n")
            f.write("or very inconsistent results depending on the dominance model implementation.\n")
        elif fitness_method == "additive":
            f.write("The additive model lacks epistatic interactions, which might be important for\n")
            f.write("realistic fitness landscapes. The observed patterns might differ from expectations\n")
            f.write("based on complex epistatic models.\n")
    
    # Return the path to the report
    return report_path

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
    Create a figure with three subplots (one per diploid model) showing the relationship 
    between mean parent fitness (x-axis) and offspring fitness (y-axis).
    
    Parameters:
    -----------
    diploid_dict : dict
        Dictionary with keys as fitness model names and values as lists of DiploidOrganism instances.
    Resu_path : str
        Directory path where the resulting figure will be saved.
    mating_strategy : str
        The mating strategy used (for plot title)
    """
    # Define the order and colors for the three models
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "blue", "recessive": "green", "codominant": "purple"}

    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    for idx, model in enumerate(tqdm(models, desc=f"Processing Models for {mating_strategy} Strategy")):
        diploids = diploid_dict.get(model, [])
        
        # Set basic subplot properties regardless of data
        axs[idx].set_title(f"{model.capitalize()} Model - {len(diploids)} organisms\n"
                          f"Mating strategy: {mating_strategy}")
        axs[idx].set_xlabel("Mean Parent Fitness")
        axs[idx].set_ylabel("Offspring Fitness")
        axs[idx].grid(True)
        
        # Handle empty data case
        if not diploids:
            axs[idx].text(0.5, 0.5, "No Data", 
                        ha='center', va='center', transform=axs[idx].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            continue
            
        # Extract fitness data
        offspring_fitness = np.array([org.fitness for org in diploids])
        parent_fitness = np.array([org.avg_parent_fitness for org in diploids])
        
        # Check if we have enough variation for meaningful analysis
        parent_unique = len(np.unique(parent_fitness))
        offspring_unique = len(np.unique(offspring_fitness))
        
        # Always plot the scatter plot if we have any data
        scatter = axs[idx].scatter(
            parent_fitness, 
            offspring_fitness, 
            alpha=0.7, 
            color=colors[model], 
            label="Data Points"
        )
        
        # Handle case with no variation
        if parent_unique <= 1:
            axs[idx].text(0.5, 0.2, "All parent fitness values are identical\n"
                         f"Value: {parent_fitness[0]:.4f}",
                        ha='center', va='center', transform=axs[idx].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            
            # Still add a horizontal line at the offspring fitness mean
            if offspring_unique > 1:
                mean_offspring = np.mean(offspring_fitness)
                axs[idx].axhline(mean_offspring, color='red', linestyle='-', label=f'Mean Offspring: {mean_offspring:.4f}')
                
            axs[idx].legend(loc='lower right')
            continue
            
        # Regular case - we have variation in both x and y
        try:
            # Scatter plot already added above
            
            # Linear regression
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(parent_fitness, offspring_fitness)
            linear_r2 = r_value**2
            
            # Plot linear fit
            x_range = np.linspace(min(parent_fitness), max(parent_fitness), 100)
            axs[idx].plot(x_range, slope * x_range + intercept, 
                        color='red', linestyle='-', 
                        label=f'Linear (R²: {linear_r2:.3f})')
            
            # Try quadratic regression if we have enough unique points
            if parent_unique >= 3:
                try:
                    # Quadratic regression
                    quad_z = np.polyfit(parent_fitness, offspring_fitness, 2)
                    quad_model = np.poly1d(quad_z)
                    
                    # Calculate quadratic R²
                    y_pred = quad_model(parent_fitness)
                    y_mean = np.mean(offspring_fitness)
                    quad_r2 = 1 - (np.sum((offspring_fitness - y_pred)**2) / 
                                  np.sum((offspring_fitness - y_mean)**2))
                    
                    # Plot quadratic fit
                    axs[idx].plot(x_range, quad_model(x_range), 
                               color='orange', linestyle='--', 
                               label=f'Quadratic (R²: {quad_r2:.3f})')
                except Exception as e:
                    print(f"Warning: Quadratic regression failed for {model} model: {e}")
            
            # Reference line (x = y)
            min_val = min(min(parent_fitness), min(offspring_fitness))
            max_val = max(max(parent_fitness), max(offspring_fitness))
            axs[idx].plot([min_val, max_val], [min_val, max_val], 
                        'k--', label='y = x (No Change Line)')
            
            # Add statistics text
            stats_text = [
                f"Linear R²: {linear_r2:.3f}",
                f"Slope: {slope:.3f}",
                f"Intercept: {intercept:.3f}",
                f"p-value: {p_value:.3e}",
            ]
            
            # Add correlation information
            pearson_r = r_value
            try:
                spearman_r, spearman_p = stats.spearmanr(parent_fitness, offspring_fitness)
                stats_text.append(f"Pearson r: {pearson_r:.3f}")
                stats_text.append(f"Spearman ρ: {spearman_r:.3f}")
            except Exception:
                stats_text.append(f"Pearson r: {pearson_r:.3f}")
            
            # Add mean values
            mean_parent = np.mean(parent_fitness)
            mean_offspring = np.mean(offspring_fitness)
            stats_text.append(f"Mean Parent: {mean_parent:.3f}")
            stats_text.append(f"Mean Offspring: {mean_offspring:.3f}")
            
            # Show improvement percentage
            improvement = ((mean_offspring - mean_parent) / abs(mean_parent)) * 100 if mean_parent != 0 else 0
            stats_text.append(f"Improvement: {improvement:.2f}%")
            
            axs[idx].text(0.05, 0.95, "\n".join(stats_text), 
                        transform=axs[idx].transAxes, 
                        fontsize=9, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Try adding a KDE plot if we have enough different points
            if parent_unique >= 5 and offspring_unique >= 5:
                try:
                    sns.kdeplot(
                        x=parent_fitness, 
                        y=offspring_fitness, 
                        ax=axs[idx], 
                        cmap="Blues", 
                        fill=True, 
                        alpha=0.3, 
                        levels=5,
                        label="Density"
                    )
                except Exception as e:
                    print(f"Warning: KDE plot failed for {model} model: {e}")
            
        except Exception as e:
            # Handle any other errors
            error_msg = f"Analysis error: {str(e)}"
            print(f"Warning: {error_msg}")
            axs[idx].text(0.5, 0.5, error_msg, 
                        ha='center', va='center', transform=axs[idx].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        # Always add a legend
        handles, labels = axs[idx].get_legend_handles_labels()
        if handles:
            axs[idx].legend(loc='lower right')

    # Global title
    plt.suptitle("Parent vs Offspring Fitness Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
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
    Create plots comparing offspring fitness to minimum and maximum parent fitness
    with robust error handling for edge cases like genome size of 1.
    
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
    
    # Create two separate figures for max and min parent fitness
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    for idx, model in enumerate(tqdm(models, desc=f"Creating min-max plots for {mating_strategy}")):
        diploids = diploid_dict.get(model, [])
        
        # Set basic subplot properties for both plots
        axs1[idx].set_title(f"{model.capitalize()} Model - {len(diploids) if diploids else 0} organisms")
        axs1[idx].set_xlabel("Maximum Parent Fitness")
        axs1[idx].set_ylabel("Offspring Fitness")
        axs1[idx].grid(True)
        
        axs2[idx].set_title(f"{model.capitalize()} Model - {len(diploids) if diploids else 0} organisms")
        axs2[idx].set_xlabel("Minimum Parent Fitness")
        axs2[idx].set_ylabel("Offspring Fitness")
        axs2[idx].grid(True)
        
        # Handle empty data case
        if not diploids:
            for ax in [axs1[idx], axs2[idx]]:
                ax.text(0.5, 0.5, "No Data", 
                      ha='center', va='center', transform=ax.transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            continue
        
        try:
            # Extract fitness data
            offspring_fitness = np.array([org.fitness for org in diploids])
            min_parent_fitness = np.array([min(org.parent1_fitness, org.parent2_fitness) for org in diploids])
            max_parent_fitness = np.array([max(org.parent1_fitness, org.parent2_fitness) for org in diploids])
            
            # Check variation in parent fitness values
            min_unique = len(np.unique(min_parent_fitness))
            max_unique = len(np.unique(max_parent_fitness))
            offspring_unique = len(np.unique(offspring_fitness))
            
            # ---- Process Max Parent Fitness Plot (Fig 1) ----
            
            # Always create scatter plot regardless of variation
            axs1[idx].scatter(
                max_parent_fitness, 
                offspring_fitness, 
                alpha=0.7, 
                color=colors[model], 
                s=30,
                label="Data Points"
            )
            
            # Handle case with no variation in max parent fitness
            if max_unique <= 1:
                axs1[idx].text(0.5, 0.2, 
                              f"All max parent fitness values are identical: {max_parent_fitness[0]:.4f}\n"
                              f"Mean offspring fitness: {np.mean(offspring_fitness):.4f}",
                              ha='center', va='center', transform=axs1[idx].transAxes,
                              bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
                
                # Add horizontal line at mean offspring fitness
                if offspring_unique > 1:
                    mean_off = np.mean(offspring_fitness)
                    axs1[idx].axhline(mean_off, color='red', linestyle='-', 
                                    label=f'Mean Offspring: {mean_off:.4f}')
            else:
                # We have variation in max parent fitness, so try regression
                try:
                    # Linear regression using scipy.stats (more robust)
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        max_parent_fitness, offspring_fitness
                    )
                    r_squared = r_value**2
                    
                    # Create range for plotting
                    x_range = np.linspace(min(max_parent_fitness), max(max_parent_fitness), 100)
                    y_pred = slope * x_range + intercept
                    
                    # Plot regression line
                    axs1[idx].plot(x_range, y_pred, 'r-', 
                                  label=f'Linear (R²: {r_squared:.3f})')
                    
                    # Try quadratic regression if we have enough unique points
                    if max_unique >= 3:
                        try:
                            # Define quadratic function for curve_fit
                            def quadratic(x, a, b, c):
                                return a * x**2 + b * x + c
                            
                            # Use curve_fit for robust fitting
                            from scipy.optimize import curve_fit
                            popt, _ = curve_fit(quadratic, max_parent_fitness, offspring_fitness)
                            
                            # Calculate predictions for plotting
                            quad_y_pred = quadratic(x_range, *popt)
                            
                            # Calculate R² for quadratic fit
                            all_y_pred = quadratic(max_parent_fitness, *popt)
                            y_mean = np.mean(offspring_fitness)
                            ss_total = np.sum((offspring_fitness - y_mean)**2)
                            ss_residual = np.sum((offspring_fitness - all_y_pred)**2)
                            quad_r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                            
                            # Plot quadratic fit
                            axs1[idx].plot(x_range, quad_y_pred, 'g--', 
                                         label=f'Quadratic (R²: {quad_r2:.3f})')
                        except Exception as e:
                            print(f"Warning: Quadratic fit failed for max parent in {model} model: {e}")
                    
                    # Add reference line (x = y)
                    min_val = min(min(max_parent_fitness), min(offspring_fitness))
                    max_val = max(max(max_parent_fitness), max(offspring_fitness))
                    axs1[idx].plot([min_val, max_val], [min_val, max_val], 
                                  'k--', label='y = x (No Change)')
                    
                    # Add statistics text
                    stats_text = [
                        f"Linear R²: {r_squared:.3f}",
                        f"Slope: {slope:.3f}",
                        f"Intercept: {intercept:.3f}",
                        f"Correlation: {r_value:.3f}",
                        f"Mean Max Parent: {np.mean(max_parent_fitness):.3f}",
                        f"Mean Offspring: {np.mean(offspring_fitness):.3f}"
                    ]
                    
                    axs1[idx].text(0.05, 0.95, "\n".join(stats_text), 
                                  transform=axs1[idx].transAxes, 
                                  fontsize=9, va='top', ha='left',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception as e:
                    # Handle regression errors for max parent
                    error_msg = str(e).split('\n')[0]  # First line only
                    print(f"Warning: Max parent regression failed for {model} model: {error_msg}")
                    axs1[idx].text(0.5, 0.5, f"Regression analysis failed:\n{error_msg}", 
                                  ha='center', va='center', transform=axs1[idx].transAxes,
                                  bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            
            # Add KDE plot if we have enough different points
            if max_unique >= 5 and offspring_unique >= 5:
                try:
                    sns.kdeplot(
                        x=max_parent_fitness, 
                        y=offspring_fitness, 
                        ax=axs1[idx], 
                        cmap="Blues", 
                        fill=True, 
                        alpha=0.3, 
                        levels=5,
                        label="Density"
                    )
                except Exception as e:
                    print(f"Warning: KDE plot failed for max parent in {model} model: {e}")
            
            # Always add legend if we have any labeled elements
            handles1, labels1 = axs1[idx].get_legend_handles_labels()
            if handles1:
                axs1[idx].legend(loc='lower right')
                
            # ---- Process Min Parent Fitness Plot (Fig 2) ----
            
            # Always create scatter plot regardless of variation
            axs2[idx].scatter(
                min_parent_fitness, 
                offspring_fitness, 
                alpha=0.7, 
                color=colors[model], 
                s=30,
                label="Data Points"
            )
            
            # Handle case with no variation in min parent fitness
            if min_unique <= 1:
                axs2[idx].text(0.5, 0.2, 
                              f"All min parent fitness values are identical: {min_parent_fitness[0]:.4f}\n"
                              f"Mean offspring fitness: {np.mean(offspring_fitness):.4f}",
                              ha='center', va='center', transform=axs2[idx].transAxes,
                              bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
                
                # Add horizontal line at mean offspring fitness
                if offspring_unique > 1:
                    mean_off = np.mean(offspring_fitness)
                    axs2[idx].axhline(mean_off, color='red', linestyle='-', 
                                    label=f'Mean Offspring: {mean_off:.4f}')
            else:
                # We have variation in min parent fitness, so try regression
                try:
                    # Linear regression using scipy.stats (more robust)
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        min_parent_fitness, offspring_fitness
                    )
                    r_squared = r_value**2
                    
                    # Create range for plotting
                    x_range = np.linspace(min(min_parent_fitness), max(min_parent_fitness), 100)
                    y_pred = slope * x_range + intercept
                    
                    # Plot regression line
                    axs2[idx].plot(x_range, y_pred, 'r-', 
                                  label=f'Linear (R²: {r_squared:.3f})')
                    
                    # Try quadratic regression if we have enough unique points
                    if min_unique >= 3:
                        try:
                            # Define quadratic function for curve_fit
                            def quadratic(x, a, b, c):
                                return a * x**2 + b * x + c
                            
                            # Use curve_fit for robust fitting
                            from scipy.optimize import curve_fit
                            popt, _ = curve_fit(quadratic, min_parent_fitness, offspring_fitness)
                            
                            # Calculate predictions for plotting
                            quad_y_pred = quadratic(x_range, *popt)
                            
                            # Calculate R² for quadratic fit
                            all_y_pred = quadratic(min_parent_fitness, *popt)
                            y_mean = np.mean(offspring_fitness)
                            ss_total = np.sum((offspring_fitness - y_mean)**2)
                            ss_residual = np.sum((offspring_fitness - all_y_pred)**2)
                            quad_r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                            
                            # Plot quadratic fit
                            axs2[idx].plot(x_range, quad_y_pred, 'g--', 
                                         label=f'Quadratic (R²: {quad_r2:.3f})')
                        except Exception as e:
                            print(f"Warning: Quadratic fit failed for min parent in {model} model: {e}")
                    
                    # Add reference line (x = y)
                    min_val = min(min(min_parent_fitness), min(offspring_fitness))
                    max_val = max(max(min_parent_fitness), max(offspring_fitness))
                    axs2[idx].plot([min_val, max_val], [min_val, max_val], 
                                  'k--', label='y = x (No Change)')
                    
                    # Add statistics text
                    stats_text = [
                        f"Linear R²: {r_squared:.3f}",
                        f"Slope: {slope:.3f}",
                        f"Intercept: {intercept:.3f}",
                        f"Correlation: {r_value:.3f}",
                        f"Mean Min Parent: {np.mean(min_parent_fitness):.3f}",
                        f"Mean Offspring: {np.mean(offspring_fitness):.3f}"
                    ]
                    
                    axs2[idx].text(0.05, 0.95, "\n".join(stats_text), 
                                  transform=axs2[idx].transAxes, 
                                  fontsize=9, va='top', ha='left',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except Exception as e:
                    # Handle regression errors for min parent
                    error_msg = str(e).split('\n')[0]  # First line only
                    print(f"Warning: Min parent regression failed for {model} model: {error_msg}")
                    axs2[idx].text(0.5, 0.5, f"Regression analysis failed:\n{error_msg}", 
                                  ha='center', va='center', transform=axs2[idx].transAxes,
                                  bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            
            # Add KDE plot if we have enough different points
            if min_unique >= 5 and offspring_unique >= 5:
                try:
                    sns.kdeplot(
                        x=min_parent_fitness, 
                        y=offspring_fitness, 
                        ax=axs2[idx], 
                        cmap="Blues", 
                        fill=True, 
                        alpha=0.3, 
                        levels=5,
                        label="Density"
                    )
                except Exception as e:
                    print(f"Warning: KDE plot failed for min parent in {model} model: {e}")
            
            # Always add legend if we have any labeled elements
            handles2, labels2 = axs2[idx].get_legend_handles_labels()
            if handles2:
                axs2[idx].legend(loc='lower right')
                
        except Exception as e:
            # Handle any general processing errors
            error_msg = str(e)
            print(f"Error processing {model} model: {error_msg}")
            for ax in [axs1[idx], axs2[idx]]:
                ax.text(0.5, 0.5, f"Error processing data:\n{error_msg}", 
                      ha='center', va='center', transform=ax.transAxes,
                      bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red", alpha=0.8))
    
    # Set global titles and adjust layout
    fig1.suptitle(f"Offspring Fitness vs Maximum Parent Fitness\n({mating_strategy} Strategy)", fontsize=16)
    fig2.suptitle(f"Offspring Fitness vs Minimum Parent Fitness\n({mating_strategy} Strategy)", fontsize=16)
    
    plt.figure(fig1.number)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    max_parent_path = os.path.join(Resu_path, f'offspring_vs_max_parent_fitness_{mating_strategy}.png')
    plt.savefig(max_parent_path, dpi=300, bbox_inches='tight')
    
    plt.figure(fig2.number)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    min_parent_path = os.path.join(Resu_path, f'offspring_vs_min_parent_fitness_{mating_strategy}.png')
    plt.savefig(min_parent_path, dpi=300, bbox_inches='tight')
    
    # Close figures to free memory
    plt.close(fig1)
    plt.close(fig2)

def plot_parent_genomic_distance_vs_offspring_fitness(diploid_dict, Resu_path, mating_strategy):
    """
    Create a figure showing the relationship between parent-parent genomic distance (x-axis)
    and offspring fitness (y-axis) with robust error handling.
    
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
    
    for idx, model in enumerate(tqdm(models, desc=f"Creating genomic distance plots for {mating_strategy}")):
        diploids = diploid_dict.get(model, [])
        
        # Set basic subplot properties
        axs[idx].set_title(f"{model.capitalize()} Model - {len(diploids) if diploids else 0} organisms")
        axs[idx].set_xlabel("Genomic Distance Between Parents")
        axs[idx].set_ylabel("Offspring Fitness")
        axs[idx].grid(True)
        
        # Handle empty data case
        if not diploids:
            axs[idx].text(0.5, 0.5, "No Data", 
                        ha='center', va='center', transform=axs[idx].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            continue
        
        # Calculate genomic distances and collect fitness values
        parent_distances = []
        offspring_fitness = []
        
        try:
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
            
            # Check if we have unique distance values
            unique_distances = len(np.unique(parent_distances))
            
            # Always create scatter plot if we have data
            axs[idx].scatter(
                parent_distances, 
                offspring_fitness, 
                alpha=0.6, 
                color='blue', 
                s=30,
                label="Data Points"
            )
            
            # Handle cases with no variation in genomic distance
            if unique_distances <= 1:
                axs[idx].text(0.5, 0.2, 
                             f"All parent genomic distances are identical: {parent_distances[0]:.4f}\n"
                             f"Mean offspring fitness: {np.mean(offspring_fitness):.4f}",
                             ha='center', va='center', transform=axs[idx].transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
                
                # Add statistical info
                axs[idx].axhline(np.mean(offspring_fitness), color='red', linestyle='-', 
                               label=f'Mean Offspring Fitness: {np.mean(offspring_fitness):.4f}')
            else:
                # We have variation in distances, so try regressions
                try:
                    # Linear regression
                    from scipy import stats
                    
                    # Use scipy.stats.linregress which is more robust than np.polyfit
                    slope, intercept, r_value, p_value, std_err = stats.linregress(parent_distances, offspring_fitness)
                    r_squared = r_value**2
                    
                    # Create line for plotting
                    x_range = np.linspace(min(parent_distances), max(parent_distances), 100)
                    y_pred = slope * x_range + intercept
                    
                    # Plot regression line
                    axs[idx].plot(x_range, y_pred, 'r-', 
                                 label=f'Linear (R²: {r_squared:.3f})')
                    
                    # Try quadratic regression if we have at least 3 unique points
                    if unique_distances >= 3:
                        try:
                            # Use numpy's polynomial fitting with extra error checking
                            quad_fit = np.polynomial.polynomial.polyfit(
                                parent_distances, offspring_fitness, 2, full=True
                            )
                            
                            # Extract coefficients
                            quad_coeffs = quad_fit[0]
                            
                            # Function for quadratic prediction
                            def quad_predict(x):
                                return quad_coeffs[0] + quad_coeffs[1]*x + quad_coeffs[2]*x*x
                            
                            # Calculate predictions
                            quad_y_pred = [quad_predict(x) for x in x_range]
                            
                            # Calculate R² for quadratic fit
                            y_pred_quad = [quad_predict(x) for x in parent_distances]
                            y_mean = np.mean(offspring_fitness)
                            ss_total = np.sum((offspring_fitness - y_mean)**2)
                            ss_residual = np.sum((offspring_fitness - y_pred_quad)**2)
                            quad_r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                            
                            # Plot quadratic fit
                            axs[idx].plot(x_range, quad_y_pred, 'g--', 
                                         label=f'Quadratic (R²: {quad_r2:.3f})')
                        except Exception as e:
                            print(f"Warning: Quadratic regression failed for {model} model: {e}")
                    
                    # Add correlation statistics
                    stats_text = [
                        f"Linear R²: {r_squared:.3f}",
                        f"Slope: {slope:.3f}",
                        f"Correlation: {r_value:.3f}",
                        f"p-value: {p_value:.3e}",
                        f"Mean Distance: {np.mean(parent_distances):.3f}",
                        f"Mean Fitness: {np.mean(offspring_fitness):.3f}"
                    ]
                    
                    axs[idx].text(0.05, 0.95, "\n".join(stats_text), 
                                transform=axs[idx].transAxes, 
                                fontsize=9, va='top', ha='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    # Handle any regression errors
                    error_msg = str(e).split('\n')[0]  # Get first line of error
                    print(f"Warning: Regression analysis failed: {error_msg}")
                    axs[idx].text(0.5, 0.5, f"Regression analysis failed:\n{error_msg}", 
                                ha='center', va='center', transform=axs[idx].transAxes,
                                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            
            # Always add legend if we have any labeled elements
            handles, labels = axs[idx].get_legend_handles_labels()
            if handles:
                axs[idx].legend(loc='lower right')
                
        except Exception as e:
            # Handle any general processing errors
            print(f"Error processing {model} model: {e}")
            axs[idx].text(0.5, 0.5, f"Error processing data:\n{str(e)}", 
                        ha='center', va='center', transform=axs[idx].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red", alpha=0.8))
    
    # Global title
    plt.suptitle(f"Parent Genomic Distance vs Offspring Fitness\n({mating_strategy} Strategy)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(Resu_path, f'parent_distance_vs_offspring_fitness_{mating_strategy}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_offspring_fitness_vs_offspring_prs(diploid_dict, Resu_path, mating_strategy):
    """
    Create plots showing the relationship between offspring's own PRS scores and offspring fitness,
    with robust error handling for edge cases like genome size of 1.
    
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
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    
    for idx, model in enumerate(tqdm(models, desc=f"Creating Offspring PRS plots for {mating_strategy}")):
        diploids = diploid_dict.get(model, [])
        
        # Set basic subplot properties
        axs[idx].set_title(f"{model.capitalize()} Model - {len(diploids) if diploids else 0} organisms")
        axs[idx].set_xlabel("Offspring PRS")
        axs[idx].set_ylabel("Offspring Fitness")
        axs[idx].grid(True)
        
        # Handle empty data case
        if not diploids:
            axs[idx].text(0.5, 0.5, "No Data", 
                        ha='center', va='center', transform=axs[idx].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            continue
        
        # Collect data with error handling
        try:
            offspring_prs = []
            offspring_fitness = []
            parent_avg_prs = []
            
            # Process in small batches to avoid memory issues
            for i in range(0, len(diploids), 100):
                batch = diploids[i:i+100]
                for org in batch:
                    try:
                        # Calculate offspring's own PRS
                        org_prs = calculate_diploid_prs(org)
                        offspring_prs.append(org_prs)
                        
                        # Calculate average parent PRS for comparison
                        p1_prs = calculate_prs(org.allele1)
                        p2_prs = calculate_prs(org.allele2)
                        parent_avg_prs.append((p1_prs + p2_prs) / 2)
                        
                        # Get offspring fitness
                        offspring_fitness.append(org.fitness)
                    except Exception as e:
                        print(f"Warning: Error processing organism {org.id}: {e}")
                
                gc.collect()  # Force garbage collection after each batch
            
            # Convert to numpy arrays
            offspring_prs = np.array(offspring_prs)
            offspring_fitness = np.array(offspring_fitness)
            parent_avg_prs = np.array(parent_avg_prs)
            
            if len(offspring_prs) == 0:
                axs[idx].text(0.5, 0.5, "PRS calculation failed for all organisms", 
                            ha='center', va='center', transform=axs[idx].transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
                continue
            
            # Check for variation in PRS values
            unique_prs = len(np.unique(offspring_prs))
            
            # Always plot the scatter regardless of variation
            axs[idx].scatter(
                offspring_prs, 
                offspring_fitness, 
                alpha=0.6, 
                color='purple', 
                s=30,
                label="Offspring PRS"
            )
            
            # Add x=y line
            # Determine the appropriate range based on all data
            min_val = min(min(offspring_prs), min(offspring_fitness))
            max_val = max(max(offspring_prs), max(offspring_fitness))
            # Add some padding
            range_padding = (max_val - min_val) * 0.05
            line_min = min_val - range_padding
            line_max = max_val + range_padding
            # Plot the line
            axs[idx].axhline(np.mean(offspring_fitness), color='red', linestyle=':', 
                             alpha=0.5, label=f'Mean Fitness: {np.mean(offspring_fitness):.2f}')
            axs[idx].axvline(np.mean(offspring_prs), color='blue', linestyle=':', 
                             alpha=0.5, label=f'Mean PRS: {np.mean(offspring_prs):.2f}')
            
            # Handle case with no PRS variation
            if unique_prs <= 1:
                axs[idx].text(0.5, 0.2, 
                             f"All offspring PRS values are identical: {offspring_prs[0]:.4f}\n"
                             f"Mean fitness: {np.mean(offspring_fitness):.4f}",
                             ha='center', va='center', transform=axs[idx].transAxes,
                             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
                
                # Add mean fitness line
                axs[idx].axhline(np.mean(offspring_fitness), color='red', linestyle='-', 
                               label=f'Mean Fitness: {np.mean(offspring_fitness):.4f}')
            else:
                # We have variation, so try regression
                try:
                    # Linear regression using scipy.stats.linregress (more robust)
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(offspring_prs, offspring_fitness)
                    r_squared = r_value**2
                    
                    # Calculate points for plotting the regression line
                    x_range = np.linspace(min(offspring_prs), max(offspring_prs), 100)
                    y_pred = slope * x_range + intercept
                    
                    # Plot the regression line
                    axs[idx].plot(x_range, y_pred, 'r-', 
                                 label=f'Linear (R²: {r_squared:.3f})')
                    
                    # Try quadratic regression with proper error handling
                    if unique_prs >= 3:
                        try:
                            # Use scipy.optimize curve_fit for more robust fitting
                            from scipy.optimize import curve_fit
                            
                            # Define quadratic function
                            def quadratic(x, a, b, c):
                                return a * x**2 + b * x + c
                            
                            # Fit curve with bounds to prevent overflow
                            popt, _ = curve_fit(quadratic, offspring_prs, offspring_fitness)
                            
                            # Calculate predictions for plotting
                            quad_y_pred = quadratic(x_range, *popt)
                            
                            # Calculate R² for quadratic fit
                            all_y_pred = quadratic(offspring_prs, *popt)
                            y_mean = np.mean(offspring_fitness)
                            ss_total = np.sum((offspring_fitness - y_mean)**2)
                            ss_residual = np.sum((offspring_fitness - all_y_pred)**2)
                            quad_r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                            
                            # Plot quadratic fit
                            axs[idx].plot(x_range, quad_y_pred, 'g--', 
                                         label=f'Quadratic (R²: {quad_r2:.3f})')
                        except Exception as e:
                            print(f"Warning: Quadratic fit failed for model {model}: {e}")
                    
                    # Calculate correlations for offspring and parent PRS
                    offspring_corr = r_value  # Already calculated above
                    
                    # Calculate parent-fitness correlation only if we have non-constant parent PRS
                    if len(np.unique(parent_avg_prs)) > 1:
                        try:
                            parent_corr, _ = stats.pearsonr(parent_avg_prs, offspring_fitness)
                        except Exception:
                            parent_corr = np.nan
                    else:
                        parent_corr = np.nan
                    
                    # Add statistics text
                    stats_text = [
                        f"Linear R²: {r_squared:.3f}",
                        f"Slope: {slope:.3f}",
                        f"Offspring PRS-Fitness Corr: {offspring_corr:.3f}"
                    ]
                    
                    if not np.isnan(parent_corr):
                        stats_text.append(f"Parent PRS-Fitness Corr: {parent_corr:.3f}")
                    
                    stats_text.extend([
                        f"Mean PRS: {np.mean(offspring_prs):.3f}",
                        f"Mean Fitness: {np.mean(offspring_fitness):.3f}"
                    ])
                    
                    axs[idx].text(0.05, 0.95, "\n".join(stats_text), 
                                transform=axs[idx].transAxes, 
                                fontsize=9, va='top', ha='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    # Handle regression errors
                    error_msg = str(e).split('\n')[0]  # Get first line only
                    print(f"Warning: Regression failed for {model} model: {error_msg}")
                    axs[idx].text(0.5, 0.5, f"Regression analysis failed:\n{error_msg}", 
                                ha='center', va='center', transform=axs[idx].transAxes,
                                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
            
            # Always add legend if we have any labeled elements
            handles, labels = axs[idx].get_legend_handles_labels()
            if handles:
                axs[idx].legend(loc='lower right')
                
        except Exception as e:
            # Handle any general errors
            print(f"Error processing model {model}: {e}")
            axs[idx].text(0.5, 0.5, f"Error processing data:\n{str(e)}", 
                        ha='center', va='center', transform=axs[idx].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red", alpha=0.8))
    
    # Global title
    plt.suptitle(f"Offspring PRS vs Offspring Fitness\n({mating_strategy} Strategy)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(Resu_path, f'offspring_prs_vs_offspring_fitness_{mating_strategy}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_slope_sign_analysis(metrics_df, Resu_path):
    """
    Create simple visualizations to summarize slope trends across runs.
    Focuses on the relationship between parent fitness and genomic distance effects.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame containing metrics from all runs
    Resu_path : str
        Directory path where to save the plots
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(Resu_path, exist_ok=True)
    
    # List of models to analyze
    models = ["dominant", "recessive", "codominant"]
    colors = {"dominant": "blue", "recessive": "green", "codominant": "purple"}
    
    # 1. SIMPLE SUMMARY TABLE
    # Create a simple text file that summarizes the key trends
    summary_file = os.path.join(Resu_path, "slope_trends_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("SLOPE TREND SUMMARY\n")
        f.write("==================\n\n")
        
        for model in models:
            f.write(f"{model.upper()} MODEL\n")
            f.write("-" * len(f"{model.upper()} MODEL") + "\n")
            
            parent_key = f"{model}_parent_fitness_slope"
            distance_key = f"{model}_distance_fitness_slope"
            
            if parent_key in metrics_df.columns:
                parent_avg = metrics_df[parent_key].mean()
                parent_pos = np.sum(metrics_df[parent_key] > 0)
                parent_neg = np.sum(metrics_df[parent_key] < 0)
                total = len(metrics_df[parent_key])
                
                f.write(f"Parent-Offspring Fitness: {parent_avg:.4f} ")
                f.write(f"({parent_pos}/{total} positive, {parent_neg}/{total} negative)\n")
                
                f.write(f"Interpretation: ")
                if parent_avg > 0:
                    f.write("Higher parent fitness tends to produce higher offspring fitness\n")
                else:
                    f.write("Higher parent fitness tends to produce lower offspring fitness\n")
            
            if distance_key in metrics_df.columns:
                distance_avg = metrics_df[distance_key].mean()
                distance_pos = np.sum(metrics_df[distance_key] > 0)
                distance_neg = np.sum(metrics_df[distance_key] < 0)
                total = len(metrics_df[distance_key])
                
                f.write(f"Genomic Distance Effect: {distance_avg:.4f} ")
                f.write(f"({distance_pos}/{total} positive, {distance_neg}/{total} negative)\n")
                
                f.write(f"Interpretation: ")
                if distance_avg > 0:
                    f.write("Greater genomic distance tends to increase offspring fitness\n")
                else:
                    f.write("Greater genomic distance tends to decrease offspring fitness\n")
            
            # If both exist, look at relationship
            if parent_key in metrics_df.columns and distance_key in metrics_df.columns:
                if (parent_avg > 0 and distance_avg < 0) or (parent_avg < 0 and distance_avg > 0):
                    f.write("TRADEOFF DETECTED: Parent fitness and genomic distance have opposing effects\n")
                else:
                    f.write("NO TRADEOFF: Parent fitness and genomic distance work in the same direction\n")
            
            f.write("\n")
    
    # 2. SIMPLE BAR CHART - Average slopes by model
    plt.figure(figsize=(10, 6))
    
    # Collect data
    parent_avgs = []
    distance_avgs = []
    model_labels = []
    
    for model in models:
        parent_key = f"{model}_parent_fitness_slope"
        distance_key = f"{model}_distance_fitness_slope"
        
        if parent_key in metrics_df.columns and distance_key in metrics_df.columns:
            parent_avgs.append(metrics_df[parent_key].mean())
            distance_avgs.append(metrics_df[distance_key].mean())
            model_labels.append(model.capitalize())
    
    # Create grouped bar chart
    x = np.arange(len(model_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, parent_avgs, width, label='Parent Fitness Effect', color=['blue', 'green', 'purple'])
    rects2 = ax.bar(x + width/2, distance_avgs, width, label='Genomic Distance Effect', color=['lightblue', 'lightgreen', 'plum'])
    
    # Add zero line
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Add labels and legend
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Slope')
    ax.set_title('Effect of Parent Fitness vs. Genomic Distance on Offspring Fitness')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    
    # Add value labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top')
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, "average_slopes_by_model.png"), dpi=300)
    plt.close()
    
    # 3. SCATTER PLOT - Parent vs Distance Slopes
    # Simple scatter plot showing relationship between parent and distance slopes
    plt.figure(figsize=(10, 8))
    
    # Collect data for all models
    for model in models:
        parent_key = f"{model}_parent_fitness_slope"
        distance_key = f"{model}_distance_fitness_slope"
        
        if parent_key in metrics_df.columns and distance_key in metrics_df.columns:
            plt.scatter(
                metrics_df[parent_key], 
                metrics_df[distance_key],
                alpha=0.7,
                s=80,
                c=colors[model],
                label=model.capitalize()
            )
            
            # Add trendline for each model
            if len(metrics_df) > 2:
                try:
                    z = np.polyfit(metrics_df[parent_key], metrics_df[distance_key], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(metrics_df[parent_key].min(), metrics_df[parent_key].max(), 100)
                    plt.plot(x_range, p(x_range), '--', color=colors[model], alpha=0.5)
                except Exception as e:
                    print(f"Could not fit trendline for {model}: {e}")
    
    # Add quadrant lines
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add quadrant labels
    plt.text(0.98, 0.98, "Parent+, Distance+\nBoth increase fitness", ha="right", va="top", transform=plt.gca().transAxes,
             bbox=dict(facecolor='lightgray', alpha=0.5))
    plt.text(0.02, 0.98, "Parent-, Distance+\nTRADEOFF", ha="left", va="top", transform=plt.gca().transAxes,
             bbox=dict(facecolor='lightgray', alpha=0.5))
    plt.text(0.98, 0.02, "Parent+, Distance-\nTRADEOFF", ha="right", va="bottom", transform=plt.gca().transAxes,
             bbox=dict(facecolor='lightgray', alpha=0.5))
    plt.text(0.02, 0.02, "Parent-, Distance-\nBoth decrease fitness", ha="left", va="bottom", transform=plt.gca().transAxes,
             bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.xlabel("Parent-Offspring Fitness Slope")
    plt.ylabel("Genomic Distance-Fitness Slope")
    plt.title("Relationship Between Parent Fitness and Genomic Distance Effects")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, "parent_vs_distance_slopes.png"), dpi=300)
    plt.close()
    
    return summary_file

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
    aparser.add_argument("--random_seed_env", type=int, default=None, help="Fixed random seed for reproducibility.")
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
    aggregated_stats, metrics_df = aggregate_simulation_results(all_runs_data, main_path)
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


    # If multiple runs were performed, analyze the impact of the fitness method
    if args.num_runs > 1:
        main_log.info("\n==== Analyzing Impact of Fitness Method ====")
        # Store fitness method in run data
        for run_data in all_runs_data:
            run_data["fitness_method"] = args.fitness_method
            
        # Run the specialized analysis
        fitness_analysis_report = analyze_fitness_model_impact(all_runs_data, main_path)
        main_log.info(f"Fitness method impact analysis saved to: {fitness_analysis_report}")

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
