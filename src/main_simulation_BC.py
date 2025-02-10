#!/home/labs/pilpel/barc/.conda/envs/sexy_yeast_env/bin/python
import numpy as np
import uuid
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap
from collections import defaultdict
import logging as log
import os

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
    
    def mutate(self, mutation_rate=0.01):
        """Randomly mutate the genome"""
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

class DiploidOrganism:
    def __init__(self, parent1, parent2, fitness_model="dominant"):
        if len(parent1.genome) != len(parent2.genome):
            raise ValueError("Parent genomes must have the same length.")
        
        # Store both alleles from parents
        self.allele1 = parent1.genome.copy()
        self.allele2 = parent2.genome.copy()
        self.fitness_model = fitness_model
        self.environment = parent1.environment
        self.id = str(uuid.uuid4())
        self.parent1_id = parent1.id
        self.parent2_id = parent2.id
        self.fitness = self.calculate_fitness()
    
    def _get_effective_genome(self):
        """
        Calculate effective genome based on inheritance model.
        Returns a single array of -1/1 values representing the phenotype.
        """
        if self.fitness_model == "dominant":
            # If either allele is -1, the result is -1 (dominant)
            return np.where((self.allele1 == -1) | (self.allele2 == -1), -1, 1)
        
        elif self.fitness_model == "recessive":
            # if both alleles are -1, the result is -1 (recessive)
            return np.where((self.allele1 == -1) & (self.allele2 == -1), -1, 1)
        
        elif self.fitness_model == "codominant":
            # Average effect of both alleles
            return np.where(self.allele1 == self.allele2, self.allele1, 1)
        
        else:
            raise ValueError(f"Unknown fitness model: {self.fitness_model}")

    def calculate_fitness(self):
        """
        Calculate fitness using the Sherrington-Kirkpatrick model
        with the effective genome based on inheritance model
        """
        effective_genome = self._get_effective_genome()
        return compute_fit_slow(
            effective_genome,
            self.environment.h,
            self.environment.J,
            F_off=0.0
        )

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

def plot_fitness_comparisons(last_generation, diploid_offspring_dict, Resu_path):
    """
    Create separate fitness comparison plots for each inheritance model.
    
    Parameters:
    ----------
    last_generation : list
        List of haploid parent organisms
    diploid_offspring_dict : dict
        Dictionary with fitness models as keys and lists of diploid offspring as values
    Resu_path : str
        Path to save the results
    """
    # Create subplots for all models in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Parent vs Offspring Fitness Comparison Across Models', fontsize=14)
    
    for idx, (model, offspring) in enumerate(diploid_offspring_dict.items()):
        # Get fitness values
        parent_fitness = [org.fitness for org in last_generation]
        diploid_fitness = [org.fitness for org in offspring]
        
        log.info(f"Parent fitness: {parent_fitness} the len is {len(parent_fitness)}")
        log.info(f"Diploid fitness: {diploid_fitness} the len is {len(diploid_fitness)}")
        
        # Create scatter plot
        ax = axes[idx]
        ax.scatter(diploid_fitness, parent_fitness, alpha=0.7, edgecolors='k')
        
        # Add diagonal line
        min_val = min(min(parent_fitness), min(diploid_fitness))
        max_val = max(max(parent_fitness), max(diploid_fitness))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        
        # Customize subplot
        ax.set_xlabel('Diploid Offspring Fitness')
        ax.set_ylabel('Parent Fitness')
        ax.set_title(f'{model.capitalize()} Model')
        ax.legend()
        ax.grid(True)
        
        # Add statistical annotations
        stats_text = (f'Mean Parent: {np.mean(parent_fitness):.3f}\n'
                     f'Mean Offspring: {np.mean(diploid_fitness):.3f}')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, 'fitness_comparisons.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_fitness_GD(last_generation, diploid_offspring_dict , Resu_path):
    """
    Create scatter plots of parent fitness vs genomic distance for each inheritance model.
    
    Parameters:
    -----------
    last_generation : list
        List of haploid parent organisms
    diploid_offspring_dict : dict
        Dictionary with fitness models as keys and lists of diploid offspring as values
    """
    # Create subplots for all models in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Parent Fitness vs Genomic Distance Across Models', fontsize=14)
    
    # Store correlation coefficients for each model
    correlations = {}
    
    for idx, (model, offspring) in enumerate(diploid_offspring_dict.items()):
        parent_fitness = []
        genomic_distances = []
        
        # Calculate fitness and genomic distances for each pair
        for i in range(0, len(last_generation) - 1, 2):
            parent1 = last_generation[i]
            parent2 = last_generation[i + 1]
            
            # Calculate average parent fitness
            avg_parent_fitness = (parent1.fitness + parent2.fitness) / 2
            parent_fitness.append(avg_parent_fitness)
            
            # Calculate genomic distance between parents
            gd = calculate_genomic_distance(parent1.genome, parent2.genome)
            genomic_distances.append(gd)
        
        # Calculate correlation
        correlation = np.corrcoef(genomic_distances, parent_fitness)[0, 1]
        correlations[model] = correlation
        
        # Create scatter plot
        ax = axes[idx]
        scatter = ax.scatter(genomic_distances, parent_fitness, 
                           alpha=0.7, edgecolors='k', c=offspring[0:len(parent_fitness)].fitness,
                           cmap='viridis')
        
        # Add trend line
        z = np.polyfit(genomic_distances, parent_fitness, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(genomic_distances), max(genomic_distances), 100)
        ax.plot(x_range, p(x_range), 'r--', 
                label=f'Correlation: {correlation:.3f}')
        
        # Add colorbar for offspring fitness
        plt.colorbar(scatter, ax=ax, label='Offspring Fitness')
        
        # Customize subplot
        ax.set_xlabel('Genomic Distance')
        ax.set_ylabel('Average Parent Fitness')
        ax.set_title(f'{model.capitalize()} Model')
        ax.legend()
        ax.grid(True)
        
        # Add statistical annotations
        stats_text = (f'Mean GD: {np.mean(genomic_distances):.3f}\n'
                     f'Mean Parent Fitness: {np.mean(parent_fitness):.3f}\n'
                     f'GD-Fitness Corr: {correlation:.3f}')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Resu_path, 'fitness_GD.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlations

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
        
        print(f"Generation {gen+1}: Pop size = {stats['population_size']}, "
              f"Avg fitness = {stats['avg_fitness']:.4f}, "
              f"Max fitness = {stats['max_fitness']:.4f}, "
              f"Min fitness = {stats['min_fitness']:.4f}")

    
    return all_organisms, individual_fitness, generation_stats

def mate_last_generation(last_generation, fitness_models=["dominant", "recessive", "codominant"]):
    """
    Mate haploid organisms to create diploid organisms using multiple fitness models.
    
    Parameters:
    -----------
    last_generation : list
        List of haploid organisms from the last generation
    fitness_models : list
        List of fitness models to use for creating diploid offspring
        
    Returns:
    --------
    dict : Dictionary with fitness models as keys and lists of diploid organisms as values
    """
    diploid_offspring = defaultdict(list)
    
    # Ensure even number of parents
    if len(last_generation) % 2 != 0:
        last_generation = last_generation[:-1]
    
    # Create diploid offspring for each fitness model
    for model in fitness_models:
        for i in range(0, len(last_generation), 2):
            parent1 = last_generation[i]
            parent2 = last_generation[i + 1]
            
            offspring = DiploidOrganism(parent1, parent2, fitness_model=model)
            diploid_offspring[model].append(offspring)
            
        print(f"Created {len(diploid_offspring[model])} diploid organisms using {model} model")
    
    return diploid_offspring

def analyze_diploid_fitness(diploid_offspring_dict, last_generation, Resu_path):
    """
    Analyze and compare fitness between haploid parents and diploid offspring
    for different inheritance models.
    
    Parameters:
    -----------
    diploid_offspring_dict : dict
        Dictionary with fitness models as keys and lists of diploid organisms as values
    last_generation : list
        List of haploid organisms from the last generation
    """
    # Create plots and get correlations
    correlations = plot_fitness_comparisons(last_generation, diploid_offspring_dict, Resu_path)
    
    # Print detailed analysis for each model
    log.info("\nDetailed Fitness Analysis:")
    log.info("=" * 50)
    
    for model, offspring in diploid_offspring_dict.items():
        parent_fitness = []
        offspring_fitness = []
        
        for i in range(0, len(last_generation) - 1, 2):
            parent1 = last_generation[i]
            parent2 = last_generation[i + 1]
            avg_parent_fitness = (parent1.fitness + parent2.fitness) / 2
            parent_fitness.append(avg_parent_fitness)
            offspring_fitness.append(offspring[i // 2].fitness)
            
        log.info(f"\nModel: {model.upper()}")
        log.info("=" * 50)
        log.info(f"Parent Statistics:")
        log.info(f"  Mean Fitness: {np.mean(parent_fitness):.4f}")
        log.info(f"  Std Deviation: {np.std(parent_fitness):.4f}")
        log.info(f"  Min Fitness: {np.min(parent_fitness):.4f}")
        log.info(f"  Max Fitness: {np.max(parent_fitness):.4f}")
        
        log.info(f"\nOffspring Statistics:")
        log.info(f"  Mean Fitness: {np.mean(offspring_fitness):.4f}")
        log.info(f"  Std Deviation: {np.std(offspring_fitness):.4f}")
        log.info(f"  Min Fitness: {np.min(offspring_fitness):.4f}")
        log.info(f"  Max Fitness: {np.max(offspring_fitness):.4f}")
        
        log.info(f"\nComparison:")
        log.info(f"  Correlation: {correlations[model]:.4f}")
        log.info(f"  Mean Fitness Difference: {np.mean(offspring_fitness) - np.mean(parent_fitness):.4f}")
        
        # Calculate if offspring tend to be better or worse than parents
        better_count = sum(1 for i in range(len(offspring_fitness)) 
                         if offspring_fitness[i] > parent_fitness[i])
        log.info(f"  Offspring better than parents: {better_count}/{len(offspring_fitness)}")
        log.info(f"  Percentage better: {(better_count/len(offspring_fitness))*100:.1f}%")

    
    # Create scatter plots of fitness vs genomic distance
    plot_fitness_GD(last_generation, diploid_offspring_dict, Resu_path) 
                      
#######################################################################
# Main function
#######################################################################
def main():
    aparser = ap.ArgumentParser(description="Run the sexy yeast simulation with detailed fitness tracking")
    aparser.add_argument("--generations", type=int, default=5, help="Number of generations to simulate")
    aparser.add_argument("--genome_size", type=int, default=100, help="Size of the genome")
    aparser.add_argument("--beta", type=float, default=0.5, help="Beta parameter")
    aparser.add_argument("--rho", type=float, default=0.25, help="Rho parameter")

    args = aparser.parse_args()

    # val Resu directory one back from the dir __file__
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Resu')):
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Resu'))

    Resu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Resu')
    
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
    log.info("Mating last generation to create diploid organisms")
    last_generation = [org for org in all_organisms if org.generation == args.generations]
    fitness_models = ["dominant", "recessive", "codominant"]
    diploid_offspring_dict = mate_last_generation(last_generation, fitness_models=fitness_models)
    analyze_diploid_fitness(diploid_offspring_dict, last_generation, Resu_path)



   # Log the final results
    log.info("Simulation complete")

if __name__ == "__main__":
    main()
    

"""
example usage:
/home/labs/pilpel/barc/sexy_yeast/main_simulation_BC.py --generations 10 --genome_size 100 --beta 0.5 --rho 0.25
"""

