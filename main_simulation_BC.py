#!/home/labs/pilpel/barc/.conda/envs/sexy_yeast_env/bin/python
import numpy as np
import uuid
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap
from collections import defaultdict
import logging as log

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

def init_log():
    """Initialize logging"""
    # init logging
    log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    # add file handler
    fh = log.FileHandler("sexy_yeast.log", mode='w')
    fh.setLevel(log.INFO)
    log.getLogger().addHandler(fh)

    return log

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
        energy = compute_fit_slow(genome, self.h, self.J , F_off=0.01)
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

class DiploidOrganism(Organism):
    def __init__(self, parent1, parent2, fitness_model="dominant"):
        if len(parent1.genome) != len(parent2.genome):
            raise ValueError("Parent genomes must have the same length.")
        
        self.genome = [(a, b) for a, b in zip(parent1.genome, parent2.genome)]
        self.fitness_model = fitness_model
        self.environment = parent1.environment  # Same environment as parents
        self.fitness = self.calculate_fitness()
    
    def calculate_fitness(self):
        if self.fitness_model == "dominant":
            return self._calculate_dominant_fitness()
        elif self.fitness_model == "recessive":
            return self._calculate_recessive_fitness()
        elif self.fitness_model == "codominant":
            return self._calculate_codominant_fitness()
        else:
            raise ValueError("Unknown fitness model: Choose 'dominant', 'recessive', or 'codominant'")
    
    def _calculate_dominant_fitness(self):
        fitness = 1.0
        for a, b in self.genome:
            if a == -1 or b == -1:
                fitness -= 0.1  # Reduce fitness for each bad site
        return max(fitness, 0.0)  # Ensure fitness is non-negative

    def _calculate_recessive_fitness(self):
        fitness = 1.0
        for a, b in self.genome:
            if a == -1 and b == -1:
                fitness -= 0.2  # Reduce fitness more if both alleles are bad
        return max(fitness, 0.0)

    def _calculate_codominant_fitness(self):
        fitness = 1.0
        for a, b in self.genome:
            bad_alleles = [a, b].count(-1)
            fitness -= 0.05 * bad_alleles  # Reduce fitness by 0.05 for each bad allele
        return max(fitness, 0.0)

#######################################################################
# plotting functions
#######################################################################
def plot_detailed_fitness(individual_fitness, generation_stats):
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
    plt.savefig('detailed_fitness.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_relationship_tree(organisms):
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
    plt.savefig("relationship_tree.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_fitness_comparison(parents, diploid_offspring, fitness_model):
    """
    Plot a scatter plot comparing the fitness of haploid parents (average of two parents)
    against the fitness of their diploid offspring.

    Parameters:
    ----------
    parents: list of Organism
        List of haploid parent organisms (must be paired).
    diploid_offspring: list of DiploidOrganism
        List of diploid offspring.
    fitness_model: str
        The inheritance model used for diploid fitness calculation (displayed in the title).
    """
    parent_fitness = []
    diploid_fitness = []

    for i in range(0, len(parents) - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        avg_parent_fitness = (parent1.fitness + parent2.fitness) / 2
        parent_fitness.append(avg_parent_fitness)
        diploid_fitness.append(diploid_offspring[i // 2].fitness)

    plt.figure(figsize=(10, 6))
    plt.scatter(parent_fitness, diploid_fitness, alpha=0.7, edgecolors='k')
    plt.plot([min(parent_fitness), max(parent_fitness)], [min(parent_fitness), max(parent_fitness)], 'r--', label='y = x')
    plt.xlabel("Average Fitness of Haploid Parents")
    plt.ylabel("Fitness of Diploid Offspring")
    plt.title(f"Fitness Comparison (Model: {fitness_model.capitalize()})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fitness_comparison_{fitness_model}.png", dpi=300)
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
        
        print(f"Generation {gen+1}: Pop size = {stats['population_size']}, "
              f"Avg fitness = {stats['avg_fitness']:.4f}, "
              f"Max fitness = {stats['max_fitness']:.4f}, "
              f"Min fitness = {stats['min_fitness']:.4f}")

    
    return all_organisms, individual_fitness, generation_stats

def mate_last_generation(last_generation, fitness_model="dominant"):
    """Mate haploid organisms to create diploid organisms."""
    diploid_offspring = []
    for i in range(0, len(last_generation) - 1, 2):
        parent1 = last_generation[i]
        parent2 = last_generation[i + 1]
        diploid_offspring.append(DiploidOrganism(parent1, parent2, fitness_model))
    
    print(f"Created {len(diploid_offspring)} diploid organisms using {fitness_model} model.")
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
    aparser.add_argument("--model", type=str, default="dominant", help="Fitness model for diploid organisms") # dominant, recessive, codominant


    args = aparser.parse_args()

    # init logging
    log = init_log()
    
    # Create environment
    environment = Environment(genome_size=args.genome_size, beta=args.beta, rho=args.rho)
    log.info(f"Environment created with genome size = {args.genome_size}, beta = {args.beta}, rho = {args.rho}")

    
    # Run simulation
    log.info(f"Running simulation for {args.generations} generations")
    all_organisms, individual_fitness, generation_stats = run_simulation(args.generations, environment)
    log.info(f"all_organisms: {all_organisms}")
    log.info(f"individual_fitness: {individual_fitness}")
    log.info(f"generation_stats: {generation_stats}")

    # Create visualizations
    print("Creating visualizations...")
    plot_detailed_fitness(individual_fitness, generation_stats)
    plot_relationship_tree(all_organisms)  # Original relationship tree visualization

    # Mate last generation to create diploid organisms
    last_generation = [org for org in all_organisms if org.generation == args.generations]
    diploid_offspring = mate_last_generation(last_generation, fitness_model=args.model)
    # Create fitness comparison plot
    plot_fitness_comparison(last_generation, diploid_offspring, fitness_model=args.model)

    
    print("Simulation complete! Check detailed_fitness.png and relationship_tree.png")

if __name__ == "__main__":
    main()
    

"""
example usage:
/home/labs/pilpel/barc/sexy_yeast/main_simulation_BC.py --generations 10 --genome_size 100 --beta 0.5 --rho 0.25
"""

