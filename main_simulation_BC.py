#!/home/labs/pilpel/barc/.conda/envs/sexy_yeast_env/bin/python
import numpy as np
import uuid
import matplotlib.pyplot as plt
import networkx as nx
import argparse as ap
from collections import defaultdict
import logging as log
import cmn

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
        return cmn.init_h(self.genome_size, self.beta)
    
    def _init_J(self):
        """Initialize coupling matrix with sparsity"""
        return cmn.init_J(self.genome_size, self.beta, self.rho)
            
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
        energy = cmn.compute_fit_slow(genome, self.h, self.J)
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

def init_log():
    """Initialize logging"""
    # init logging
    log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    # add file handler
    fh = log.FileHandler("sexy_yeast.log", mode='w')
    fh.setLevel(log.INFO)
    log.getLogger().addHandler(fh)

    return log

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

def main():
    aparser = ap.ArgumentParser(description="Run the sexy yeast simulation with detailed fitness tracking")
    aparser.add_argument("--generations", type=int, default=5, help="Number of generations to simulate")
    aparser.add_argument("--genome_size", type=int, default=100, help="Size of the genome")
    aparser.add_argument("--beta", type=float, default=0.5, help="Beta parameter")
    aparser.add_argument("--rho", type=float, default=0.25, help="Rho parameter")

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
    
    print("Simulation complete! Check detailed_fitness.png and relationship_tree.png")

if __name__ == "__main__":
    main()
    

"""
example usage:
/home/labs/pilpel/barc/sexy_yeast/main_simulation_BC.py --generations 10 --genome_size 100 --beta 0.5 --rho 0.25
"""

