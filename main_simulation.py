#!/home/labs/pilpel/barc/.conda/envs/sexy_yeast_env/bin/python
import numpy as np
import uuid  # For unique IDs for each organism
import logging as log
import cmn  # Assume all the previously defined functions are in the cmn module
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
import argparse as ap

# init logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
# add file handler
fh = log.FileHandler("sexy_yeast.log", mode='w')
fh.setLevel(log.INFO)
log.getLogger().addHandler(fh)


class Organism:
    def __init__(self, sigma, generation=0, parent_id=None):
        self.id = str(uuid.uuid4())  # Unique ID for each organism
        self.sigma = sigma.copy()  # Current sequence
        self.generation = generation
        self.parent_id = parent_id  # ID of the parent organism
        self.h = cmn.init_h(len(sigma), beta=0.5)
        self.J = cmn.init_J(len(sigma), beta=0.5, rho=0.25)
        self.F_off = cmn.calc_F_off(self.sigma, self.h, self.J)
        self.fitness = cmn.compute_fit_slow(self.sigma, self.h, self.J, self.F_off)
        self.relative_fitness = self.fitness  # Relative to initial ancestor
        
    def mutate(self):
        """Mutate the sequence by flipping a random spin."""
        idx = cmn.sswm_flip(self.sigma, self.h, self.J)
        self.sigma[idx] *= -1
    
    def reproduce(self):
        """Reproduce and generate two offspring."""
        child1 = Organism(self.sigma, generation=self.generation + 1, parent_id=self.id)
        child2 = Organism(self.sigma, generation=self.generation + 1, parent_id=self.id)
        return child1, child2

def run_simulation(num_generations):
    """Run the evolution simulation for num_generations."""
    initial_sigma = cmn.init_sigma(1024)  # Starting sequence
    ancestor = Organism(initial_sigma)  # Create the initial organism
    
    population = [ancestor]
    all_organisms = [ancestor]
    
    for gen in range(1, num_generations + 1):
        next_generation = []
        for organism in population:
            organism.mutate()  # Mutate before reproducing
            child1, child2 = organism.reproduce()
            next_generation.extend([child1, child2])
            all_organisms.extend([child1, child2])
        
        log.info(f"Generation {gen}: Population size = {len(next_generation)}")
        population = next_generation  # Move to the next generation
    
    return all_organisms

def plot_fitness_trajectory(organisms):
    """Plot the fitness trajectory over generations."""
    generations = [org.generation for org in organisms]
    fitness_values = [org.fitness for org in organisms]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(generations, fitness_values, alpha=0.7)
    plt.plot(generations, fitness_values, alpha=0.3, linestyle='dashed', color='gray')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Trajectory Over Generations")
    plt.grid(True)
    plt.savefig("fitness_trajectory.png")

def plot_lineage_tree(organisms):
    """Plot the evolutionary tree using NetworkX."""
    G = nx.DiGraph()
    
    # Add nodes and edges for each organism and its parent
    for org in organisms:
        G.add_node(org.id, generation=org.generation, fitness=org.fitness)
        if org.parent_id:
            G.add_edge(org.parent_id, org.id)
    
    pos = nx.spring_layout(G)  # Position nodes with spring layout
    plt.figure(figsize=(16, 10))
    
    # Draw nodes and edges with labels for generations
    nx.draw(G, pos, with_labels=False, node_size=50, alpha=0.8)
    node_labels = {org.id: f"G{org.generation}" for org in organisms}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title("Lineage Tree")
    plt.savefig("lineage_tree.png")
        
def main():
    aparser = ap.ArgumentParser(description="Run the sexy yeast simulation")
    aparser.add_argument("--num_generations", type=int, help="Number of generations to simulate" , default=5)
    args = aparser.parse_args()

    all_organisms = run_simulation(args.num_generations)
    log.info(f"Total number of organisms: {len(all_organisms)}")
    log.info(f"Fitness of the last organism: {all_organisms[-1].fitness}")
    log.info(f"Fitness of the ancestor: {all_organisms[0].fitness}")

    plot_fitness_trajectory(all_organisms)
    plot_lineage_tree(all_organisms)

if __name__ == "__main__":
    main()
