#!/usr/bin/env python3
"""
Simulation engine for evolutionary dynamics.

This module handles the main simulation logic, including running generations,
managing populations, and coordinating mating strategies.
"""

import numpy as np
import random
from collections import defaultdict
from itertools import combinations, product
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import logging
import gc
import psutil

from core_models import (
    Environment, Organism, DiploidOrganism, OrganismWithMatingType,
    MatingStrategy, MatingType, FitnessMethod
)


class EvolutionarySimulation:
    """
    Main simulation engine for evolutionary dynamics.
    
    This class orchestrates the simulation, managing populations through
    generations and handling reproduction, mutation, and selection.
    """
    
    def __init__(self, environment: Environment, logger: Optional[logging.Logger] = None):
        """
        Initialize the simulation.
        
        Args:
            environment: The environment in which evolution occurs
            logger: Logger for tracking simulation progress
        """
        self.environment = environment
        self.logger = logger or logging.getLogger(__name__)
        
        # Simulation state
        self.population: List[Organism] = []
        self.all_organisms: List[Organism] = []
        self.individual_fitness: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.generation_stats: List[Dict[str, Any]] = []
        self.current_generation = 0
    
    def create_initial_organism(self, target_fitness: Optional[float] = None,
                              initial_genome_seed: Optional[int] = None,
                              mutation_seed: Optional[int] = None,
                              max_attempts: int = 10000) -> Organism:
        """
        Create the initial organism for the simulation.
        
        Args:
            target_fitness: If specified, search for genome with this fitness
            initial_genome_seed: Seed for genome generation
            mutation_seed: Seed for mutations
            max_attempts: Maximum attempts to find target fitness
            
        Returns:
            The initial organism
        """
        if target_fitness is not None:
            return self._find_organism_with_target_fitness(
                target_fitness, initial_genome_seed, mutation_seed, max_attempts
            )
        else:
            organism = Organism(
                environment=self.environment,
                genome_seed=initial_genome_seed,
                mutation_seed=mutation_seed
            )
            self.logger.info(f"Created random initial organism with fitness {organism.fitness:.4f}")
            return organism
    
    def _find_organism_with_target_fitness(self, target_fitness: float,
                                         initial_genome_seed: Optional[int],
                                         mutation_seed: Optional[int],
                                         max_attempts: int) -> Organism:
        """Search for an organism with fitness close to the target."""
        self.logger.info(f"Searching for organism with fitness close to {target_fitness}")
        
        best_organism = None
        best_diff = float('inf')
        
        for attempt in tqdm(range(max_attempts), desc="Searching for target fitness"):
            test_seed = None if initial_genome_seed is None else initial_genome_seed + attempt
            test_organism = Organism(
                environment=self.environment,
                genome_seed=test_seed,
                mutation_seed=mutation_seed
            )
            
            diff = abs(test_organism.fitness - target_fitness)
            if diff < best_diff:
                best_diff = diff
                best_organism = test_organism
                
                if diff < 0.01:  # Close enough
                    self.logger.info(f"Found suitable organism after {attempt + 1} attempts")
                    break
        
        self.logger.info(f"Best organism: fitness {best_organism.fitness:.4f}, "
                        f"target {target_fitness:.4f}, diff {best_diff:.4f}")
        return best_organism
    
    def initialize_population(self, initial_organism: Organism) -> None:
        """Initialize the population with the given organism."""
        self.population = [initial_organism]
        self.all_organisms = [initial_organism]
        self.individual_fitness[initial_organism.id].append((0, initial_organism.fitness))
        self.current_generation = 0
        
        self.logger.info(f"Population initialized with organism {initial_organism.id[:8]}")
    
    def run_generations(self, num_generations: int, 
                       max_population_size: int = 100000,
                       mutation_seed: Optional[int] = None,
                       log_genomes: bool = False) -> None:
        """
        Run the simulation for the specified number of generations.
        
        Args:
            num_generations: Number of generations to simulate
            max_population_size: Maximum population size per generation
            mutation_seed: Seed for mutations
            log_genomes: Whether to log individual genomes
        """
        for gen in tqdm(range(num_generations), desc="Running generations"):
            self._run_single_generation(gen + 1, max_population_size, 
                                      mutation_seed, log_genomes)
            self._monitor_resources()
            gc.collect()
    
    def _run_single_generation(self, generation: int, max_population_size: int,
                             mutation_seed: Optional[int], log_genomes: bool) -> None:
        """Run a single generation of evolution."""
        next_generation = []
        generation_fitness = []
        
        # Reproduction and mutation
        for organism in self.population:
            child1, child2 = organism.reproduce(mutation_seed=mutation_seed)
            child1.mutate()
            child2.mutate()
            
            # Track fitness
            self.individual_fitness[child1.id].append((generation, child1.fitness))
            self.individual_fitness[child2.id].append((generation, child2.fitness))
            generation_fitness.extend([child1.fitness, child2.fitness])
            
            next_generation.extend([child1, child2])
            self.all_organisms.extend([child1, child2])
        
        # Selection (keep best if population too large)
        if len(next_generation) > max_population_size:
            next_generation.sort(key=lambda x: (x.fitness, x.id), reverse=True)
            next_generation = next_generation[:max_population_size]
        
        self.population = next_generation
        self.current_generation = generation
        
        # Record statistics
        self._record_generation_stats(generation, generation_fitness)
        
        # Optional detailed logging
        if log_genomes and len(self.population) < 50:
            self._log_population_details(generation)
    
    def _record_generation_stats(self, generation: int, fitness_values: List[float]) -> None:
        """Record statistics for this generation."""
        stats = {
            'generation': generation,
            'population_size': len(self.population),
            'avg_fitness': np.mean(fitness_values),
            'max_fitness': np.max(fitness_values),
            'min_fitness': np.min(fitness_values),
            'std_fitness': np.std(fitness_values)
        }
        self.generation_stats.append(stats)
        
        self.logger.info(f"Generation {generation}: "
                        f"Pop={stats['population_size']}, "
                        f"Avg fit={stats['avg_fitness']:.4f}")
    
    def _log_population_details(self, generation: int) -> None:
        """Log detailed information about each organism in the population."""
        for org in self.population:
            self.logger.info(f"[GEN {generation}] {org}, Genome={org.genome}")
    
    def _monitor_resources(self) -> None:
        """Monitor memory usage."""
        memory_gb = psutil.virtual_memory().used / (1024 ** 3)
        self.logger.debug(f"Memory usage: {memory_gb:.2f} GB")
    
    def get_last_generation(self) -> List[Organism]:
        """Get organisms from the last generation."""
        if not self.all_organisms:
            return []
        
        max_gen = max(org.generation for org in self.all_organisms)
        return [org for org in self.all_organisms if org.generation == max_gen]


class MatingEngine:
    """
    Handles mating strategies and diploid organism creation.
    
    This class manages different mating strategies and creates diploid
    offspring from haploid parents.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the mating engine."""
        self.logger = logger or logging.getLogger(__name__)
    
    def mate_organisms(self, organisms: List[Organism], 
                      strategy: MatingStrategy,
                      fitness_models: List[str] = None,
                      log_crosses: bool = True) -> Dict[str, List[DiploidOrganism]]:
        """
        Create diploid offspring using the specified mating strategy.
        
        Args:
            organisms: List of organisms to mate
            strategy: Mating strategy to use
            fitness_models: List of fitness models to test
            log_crosses: Whether to log individual crosses
            
        Returns:
            Dictionary mapping fitness models to lists of diploid offspring
        """
        if fitness_models is None:
            fitness_models = ["dominant", "recessive", "codominant"]
        
        # Sort organisms for consistent results
        sorted_organisms = sorted(organisms, key=lambda x: x.id)
        
        if strategy == MatingStrategy.ONE_TO_ONE:
            return self._mate_one_to_one(sorted_organisms, fitness_models, log_crosses)
        elif strategy == MatingStrategy.ALL_VS_ALL:
            return self._mate_all_vs_all(sorted_organisms, fitness_models, log_crosses)
        elif strategy == MatingStrategy.MATING_TYPES:
            return self._mate_by_types(sorted_organisms, fitness_models, log_crosses)
        else:
            raise ValueError(f"Unknown mating strategy: {strategy}")
    
    def _mate_one_to_one(self, organisms: List[Organism], 
                        fitness_models: List[str],
                        log_crosses: bool) -> Dict[str, List[DiploidOrganism]]:
        """Mate organisms in adjacent pairs."""
        diploid_offspring = defaultdict(list)
        
        # Ensure even number of organisms
        if len(organisms) % 2 != 0:
            organisms = organisms[:-1]
            self.logger.info(f"Removed one organism to get even number: {len(organisms)}")
        
        for model in fitness_models:
            for i in range(0, len(organisms), 2):
                parent1, parent2 = organisms[i], organisms[i + 1]
                offspring = DiploidOrganism(parent1, parent2, fitness_model=model)
                diploid_offspring[model].append(offspring)
                
                if log_crosses:
                    self._log_cross(parent1, parent2, offspring, model)
        
        return diploid_offspring
    
    def _mate_all_vs_all(self, organisms: List[Organism],
                        fitness_models: List[str],
                        log_crosses: bool) -> Dict[str, List[DiploidOrganism]]:
        """Mate every organism with every other organism."""
        diploid_offspring = defaultdict(list)
        
        for model in fitness_models:
            for parent1, parent2 in combinations(organisms, 2):
                offspring = DiploidOrganism(parent1, parent2, fitness_model=model)
                diploid_offspring[model].append(offspring)
                
                if log_crosses and len(organisms) <= 5:  # Only log for small populations
                    self._log_cross(parent1, parent2, offspring, model)
        
        self.logger.info(f"All-vs-all mating: {len(organisms)} organisms, "
                        f"{len(list(combinations(organisms, 2)))} crosses per model")
        return diploid_offspring
    
    def _mate_by_types(self, organisms: List[Organism],
                      fitness_models: List[str],
                      log_crosses: bool) -> Dict[str, List[DiploidOrganism]]:
        """Mate organisms based on assigned mating types."""
        # Assign mating types
        typed_organisms = self._assign_mating_types(organisms)
        
        # Separate by type
        type_a = [org for org in typed_organisms if org.mating_type == MatingType.A]
        type_alpha = [org for org in typed_organisms if org.mating_type == MatingType.ALPHA]
        
        self.logger.info(f"Mating types: {len(type_a)} type A, {len(type_alpha)} type alpha")
        
        diploid_offspring = defaultdict(list)
        
        for model in fitness_models:
            for a_org, alpha_org in product(type_a, type_alpha):
                offspring = DiploidOrganism(
                    a_org.organism, alpha_org.organism, 
                    fitness_model=model, mating_type=None
                )
                diploid_offspring[model].append(offspring)
                
                if log_crosses and len(type_a) <= 5 and len(type_alpha) <= 5:
                    self._log_cross(a_org.organism, alpha_org.organism, offspring, model)
        
        return diploid_offspring
    
    def _assign_mating_types(self, organisms: List[Organism]) -> List[OrganismWithMatingType]:
        """Randomly assign mating types to organisms."""
        typed_organisms = []
        for org in organisms:
            mating_type = random.choice([MatingType.A, MatingType.ALPHA])
            typed_organisms.append(OrganismWithMatingType(org, mating_type))
        return typed_organisms
    
    def _log_cross(self, parent1: Organism, parent2: Organism, 
                   offspring: DiploidOrganism, model: str) -> None:
        """Log details of a mating cross."""
        from analysis_tools import calculate_genomic_distance  # Import here to avoid circular import
        
        distance = calculate_genomic_distance(parent1.genome, parent2.genome)
        self.logger.info(f"[CROSS {model}] P1({parent1.id[:8]}) fit={parent1.fitness:.4f}, "
                        f"P2({parent2.id[:8]}) fit={parent2.fitness:.4f}, "
                        f"dist={distance}, offspring fit={offspring.fitness:.4f}")


class SimulationRunner:
    """
    High-level interface for running complete evolutionary simulations.
    
    This class combines the EvolutionarySimulation and MatingEngine to
    provide a simple interface for running full simulations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the simulation runner."""
        self.logger = logger or logging.getLogger(__name__)
        self.simulation = None
        self.mating_engine = MatingEngine(logger)
    
    def run_complete_simulation(self, environment: Environment,
                              num_generations: int,
                              mating_strategy: MatingStrategy,
                              initial_fitness: Optional[float] = None,
                              initial_genome_seed: Optional[int] = None,
                              mutation_seed: Optional[int] = None,
                              max_population_size: int = 100000,
                              log_genomes: bool = False,
                              fitness_models: List[str] = None) -> Dict[str, Any]:
        """
        Run a complete simulation from start to finish.
        
        Args:
            environment: Environment for the simulation
            num_generations: Number of generations to run
            mating_strategy: Strategy for mating organisms
            initial_fitness: Target fitness for initial organism
            initial_genome_seed: Seed for initial genome
            mutation_seed: Seed for mutations
            max_population_size: Maximum population size
            log_genomes: Whether to log individual genomes
            fitness_models: Fitness models to test in diploid phase
            
        Returns:
            Dictionary containing all simulation results
        """
        if fitness_models is None:
            fitness_models = ["dominant", "recessive", "codominant"]
        
        self.logger.info("Starting complete simulation")
        self.logger.info(f"Environment: {environment.get_description()}")
        self.logger.info(f"Generations: {num_generations}")
        self.logger.info(f"Mating strategy: {mating_strategy.value}")
        
        # Initialize simulation
        self.simulation = EvolutionarySimulation(environment, self.logger)
        
        # Create initial organism
        initial_organism = self.simulation.create_initial_organism(
            target_fitness=initial_fitness,
            initial_genome_seed=initial_genome_seed,
            mutation_seed=mutation_seed
        )
        
        # Initialize population
        self.simulation.initialize_population(initial_organism)
        
        # Run evolution
        self.simulation.run_generations(
            num_generations=num_generations,
            max_population_size=max_population_size,
            mutation_seed=mutation_seed,
            log_genomes=log_genomes
        )
        
        # Get final generation for mating
        last_generation = self.simulation.get_last_generation()
        self.logger.info(f"Final generation has {len(last_generation)} organisms")
        
        # Perform mating
        diploid_offspring = self.mating_engine.mate_organisms(
            organisms=last_generation,
            strategy=mating_strategy,
            fitness_models=fitness_models,
            log_crosses=len(last_generation) <= 20  # Only log for small populations
        )
        
        # Log mating results
        for model, offspring_list in diploid_offspring.items():
            self.logger.info(f"{model} model: {len(offspring_list)} diploid offspring")
        
        return {
            "simulation": self.simulation,
            "diploid_offspring": diploid_offspring,
            "environment": environment,
            "parameters": {
                "num_generations": num_generations,
                "mating_strategy": mating_strategy.value,
                "initial_fitness": initial_fitness,
                "genome_size": environment.genome_size,
                "fitness_method": environment.fitness_method.value,
                "mutation_seed": mutation_seed,
                "initial_genome_seed": initial_genome_seed
            }
        }