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
    MatingStrategy, MatingType, FitnessMethod, calculate_genomic_distance , CrossDataCollector
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
        
        # Record initial statistics
        self._record_generation_stats(0, [initial_organism.fitness])
        
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
        if not fitness_values:
            return
            
        stats = {
            'generation': generation,
            'population_size': len(self.population),
            'avg_fitness': float(np.mean(fitness_values)),
            'max_fitness': float(np.max(fitness_values)),
            'min_fitness': float(np.min(fitness_values)),
            'std_fitness': float(np.std(fitness_values))
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
    Handles mating between organisms according to different strategies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the mating engine."""
        self.logger = logger or logging.getLogger(__name__)
        self.data_collector = CrossDataCollector()
    
    def start_new_run(self) -> None:
        """Start a new simulation run."""
        self.data_collector.start_new_run()
    
    def mate_organisms(self, organisms: List[Organism], 
                      strategy: MatingStrategy,
                      fitness_models: List[str] = None,
                      log_crosses: bool = True) -> Dict[str, List[DiploidOrganism]]:
        """
        Mate organisms according to the specified strategy.
        """
        if not fitness_models:
            fitness_models = ["dominant", "recessive", "codominant"]
            
        offspring_by_model = {model: [] for model in fitness_models}
        
        # Log mating setup
        self.logger.info(f"Mating {len(organisms)} organisms using {strategy.value} strategy")
        self.logger.info(f"Testing inheritance modes: {fitness_models}")
        
        if strategy == MatingStrategy.ONE_TO_ONE:
            self._mate_one_to_one(organisms, fitness_models, log_crosses, offspring_by_model)
        elif strategy == MatingStrategy.ALL_VS_ALL:
            self._mate_all_vs_all(organisms, fitness_models, log_crosses, offspring_by_model)
        elif strategy == MatingStrategy.MATING_TYPES:
            self._mate_by_types(organisms, fitness_models, log_crosses, offspring_by_model)
        else:
            raise ValueError(f"Unknown mating strategy: {strategy}")
        
        # Validate offspring creation
        for model, offspring_list in offspring_by_model.items():
            self.logger.info(f"Created {len(offspring_list)} offspring for model {model}")
            if not offspring_list:
                self.logger.warning(f"No offspring created for model {model}")
        
        # Validate collected cross data
        if not self.data_collector.validate_data():
            self.logger.error("Cross data validation failed")
            raise ValueError("Invalid cross data collected")
            
        return offspring_by_model
    
    def _mate_one_to_one(self, organisms: List[Organism], 
                        fitness_models: List[str],
                        log_crosses: bool,
                        offspring_by_model: Dict[str, List[DiploidOrganism]]) -> None:
        """Mate organisms one-to-one."""
        for i in range(0, len(organisms) - 1, 2):
            parent1, parent2 = organisms[i], organisms[i + 1]
            self._create_offspring_for_models(parent1, parent2, fitness_models, 
                                           log_crosses, offspring_by_model)
    
    def _mate_all_vs_all(self, organisms: List[Organism],
                        fitness_models: List[str],
                        log_crosses: bool,
                        offspring_by_model: Dict[str, List[DiploidOrganism]]) -> None:
        """Mate all organisms with all other organisms."""
        for parent1, parent2 in combinations(organisms, 2):
            self._create_offspring_for_models(parent1, parent2, fitness_models,
                                           log_crosses, offspring_by_model)
    
    def _mate_by_types(self, organisms: List[Organism],
                      fitness_models: List[str],
                      log_crosses: bool,
                      offspring_by_model: Dict[str, List[DiploidOrganism]]) -> None:
        """Mate organisms based on mating types."""
        typed_organisms = self._assign_mating_types(organisms)
        type_a = [org for org in typed_organisms if org.mating_type == MatingType.A]
        type_alpha = [org for org in typed_organisms if org.mating_type == MatingType.ALPHA]
        
        for parent1, parent2 in product(type_a, type_alpha):
            self._create_offspring_for_models(parent1.organism, parent2.organism,
                                           fitness_models, log_crosses, offspring_by_model)
    
    def _create_offspring_for_models(self, parent1: Organism, parent2: Organism,
                                   fitness_models: List[str],
                                   log_crosses: bool,
                                   offspring_by_model: Dict[str, List[DiploidOrganism]]) -> None:
        """Create offspring for each inheritance mode."""
        for model in fitness_models:
            try:
                offspring = DiploidOrganism(parent1, parent2, fitness_model=model)
                offspring_by_model[model].append(offspring)
                
                # Record cross data
                self.data_collector.record_cross(model, parent1, parent2, offspring)
                
                if log_crosses:
                    self._log_cross(parent1, parent2, offspring, model)
                
                self.logger.debug(
                    f"Created offspring: mode={model}, "
                    f"parent1_fit={parent1.fitness:.4f}, "
                    f"parent2_fit={parent2.fitness:.4f}, "
                    f"offspring_fit={offspring.fitness:.4f}"
                )
            except Exception as e:
                self.logger.error(f"Failed to create offspring for model {model}: {e}")
                self.logger.error("Error details:", exc_info=True)
    
    def get_cross_data(self) -> List[Dict[str, Any]]:
        """Get all collected cross data."""
        data = self.data_collector.get_data_as_list()
        self.logger.info(f"Retrieved {len(data)} crosses from data collector")
        
        # Validate data before returning
        if not data:
            self.logger.warning("No cross data collected")
        else:
            self.logger.info(f"Sample cross data: {data[0]}")
        
        return data
    
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
        distance = calculate_genomic_distance(parent1.genome, parent2.genome)
        self.logger.info(f"[CROSS {model}] P1({parent1.id[:8]}) fit={parent1.fitness:.4f}, "
                        f"P2({parent2.id[:8]}) fit={parent2.fitness:.4f}, "
                        f"dist={distance}, offspring fit={offspring.fitness:.4f}")


class SimulationRunner:
    """
    Coordinates running complete simulations with different parameters.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the simulation runner."""
        self.logger = logger or logging.getLogger(__name__)
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
        Run a complete simulation with the given parameters.
        
        Args:
            environment: The environment for evolution
            num_generations: Number of generations to simulate
            mating_strategy: Strategy for organism mating
            initial_fitness: Target fitness for initial organism
            initial_genome_seed: Seed for initial genome
            mutation_seed: Seed for mutations
            max_population_size: Maximum population size
            log_genomes: Whether to log individual genomes
            fitness_models: List of inheritance modes to test
            
        Returns:
            Dictionary containing simulation results and statistics
        """
        # Start new run in data collector
        self.mating_engine.start_new_run()
        self.logger.info("Started new simulation run")
        
        # Initialize simulation
        sim = EvolutionarySimulation(environment, self.logger)
        initial_org = sim.create_initial_organism(
            target_fitness=initial_fitness,
            initial_genome_seed=initial_genome_seed,
            mutation_seed=mutation_seed
        )
        sim.initialize_population(initial_org)
        
        # Run generations
        sim.run_generations(
            num_generations=num_generations,
            max_population_size=max_population_size,
            mutation_seed=mutation_seed,
            log_genomes=log_genomes
        )
        
        # Get final population and mate organisms
        final_population = sim.get_last_generation()
        self.logger.info(f"Final population size: {len(final_population)}")
        
        # Perform mating and collect offspring
        offspring_by_model = self.mating_engine.mate_organisms(
            organisms=final_population,
            strategy=mating_strategy,
            fitness_models=fitness_models,
            log_crosses=True
        )
        
        # Get cross data
        cross_data = self.mating_engine.get_cross_data()
        self.logger.info(f"Collected {len(cross_data)} crosses")
        
        # Validate data before returning
        if not cross_data:
            self.logger.error("No cross data collected")
            raise ValueError("No cross data collected during simulation")
        
        # Log sample data
        self.logger.info(f"Sample cross data: {cross_data[0]}")
        
        # Collect results
        results = {
            "generation_stats": sim.generation_stats,
            "individual_fitness": dict(sim.individual_fitness),
            "cross_data": cross_data,
            "diploid_offspring": {
                model: [offspring.to_dict() for offspring in offspring_list]
                for model, offspring_list in offspring_by_model.items()
            },
            "parameters": {
                "num_generations": num_generations,
                "mating_strategy": mating_strategy.value,
                "initial_fitness": initial_fitness,
                "genome_size": environment.genome_size,
                "fitness_method": environment.fitness_method.value,
                "mutation_seed": mutation_seed,
                "initial_genome_seed": initial_genome_seed,
                "max_population_size": max_population_size
            },
            "simulation": sim  # Add the simulation object back
        }
        
        # Validate results structure
        required_keys = ["generation_stats", "individual_fitness", "cross_data", 
                        "diploid_offspring", "parameters", "simulation"]
        for key in required_keys:
            if key not in results:
                self.logger.error(f"Missing required key in results: {key}")
                raise ValueError(f"Invalid results structure: missing {key}")
        
        self.logger.info("Simulation completed successfully")
        return results