#!/usr/bin/env python3
"""
Core models and enums for the evolutionary simulation.

This module contains the fundamental data structures and enumerations
used throughout the simulation.
"""

import numpy as np
import uuid
from enum import Enum
from typing import Optional, Union, Dict, Any
import logging


class MatingStrategy(Enum):
    """Strategies for how organisms mate with each other."""
    ONE_TO_ONE = "one_to_one"
    ALL_VS_ALL = "all_vs_all"
    MATING_TYPES = "mating_types"


class MatingType(Enum):
    """Mating types for organisms (used in MATING_TYPES strategy)."""
    A = "A"
    ALPHA = "alpha"


class FitnessMethod(Enum):
    """Available fitness calculation methods."""
    SHERRINGTON_KIRKPATRICK = "sherrington_kirkpatrick"  # Complex interaction model
    SINGLE_POSITION = "single_position"                  # Simple single-locus model
    ADDITIVE = "additive"                               # Independent additive effects


class Environment:
    """
    Represents the fitness landscape for evolutionary simulation.
    
    The environment defines how genomes map to fitness values. It supports
    multiple fitness calculation methods, from complex interaction models
    to simple additive effects.
    """
    
    def __init__(self, genome_size: int, beta: float = 0.5, rho: float = 0.25, 
                 seed: Optional[int] = None, 
                 fitness_method: FitnessMethod = FitnessMethod.SHERRINGTON_KIRKPATRICK):
        """
        Initialize the environment.
        
        Args:
            genome_size: Number of loci in the genome
            beta: Controls fitness landscape ruggedness (for Sherrington-Kirkpatrick)
            rho: Controls correlation between sites (for Sherrington-Kirkpatrick)
            seed: Random seed for reproducible environments
            fitness_method: Which fitness calculation method to use
        """
        self.genome_size = genome_size
        self.beta = beta
        self.rho = rho
        self.seed = seed
        self.fitness_method = fitness_method
        
        # Initialize fitness landscape based on method
        self._rng = np.random.default_rng(seed)
        self._initialize_fitness_landscape()
    
    def _initialize_fitness_landscape(self) -> None:
        """Initialize the fitness landscape based on the chosen method."""
        if self.fitness_method == FitnessMethod.SHERRINGTON_KIRKPATRICK:
            self.h = self._init_h()
            self.J = self._init_J()
            self.alternative_params = None
        else:
            self.h = None
            self.J = None
            self.alternative_params = self._init_alternative_fitness()
    
    def _init_h(self) -> np.ndarray:
        """Initialize external fields for Sherrington-Kirkpatrick model."""
        sig_h = np.sqrt(1 - self.beta)
        return self._rng.normal(0.0, sig_h, self.genome_size)
    
    def _init_J(self) -> np.ndarray:
        """Initialize coupling matrix for Sherrington-Kirkpatrick model."""
        if self.genome_size == 1:
            return np.zeros((1, 1))
        
        if not (0 < self.rho <= 1):
            raise ValueError("rho must be between 0 (exclusive) and 1 (inclusive)")
        
        sig_J = np.sqrt(self.beta / (self.genome_size * self.rho))
        J_upper = np.zeros((self.genome_size, self.genome_size))
        
        # Calculate sparsity
        total_elements = self.genome_size * (self.genome_size - 1) // 2
        num_nonzero = max(1, int(np.floor(self.rho * total_elements))) if self.rho > 0 else 0
        
        if total_elements > 0 and num_nonzero > 0:
            triu_indices = np.triu_indices(self.genome_size, k=1)
            selected_indices = self._rng.choice(total_elements, size=num_nonzero, replace=False)
            rows = triu_indices[0][selected_indices]
            cols = triu_indices[1][selected_indices]
            J_upper[rows, cols] = self._rng.normal(loc=0.0, scale=sig_J, size=num_nonzero)
        
        return J_upper + J_upper.T  # Symmetrize
    
    def _init_alternative_fitness(self) -> Dict[str, Any]:
        """Initialize parameters for alternative fitness methods."""
        if self.fitness_method == FitnessMethod.SINGLE_POSITION:
            return {
                "method": self.fitness_method,
                "position": self._rng.integers(0, self.genome_size),
                "favorable_value": 1
            }
        elif self.fitness_method == FitnessMethod.ADDITIVE:
            return {
                "method": self.fitness_method,
                "weights": self._rng.normal(0, 1, self.genome_size)
            }
        else:
            raise ValueError(f"Unknown fitness method: {self.fitness_method}")
    
    def calculate_fitness(self, genome: np.ndarray) -> float:
        """
        Calculate fitness for a given genome.
        
        Args:
            genome: Array representing the genome (values of -1 or 1)
            
        Returns:
            Fitness value for this genome in this environment
        """
        if self.fitness_method == FitnessMethod.SHERRINGTON_KIRKPATRICK:
            return self._calculate_sk_fitness(genome)
        else:
            return self._calculate_alternative_fitness(genome)
    
    def _calculate_sk_fitness(self, genome: np.ndarray) -> float:
        """Calculate fitness using Sherrington-Kirkpatrick model."""
        return genome @ (self.h + 0.5 * self.J @ genome)
    
    def _calculate_alternative_fitness(self, genome: np.ndarray) -> float:
        """Calculate fitness using alternative methods."""
        method = self.alternative_params["method"]
        
        if method == FitnessMethod.SINGLE_POSITION:
            position = self.alternative_params["position"]
            favorable_value = self.alternative_params["favorable_value"]

            if position >= len(genome):
                return 0.0
            
            # Normalize values between 0 and 1
            return float(genome[position])  # This would use 0, 0.5, or 1.0

            
        elif method == FitnessMethod.ADDITIVE:
            weights = self.alternative_params["weights"]
            genome_01 = (genome + 1) / 2  # Convert -1/1 to 0/1
            fitness = np.dot(genome_01, weights)
            return 0.1 + 0.9 / (1 + np.exp(-fitness))  # Sigmoid normalization
        
        return 0.5  # Fallback
    
    def get_description(self) -> str:
        """Get a human-readable description of this environment."""
        if self.fitness_method == FitnessMethod.SHERRINGTON_KIRKPATRICK:
            return f"Sherrington-Kirkpatrick (β={self.beta}, ρ={self.rho})"
        elif self.fitness_method == FitnessMethod.SINGLE_POSITION:
            pos = self.alternative_params["position"]
            return f"Single position model (position {pos})"
        elif self.fitness_method == FitnessMethod.ADDITIVE:
            return "Additive model (independent effects)"
        return "Unknown fitness method"


class Organism:
    """
    Represents a haploid organism in the simulation.
    
    Each organism has a genome, can calculate its fitness in an environment,
    can mutate, and can reproduce to create offspring.
    """
    
    def __init__(self, environment: Environment, genome: Optional[np.ndarray] = None,
                 generation: int = 0, parent_id: Optional[str] = None,
                 mutation_rate: Optional[float] = None,
                 genome_seed: Optional[int] = None,
                 mutation_seed: Optional[int] = None):
        """
        Create a new organism.
        
        Args:
            environment: The environment this organism lives in
            genome: Specific genome to use (if None, generates random)
            generation: Which generation this organism belongs to
            parent_id: ID of parent organism (if any)
            mutation_rate: Probability of mutation per locus
            genome_seed: Seed for genome generation
            mutation_seed: Seed for mutations
        """
        self.id = str(uuid.uuid4())
        self.environment = environment
        self.generation = generation
        self.parent_id = parent_id
        
        # Set up random number generators
        self._setup_rngs(genome_seed, mutation_seed)
        
        # Initialize genome
        if genome is None:
            genome_rng = np.random.default_rng(genome_seed)
            self.genome = genome_rng.choice([-1, 1], environment.genome_size)
        else:
            self.genome = genome.copy()
        
        # Set mutation rate
        self.mutation_rate = mutation_rate if mutation_rate is not None else 1.0 / environment.genome_size
        
        # Calculate initial fitness
        self.fitness = self.calculate_fitness()
    
    def _setup_rngs(self, genome_seed: Optional[int], mutation_seed: Optional[int]) -> None:
        """Set up random number generators for this organism."""
        if mutation_seed is not None:
            # Create unique seed for this organism
            genome_rng = np.random.default_rng(genome_seed)
            unique_addition = self.generation * 1000 + genome_rng.integers(1000)
            organism_mutation_seed = mutation_seed + unique_addition
        else:
            organism_mutation_seed = None
        
        self.rng = np.random.default_rng(organism_mutation_seed)
    
    def calculate_fitness(self) -> float:
        """Calculate this organism's fitness in its environment."""
        return self.environment.calculate_fitness(self.genome)
    
    def mutate(self) -> None:
        """Apply mutations to this organism's genome."""
        mutation_sites = self.rng.random(len(self.genome)) < self.mutation_rate
        self.genome[mutation_sites] *= -1
        self.fitness = self.calculate_fitness()
    
    def reproduce(self, mutation_seed: Optional[int] = None) -> tuple['Organism', 'Organism']:
        """
        Create two offspring from this organism.
        
        Args:
            mutation_seed: Seed for offspring mutations
            
        Returns:
            Tuple of two child organisms
        """
        child1 = Organism(
            environment=self.environment,
            genome=self.genome,
            generation=self.generation + 1,
            parent_id=self.id,
            mutation_rate=self.mutation_rate,
            mutation_seed=mutation_seed
        )
        
        child2 = Organism(
            environment=self.environment,
            genome=self.genome,
            generation=self.generation + 1,
            parent_id=self.id,
            mutation_rate=self.mutation_rate,
            mutation_seed=mutation_seed
        )
        
        return child1, child2
    
    def __repr__(self) -> str:
        return f"Organism(id={self.id[:8]}, gen={self.generation}, fit={self.fitness:.4f})"


class DiploidOrganism:
    """
    Represents a diploid organism created from two haploid parents.
    
    Diploid organisms have two alleles for each locus and calculate fitness
    based on dominance relationships between alleles.
    """
    
    def __init__(self, parent1: Organism, parent2: Organism, 
                 fitness_model: str = "dominant", mating_type: Optional[MatingType] = None):
        """
        Create a diploid organism from two haploid parents.
        
        Args:
            parent1: First parent organism
            parent2: Second parent organism
            fitness_model: How to handle dominance ("dominant", "recessive", "codominant")
            mating_type: Mating type if applicable
        """
        if len(parent1.genome) != len(parent2.genome):
            raise ValueError("Parent genomes must have the same length")
        
        self.allele1 = parent1.genome.copy()
        self.allele2 = parent2.genome.copy()
        self.fitness_model = fitness_model
        self.environment = parent1.environment
        self.mating_type = mating_type
        
        # Store IDs and fitness values
        self.id = str(uuid.uuid4())
        self.parent1_id = parent1.id
        self.parent2_id = parent2.id
        self.parent1_fitness = parent1.fitness
        self.parent2_fitness = parent2.fitness
        self.avg_parent_fitness = (parent1.fitness + parent2.fitness) / 2
        
        # Calculate fitness
        self.fitness = self.calculate_fitness()
    
    def _get_effective_genome(self) -> np.ndarray:
        """Calculate the effective genome based on fitness model."""
        if self.fitness_model == "codominant":
            return (self.allele1 + self.allele2) / 2
        
        elif self.fitness_model == "dominant":
            # Take locus-wise max (i.e., at least one '1' gives '1')
            return np.maximum(self.allele1, self.allele2)

        elif self.fitness_model == "recessive":
            # Take locus-wise min (i.e., both must be '1' to stay '1')
            return np.minimum(self.allele1, self.allele2)

        else:
            raise ValueError(f"Unknown fitness model: {self.fitness_model}")


    def calculate_fitness(self) -> float:
        """Calculate fitness using the effective genome."""
        effective_genome = self._get_effective_genome()
        return self.environment.calculate_fitness(effective_genome)
    
    def __repr__(self) -> str:
        return f"DiploidOrganism(id={self.id[:8]}, model={self.fitness_model}, fit={self.fitness:.4f})"


class OrganismWithMatingType:
    """Wrapper to add mating type to an organism."""
    
    def __init__(self, organism: Organism, mating_type: MatingType):
        self.organism = organism
        self.mating_type = mating_type
    
    def __repr__(self) -> str:
        return f"TypedOrganism({self.organism}, type={self.mating_type.value})"