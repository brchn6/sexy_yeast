#!/usr/bin/env python3
"""
Core models and enums for the evolutionary simulation.

This module contains the fundamental data structures and enumerations
used throughout the simulation.
"""

import numpy as np
import uuid
from enum import Enum
from typing import Optional, Union, Dict, Any, List
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
        return float(genome @ (self.h + 0.5 * self.J @ genome))
    
    def _calculate_alternative_fitness(self, genome: np.ndarray) -> float:
        """Calculate fitness using alternative methods."""
        method = self.alternative_params["method"]
        
        if method == FitnessMethod.SINGLE_POSITION:
            position = self.alternative_params["position"]
            
            if position >= len(genome):
                return 0.0
            
            # Normalize values from [-1, 1] to [0, 1]
            return float((genome[position] + 1) / 2.0)
            
        elif method == FitnessMethod.ADDITIVE:
            weights = self.alternative_params["weights"]
            genome_01 = (genome + 1) / 2  # Convert -1/1 to 0/1
            fitness = np.dot(genome_01, weights)
            return float(0.1 + 0.9 / (1 + np.exp(-fitness)))  # Sigmoid normalization
        
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert organism to dictionary for serialization."""
        return {
            "id": self.id,
            "generation": self.generation,
            "fitness": float(self.fitness),
            "genome": self.genome.tolist(),
            "parent_id": self.parent_id,
            "mutation_rate": self.mutation_rate
        }
    
    def __repr__(self) -> str:
        return f"Organism(id={self.id[:8]}, gen={self.generation}, fit={self.fitness:.4f})"


class DiploidOrganism:
    """
    Represents a diploid organism resulting from mating two haploid organisms.
    
    This class handles different fitness models (dominant, recessive, codominant)
    and calculates fitness based on the combined genomes of the parents.
    """
    
    def __init__(self, parent1: Organism, parent2: Organism, 
                 fitness_model: str = "dominant", mating_type: Optional[MatingType] = None):
        """
        Create a diploid organism from two parents.
        
        Args:
            parent1: First parent organism
            parent2: Second parent organism
            fitness_model: How to calculate fitness from alleles
            mating_type: Optional mating type for this organism
        """
        self.parent1 = parent1
        self.parent2 = parent2
        self.fitness_model = fitness_model
        self.mating_type = mating_type
        
        # Store parent genomes as alleles
        self.allele1 = parent1.genome.copy()
        self.allele2 = parent2.genome.copy()
        
        # Calculate fitness
        self.fitness = self.calculate_fitness()
        
        # Store parent fitness for analysis
        self.parent1_fitness = parent1.fitness
        self.parent2_fitness = parent2.fitness
        self.avg_parent_fitness = (parent1.fitness + parent2.fitness) / 2
        
        # Calculate additional metrics for analysis
        self.prs = self.calculate_prs()
        self.genomic_distance = self.calculate_genomic_distance()
    
    def calculate_prs(self) -> float:
        """Calculate polygenic risk score for this diploid organism."""
        return float((np.sum(self.allele1) + np.sum(self.allele2)) / 2)
    
    def calculate_genomic_distance(self) -> int:
        """Calculate Hamming distance between the two alleles."""
        return int(np.sum(self.allele1 != self.allele2))
    
    def _get_effective_genome(self) -> np.ndarray:
        """Get effective genome based on fitness model."""
        if self.parent1.environment.fitness_method == FitnessMethod.SINGLE_POSITION:
            # For single position model, handle alleles at the target position
            position = self.parent1.environment.alternative_params["position"]
            if self.fitness_model == "dominant":
                # 1 dominates -1 at the target position
                effective_genome = self.allele1.copy()
                effective_genome[position] = max(self.allele1[position], self.allele2[position])
            elif self.fitness_model == "recessive":
                # -1 is recessive at the target position
                effective_genome = self.allele1.copy()
                effective_genome[position] = min(self.allele1[position], self.allele2[position])
            else:  # codominant
                # Average effect at the target position
                effective_genome = self.allele1.copy()
                effective_genome[position] = (self.allele1[position] + self.allele2[position]) / 2
            return effective_genome
        else:
            # For other fitness methods, use the original logic
            if self.fitness_model == "dominant":
                # 1 dominates -1
                return np.maximum(self.allele1, self.allele2)
            elif self.fitness_model == "recessive":
                # -1 is recessive
                return np.minimum(self.allele1, self.allele2)
            elif self.fitness_model == "codominant":
                # Average effect
                return (self.allele1 + self.allele2) / 2
            else:
                raise ValueError(f"Unknown fitness model: {self.fitness_model}")
    
    def calculate_fitness(self) -> float:
        """Calculate fitness based on the effective genome."""
        effective_genome = self._get_effective_genome()
        return self.parent1.environment.calculate_fitness(effective_genome)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the diploid organism to a dictionary for serialization."""
        return {
            "fitness": float(self.fitness),
            "fitness_model": self.fitness_model,
            "parent1_fitness": float(self.parent1_fitness),
            "parent2_fitness": float(self.parent2_fitness),
            "avg_parent_fitness": float(self.avg_parent_fitness),
            "allele1": self.allele1.tolist(),
            "allele2": self.allele2.tolist(),
            "mating_type": self.mating_type.value if self.mating_type else None,
            "prs": float(self.prs),
            "genomic_distance": int(self.genomic_distance)
        }
    
    def __repr__(self) -> str:
        """String representation of the diploid organism."""
        return f"DiploidOrganism(fitness={self.fitness:.4f}, model={self.fitness_model})"


class OrganismWithMatingType:
    """Wrapper to add mating type to an organism."""
    
    def __init__(self, organism: Organism, mating_type: MatingType):
        self.organism = organism
        self.mating_type = mating_type
        
        # Expose organism attributes for compatibility
        self.id = organism.id
        self.generation = organism.generation
        self.fitness = organism.fitness
        self.genome = organism.genome
        self.parent_id = organism.parent_id
    
    def __repr__(self) -> str:
        return f"TypedOrganism({self.organism}, type={self.mating_type.value})"


# Utility functions that were causing circular imports
def calculate_genomic_distance(genome1: np.ndarray, genome2: np.ndarray) -> float:
    """Calculate Hamming distance between two genomes.
    
    Args:
        genome1: First genome array
        genome2: Second genome array
        
    Returns:
        Hamming distance between the genomes
    """
    return float(np.sum(genome1 != genome2))


def calculate_diploid_prs(organism: DiploidOrganism) -> float:
    """Calculate polygenic risk score for a diploid organism as the total count of minor (-1) alleles across both alleles."""
    minor_allele_count = np.sum(organism.allele1 == -1) + np.sum(organism.allele2 == -1)
    return minor_allele_count  # total count of minor alleles


def calculate_polygenic_risk_score(genome: np.ndarray) -> float:
    """Calculate polygenic risk score as sum of genome values."""
    return float(np.sum(genome))


class CrossData:
    """
    Stores data for a single cross between two parents.
    """
    def __init__(self, 
                 inheritance_mode: str,
                 avg_parent_fitness: float,
                 genomic_distance: int,
                 combined_prs: float,
                 offspring_fitness: float,
                 run_id: int):
        self.inheritance_mode = inheritance_mode
        self.avg_parent_fitness = avg_parent_fitness
        self.genomic_distance = genomic_distance
        self.combined_prs = combined_prs
        self.offspring_fitness = offspring_fitness
        self.run_id = run_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cross data to dictionary format."""
        return {
            "inheritance_mode": self.inheritance_mode,
            "avg_parent_fitness": self.avg_parent_fitness,
            "genomic_distance": self.genomic_distance,
            "combined_prs": self.combined_prs,
            "offspring_fitness": self.offspring_fitness,
            "run_id": self.run_id
        }

class CrossDataCollector:
    """
    Manages collection and storage of cross data across simulation runs.
    """
    def __init__(self):
        self.crosses: List[CrossData] = []
        self.current_run_id = 0
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized CrossDataCollector")
    
    def start_new_run(self) -> None:
        """Start a new simulation run."""
        self.current_run_id += 1
        self.logger.info(f"Starting new run {self.current_run_id}")
        self.logger.info(f"Current crosses count: {len(self.crosses)}")
    
    def record_cross(self, 
                    inheritance_mode: str,
                    parent1: Organism,
                    parent2: Organism,
                    offspring: DiploidOrganism) -> None:
        """Record data for a single cross."""
        try:
            # Validate inputs
            if not isinstance(inheritance_mode, str):
                raise ValueError(f"Invalid inheritance_mode type: {type(inheritance_mode)}")
            if not isinstance(parent1, Organism) or not isinstance(parent2, Organism):
                raise ValueError(f"Invalid parent types: {type(parent1)}, {type(parent2)}")
            if not isinstance(offspring, DiploidOrganism):
                raise ValueError(f"Invalid offspring type: {type(offspring)}")
            
            # Calculate values
            avg_parent_fitness = (parent1.fitness + parent2.fitness) / 2
            genomic_distance = offspring.calculate_genomic_distance()
            combined_prs = offspring.calculate_prs()
            offspring_fitness = offspring.fitness
            
            # Validate calculated values
            if not isinstance(avg_parent_fitness, (int, float)):
                raise ValueError(f"Invalid avg_parent_fitness type: {type(avg_parent_fitness)}")
            if not isinstance(genomic_distance, int):
                raise ValueError(f"Invalid genomic_distance type: {type(genomic_distance)}")
            if not isinstance(combined_prs, (int, float)):
                raise ValueError(f"Invalid combined_prs type: {type(combined_prs)}")
            if not isinstance(offspring_fitness, (int, float)):
                raise ValueError(f"Invalid offspring_fitness type: {type(offspring_fitness)}")
            
            cross_data = CrossData(
                inheritance_mode=inheritance_mode,
                avg_parent_fitness=avg_parent_fitness,
                genomic_distance=genomic_distance,
                combined_prs=combined_prs,
                offspring_fitness=offspring_fitness,
                run_id=self.current_run_id
            )
            
            self.crosses.append(cross_data)
            self.logger.debug(
                f"Recorded cross: mode={inheritance_mode}, "
                f"parent_fitness={avg_parent_fitness:.4f}, "
                f"offspring_fitness={offspring_fitness:.4f}, "
                f"genomic_distance={genomic_distance}, "
                f"prs={combined_prs:.4f}"
            )
        except Exception as e:
            self.logger.error(f"Failed to record cross: {e}")
            self.logger.error("Error details:", exc_info=True)
            raise  # Re-raise to handle in calling code
    
    def get_data_as_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all cross data as a dictionary."""
        self.logger.info(f"Retrieved {len(self.crosses)} crosses for run {self.current_run_id}")
        
        # Validate data before returning
        for cross in self.crosses:
            if not isinstance(cross, CrossData):
                raise ValueError(f"Invalid cross type in collection: {type(cross)}")
        
        return {
            "run_id": self.current_run_id,
            "crosses": [cross.to_dict() for cross in self.crosses]
        }
    
    def get_data_as_list(self) -> List[Dict[str, Any]]:
        """Get all cross data as a flat list of dictionaries."""
        self.logger.info(f"Retrieved {len(self.crosses)} crosses as list")
        
        # Validate data before returning
        for cross in self.crosses:
            if not isinstance(cross, CrossData):
                raise ValueError(f"Invalid cross type in collection: {type(cross)}")
        
        return [cross.to_dict() for cross in self.crosses]
    
    def clear(self) -> None:
        """Clear all collected data."""
        self.logger.info("Clearing all collected cross data")
        self.crosses = []
        self.current_run_id = 0
    
    def validate_data(self) -> bool:
        """Validate all collected data."""
        try:
            self.logger.info("Validating collected data...")
            
            # Check if we have any data
            if not self.crosses:
                self.logger.warning("No crosses collected")
                return False
            
            # Validate each cross
            for i, cross in enumerate(self.crosses):
                if not isinstance(cross, CrossData):
                    self.logger.error(f"Invalid cross type at index {i}: {type(cross)}")
                    return False
                
                # Validate cross data
                cross_dict = cross.to_dict()
                required_fields = [
                    "inheritance_mode", "avg_parent_fitness", "genomic_distance",
                    "combined_prs", "offspring_fitness", "run_id"
                ]
                
                for field in required_fields:
                    if field not in cross_dict:
                        self.logger.error(f"Missing field '{field}' in cross at index {i}")
                        return False
                
                # Validate field types
                if not isinstance(cross_dict["inheritance_mode"], str):
                    self.logger.error(f"Invalid inheritance_mode type in cross {i}")
                    return False
                if not isinstance(cross_dict["avg_parent_fitness"], (int, float)):
                    self.logger.error(f"Invalid avg_parent_fitness type in cross {i}")
                    return False
                if not isinstance(cross_dict["genomic_distance"], int):
                    self.logger.error(f"Invalid genomic_distance type in cross {i}")
                    return False
                if not isinstance(cross_dict["combined_prs"], (int, float)):
                    self.logger.error(f"Invalid combined_prs type in cross {i}")
                    return False
                if not isinstance(cross_dict["offspring_fitness"], (int, float)):
                    self.logger.error(f"Invalid offspring_fitness type in cross {i}")
                    return False
                if not isinstance(cross_dict["run_id"], int):
                    self.logger.error(f"Invalid run_id type in cross {i}")
                    return False
            
            self.logger.info(f"Data validation successful: {len(self.crosses)} valid crosses")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            self.logger.error("Error details:", exc_info=True)
            return False