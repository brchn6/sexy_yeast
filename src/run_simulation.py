#!/usr/bin/env python3
"""
Script to run genetic simulation and save results for analysis.

This script runs the genetic simulation with specified parameters and saves
the results in a format suitable for analysis.
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional

from core_models import Environment, Organism, DiploidOrganism, FitnessMethod
from simulation_engine import SimulationRunner, MatingStrategy


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def save_simulation_data(crosses: Dict[str, List[DiploidOrganism]], output_path: Path) -> None:
    """Save simulation data to JSON file."""
    data = {}
    for model, organisms in crosses.items():
        data[model] = []
        for org in organisms:
            org_data = {
                'parent1': {
                    'genome': org.parent1.genome.tolist(),
                    'fitness': org.parent1.fitness,
                    'generation': org.parent1.generation,
                    'parent_id': org.parent1.parent_id
                },
                'parent2': {
                    'genome': org.parent2.genome.tolist(),
                    'fitness': org.parent2.fitness,
                    'generation': org.parent2.generation,
                    'parent_id': org.parent2.parent_id
                },
                'fitness': org.fitness
            }
            data[model].append(org_data)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description="Run genetic simulation")
    parser.add_argument("--genome-size", type=int, default=100,
                      help="Size of the genome")
    parser.add_argument("--num-generations", type=int, default=50,
                      help="Number of generations to simulate")
    parser.add_argument("--population-size", type=int, default=1000,
                      help="Maximum population size per generation")
    parser.add_argument("--mutation-rate", type=float, default=0.01,
                      help="Mutation rate per locus")
    parser.add_argument("--fitness-method", type=str, default="sherrington_kirkpatrick",
                      choices=["sherrington_kirkpatrick", "single_position", "additive"],
                      help="Fitness calculation method")
    parser.add_argument("--output", type=str, default="simulation_output",
                      help="Output directory for simulation results")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    environment = Environment(
        genome_size=args.genome_size,
        fitness_method=FitnessMethod(args.fitness_method)
    )
    
    # Initialize simulation
    runner = SimulationRunner(logger=logger)
    
    # Run simulation
    logger.info("Starting simulation...")
    result = runner.run_complete_simulation(
        environment=environment,
        num_generations=args.num_generations,
        mating_strategy=MatingStrategy.ALL_VS_ALL,
        max_population_size=args.population_size,
        fitness_models=["dominant", "recessive", "codominant"]
    )
    
    # Save results
    output_path = output_dir / "simulation_data.json"
    logger.info(f"Saving simulation data to {output_path}")
    save_simulation_data(result["diploid_offspring"], output_path)
    
    logger.info("Simulation complete")


if __name__ == "__main__":
    main() 