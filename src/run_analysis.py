#!/usr/bin/env python3
"""
Script to run genetic analysis on simulation results.

This script provides a command-line interface to run the genetic analysis
on simulation results, generating plots and statistics for parent-offspring
relationships.
"""

import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional

from core_models import Environment, Organism, DiploidOrganism
from analysis_tools import GeneticAnalysis
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


def load_simulation_data(data_path: Path) -> Dict[str, List[DiploidOrganism]]:
    """Load simulation data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Convert JSON data back to DiploidOrganism objects
    crosses = {}
    for model, organisms in data.items():
        crosses[model] = []
        for org_data in organisms:
            # Recreate parent organisms
            parent1 = Organism(
                environment=None,  # Will be set later
                genome=np.array(org_data['parent1']['genome']),
                generation=org_data['parent1']['generation'],
                parent_id=org_data['parent1']['parent_id']
            )
            parent1.fitness = org_data['parent1']['fitness']
            
            parent2 = Organism(
                environment=None,  # Will be set later
                genome=np.array(org_data['parent2']['genome']),
                generation=org_data['parent2']['generation'],
                parent_id=org_data['parent2']['parent_id']
            )
            parent2.fitness = org_data['parent2']['fitness']
            
            # Create diploid organism
            diploid = DiploidOrganism(
                parent1=parent1,
                parent2=parent2,
                fitness_model=model
            )
            diploid.fitness = org_data['fitness']
            crosses[model].append(diploid)
    
    return crosses


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Run genetic analysis on simulation results")
    parser.add_argument("--data", type=str, required=True,
                      help="Path to simulation data JSON file")
    parser.add_argument("--output", type=str, default="analysis_output",
                      help="Output directory for analysis results")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load simulation data
    logger.info(f"Loading simulation data from {args.data}")
    crosses = load_simulation_data(Path(args.data))
    
    # Initialize analysis
    analyzer = GeneticAnalysis(output_dir=output_dir, logger=logger)
    
    # Run analysis for each model
    for model, organisms in crosses.items():
        logger.info(f"\nAnalyzing {model} model...")
        analyzer.analyze_crosses(organisms, run_id=model)
    
    logger.info(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 