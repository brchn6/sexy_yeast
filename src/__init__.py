#!/usr/bin/env python3
"""
Evolutionary Simulation Package

A comprehensive toolkit for simulating evolutionary dynamics with diploid analysis.

This package provides a clean, object-oriented framework for:
- Running evolutionary simulations with customizable fitness landscapes
- Analyzing parent-offspring relationships in diploid organisms
- Visualizing evolutionary dynamics and comparative results
- Managing multiple simulation runs with statistical analysis

Main Components:
- core_models: Fundamental data structures (Environment, Organism, DiploidOrganism)
- simulation_engine: Simulation orchestration and execution
- analysis_tools: Statistical analysis and data processing
- visualization: Publication-quality plotting and visualization
- config_utils: Configuration management and utilities
- main_application: Command-line interface and workflow coordination

Example Usage:
    # Quick start with default parameters
    from evolutionary_simulation import SimulationRunner, Environment, FitnessMethod
    
    env = Environment(genome_size=100, fitness_method=FitnessMethod.SHERRINGTON_KIRKPATRICK)
    runner = SimulationRunner()
    results = runner.run_complete_simulation(env, num_generations=10, ...)
    
    # Or use the command-line interface
    python -m evolutionary_simulation.main_application --generations 10 --genome_size 100

Author: [Your Name]
Version: 1.0.0
"""

# Import main classes and functions for easy access
from .core_models import (
    # Enums
    MatingStrategy,
    MatingType, 
    FitnessMethod,
    
    # Core classes
    Environment,
    Organism,
    DiploidOrganism,
    OrganismWithMatingType
)

from .simulation_engine import (
    EvolutionarySimulation,
    MatingEngine,
    SimulationRunner
)

from .analysis_tools import (
    SimulationAnalyzer,
    MultiRunAnalyzer,
    calculate_genomic_distance,
    calculate_polygenic_risk_score,
    calculate_diploid_prs,
    calculate_regression_stats,
    save_analysis_results
)

from .visualization import (
    SimulationVisualizer,
    MultiRunVisualizer
)

from .config_utils import (
    SimulationConfig,
    NumpyJSONEncoder,
    save_json_with_numpy,
    load_json,
    setup_directory_structure,
    ProgressTracker,
    format_time_duration,
    format_memory_size,
    validate_file_path,
    summarize_simulation_parameters,
    
    # Pre-defined configurations
    DEFAULT_QUICK_CONFIG,
    DEFAULT_STANDARD_CONFIG,
    DEFAULT_EXTENSIVE_CONFIG
)

from .main_application import (
    EvolutionarySimulationApp,
    setup_logging,
    create_output_directory,
    get_lsf_job_info
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Comprehensive evolutionary simulation with diploid analysis"

# Define what gets imported with "from evolutionary_simulation import *"
__all__ = [
    # Enums
    "MatingStrategy",
    "MatingType", 
    "FitnessMethod",
    
    # Core classes
    "Environment",
    "Organism", 
    "DiploidOrganism",
    "OrganismWithMatingType",
    
    # Simulation engine
    "EvolutionarySimulation",
    "MatingEngine", 
    "SimulationRunner",
    
    # Analysis tools
    "SimulationAnalyzer",
    "MultiRunAnalyzer",
    "calculate_genomic_distance",
    "calculate_polygenic_risk_score", 
    "calculate_diploid_prs",
    "calculate_regression_stats",
    "save_analysis_results",
    
    # Visualization
    "SimulationVisualizer",
    "MultiRunVisualizer",
    
    # Configuration and utilities
    "SimulationConfig",
    "NumpyJSONEncoder",
    "save_json_with_numpy",
    "load_json",
    "setup_directory_structure",
    "ProgressTracker",
    "format_time_duration",
    "format_memory_size", 
    "validate_file_path",
    "summarize_simulation_parameters",
    "DEFAULT_QUICK_CONFIG",
    "DEFAULT_STANDARD_CONFIG", 
    "DEFAULT_EXTENSIVE_CONFIG",
    
    # Main application
    "EvolutionarySimulationApp",
    "setup_logging",
    "create_output_directory",
    "get_lsf_job_info"
]


def quick_simulation(genome_size: int = 100, generations: int = 10, 
                    num_runs: int = 3, fitness_method: str = "sherrington_kirkpatrick",
                    output_dir: str = "QuickResults") -> dict:
    """
    Run a quick simulation with sensible defaults.
    
    This is a convenience function for getting started quickly without
    having to configure all the parameters manually.
    
    Args:
        genome_size: Size of the genome
        generations: Number of generations to simulate
        num_runs: Number of independent runs
        fitness_method: Fitness calculation method
        output_dir: Directory for results
        
    Returns:
        Dictionary containing aggregated results
        
    Example:
        >>> from evolutionary_simulation import quick_simulation
        >>> results = quick_simulation(genome_size=50, generations=5, num_runs=2)
        >>> print(f"Final fitness: {results['haploid_evolution']['final_fitness']['mean']:.3f}")
    """
    import tempfile
    import logging
    from pathlib import Path
    
    # Create temporary logger
    logger = logging.getLogger("quick_sim")
    logger.setLevel(logging.INFO)
    
    # If no handlers, add a simple console handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Create environment
    env = Environment(
        genome_size=genome_size,
        fitness_method=FitnessMethod(fitness_method)
    )
    
    # Run simulations
    runner = SimulationRunner(logger)
    analyzer = SimulationAnalyzer(logger)
    multi_analyzer = MultiRunAnalyzer(logger)
    
    all_results = []
    
    logger.info(f"Running {num_runs} quick simulations...")
    
    for run_id in range(1, num_runs + 1):
        logger.info(f"Run {run_id}/{num_runs}")
        
        # Run single simulation
        sim_result = runner.run_complete_simulation(
            environment=env,
            num_generations=generations,
            mating_strategy=MatingStrategy.ALL_VS_ALL,
            max_population_size=10000,
            log_genomes=False
        )
        
        # Analyze results
        analysis_result = analyzer.analyze_simulation(sim_result)
        analysis_result["run_id"] = run_id
        all_results.append(analysis_result)
    
    # Aggregate results
    aggregated = multi_analyzer.aggregate_runs(all_results)
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        save_analysis_results(aggregated, output_path, "quick_simulation_results.json")
        logger.info(f"Results saved to {output_path}")
    
    logger.info("Quick simulation complete!")
    return aggregated


def create_example_config(config_type: str = "standard") -> SimulationConfig:
    """
    Create an example configuration for different use cases.
    
    Args:
        config_type: Type of configuration ("quick", "standard", "extensive")
        
    Returns:
        SimulationConfig object
        
    Example:
        >>> from evolutionary_simulation import create_example_config
        >>> config = create_example_config("quick")
        >>> config.save_to_file("my_config.json")
    """
    config_map = {
        "quick": DEFAULT_QUICK_CONFIG,
        "standard": DEFAULT_STANDARD_CONFIG, 
        "extensive": DEFAULT_EXTENSIVE_CONFIG
    }
    
    if config_type not in config_map:
        raise ValueError(f"Unknown config type: {config_type}. "
                        f"Options are: {list(config_map.keys())}")
    
    return config_map[config_type]


def get_package_info() -> dict:
    """
    Get information about the package and its components.
    
    Returns:
        Dictionary with package information
    """
    return {
        "name": "evolutionary_simulation",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "core_models": "Fundamental data structures and enums",
            "simulation_engine": "Simulation execution and coordination", 
            "analysis_tools": "Statistical analysis and data processing",
            "visualization": "Plotting and visualization tools",
            "config_utils": "Configuration management and utilities",
            "main_application": "Command-line interface and workflow"
        },
        "key_classes": [
            "Environment", "Organism", "DiploidOrganism",
            "SimulationRunner", "SimulationAnalyzer", "SimulationVisualizer"
        ],
        "supported_fitness_methods": [
            "sherrington_kirkpatrick", "single_position", "additive"
        ],
        "supported_mating_strategies": [
            "one_to_one", "all_vs_all", "mating_types"
        ]
    }


# Convenience function for command-line usage
def main():
    """
    Entry point for command-line usage.
    
    This allows the package to be run as:
    python -m evolutionary_simulation
    """
    from .main_application import main as app_main
    app_main()


# Package-level configuration
import logging
import warnings

# Set up a null handler to prevent logging messages if no handler is configured
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Filter some common warnings that might clutter output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


# Print welcome message when package is imported (optional)
def _welcome_message():
    """Print a brief welcome message when package is imported."""
    import sys
    if hasattr(sys, 'ps1'):  # Only in interactive mode
        print(f"Evolutionary Simulation Package v{__version__}")
        print("Use quick_simulation() for a fast start or create_example_config() for configuration.")
        print("Type help(evolutionary_simulation) for more information.")

# Uncomment the next line if you want the welcome message
# _welcome_message()