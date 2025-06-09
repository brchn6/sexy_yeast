#!/usr/bin/env python3
"""
Configuration and utility functions for the evolutionary simulation.

This module provides configuration management, helper functions, and
common utilities used throughout the simulation package.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass, asdict
import logging


@dataclass
class SimulationConfig:
    """
    Configuration dataclass for simulation parameters.
    
    This provides a structured way to manage simulation parameters
    with validation and easy serialization.
    """
    # Basic parameters
    generations: int = 5
    genome_size: int = 100
    num_runs: int = 1
    
    # Environment parameters
    fitness_method: str = "sherrington_kirkpatrick"
    beta: float = 0.5
    rho: float = 0.25
    
    # Mating strategy
    mating_strategy: str = "all_vs_all"
    
    # Random seeds
    random_seed_env: Optional[int] = None
    initial_genome_seed: Optional[int] = None
    mutation_seed: Optional[int] = None
    
    # Initial conditions
    initial_fitness: Optional[float] = None
    
    # Population control
    max_population_size: int = 100000
    
    # Output and logging
    output_dir: str = "Results"
    log_level: str = "INFO"
    log_genomes: bool = False
    
    # Analysis options
    save_individual_runs: bool = False
    create_plots: bool = True
    plot_individual_runs: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        
        if self.genome_size < 1:
            raise ValueError("genome_size must be at least 1")
        
        if self.num_runs < 1:
            raise ValueError("num_runs must be at least 1")
        
        if self.fitness_method not in ["sherrington_kirkpatrick", "single_position", "additive"]:
            raise ValueError(f"Invalid fitness_method: {self.fitness_method}")
        
        if self.fitness_method == "sherrington_kirkpatrick":
            if not (0 < self.rho <= 1):
                raise ValueError("rho must be between 0 (exclusive) and 1 (inclusive)")
        
        if self.mating_strategy not in ["one_to_one", "all_vs_all", "mating_types"]:
            raise ValueError(f"Invalid mating_strategy: {self.mating_strategy}")
        
        if self.plot_individual_runs and not self.save_individual_runs:
            raise ValueError("plot_individual_runs requires save_individual_runs to be True")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'SimulationConfig':
        """Load configuration from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create configuration from dictionary."""
        return cls(**data)


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy data types.
    
    This allows NumPy arrays, integers, and floats to be serialized
    as regular Python types for JSON compatibility.
    """
    
    def default(self, obj):
        """Convert NumPy types to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        else:
            return super().default(obj)


def save_json_with_numpy(data: Any, filepath: Union[str, Path], 
                        indent: int = 2) -> None:
    """
    Save data to JSON file with NumPy type handling.
    
    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=indent)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        return json.load(f)


def setup_directory_structure(base_path: Path, 
                            include_individual_runs: bool = False) -> Dict[str, Path]:
    """
    Set up standard directory structure for simulation outputs.
    
    Args:
        base_path: Base output directory
        include_individual_runs: Whether to create individual run directories
        
    Returns:
        Dictionary mapping directory names to paths
    """
    base_path = Path(base_path)
    
    directories = {
        "base": base_path,
        "logs": base_path / "logs",
        "plots": base_path / "plots",
        "analysis": base_path / "analysis",
        "raw_data": base_path / "raw_data"
    }
    
    if include_individual_runs:
        directories["individual_runs"] = base_path / "individual_runs"
    
    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    
    This provides a consistent way to track and report progress
    across different components of the simulation.
    """
    
    def __init__(self, total_steps: int, description: str = "Progress",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the progress tracker.
        
        Args:
            total_steps: Total number of steps to track
            description: Description of the operation
            logger: Logger for progress messages
        """
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.logger = logger or logging.getLogger(__name__)
    
    def update(self, steps: int = 1, message: str = "") -> None:
        """
        Update progress by the specified number of steps.
        
        Args:
            steps: Number of steps to advance
            message: Optional message to include
        """
        self.current_step += steps
        progress_percent = (self.current_step / self.total_steps) * 100
        
        log_message = f"{self.description}: {self.current_step}/{self.total_steps} " \
                     f"({progress_percent:.1f}%)"
        
        if message:
            log_message += f" - {message}"
        
        self.logger.info(log_message)
    
    def complete(self, final_message: str = "Complete") -> None:
        """Mark the operation as complete."""
        self.current_step = self.total_steps
        self.logger.info(f"{self.description}: {final_message}")


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in a human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def format_memory_size(bytes_size: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def validate_file_path(filepath: Union[str, Path], 
                      must_exist: bool = False,
                      create_parent: bool = False) -> Path:
    """
    Validate and process file path.
    
    Args:
        filepath: Path to validate
        must_exist: Whether file must already exist
        create_parent: Whether to create parent directories
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file must exist but doesn't
        ValueError: If path is invalid
    """
    filepath = Path(filepath)
    
    if must_exist and not filepath.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")
    
    if create_parent:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    return filepath


def summarize_simulation_parameters(config: SimulationConfig) -> str:
    """
    Create a human-readable summary of simulation parameters.
    
    Args:
        config: Simulation configuration
        
    Returns:
        Formatted parameter summary
    """
    summary_lines = [
        "Simulation Parameters:",
        "=" * 40,
        f"Generations: {config.generations}",
        f"Genome size: {config.genome_size}",
        f"Number of runs: {config.num_runs}",
        f"Fitness method: {config.fitness_method}",
        f"Mating strategy: {config.mating_strategy}",
        "",
        "Environment parameters:",
    ]
    
    if config.fitness_method == "sherrington_kirkpatrick":
        summary_lines.extend([
            f"  Beta: {config.beta}",
            f"  Rho: {config.rho}"
        ])
    
    summary_lines.extend([
        "",
        "Random seeds:",
        f"  Environment: {config.random_seed_env}",
        f"  Initial genome: {config.initial_genome_seed}",
        f"  Mutations: {config.mutation_seed}",
        "",
        f"Initial fitness target: {config.initial_fitness}",
        f"Max population size: {config.max_population_size}",
        f"Log level: {config.log_level}"
    ])
    
    return "\n".join(summary_lines)


# Default configuration for common use cases
DEFAULT_QUICK_CONFIG = SimulationConfig(
    generations=3,
    genome_size=50,
    num_runs=2,
    fitness_method="single_position",
    create_plots=True,
    log_level="INFO"
)

DEFAULT_STANDARD_CONFIG = SimulationConfig(
    generations=10,
    genome_size=100,
    num_runs=5,
    fitness_method="sherrington_kirkpatrick",
    beta=0.5,
    rho=0.25,
    create_plots=True,
    save_individual_runs=True,
    log_level="INFO"
)

DEFAULT_EXTENSIVE_CONFIG = SimulationConfig(
    generations=50,
    genome_size=200,
    num_runs=20,
    fitness_method="sherrington_kirkpatrick",
    beta=0.5,
    rho=0.25,
    create_plots=True,
    save_individual_runs=True,
    plot_individual_runs=True,
    log_level="INFO"
)