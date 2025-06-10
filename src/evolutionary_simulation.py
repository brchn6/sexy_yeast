#!/home/labs/pilpel/barc/.conda/envs/sexy_yeast_env/bin/python
"""
Main application for the evolutionary simulation.

This module provides the command-line interface and orchestrates all
components of the simulation system.
"""

import argparse
import logging
import sys
import time
import os
import psutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from core_models import Environment, FitnessMethod, MatingStrategy
from simulation_engine import SimulationRunner
from analysis_tools import SimulationAnalyzer, MultiRunAnalyzer, save_analysis_results
from visualization import SimulationVisualizer


def setup_logging(log_level: str = "INFO", log_directory: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging with both console and file handlers.
    
    Args:
        log_level: Logging level
        log_directory: Directory for log files (optional)
        
    Returns:
        Configured logger
    """
    # Configure logging format
    log_format = "%(asctime)s | %(name)s | %(levelname)s | L%(lineno)d | %(message)s"
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (if directory specified)
    if log_directory:
        log_directory = Path(log_directory)
        log_directory.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_directory / "simulation.log", mode="w")
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def get_lsf_job_info() -> Dict[str, str]:
    """Get LSF job information if running on a cluster."""
    lsf_vars = ['LSB_JOBID', 'LSB_QUEUE', 'LSB_HOSTS', 'LSB_JOBNAME', 'LSB_CMD']
    return {var: os.environ.get(var, 'N/A') for var in lsf_vars}


def create_output_directory(base_dir: str, args: argparse.Namespace) -> Path:
    """
    Create a descriptive output directory with timestamp and parameters.
    
    Args:
        base_dir: Base directory name
        args: Command line arguments
        
    Returns:
        Path to the created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive directory name
    params = [
        f"L{args.genome_size}",
        f"gen{args.generations}",
        f"fitness_{args.fitness_method}",
        f"runs{args.num_runs}"
    ]
    
    if args.fitness_method == "sherrington_kirkpatrick":
        params.extend([f"beta{args.beta}", f"rho{args.rho}"])
    
    if args.random_seed_env is not None:
        params.append(f"envseed{args.random_seed_env}")
    
    if args.initial_fitness is not None:
        params.append(f"initfit{args.initial_fitness}")
    
    if args.mutation_seed is not None:
        params.append(f"mutseed{args.mutation_seed}")
    
    # Combine all parts
    dir_name = f"{base_dir}_{timestamp}_{'_'.join(params)}"
    
    # Create the directory
    output_path = Path(dir_name)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


class EvolutionarySimulationApp:
    """
    Main application class that orchestrates the entire simulation workflow.
    
    This class handles command-line arguments, coordinates all components,
    and manages the overall simulation workflow.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.logger = None
        self.args = None
        self.output_path = None
        
        # Initialize components
        self.simulation_runner = None
        self.analyzer = None
        self.visualizer = None
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Run evolutionary simulation with diploid analysis",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
    def parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Run evolutionary simulation with diploid analysis",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Basic simulation parameters
        parser.add_argument("--generations", type=int, default=5,
                          help="Number of generations to simulate")
        parser.add_argument("--genome_size", type=int, default=100,
                          help="Size of the genome")
        parser.add_argument("--num_runs", type=int, default=1,
                          help="Number of independent simulation runs")
        
        # Environment parameters
        parser.add_argument("--fitness_method", type=str, default="sherrington_kirkpatrick",
                          choices=["sherrington_kirkpatrick", "single_position", "additive"],
                          help="Method for calculating fitness")
        parser.add_argument("--beta", type=float, default=0.5,
                          help="Beta parameter for Sherrington-Kirkpatrick model")
        parser.add_argument("--rho", type=float, default=0.25,
                          help="Rho parameter for Sherrington-Kirkpatrick model")
        
        # Mating strategy
        parser.add_argument("--mating_strategy", type=str, default="all_vs_all",
                          choices=["one_to_one", "all_vs_all", "mating_types"],
                          help="Strategy for organism mating")
        
        # Random seeds
        parser.add_argument("--random_seed_env", type=int, default=None,
                          help="Seed for environment initialization (same across runs)")
        parser.add_argument("--initial_genome_seed", type=int, default=None,
                          help="Seed for initial genome generation")
        parser.add_argument("--mutation_seed", type=int, default=None,
                          help="Seed for mutations during reproduction")
        
        # Initial conditions
        parser.add_argument("--initial_fitness", type=float, default=None,
                          help="Target fitness for initial organism")
        
        # Output and logging
        parser.add_argument("--output_dir", type=str, default="Results",
                          help="Base name for output directory")
        parser.add_argument("--log_level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                          help="Logging level")
        parser.add_argument("--log_genomes", action="store_true",
                          help="Log individual genomes each generation")
        
        # Analysis and visualization options
        parser.add_argument("--save_individual_runs", action="store_true",
                          help="Save detailed data for individual runs")
        parser.add_argument("--create_plots", action="store_true", default=True,
                          help="Create visualization plots")
        parser.add_argument("--plot_individual_runs", action="store_true",
                          help="Create plots for individual runs (requires --save_individual_runs)")
        
        # Population control
        parser.add_argument("--max_population_size", type=int, default=100000,
                          help="Maximum population size per generation")
        
        # Show help if no arguments provided
        if len(sys.argv) == 1:
            parser.print_help(sys.stderr)
            sys.exit(1)
        
        args = parser.parse_args()
        
        # Validate arguments
        self._validate_arguments(args)
        
        return args
    
    def _validate_arguments(self, args: argparse.Namespace) -> None:
        """Validate command-line arguments."""
        if args.generations < 1:
            raise ValueError("Number of generations must be at least 1")
        
        if args.genome_size < 1:
            raise ValueError("Genome size must be at least 1")
        
        if args.num_runs < 1:
            raise ValueError("Number of runs must be at least 1")
        
        if args.fitness_method == "sherrington_kirkpatrick":
            if not (0 < args.rho <= 1):
                raise ValueError("Rho must be between 0 (exclusive) and 1 (inclusive)")
        
        if args.plot_individual_runs and not args.save_individual_runs:
            raise ValueError("--plot_individual_runs requires --save_individual_runs")
    
    def initialize_components(self) -> None:
        """Initialize all application components."""
        self.simulation_runner = SimulationRunner(self.logger)
        self.analyzer = SimulationAnalyzer(self.logger)
        self.visualizer = SimulationVisualizer()
    
    def run_single_simulation(self, run_id: int) -> Dict[str, Any]:
        """
        Run a single simulation and return results.
        
        Args:
            run_id: Identifier for this run
            
        Returns:
            Dictionary containing simulation results and analysis
        """
        self.logger.info(f"\n==== Starting Run {run_id}/{self.args.num_runs} ====")
        
        start_time = time.time()
        
        # Create environment
        environment = Environment(
            genome_size=self.args.genome_size,
            beta=self.args.beta,
            rho=self.args.rho,
            seed=self.args.random_seed_env,  # Same seed for all runs
            fitness_method=FitnessMethod(self.args.fitness_method)
        )
        
        self.logger.info(f"Environment: {environment.get_description()}")
        
        # Run simulation
        simulation_result = self.simulation_runner.run_complete_simulation(
            environment=environment,
            num_generations=self.args.generations,
            mating_strategy=MatingStrategy(self.args.mating_strategy),
            initial_fitness=self.args.initial_fitness,
            initial_genome_seed=self.args.initial_genome_seed,
            mutation_seed=self.args.mutation_seed,
            max_population_size=self.args.max_population_size,
            log_genomes=self.args.log_genomes
        )
        
        # Analyze results
        analysis_result = self.analyzer.analyze_simulation(simulation_result)
        analysis_result["run_id"] = run_id
        analysis_result["run_time_seconds"] = time.time() - start_time
        
        # Save individual run data if requested
        if self.args.save_individual_runs:
            self._save_individual_run(run_id, simulation_result, analysis_result)
        
        # Create plots for individual run if requested
        if self.args.plot_individual_runs:
            self._create_individual_plots(run_id, simulation_result)
        
        # Log run completion
        self.logger.info(f"Run {run_id} completed in {analysis_result['run_time_seconds']:.2f} seconds")
        
        return analysis_result
    
    def _save_individual_run(self, run_id: int, simulation_result: Dict[str, Any], 
                           analysis_result: Dict[str, Any]) -> None:
        """Save data for an individual run."""
        run_dir = self.output_path / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        # Save analysis results
        save_analysis_results(analysis_result, run_dir, "analysis.json")
        
        # Save simulation parameters
        with open(run_dir / "parameters.json", 'w') as f:
            json.dump(simulation_result["parameters"], f, indent=2)
        
        self.logger.info(f"Saved individual run data to {run_dir}")
    
    def _create_individual_plots(self, run_id: int, simulation_result: Dict[str, Any]) -> None:
        """Create plots for an individual run."""
        if not self.args.create_plots:
            return
        
        run_dir = self.output_path / f"run_{run_id}"
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        simulation = simulation_result["simulation"]
        diploid_offspring = simulation_result["diploid_offspring"]
        mating_strategy = self.args.mating_strategy

        self.logger.debug(f"Simulation object type: {type(simulation)}")
        self.logger.debug(f"Simulation object attributes: {dir(simulation)}")

        
        # Get all organisms from the simulation for tree visualization
        all_organisms = self._extract_organisms_for_tree(simulation)

        try:
            # Fitness evolution plot
            self.visualizer.plot_fitness_evolution(simulation, plots_dir)
    
            # Basic relationship tree (limited to 800 organisms for performance)
            self.visualizer.plot_relationship_tree(all_organisms, plots_dir, filename="relationship_tree.png",max_organisms=800)
            
            # Parent-offspring relationships
            self.visualizer.plot_parent_offspring_relationships(
                diploid_offspring, plots_dir, mating_strategy
            )

            self.visualizer.plot_min_max_parent_offspring_fitness(
                diploid_offspring, plots_dir, mating_strategy
            )
            
            # Genomic distance effects
            self.visualizer.plot_genomic_distance_effects(
                diploid_offspring, plots_dir, mating_strategy
            )
            
            # PRS analysis
            self.visualizer.plot_prs_analysis(
                diploid_offspring, plots_dir, mating_strategy
            )
            
            # Fitness heatmap
            self.visualizer.plot_fitness_heatmap(diploid_offspring, plots_dir)

            
            self.logger.info(f"Created plots for run {run_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create plots for run {run_id}: {e}")

    def _extract_organisms_for_tree(self, simulation) -> List[Any]:
        """
        Extract organism data from the simulation for relationship tree visualization.

        This assumes that organisms have `id`, `generation`, `fitness`, and `parent_id` attributes.

        Args:
            simulation: The simulation object

        Returns:
            List of organisms suitable for tree plotting
        """
        try:
            candidates: List[Any] = []

            # Preferred: all_organisms (used in EvolutionarySimulation)
            if hasattr(simulation, 'all_organisms') and simulation.all_organisms:
                candidates = simulation.all_organisms
                self.logger.debug(f"Using 'all_organisms': {len(candidates)} organisms")
            
            # Fallback: get_last_generation (usually returns only the final generation)
            elif hasattr(simulation, 'get_last_generation'):
                candidates = simulation.get_last_generation()
                self.logger.debug(f"Using 'get_last_generation': {len(candidates)} organisms")
            
            # You could add more extraction options here if your structure grows

            # Validate a few sample organisms
            for i, org in enumerate(candidates[:5]):
                self.logger.debug(
                    f"Sample Org[{i}] — id: {getattr(org, 'id', None)}, "
                    f"gen: {getattr(org, 'generation', None)}, "
                    f"fit: {getattr(org, 'fitness', None)}, "
                    f"parent: {getattr(org, 'parent_id', None)}"
                )

            return candidates

        except Exception as e:
            self.logger.warning(f"Failed to extract organisms for tree visualization: {e}")
            return []

    def run_multiple_simulations(self) -> List[Dict[str, Any]]:
        """
        Run multiple simulations and collect results.
        
        Returns:
            List of analysis results from all runs
        """
        all_results = []
        
        for run_id in range(1, self.args.num_runs + 1):
            try:
                result = self.run_single_simulation(run_id)
                all_results.append(result)
                
                # Monitor memory usage
                memory_percent = psutil.virtual_memory().percent
                self.logger.info(f"Memory usage after run {run_id}: {memory_percent:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Run {run_id} failed: {e}")
                # Continue with other runs
                continue
        
        return all_results
    
    def aggregate_and_analyze_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple runs and perform comparative analysis.
        
        Args:
            all_results: List of individual run results
            
        Returns:
            Aggregated analysis results
        """
        self.logger.info("\n==== Aggregating Results ====")
        
        # Use MultiRunAnalyzer to aggregate results
        multi_analyzer = MultiRunAnalyzer(self.logger)
        aggregated_stats = multi_analyzer.aggregate_runs(all_results)
        
        # Add metadata
        aggregated_stats["simulation_metadata"] = {
            "total_runs": len(all_results),
            "successful_runs": len([r for r in all_results if "error" not in r]),
            "parameters": self.args.__dict__,
            "system_info": {
                "cpu_count": os.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "lsf_info": get_lsf_job_info()
            }
        }
        
        return aggregated_stats
    
    def save_final_results(self, aggregated_stats: Dict[str, Any], 
                          all_results: List[Dict[str, Any]]) -> None:
        """Save final aggregated results and summary."""
        # Save aggregated statistics
        save_analysis_results(aggregated_stats, self.output_path, "aggregated_analysis.json")
        
        # Save summary of all runs
        summary_data = {
            "individual_runs": all_results,
            "aggregated_stats": aggregated_stats
        }
        save_analysis_results(summary_data, self.output_path, "complete_results.json")
        
        # Create summary text file
        self._create_summary_report(aggregated_stats)
        
        self.logger.info(f"Final results saved to {self.output_path}")
    
    def _create_summary_report(self, aggregated_stats: Dict[str, Any]) -> None:
        """Create a human-readable summary report."""
        summary_file = self.output_path / "SUMMARY.txt"
        
        with open(summary_file, 'w') as f:
            f.write("EVOLUTIONARY SIMULATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            metadata = aggregated_stats.get("simulation_metadata", {})
            f.write(f"Total Runs: {metadata.get('total_runs', 'N/A')}\n")
            f.write(f"Successful Runs: {metadata.get('successful_runs', 'N/A')}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Parameters
            params = metadata.get("parameters", {})
            f.write("PARAMETERS:\n")
            f.write("-" * 20 + "\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Haploid evolution results
            haploid = aggregated_stats.get("haploid_evolution", {})
            if haploid:
                f.write("HAPLOID EVOLUTION RESULTS:\n")
                f.write("-" * 30 + "\n")
                for metric, stats in haploid.items():
                    if isinstance(stats, dict) and "mean" in stats:
                        f.write(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write("\n")
            
            # Diploid model comparison
            diploid = aggregated_stats.get("diploid_models", {})
            if diploid:
                f.write("DIPLOID MODEL COMPARISON:\n")
                f.write("-" * 30 + "\n")
                for model, stats in diploid.items():
                    f.write(f"\n{model.upper()}:\n")
                    if "avg_offspring_fitness" in stats:
                        fit_stats = stats["avg_offspring_fitness"]
                        f.write(f"  Avg offspring fitness: {fit_stats['mean']:.4f} ± {fit_stats['std']:.4f}\n")
                    if "fitness_improvement" in stats:
                        imp_stats = stats["fitness_improvement"]
                        f.write(f"  Fitness improvement: {imp_stats['mean']:.4f} ± {imp_stats['std']:.4f}\n")
            
            f.write(f"\nDetailed results available in: {self.output_path}\n")
    
    
    def run(self) -> None:
        """Main application entry point."""
        try:
            # Parse arguments
            self.args = self.parse_arguments()
            
            # Create output directory
            self.output_path = create_output_directory(self.args.output_dir, self.args)
            
            # Set up logging
            self.logger = setup_logging(self.args.log_level, self.output_path)
            
            # Log startup information
            self._log_startup_info()
            
            # Initialize components
            self.initialize_components()
            
            # Run simulations
            start_time = time.time()
            all_results = self.run_multiple_simulations()
            
            if not all_results:
                self.logger.error("No successful runs completed")
                return
            
            # Aggregate and analyze results
            aggregated_stats = self.aggregate_and_analyze_results(all_results)
            
            # Save results
            self.save_final_results(aggregated_stats, all_results)
                        
            # Log completion
            total_time = time.time() - start_time
            self.logger.info(f"\n==== SIMULATION COMPLETE ====")
            self.logger.info(f"Total time: {total_time:.2f} seconds")
            self.logger.info(f"Average time per run: {total_time/len(all_results):.2f} seconds")
            self.logger.info(f"Results saved to: {self.output_path}")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Application failed: {e}")
            else:
                print(f"Application failed: {e}")
            sys.exit(1)
    
    def _log_startup_info(self) -> None:
        """Log startup information."""
        self.logger.info("Starting Evolutionary Simulation")
        self.logger.info(f"Command: {' '.join(sys.argv)}")
        self.logger.info(f"Output directory: {self.output_path}")
        self.logger.info(f"Arguments: {self.args}")
        
        # System information
        self.logger.info(f"CPU count: {os.cpu_count()}")
        self.logger.info(f"Total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        
        # LSF information (if available)
        lsf_info = get_lsf_job_info()
        if lsf_info['LSB_JOBID'] != 'N/A':
            self.logger.info("LSF Job Information:")
            for key, value in lsf_info.items():
                self.logger.info(f"  {key}: {value}")


def main():
    """Main entry point for the application."""
    app = EvolutionarySimulationApp()
    app.run()


if __name__ == "__main__":
    main()