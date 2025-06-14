#!/usr/bin/env python3
"""
Analysis tools for genetic simulation data.

This module provides functions for statistical analysis and visualization
of genetic simulation results, including regression analysis and plotting
of various genetic metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json

# Import utility functions from core_models to avoid circular imports
from core_models import calculate_genomic_distance, calculate_diploid_prs, calculate_polygenic_risk_score


class SimulationDataCollector:
    """Collects and manages data from multiple simulation runs."""
    
    def __init__(self):
        """Initialize the data collector."""
        self.runs_data = []
        self.summary_data = {}
    
    def add_run_data(self, run_data: Dict[str, Any]) -> None:
        """Add data from a single run to the collector.
        
        Args:
            run_data: Dictionary containing run results and analysis
        """
        # Convert diploid offspring to the correct format if needed
        if "diploid_offspring" in run_data:
            for model in run_data["diploid_offspring"]:
                if isinstance(run_data["diploid_offspring"][model], list):
                    # Check if it's already in dict format
                    if run_data["diploid_offspring"][model] and isinstance(run_data["diploid_offspring"][model][0], dict):
                        # Already in correct format, convert to analysis format
                        run_data["diploid_offspring"][model] = self._analyze_diploid_model(
                            run_data["diploid_offspring"][model]
                        )
        
        self.runs_data.append(run_data)
        self._update_summary()
    
    def _analyze_diploid_model(self, organisms: List[Dict[str, Any]]) -> Dict[str, List]:
        """Analyze statistics for a specific diploid model.
        
        Args:
            organisms: List of diploid offspring (as dicts)
            
        Returns:
            Dictionary containing lists of parent-offspring relationships,
            PRS scores, and genomic distances
        """
        if not organisms:
            return {}
        
        parent_offspring = []
        prs = []
        genomic_distance = []
        
        for org in organisms:
            # Parent-offspring fitness relationship
            parent_offspring.append((
                org["avg_parent_fitness"],
                org["fitness"]
            ))
            
            # PRS analysis
            if "prs" in org:
                prs_val = org["prs"]
            elif "allele1" in org and "allele2" in org:
                prs_val = (np.sum(org["allele1"]) + np.sum(org["allele2"])) / 2
            else:
                prs_val = None
            
            prs.append((prs_val, org["fitness"]))
            
            # Genomic distance
            if "genomic_distance" in org:
                distance = org["genomic_distance"]
            elif "allele1" in org and "allele2" in org:
                distance = np.sum(np.array(org["allele1"]) != np.array(org["allele2"]))
            else:
                distance = None
                
            genomic_distance.append((distance, org["fitness"]))
            
        return {
            "parent_offspring": parent_offspring,
            "prs": prs,
            "genomic_distance": genomic_distance
        }
    
    def _update_summary(self) -> None:
        """Update summary statistics based on collected runs."""
        if not self.runs_data:
            return
            
        # Extract key metrics from all runs
        fitness_evolution = []
        diploid_stats = {
            "dominant": {"parent_offspring": [], "prs": [], "genomic_distance": []},
            "recessive": {"parent_offspring": [], "prs": [], "genomic_distance": []},
            "codominant": {"parent_offspring": [], "prs": [], "genomic_distance": []}
        }
        
        for run in self.runs_data:
            # Collect fitness evolution data
            if "fitness_evolution" in run:
                fitness_evolution.append(run["fitness_evolution"])
            
            # Collect diploid statistics
            if "diploid_offspring" in run:
                for model, model_data in run["diploid_offspring"].items():
                    if model in diploid_stats:
                        # Handle both list of organisms and analyzed data
                        if isinstance(model_data, list) and model_data:
                            if isinstance(model_data[0], dict) and "fitness" in model_data[0]:
                                # List of organism dicts
                                for org in model_data:
                                    diploid_stats[model]["parent_offspring"].append([
                                        org["avg_parent_fitness"],
                                        org["fitness"]
                                    ])
                                    
                                    if "prs" in org:
                                        diploid_stats[model]["prs"].append([org["prs"], org["fitness"]])
                                    
                                    if "genomic_distance" in org:
                                        diploid_stats[model]["genomic_distance"].append([org["genomic_distance"], org["fitness"]])
                        elif isinstance(model_data, dict):
                            # Analyzed data format
                            for metric in ["parent_offspring", "prs", "genomic_distance"]:
                                if metric in model_data:
                                    diploid_stats[model][metric].extend(model_data[metric])
        
        # Calculate summary statistics
        self.summary_data = {
            "fitness_evolution": self._calculate_fitness_summary(fitness_evolution),
            "diploid_stats": self._calculate_diploid_summary(diploid_stats)
        }
    
    def _calculate_fitness_summary(self, fitness_data: List[List[float]]) -> Dict[str, Any]:
        """Calculate summary statistics for fitness evolution."""
        if not fitness_data:
            return {}
            
        # Convert to numpy array for easier calculation
        fitness_array = np.array(fitness_data)
        
        return {
            "mean": np.mean(fitness_array, axis=0).tolist(),
            "std": np.std(fitness_array, axis=0).tolist(),
            "min": np.min(fitness_array, axis=0).tolist(),
            "max": np.max(fitness_array, axis=0).tolist()
        }
    
    def _calculate_diploid_summary(self, diploid_stats: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, Any]]:
        """Calculate summary statistics for diploid metrics."""
        summary = {}
        
        for model, metrics in diploid_stats.items():
            summary[model] = {}
            for metric, data in metrics.items():
                if data:
                    # Convert data to numpy arrays for calculations
                    data_array = np.array(data)
                    if len(data_array.shape) > 1 and data_array.shape[1] == 2:  # For paired data (e.g., parent-offspring)
                        x_data = data_array[:, 0]
                        y_data = data_array[:, 1]
                        # Filter out None values
                        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                        if np.any(valid_mask):
                            x_valid = x_data[valid_mask]
                            y_valid = y_data[valid_mask]
                            summary[model][metric] = {
                                "x_mean": float(np.mean(x_valid)),
                                "x_std": float(np.std(x_valid)) if len(x_valid) > 1 else 0.0,
                                "y_mean": float(np.mean(y_valid)),
                                "y_std": float(np.std(y_valid)) if len(y_valid) > 1 else 0.0
                            }
                    else:  # For single value data
                        valid_data = data_array[~np.isnan(data_array)] if data_array.dtype == float else data_array
                        if len(valid_data) > 0:
                            summary[model][metric] = {
                                "mean": float(np.mean(valid_data)),
                                "std": float(np.std(valid_data)) if len(valid_data) > 1 else 0.0
                            }
        
        return summary
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all collected data and summary statistics.
        
        Returns:
            Dictionary containing all run data and summary statistics
        """
        return {
            "runs": self.runs_data,
            "summary": self.summary_data
        }


class SimulationAnalyzer:
    """Analyzes results from a single simulation run."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the analyzer.
        
        Args:
            logger: Optional logger for tracking analysis progress
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_simulation(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results from a single simulation run.
        
        Args:
            simulation_result: Dictionary containing simulation results
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "fitness_evolution": self._analyze_fitness_evolution(simulation_result),
            "diploid_offspring": self._analyze_diploid_offspring(simulation_result),
            "population_stats": self._analyze_population_stats(simulation_result)
        }
        
        return analysis
    
    def _analyze_fitness_evolution(self, simulation_result: Dict[str, Any]) -> List[float]:
        """Analyze fitness evolution across generations."""
        if "simulation" in simulation_result:
            simulation = simulation_result["simulation"]
            if hasattr(simulation, 'generation_stats') and simulation.generation_stats:
                return [stat['avg_fitness'] for stat in simulation.generation_stats]
        
        # Fallback to fitness_history if available
        if "fitness_history" in simulation_result:
            return [np.mean(gen_fitness) for gen_fitness in simulation_result["fitness_history"]]
            
        return []
    
    def _analyze_diploid_offspring(self, simulation_result: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
        """Analyze diploid offspring statistics."""
        if "diploid_offspring" not in simulation_result:
            return {}
            
        diploid_data = simulation_result["diploid_offspring"]
        analysis = {}
        
        for model in ["dominant", "recessive", "codominant"]:
            if model in diploid_data:
                organisms = diploid_data[model]
                # Convert objects to dicts if needed
                if organisms and hasattr(organisms[0], 'to_dict'):
                    organisms = [org.to_dict() for org in organisms]
                analysis[model] = self._analyze_diploid_model(organisms)
        
        return analysis
    
    def _analyze_diploid_model(self, organisms: List[Dict[str, Any]]) -> Dict[str, List]:
        """Analyze statistics for a specific diploid model.
        
        Args:
            organisms: List of diploid offspring (as dicts)
            
        Returns:
            Dictionary containing lists of parent-offspring relationships,
            PRS scores, and genomic distances
        """
        if not organisms:
            return {}
        
        parent_offspring = []
        prs = []
        genomic_distance = []
        
        for org in organisms:
            # Parent-offspring fitness relationship
            parent_offspring.append((
                org["avg_parent_fitness"],
                org["fitness"]
            ))
            
            # PRS analysis
            if "prs" in org:
                prs_val = org["prs"]
            elif "allele1" in org and "allele2" in org:
                prs_val = (np.sum(org["allele1"]) + np.sum(org["allele2"])) / 2
            else:
                prs_val = None
            
            prs.append((prs_val, org["fitness"]))
            
            # Genomic distance
            if "genomic_distance" in org:
                distance = org["genomic_distance"]
            elif "allele1" in org and "allele2" in org:
                distance = np.sum(np.array(org["allele1"]) != np.array(org["allele2"]))
            else:
                distance = None
                
            genomic_distance.append((distance, org["fitness"]))
            
        return {
            "parent_offspring": parent_offspring,
            "prs": prs,
            "genomic_distance": genomic_distance
        }
    
    def _analyze_population_stats(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze population-level statistics."""
        if "simulation" not in simulation_result:
            return {}
            
        simulation = simulation_result["simulation"]
        
        if hasattr(simulation, 'generation_stats') and simulation.generation_stats:
            stats = {
                "size": [stat['population_size'] for stat in simulation.generation_stats],
                "fitness_mean": [stat['avg_fitness'] for stat in simulation.generation_stats],
                "fitness_std": [stat['std_fitness'] for stat in simulation.generation_stats]
            }
            return stats
        
        return {}


class MultiRunAnalyzer:
    """Analyzes results from multiple simulation runs."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the analyzer.
        
        Args:
            logger: Optional logger for tracking analysis progress
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def aggregate_runs(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate and analyze results from multiple runs.
        
        Args:
            all_results: List of analysis results from individual runs
            
        Returns:
            Dictionary containing aggregated statistics
        """
        if not all_results:
            return {}
            
        aggregated = {
            "haploid_evolution": self._aggregate_haploid_evolution(all_results),
            "diploid_models": self._aggregate_diploid_models(all_results),
            "population_stats": self._aggregate_population_stats(all_results)
        }
        
        return aggregated
    
    def _aggregate_haploid_evolution(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate haploid evolution statistics."""
        fitness_evolution = []
        
        for result in results:
            if "fitness_evolution" in result and result["fitness_evolution"]:
                fitness_evolution.append(result["fitness_evolution"])
        
        if not fitness_evolution:
            return {}
            
        # Convert to numpy array for easier calculation
        fitness_array = np.array(fitness_evolution)
        
        # Calculate statistics only if we have data
        stats = {}
        
        if len(fitness_array) > 0:
            stats["mean_fitness"] = {
                "mean": float(np.mean(fitness_array)),
                "std": float(np.std(fitness_array)) if len(fitness_array) > 1 else 0.0
            }
            
            if len(fitness_array.shape) > 1 and fitness_array.shape[1] > 0:
                max_fitness = np.max(fitness_array, axis=1)
                stats["max_fitness"] = {
                    "mean": float(np.mean(max_fitness)),
                    "std": float(np.std(max_fitness)) if len(max_fitness) > 1 else 0.0
                }
        
        return stats
    
    def _aggregate_diploid_models(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Aggregate statistics for different diploid models."""
        model_stats = {
            "dominant": self._aggregate_diploid_model_stats(results, "dominant"),
            "recessive": self._aggregate_diploid_model_stats(results, "recessive"),
            "codominant": self._aggregate_diploid_model_stats(results, "codominant")
        }
        
        return model_stats
    
    def _aggregate_diploid_model_stats(self, results: List[Dict[str, Any]], model: str) -> Dict[str, Dict[str, float]]:
        """Aggregate statistics for a specific diploid model."""
        all_offspring_fitness = []
        all_fitness_improvement = []
        
        for result in results:
            if "diploid_offspring" in result and model in result["diploid_offspring"]:
                model_data = result["diploid_offspring"][model]
                if "parent_offspring" in model_data and model_data["parent_offspring"]:
                    for parent_fit, offspring_fit in model_data["parent_offspring"]:
                        all_offspring_fitness.append(offspring_fit)
                        all_fitness_improvement.append(offspring_fit - parent_fit)
        
        if not all_offspring_fitness:
            return {}
            
        # Convert to numpy arrays for calculations
        offspring_fitness_array = np.array(all_offspring_fitness)
        fitness_improvement_array = np.array(all_fitness_improvement)
        
        # Calculate statistics only if we have data
        stats = {}
        
        if len(offspring_fitness_array) > 0:
            stats["avg_offspring_fitness"] = {
                "mean": float(np.mean(offspring_fitness_array)),
                "std": float(np.std(offspring_fitness_array)) if len(offspring_fitness_array) > 1 else 0.0
            }
        
        if len(fitness_improvement_array) > 0:
            stats["fitness_improvement"] = {
                "mean": float(np.mean(fitness_improvement_array)),
                "std": float(np.std(fitness_improvement_array)) if len(fitness_improvement_array) > 1 else 0.0
            }
        
        return stats
    
    def _aggregate_population_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate population-level statistics."""
        all_sizes = []
        all_fitness_means = []
        all_fitness_stds = []
        
        for result in results:
            if "population_stats" in result:
                stats = result["population_stats"]
                if "size" in stats:
                    all_sizes.extend(stats["size"])
                if "fitness_mean" in stats:
                    all_fitness_means.extend(stats["fitness_mean"])
                if "fitness_std" in stats:
                    all_fitness_stds.extend(stats["fitness_std"])
        
        if not all_sizes:
            return {}
            
        return {
            "population_size": {
                "mean": float(np.mean(all_sizes)),
                "std": float(np.std(all_sizes))
            },
            "fitness_mean": {
                "mean": float(np.mean(all_fitness_means)),
                "std": float(np.std(all_fitness_means))
            },
            "fitness_std": {
                "mean": float(np.mean(all_fitness_stds)),
                "std": float(np.std(all_fitness_stds))
            }
        }


def calculate_regression_stats(x: np.ndarray, y: np.ndarray, degree: int = 1) -> Dict[str, float]:
    """
    Calculate regression statistics for x and y data.
    
    Args:
        x: Independent variable
        y: Dependent variable
        degree: Polynomial degree
        
    Returns:
        Dictionary containing regression statistics
    """
    try:
        # Reshape x if needed
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x)
        
        # Fit regression
        model = LinearRegression()
        model.fit(x_poly, y)
        
        # Calculate predictions
        y_pred = model.predict(x_poly)
        
        # Calculate R²
        r2 = model.score(x_poly, y)
        
        # Calculate p-value using F-test
        n = len(y)
        p = len(model.coef_)
        if n > p and r2 < 1.0:
            f_stat = (r2 / (p - 1)) / ((1 - r2) / (n - p))
            p_value = 1 - stats.f.cdf(f_stat, p - 1, n - p)
        else:
            p_value = 1.0
        
        return {
            "r2": float(r2),
            "p_value": float(p_value),
            "coefficients": [float(c) for c in model.coef_],
            "intercept": float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
        }
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Regression calculation failed: {e}")
        return {
            "r2": 0.0,
            "p_value": 1.0,
            "coefficients": [],
            "intercept": 0.0
        }


def save_analysis_results(results: Dict[str, Any], output_dir: Union[str, Path], filename: str) -> None:
    """Save analysis results to a JSON file.
    
    Args:
        results: Dictionary containing analysis results
        output_dir: Directory to save the results
        filename: Name of the output file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif hasattr(obj, 'to_dict'):  # Handle objects with to_dict method
            return convert_numpy(obj.to_dict())
        return obj
    
    # Convert and save
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)


class GeneticAnalysis:
    """Class for analyzing genetic simulation data."""
    
    def __init__(self, output_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the analysis tools.
    
        Args:
            output_dir: Directory to save plots and analysis results
            logger: Logger for tracking analysis progress
        """
        self.output_dir = Path(output_dir) if output_dir else Path("analysis_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_genetic_distance(self, genome1: np.ndarray, genome2: np.ndarray) -> float:
        """Calculate Hamming distance between two genomes."""
        return calculate_genomic_distance(genome1, genome2)
    
    def calculate_polygenic_risk_score(self, genome: np.ndarray) -> float:
        """Calculate polygenic risk score as sum of genome values."""
        return calculate_polygenic_risk_score(genome)
    
    def prepare_cross_data(self, crosses: List[dict]) -> pd.DataFrame:
        """
        Prepare data from crosses for analysis.
        Args:
            crosses: List of diploid offspring data dictionaries
        Returns:
            DataFrame with analysis-ready data
        """
        data = []
        for cross in crosses:
            # Calculate metrics
            if "genomic_distance" in cross:
                genetic_distance = cross["genomic_distance"]
            elif "allele1" in cross and "allele2" in cross:
                genetic_distance = self.calculate_genetic_distance(
                    np.array(cross["allele1"]), np.array(cross["allele2"])
                )
            else:
                genetic_distance = None
                
            mean_parent_fitness = cross.get("avg_parent_fitness", 0.0)
            
            if "prs" in cross:
                mean_parent_prs = cross["prs"]
            elif "allele1" in cross and "allele2" in cross:
                mean_parent_prs = (np.sum(cross["allele1"]) + np.sum(cross["allele2"])) / 2
            else:
                mean_parent_prs = None
                
            data.append({
                'genetic_distance': genetic_distance,
                'mean_parent_fitness': mean_parent_fitness,
                'mean_parent_prs': mean_parent_prs,
                'offspring_fitness': cross["fitness"],
                'fitness_model': cross.get("fitness_model", "unknown")
            })
        return pd.DataFrame(data)
    
    def fit_regression(self, X: np.ndarray, y: np.ndarray, degree: int = 1) -> Tuple[float, float, float, np.ndarray]:
        """
        Fit polynomial regression and return statistics.
        
        Args:
            X: Independent variable
            y: Dependent variable
            degree: Polynomial degree (1 for linear, 2 for quadratic)
            
        Returns:
            Tuple of (r2_score, coefficients, p_value, predicted_values)
        """
        stats = calculate_regression_stats(X, y, degree)
        
        # Calculate predictions for plotting
        X_reshaped = X.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_reshaped)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        return stats["r2"], stats["coefficients"], stats["p_value"], y_pred
    
    def plot_relationship(self, data: pd.DataFrame, x_col: str, y_col: str, 
                         title: str, filename: str, degree: int = 2) -> None:
        """
        Create scatter plot with regression lines.
        
        Args:
            data: DataFrame containing the data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Plot title
            filename: Output filename
            degree: Maximum polynomial degree to fit
        """
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        sns.scatterplot(data=data, x=x_col, y=y_col, alpha=0.5)
        
        # Fit and plot regressions
        X = data[x_col].dropna().values
        y = data[y_col].dropna().values
        
        if len(X) > 1 and len(y) > 1:
            # Linear regression
            r2_linear, coef_linear, p_linear, y_pred_linear = self.fit_regression(X, y, degree=1)
            
            # Quadratic regression
            r2_quad, coef_quad, p_quad, y_pred_quad = self.fit_regression(X, y, degree=2)
            
            # Sort X for smooth line plotting
            sort_idx = np.argsort(X)
            X_sorted = X[sort_idx]
            
            # Plot regression lines
            plt.plot(X_sorted, y_pred_linear[sort_idx], 'r-', 
                    label=f'Linear (R²={r2_linear:.3f}, p={p_linear:.3e})')
            plt.plot(X_sorted, y_pred_quad[sort_idx], 'g--', 
                    label=f'Quadratic (R²={r2_quad:.3f}, p={p_quad:.3e})')
            
            # Log statistics
            self.logger.info(f"\nAnalysis for {title}:")
            self.logger.info(f"Linear regression: R²={r2_linear:.3f}, p={p_linear:.3e}")
            self.logger.info(f"Quadratic regression: R²={r2_quad:.3f}, p={p_quad:.3e}")
        
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()  # Important: close figure to prevent memory leaks
    
    def analyze_crosses(self, crosses: List[Dict[str, Any]], run_id: Optional[str] = None) -> None:
        """
        Perform complete analysis of crosses.
        
        Args:
            crosses: List of diploid offspring data dictionaries
            run_id: Optional identifier for this run
        """
        # Prepare data
        data = self.prepare_cross_data(crosses)
        
        # Create run-specific directory if needed
        if run_id:
            run_dir = self.output_dir / f"run_{run_id}"
            run_dir.mkdir(exist_ok=True)
            original_output_dir = self.output_dir
            self.output_dir = run_dir
        
        try:
            # Generate plots
            if not data.empty:
                self.plot_relationship(
                    data, 'genetic_distance', 'offspring_fitness',
                    'Genetic Distance vs Offspring Fitness',
                    'genetic_distance_vs_fitness.png'
                )
                
                self.plot_relationship(
                    data, 'mean_parent_fitness', 'offspring_fitness',
                    'Mean Parental Fitness vs Offspring Fitness',
                    'parent_fitness_vs_offspring.png'
                )
                
                self.plot_relationship(
                    data, 'mean_parent_prs', 'offspring_fitness',
                    'Mean Parental PRS vs Offspring Fitness',
                    'parent_prs_vs_offspring.png'
                )
                
                # Save raw data
                data.to_csv(self.output_dir / 'cross_analysis_data.csv', index=False)
                
                # Log summary statistics
                self.logger.info("\nSummary Statistics:")
                self.logger.info(data.describe().to_string())
            else:
                self.logger.warning("No data available for cross analysis")
                
        finally:
            # Restore original output directory
            if run_id:
                self.output_dir = original_output_dir