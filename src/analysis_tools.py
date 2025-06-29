#!/usr/bin/env python3
"""
Analysis tools for genetic simulation data - FIXED VERSION.

This module provides functions for statistical analysis and data collection
of genetic simulation results, with proper data format handling for visualization.
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
import os

# Import utility functions from core_models to avoid circular imports
from core_models import calculate_genomic_distance, calculate_diploid_prs, calculate_polygenic_risk_score


class SimulationDataCollector:
    """Collects and manages data from multiple simulation runs - FIXED VERSION."""
    
    def __init__(self):
        """Initialize the data collector."""
        self.runs_data = []
        self.summary_data = {}
        self.logger = logging.getLogger(__name__)
    
    def add_run_data(self, run_data: Dict[str, Any]) -> None:
        """Add data from a single run to the collector - FIXED VERSION.
        
        Args:
            run_data: Dictionary containing run results and analysis
        """
        # Deep copy and process the diploid offspring data
        processed_run_data = self._process_run_data(run_data)
        self.runs_data.append(processed_run_data)
        self._update_summary()
        
        # Debug logging
        self.logger.debug(f"Added run data. Total runs: {len(self.runs_data)}")
        if "diploid_offspring" in processed_run_data:
            for model, organisms in processed_run_data["diploid_offspring"].items():
                self.logger.debug(f"Run has {len(organisms)} {model} organisms")
    
    def _process_run_data(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process run data to ensure proper format for visualization."""
        processed_data = run_data.copy()
        
        # Convert diploid offspring to proper dictionary format if needed
        if "diploid_offspring" in run_data:
            diploid_offspring = run_data["diploid_offspring"]
            processed_diploid = {}
            
            for model, organisms in diploid_offspring.items():
                processed_organisms = []
                
                if isinstance(organisms, list):
                    for org in organisms:
                        if hasattr(org, 'to_dict'):
                            # Object format - convert to dict
                            org_dict = org.to_dict()
                        elif isinstance(org, dict):
                            # Already dict format
                            org_dict = org.copy()
                        else:
                            # Try to extract data manually
                            try:
                                org_dict = {
                                    "fitness": org.calculate_fitness() if hasattr(org, 'calculate_fitness') else 0,
                                    "parent1_fitness": org.parent1.calculate_fitness() if hasattr(org, 'parent1') else 0,
                                    "parent2_fitness": org.parent2.calculate_fitness() if hasattr(org, 'parent2') else 0,
                                    "avg_parent_fitness": ((org.parent1.calculate_fitness() + org.parent2.calculate_fitness()) / 2) if hasattr(org, 'parent1') and hasattr(org, 'parent2') else 0,
                                    "prs": org.calculate_prs() if hasattr(org, 'calculate_prs') else 0,
                                    "genomic_distance": org.calculate_genomic_distance() if hasattr(org, 'calculate_genomic_distance') else 0,
                                    "fitness_model": getattr(org, 'fitness_model', model)
                                }
                            except Exception as e:
                                self.logger.warning(f"Failed to process organism: {e}")
                                continue
                        
                        # Ensure all required fields are present
                        if "avg_parent_fitness" not in org_dict and "parent1_fitness" in org_dict and "parent2_fitness" in org_dict:
                            org_dict["avg_parent_fitness"] = (org_dict["parent1_fitness"] + org_dict["parent2_fitness"]) / 2
                        
                        processed_organisms.append(org_dict)
                
                processed_diploid[model] = processed_organisms
                self.logger.debug(f"Processed {len(processed_organisms)} {model} organisms")
            
            processed_data["diploid_offspring"] = processed_diploid
        
        return processed_data
    
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
                for model, organisms in run["diploid_offspring"].items():
                    if model in diploid_stats and organisms:
                        for org in organisms:
                            try:
                                # Extract parent-offspring relationships
                                parent_fit = org.get("avg_parent_fitness", 0)
                                offspring_fit = org.get("fitness", 0)
                                diploid_stats[model]["parent_offspring"].append([parent_fit, offspring_fit])
                                
                                # Extract PRS relationships
                                prs_val = org.get("prs", 0)
                                diploid_stats[model]["prs"].append([prs_val, offspring_fit])
                                
                                # Extract genomic distance relationships
                                genomic_dist = org.get("genomic_distance", 0)
                                diploid_stats[model]["genomic_distance"].append([genomic_dist, offspring_fit])
                                
                            except Exception as e:
                                self.logger.debug(f"Failed to extract summary data: {e}")
                                continue
        
        # Calculate summary statistics
        self.summary_data = {
            "fitness_evolution": self._calculate_fitness_summary(fitness_evolution),
            "diploid_stats": self._calculate_diploid_summary(diploid_stats)
        }
        
        # Log summary
        for model in diploid_stats:
            count = len(diploid_stats[model]["parent_offspring"])
            self.logger.debug(f"Summary: {count} total data points for {model}")
    
    def _calculate_fitness_summary(self, fitness_data: List[List[float]]) -> Dict[str, Any]:
        """Calculate summary statistics for fitness evolution."""
        if not fitness_data:
            return {}
            
        # Convert to numpy array for easier calculation
        max_len = max(len(fd) for fd in fitness_data) if fitness_data else 0
        if max_len == 0:
            return {}
        
        # Pad shorter sequences with NaN
        fitness_array = np.full((len(fitness_data), max_len), np.nan)
        for i, fd in enumerate(fitness_data):
            fitness_array[i, :len(fd)] = fd
        
        return {
            "mean": np.nanmean(fitness_array, axis=0).tolist(),
            "std": np.nanstd(fitness_array, axis=0).tolist(),
            "min": np.nanmin(fitness_array, axis=0).tolist(),
            "max": np.nanmax(fitness_array, axis=0).tolist()
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
                    if len(data_array.shape) > 1 and data_array.shape[1] == 2:  # For paired data
                        x_data = data_array[:, 0]
                        y_data = data_array[:, 1]
                        # Filter out None/NaN values
                        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                        if np.any(valid_mask):
                            x_valid = x_data[valid_mask]
                            y_valid = y_data[valid_mask]
                            summary[model][metric] = {
                                "x_mean": float(np.mean(x_valid)),
                                "x_std": float(np.std(x_valid)) if len(x_valid) > 1 else 0.0,
                                "y_mean": float(np.mean(y_valid)),
                                "y_std": float(np.std(y_valid)) if len(y_valid) > 1 else 0.0,
                                "count": int(len(x_valid))
                            }
                    else:  # For single value data
                        valid_data = data_array[~np.isnan(data_array)] if data_array.dtype == float else data_array
                        if len(valid_data) > 0:
                            summary[model][metric] = {
                                "mean": float(np.mean(valid_data)),
                                "std": float(np.std(valid_data)) if len(valid_data) > 1 else 0.0,
                                "count": int(len(valid_data))
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
    """Analyzes results from a single simulation run - IMPROVED VERSION."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the analyzer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_simulation(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results from a single simulation run - IMPROVED VERSION."""
        analysis = {
            "fitness_evolution": self._analyze_fitness_evolution(simulation_result),
            "diploid_offspring": self._analyze_diploid_offspring(simulation_result),
            "population_stats": self._analyze_population_stats(simulation_result),
            "polynomial_fits": self._analyze_polynomial_fits(simulation_result)
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
    
    def _analyze_diploid_offspring(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze diploid offspring statistics - IMPROVED to keep original organisms."""
        if "diploid_offspring" not in simulation_result:
            return {}
            
        diploid_data = simulation_result["diploid_offspring"]
        analysis = {}
        
        for model in ["dominant", "recessive", "codominant"]:
            if model in diploid_data:
                organisms = diploid_data[model]
                # Keep the original organisms for visualization
                analysis[model] = organisms
        
        return analysis
    
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

    def _analyze_polynomial_fits(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate polynomial fits for different relationships."""
        if "diploid_offspring" not in simulation_result:
            return {}

        fits = {}
        for model, organisms in simulation_result["diploid_offspring"].items():
            if not organisms:
                continue

            # Extract data
            prs_values = []
            genomic_distances = []
            parent_fitness = []
            offspring_fitness = []

            for org in organisms:
                if isinstance(org, dict):
                    prs_values.append(org.get("prs", 0))
                    genomic_distances.append(org.get("genomic_distance", 0))
                    parent_fitness.append(org.get("avg_parent_fitness", 0))
                    offspring_fitness.append(org.get("fitness", 0))

            # Calculate polynomial fits
            fits[model] = {
                "prs_fit": self._calculate_fit(prs_values, offspring_fitness),
                "genomic_distance_fit": self._calculate_fit(genomic_distances, offspring_fitness),
                "parent_offspring_fit": self._calculate_fit(parent_fitness, offspring_fitness)
            }

        return fits

    def _calculate_fit(self, x: List[float], y: List[float], degree: int = 2) -> Dict[str, Any]:
        """Calculate polynomial fit statistics."""
        try:
            x_array = np.array(x)
            y_array = np.array(y)
            
            # Remove any NaN or infinite values
            mask = np.isfinite(x_array) & np.isfinite(y_array)
            x_clean = x_array[mask]
            y_clean = y_array[mask]
            
            if len(x_clean) < 3:
                return {
                    "coefficients": [],
                    "r2": 0.0,
                    "p_value": 1.0,
                    "equation": "insufficient data"
                }
            
            # Fit polynomial
            coeffs = np.polyfit(x_clean, y_clean, degree)
            
            # Calculate predictions
            poly = np.poly1d(coeffs)
            y_pred = poly(x_clean)
            
            # Calculate R²
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate p-value using F-test
            n = len(x_clean)
            p = len(coeffs)
            if n > p and r2 < 1.0:
                f_stat = (r2 / (p - 1)) / ((1 - r2) / (n - p))
                p_value = 1 - stats.f.cdf(f_stat, p - 1, n - p)
            else:
                p_value = 1.0
            
            # Create equation string
            terms = []
            for i, coef in enumerate(coeffs):
                power = degree - i
                if power == 0:
                    terms.append(f"{coef:.4f}")
                elif power == 1:
                    terms.append(f"{coef:.4f}x")
                else:
                    terms.append(f"{coef:.4f}x^{power}")
            equation = " + ".join(terms)
            
            return {
                "coefficients": coeffs.tolist(),
                "r2": float(r2),
                "p_value": float(p_value),
                "equation": equation
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate polynomial fit: {e}")
            return {
                "coefficients": [],
                "r2": 0.0,
                "p_value": 1.0,
                "equation": "error in calculation"
            }


class MultiRunAnalyzer:
    """Analyzes results from multiple simulation runs - IMPROVED VERSION."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the analyzer."""
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
            
        # Handle different length sequences
        max_len = max(len(fe) for fe in fitness_evolution)
        fitness_array = np.full((len(fitness_evolution), max_len), np.nan)
        
        for i, fe in enumerate(fitness_evolution):
            fitness_array[i, :len(fe)] = fe
        
        stats = {}
        
        if len(fitness_array) > 0:
            # Calculate final fitness statistics
            final_fitness = [fe[-1] for fe in fitness_evolution if fe]
            if final_fitness:
                stats["mean_fitness"] = {
                    "mean": float(np.mean(final_fitness)),
                    "std": float(np.std(final_fitness)) if len(final_fitness) > 1 else 0.0
                }
                
                stats["max_fitness"] = {
                    "mean": float(np.mean(final_fitness)),  # For final generation
                    "std": float(np.std(final_fitness)) if len(final_fitness) > 1 else 0.0
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
        all_parent_fitness = []
        
        for result in results:
            if "diploid_offspring" in result and model in result["diploid_offspring"]:
                organisms = result["diploid_offspring"][model]
                
                for org in organisms:
                    try:
                        if hasattr(org, 'calculate_fitness'):
                            # Object format
                            offspring_fit = org.calculate_fitness()
                            parent_fit = (org.parent1.calculate_fitness() + org.parent2.calculate_fitness()) / 2
                        elif isinstance(org, dict):
                            # Dictionary format
                            offspring_fit = org.get("fitness", 0)
                            parent_fit = org.get("avg_parent_fitness", 0)
                        else:
                            continue
                        
                        all_offspring_fitness.append(offspring_fit)
                        all_fitness_improvement.append(offspring_fit - parent_fit)
                        all_parent_fitness.append(parent_fit)
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to aggregate organism data: {e}")
                        continue
        
        if not all_offspring_fitness:
            return {}
            
        # Calculate statistics
        stats = {}
        
        if all_offspring_fitness:
            stats["avg_offspring_fitness"] = {
                "mean": float(np.mean(all_offspring_fitness)),
                "std": float(np.std(all_offspring_fitness)) if len(all_offspring_fitness) > 1 else 0.0
            }
        
        if all_fitness_improvement:
            stats["fitness_improvement"] = {
                "mean": float(np.mean(all_fitness_improvement)),
                "std": float(np.std(all_fitness_improvement)) if len(all_fitness_improvement) > 1 else 0.0
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
                "std": float(np.std(all_sizes)) if len(all_sizes) > 1 else 0.0
            },
            "fitness_mean": {
                "mean": float(np.mean(all_fitness_means)),
                "std": float(np.std(all_fitness_means)) if len(all_fitness_means) > 1 else 0.0
            },
            "fitness_std": {
                "mean": float(np.mean(all_fitness_stds)),
                "std": float(np.std(all_fitness_stds)) if len(all_fitness_stds) > 1 else 0.0
            }
        }


def calculate_regression_stats(x: np.ndarray, y: np.ndarray, degree: int = 1) -> Dict[str, float]:
    """Calculate regression statistics for x and y data."""
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
    """Save analysis results to a JSON file."""
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
        """Initialize the analysis tools."""
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
        """Prepare data from crosses for analysis."""
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
        """Fit polynomial regression and return statistics."""
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
        """Create scatter plot with regression lines."""
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
        """Perform complete analysis of crosses."""
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


class CrossDataAnalyzer:
    """
    Analyzes and visualizes cross data from evolutionary simulations.
    """
    
    def __init__(self, cross_data: List[Dict[str, Any]]):
        """
        Initialize the analyzer with cross data.
        
        Args:
            cross_data: List of dictionaries containing cross data
        """
        self.df = pd.DataFrame(cross_data)
    
    def plot_fitness_vs_parent_fitness(self, output_dir: str) -> None:
        """Plot offspring fitness vs average parent fitness for each inheritance mode."""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with regression
        sns.lmplot(
            data=self.df,
            x="avg_parent_fitness",
            y="offspring_fitness",
            hue="inheritance_mode",
            order=2,  # 2nd degree polynomial
            scatter_kws={"alpha": 0.3},
            height=8,
            aspect=1.5
        )
        
        plt.title("Offspring Fitness vs Average Parent Fitness")
        plt.xlabel("Average Parent Fitness")
        plt.ylabel("Offspring Fitness")
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "fitness_vs_parent_fitness.png"))
        plt.close()
    
    def plot_fitness_vs_genomic_distance(self, output_dir: str) -> None:
        """Plot offspring fitness vs genomic distance for each inheritance mode."""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with regression
        sns.lmplot(
            data=self.df,
            x="genomic_distance",
            y="offspring_fitness",
            hue="inheritance_mode",
            order=2,  # 2nd degree polynomial
            scatter_kws={"alpha": 0.3},
            height=8,
            aspect=1.5
        )
        
        plt.title("Offspring Fitness vs Genomic Distance")
        plt.xlabel("Genomic Distance")
        plt.ylabel("Offspring Fitness")
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "fitness_vs_genomic_distance.png"))
        plt.close()
    
    def plot_fitness_vs_prs(self, output_dir: str) -> None:
        """Plot offspring fitness vs combined PRS for each inheritance mode."""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with regression
        sns.lmplot(
            data=self.df,
            x="combined_prs",
            y="offspring_fitness",
            hue="inheritance_mode",
            order=2,  # 2nd degree polynomial
            scatter_kws={"alpha": 0.3},
            height=8,
            aspect=1.5
        )
        
        plt.title("Offspring Fitness vs Combined PRS")
        plt.xlabel("Combined PRS")
        plt.ylabel("Offspring Fitness")
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "fitness_vs_prs.png"))
        plt.close()
    
    def plot_fitness_distributions(self, output_dir: str) -> None:
        """Plot fitness distributions for each inheritance mode."""
        plt.figure(figsize=(12, 8))
        
        # Violin plot
        sns.violinplot(
            data=self.df,
            x="inheritance_mode",
            y="offspring_fitness"
        )
        
        plt.title("Offspring Fitness Distribution by Inheritance Mode")
        plt.xlabel("Inheritance Mode")
        plt.ylabel("Offspring Fitness")
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "fitness_distributions.png"))
        plt.close()
    
    def calculate_regression_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate regression statistics for each inheritance mode.
        
        Returns:
            Dictionary containing R² and p-values for each mode
        """
        stats_by_mode = {}
        
        for mode in self.df["inheritance_mode"].unique():
            mode_data = self.df[self.df["inheritance_mode"] == mode]
            
            # Fit 2nd degree polynomial
            x = mode_data["avg_parent_fitness"]
            y = mode_data["offspring_fitness"]
            coeffs = np.polyfit(x, y, 2)
            
            # Calculate R²
            y_pred = np.polyval(coeffs, x)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            # Calculate p-value using F-test
            n = len(x)
            p = 1 - stats.f.cdf(
                (r2 / 2) / ((1 - r2) / (n - 3)),
                2,  # degrees of freedom for regression
                n - 3  # degrees of freedom for residuals
            )
            
            stats_by_mode[mode] = {
                "r2": r2,
                "p_value": p
            }
        
        return stats_by_mode
    
    def save_analysis_results(self, output_dir: str) -> None:
        """
        Save all analysis results to the specified directory.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        self.plot_fitness_vs_parent_fitness(output_dir)
        self.plot_fitness_vs_genomic_distance(output_dir)
        self.plot_fitness_vs_prs(output_dir)
        self.plot_fitness_distributions(output_dir)
        
        # Calculate and save statistics
        stats = self.calculate_regression_stats()
        with open(os.path.join(output_dir, "regression_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save raw data
        self.df.to_csv(os.path.join(output_dir, "cross_data.csv"), index=False)
