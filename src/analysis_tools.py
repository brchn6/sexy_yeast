#!/usr/bin/env python3
"""
Analysis tools for evolutionary simulation results.

This module provides functions for analyzing simulation data, calculating
statistics, and extracting insights from evolutionary runs.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import json
import logging
from pathlib import Path
import scipy.stats

from core_models import DiploidOrganism, Environment


def calculate_genomic_distance(genome1: np.ndarray, genome2: np.ndarray) -> int:
    """
    Calculate Hamming distance between two genomes.
    
    Args:
        genome1, genome2: Binary genome arrays (-1 or 1)
        
    Returns:
        Number of positions where genomes differ
    """
    if len(genome1) != len(genome2):
        raise ValueError("Genomes must be the same length")
    
    return np.sum(genome1 != genome2)


def calculate_polygenic_risk_score(genome: np.ndarray) -> float:
    """
    Calculate a simple Polygenic Risk Score (PRS) for a genome.
    
    Args:
        genome: Array of values representing the genome
        
    Returns:
        The calculated PRS score
    """
    # Convert from -1/1 encoding to 0/1 encoding for SNP risk alleles
    # Consider 1 as the "risk allele"
    risk_alleles = (genome + 1) / 2  # Converts -1 to 0 and 1 to 1
    return np.sum(risk_alleles)


def calculate_diploid_prs(diploid_organism: DiploidOrganism) -> float:
    """
    Calculate PRS for a diploid organism with proper dominance handling.
    
    Args:
        diploid_organism: The diploid organism to calculate PRS for
        
    Returns:
        The calculated PRS score
    """
    if diploid_organism.fitness_model == "codominant":
        # For codominant model, average the two alleles
        prs1 = calculate_polygenic_risk_score(diploid_organism.allele1)
        prs2 = calculate_polygenic_risk_score(diploid_organism.allele2)
        return (prs1 + prs2) / 2
    else:
        # For dominant/recessive models, use the effective genome
        effective_genome = diploid_organism._get_effective_genome()
        
        # Handle potential float values in effective genome
        if np.any(np.abs(effective_genome) < 0.5):
            # Convert values close to 0 to 0, others to 1
            binary_genome = np.where(np.abs(effective_genome) < 0.5, -1, 1)
            return calculate_polygenic_risk_score(binary_genome)
        else:
            return calculate_polygenic_risk_score(effective_genome.astype(int))


def calculate_regression_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression statistics.
    
    Args:
        x, y: Arrays of x and y values
        
    Returns:
        Dictionary containing regression statistics
    """
    # Handle edge cases
    if len(x) <= 1 or len(np.unique(x)) <= 1:
        return {
            'linear_r_squared': 0,
            'linear_slope': 0,
            'linear_intercept': 0,
            'linear_p_value': 1,
            'quadratic_r_squared': 0,
            'quadratic_coeffs': [0, 0, 0]
        }
    
    try:
        # Linear regression
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        linear_r_squared = r_value ** 2
        
        # Quadratic regression
        quadratic_r_squared = 0
        quadratic_coeffs = [0, 0, 0]
        
        if len(np.unique(x)) >= 3:  # Need at least 3 unique points for quadratic
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)
            y_mean = np.mean(y)
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            
            if ss_tot > 0:
                quadratic_r_squared = 1 - (ss_res / ss_tot)
                quadratic_coeffs = coeffs.tolist()
        
        return {
            'linear_r_squared': linear_r_squared,
            'linear_slope': slope,
            'linear_intercept': intercept,
            'linear_p_value': p_value,
            'quadratic_r_squared': quadratic_r_squared,
            'quadratic_coeffs': quadratic_coeffs
        }
        
    except Exception as e:
        logging.warning(f"Regression analysis failed: {e}")
        return {
            'linear_r_squared': 0,
            'linear_slope': 0,
            'linear_intercept': 0,
            'linear_p_value': 1,
            'quadratic_r_squared': 0,
            'quadratic_coeffs': [0, 0, 0],
            'error': str(e)
        }


class SimulationAnalyzer:
    """
    Comprehensive analyzer for simulation results.
    
    This class provides methods to analyze single simulation runs and
    extract detailed statistics about evolutionary dynamics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the analyzer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_simulation(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a complete simulation result.
        
        Args:
            simulation_result: Result from SimulationRunner.run_complete_simulation
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        simulation = simulation_result["simulation"]
        diploid_offspring = simulation_result["diploid_offspring"]
        
        # Analyze haploid evolution
        haploid_stats = self._analyze_haploid_evolution(simulation)
        
        # Analyze diploid offspring
        diploid_stats = self._analyze_diploid_offspring(diploid_offspring)
        
        # Calculate final generation distribution
        final_distribution = self._calculate_final_distribution(simulation)
        
        return {
            "haploid_evolution": haploid_stats,
            "diploid_analysis": diploid_stats,
            "final_distribution": final_distribution,
            "parameters": simulation_result["parameters"]
        }
    
    def _analyze_haploid_evolution(self, simulation) -> Dict[str, Any]:
        """Analyze the haploid evolution phase."""
        generation_stats = simulation.generation_stats
        
        if not generation_stats:
            return {"error": "No generation statistics available"}
        
        initial_fitness = generation_stats[0]['avg_fitness']
        final_fitness = generation_stats[-1]['avg_fitness']
        fitness_improvement = ((final_fitness - initial_fitness) / initial_fitness) * 100
        
        return {
            "generations": len(generation_stats),
            "initial_fitness": initial_fitness,
            "final_fitness": final_fitness,
            "fitness_improvement_percent": fitness_improvement,
            "final_population_size": generation_stats[-1]['population_size'],
            "final_fitness_std": generation_stats[-1]['std_fitness']
        }
    
    def _analyze_diploid_offspring(self, diploid_offspring: Dict[str, List[DiploidOrganism]]) -> Dict[str, Any]:
        """Analyze diploid offspring for each fitness model."""
        results = {}
        
        for model, organisms in diploid_offspring.items():
            if not organisms:
                results[model] = {"error": "No organisms for this model"}
                continue
            
            # Extract data
            parent_fitness = np.array([org.avg_parent_fitness for org in organisms])
            offspring_fitness = np.array([org.fitness for org in organisms])
            genomic_distances = np.array([
                calculate_genomic_distance(org.allele1, org.allele2) 
                for org in organisms
            ])
            
            # Basic statistics
            model_stats = {
                "count": len(organisms),
                "avg_parent_fitness": np.mean(parent_fitness),
                "avg_offspring_fitness": np.mean(offspring_fitness),
                "fitness_improvement": np.mean(offspring_fitness - parent_fitness),
                "avg_genomic_distance": np.mean(genomic_distances),
                "std_genomic_distance": np.std(genomic_distances),
                "min_genomic_distance": np.min(genomic_distances),
                "max_genomic_distance": np.max(genomic_distances)
            }
            
            # Regression analyses
            model_stats["parent_fitness_regression"] = calculate_regression_stats(
                parent_fitness, offspring_fitness
            )
            model_stats["distance_regression"] = calculate_regression_stats(
                genomic_distances, offspring_fitness
            )
            
            # PRS analysis
            try:
                prs_scores = [calculate_diploid_prs(org) for org in organisms]
                model_stats["prs_analysis"] = {
                    "mean_prs": np.mean(prs_scores),
                    "std_prs": np.std(prs_scores),
                    "prs_fitness_regression": calculate_regression_stats(
                        np.array(prs_scores), offspring_fitness
                    )
                }
            except Exception as e:
                model_stats["prs_analysis"] = {"error": str(e)}
            
            results[model] = model_stats
        
        return results
    
    def _calculate_final_distribution(self, simulation) -> Dict[str, Any]:
        """Calculate fitness distribution of the final generation."""
        # Get fitness values for all organisms in the final generation
        final_generation = simulation.get_last_generation()
        
        if not final_generation:
            return {"error": "No final generation data"}
        
        fitness_values = np.array([org.fitness for org in final_generation])
        
        # Calculate distribution statistics
        distribution = {
            "count": len(fitness_values),
            "mean": np.mean(fitness_values),
            "median": np.median(fitness_values),
            "std": np.std(fitness_values),
            "min": np.min(fitness_values),
            "max": np.max(fitness_values),
            "percentile_25": np.percentile(fitness_values, 25),
            "percentile_75": np.percentile(fitness_values, 75)
        }
        
        # Add distribution shape statistics
        if len(fitness_values) > 2:
            distribution["skewness"] = scipy.stats.skew(fitness_values)
            distribution["kurtosis"] = scipy.stats.kurtosis(fitness_values)
        
        # Add histogram data
        hist, bin_edges = np.histogram(fitness_values, bins=min(10, len(fitness_values) // 2))
        distribution["histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }
        
        return distribution


class MultiRunAnalyzer:
    """
    Analyzer for multiple simulation runs.
    
    This class aggregates and compares results across multiple simulation
    runs to identify patterns and calculate statistical significance.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the multi-run analyzer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def aggregate_runs(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate analysis results from multiple runs.
        
        Args:
            analysis_results: List of analysis results from individual runs
            
        Returns:
            Aggregated statistics across all runs
        """
        if not analysis_results:
            return {"error": "No analysis results provided"}
        
        # Extract metrics for aggregation
        metrics_data = []
        for run_idx, result in enumerate(analysis_results):
            run_metrics = self._extract_run_metrics(result, run_idx + 1)
            metrics_data.append(run_metrics)
        
        # Create DataFrame for easy analysis
        metrics_df = pd.DataFrame(metrics_data)
        
        # Calculate aggregated statistics
        aggregated = {
            "num_runs": len(analysis_results),
            "haploid_evolution": self._aggregate_haploid_stats(metrics_df),
            "diploid_models": self._aggregate_diploid_stats(metrics_df),
            "cross_run_correlations": self._calculate_cross_run_correlations(metrics_df)
        }
        
        return aggregated
    
    def _extract_run_metrics(self, analysis_result: Dict[str, Any], run_id: int) -> Dict[str, Any]:
        """Extract key metrics from a single run analysis."""
        metrics = {"run_id": run_id}
        
        # Haploid evolution metrics
        haploid = analysis_result.get("haploid_evolution", {})
        metrics.update({
            "initial_fitness": haploid.get("initial_fitness", 0),
            "final_fitness": haploid.get("final_fitness", 0),
            "fitness_improvement_percent": haploid.get("fitness_improvement_percent", 0),
            "final_population_size": haploid.get("final_population_size", 0)
        })
        
        # Diploid model metrics
        diploid = analysis_result.get("diploid_analysis", {})
        for model in ["dominant", "recessive", "codominant"]:
            model_data = diploid.get(model, {})
            if "error" not in model_data:
                prefix = f"{model}_"
                metrics.update({
                    f"{prefix}avg_offspring_fitness": model_data.get("avg_offspring_fitness", 0),
                    f"{prefix}fitness_improvement": model_data.get("fitness_improvement", 0),
                    f"{prefix}avg_genomic_distance": model_data.get("avg_genomic_distance", 0),
                })
                
                # Regression statistics
                parent_reg = model_data.get("parent_fitness_regression", {})
                metrics.update({
                    f"{prefix}parent_r2": parent_reg.get("linear_r_squared", 0),
                    f"{prefix}parent_slope": parent_reg.get("linear_slope", 0),
                    f"{prefix}parent_p_value": parent_reg.get("linear_p_value", 1)
                })
                
                distance_reg = model_data.get("distance_regression", {})
                metrics.update({
                    f"{prefix}distance_r2": distance_reg.get("linear_r_squared", 0),
                    f"{prefix}distance_slope": distance_reg.get("linear_slope", 0),
                    f"{prefix}distance_p_value": distance_reg.get("linear_p_value", 1)
                })
        
        return metrics
    
    def _aggregate_haploid_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate haploid evolution statistics."""
        numeric_cols = ["initial_fitness", "final_fitness", "fitness_improvement_percent"]
        return {
            col: {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max()
            }
            for col in numeric_cols if col in df.columns
        }
    
    def _aggregate_diploid_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate diploid model statistics."""
        results = {}
        
        for model in ["dominant", "recessive", "codominant"]:
            model_cols = [col for col in df.columns if col.startswith(f"{model}_")]
            if not model_cols:
                continue
            
            model_stats = {}
            for col in model_cols:
                metric_name = col.replace(f"{model}_", "")
                model_stats[metric_name] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max()
                }
            
            results[model] = model_stats
        
        return results
    
    def _calculate_cross_run_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations between metrics across runs."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            
            # Convert to dictionary for JSON serialization
            return correlation_matrix.to_dict()
        except Exception as e:
            self.logger.warning(f"Could not calculate correlations: {e}")
            return {"error": str(e)}


def save_analysis_results(results: Dict[str, Any], output_path: Path,
                         filename: str = "analysis_results.json") -> None:
    """
    Save analysis results to a JSON file.
    
    Args:
        results: Analysis results dictionary
        output_path: Directory to save the file
        filename: Name of the output file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=_numpy_json_encoder)


def _numpy_json_encoder(obj):
    """Custom JSON encoder for NumPy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')