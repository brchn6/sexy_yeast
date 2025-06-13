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
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path

from core_models import Organism, DiploidOrganism

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
        return np.sum(genome1 != genome2)
    
    def calculate_polygenic_risk_score(self, genome: np.ndarray) -> float:
        """Calculate polygenic risk score as sum of genome values."""
        return np.sum(genome)
    
    def prepare_cross_data(self, crosses: List[DiploidOrganism]) -> pd.DataFrame:
        """
        Prepare data from crosses for analysis.
        
        Args:
            crosses: List of DiploidOrganism objects from crosses
            
        Returns:
            DataFrame with analysis-ready data
        """
        data = []
        for cross in crosses:
            parent1 = cross.parent1
            parent2 = cross.parent2
            
            # Calculate metrics
            genetic_distance = self.calculate_genetic_distance(parent1.genome, parent2.genome)
            mean_parent_fitness = (parent1.fitness + parent2.fitness) / 2
            mean_parent_prs = (self.calculate_polygenic_risk_score(parent1.genome) + 
                             self.calculate_polygenic_risk_score(parent2.genome)) / 2
            
            data.append({
                'genetic_distance': genetic_distance,
                'mean_parent_fitness': mean_parent_fitness,
                'mean_parent_prs': mean_parent_prs,
                'offspring_fitness': cross.fitness,
                'parent1_id': parent1.id,
                'parent2_id': parent2.id,
                'offspring_id': cross.id
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
        # Reshape X if needed
        X = X.reshape(-1, 1)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Fit regression
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Calculate predictions
        y_pred = model.predict(X_poly)
        
        # Calculate R²
        r2 = model.score(X_poly, y)
        
        # Calculate p-value using F-test
        n = len(y)
        p = len(model.coef_)
        f_stat = (r2 / (p - 1)) / ((1 - r2) / (n - p))
        p_value = 1 - stats.f.cdf(f_stat, p - 1, n - p)
        
        return r2, model.coef_, p_value, y_pred
    
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
        X = data[x_col].values
        y = data[y_col].values
        
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
        
        plt.title(title)
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.ylabel(y_col.replace('_', ' ').title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log statistics
        self.logger.info(f"\nAnalysis for {title}:")
        self.logger.info(f"Linear regression: R²={r2_linear:.3f}, p={p_linear:.3e}")
        self.logger.info(f"Quadratic regression: R²={r2_quad:.3f}, p={p_quad:.3e}")
    
    def analyze_crosses(self, crosses: List[DiploidOrganism], run_id: Optional[str] = None) -> None:
        """
        Perform complete analysis of crosses.
        
        Args:
            crosses: List of DiploidOrganism objects from crosses
            run_id: Optional identifier for this run
        """
        # Prepare data
        data = self.prepare_cross_data(crosses)
        
        # Create run-specific directory if needed
        if run_id:
            run_dir = self.output_dir / f"run_{run_id}"
            run_dir.mkdir(exist_ok=True)
            self.output_dir = run_dir
        
        # Generate plots
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
