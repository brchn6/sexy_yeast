#!/usr/bin/env python3
"""
Visualization tools for evolutionary simulation results 

This module provides publication-quality visualizations with:
- Large, clear text suitable for articles
- 2nd degree polynomial regression lines
- R² values and FDR-corrected p-values
- Professional formatting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import scipy.stats
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set up matplotlib for high-quality output
plt.switch_backend('Agg')
plt.rcParams.update({
    'font.size': 16,           # Larger base font
    'axes.titlesize': 20,      # Bigger titles
    'axes.labelsize': 18,      # Bigger axis labels
    'xtick.labelsize': 16,     # Bigger tick labels
    'ytick.labelsize': 16,     # Bigger tick labels
    'legend.fontsize': 16,     # Bigger legend
    'figure.titlesize': 24,    # Bigger figure title
    'lines.linewidth': 3,      # Thicker lines
    'lines.markersize': 10,    # Bigger markers
    'font.weight': 'bold',     # Bold text
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.dpi': 300,         # High resolution
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans']
})

# Import utility functions from core_models to avoid circular imports
from core_models import calculate_genomic_distance, calculate_diploid_prs


class SimulationVisualizer:
    """
    Creates publication-quality visualizations of simulation results.
    
    This class handles all plotting functionality with a focus on clarity,
    aesthetics, and scientific communication with statistical analysis.
    """
    
    def __init__(self, style: str = "whitegrid", palette: str = "deep"):
        """
        Initialize the visualizer with aesthetic settings.
        
        Args:
            style: Seaborn style to use
            palette: Color palette for plots
        """
        # Set up plotting style
        sns.set_style(style)
        sns.set_palette(palette)
        
        # Define consistent colors for different models
        self.model_colors = {
            "dominant": "#2E86AB",
            "recessive": "#A23B72", 
            "codominant": "#F18F01"
        }
        
        # Colors for tree visualization
        self.tree_colors = {
            "edges": "#808080",
            "nodes": "viridis",  # colormap name
            "background": "white"
        }
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking for FDR correction
        self.p_values = []
        self.test_names = []
    
    def _calculate_polynomial_fit(self, x_data: np.ndarray, y_data: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, float, float]:
        """
        Calculate polynomial fit with R² and p-value.
        
        Returns:
            fitted_line: x,y values for the fitted polynomial
            r_squared: R² value
            p_value: significance p-value
        """
        if len(x_data) < 3 or len(y_data) < 3:
            return np.array([]), 0.0, 1.0
        
        try:
            # Remove any NaN or infinite values
            mask = np.isfinite(x_data) & np.isfinite(y_data)
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            if len(x_clean) < 3:
                return np.array([]), 0.0, 1.0
            
            # Fit polynomial
            coeffs = np.polyfit(x_clean, y_clean, degree)
            poly_func = np.poly1d(coeffs)
            
            # Generate smooth line for plotting
            x_smooth = np.linspace(np.min(x_clean), np.max(x_clean), 100)
            y_fitted = poly_func(x_smooth)
            
            # Calculate R²
            y_pred = poly_func(x_clean)
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate p-value using F-test
            n = len(x_clean)
            k = degree  # number of parameters (excluding intercept)
            if n > k + 1:
                mse_res = ss_res / (n - k - 1)
                mse_reg = (ss_tot - ss_res) / k
                if mse_res > 0:
                    f_stat = mse_reg / mse_res
                    p_value = 1 - scipy.stats.f.cdf(f_stat, k, n - k - 1)
                else:
                    p_value = 0.0
            else:
                p_value = 1.0
            
            return np.column_stack([x_smooth, y_fitted]), r_squared, p_value
            
        except Exception as e:
            self.logger.debug(f"Polynomial fit failed: {e}")
            return np.array([]), 0.0, 1.0
    
    def _add_regression_line(self, ax, x_data: np.ndarray, y_data: np.ndarray, 
                           label: str, color: str = 'red') -> Tuple[float, float]:
        """Add 2nd degree polynomial regression with R² and p-values."""
        fitted_data, r_squared, p_value = self._calculate_polynomial_fit(x_data, y_data, degree=2)
        
        if len(fitted_data) > 0:
            # Plot the fitted line
            ax.plot(fitted_data[:, 0], fitted_data[:, 1], '--', 
                   linewidth=4, alpha=0.9, color=color,
                   label=f'{label} (R²={r_squared:.3f})')
            
            # Store p-value for FDR correction
            self.p_values.append(p_value)
            self.test_names.append(f"{label}_fit")
            
            # Add text box with statistics
            if r_squared > 0.01:  # Only show if meaningful relationship
                if p_value < 0.001:
                    p_text = 'p < 0.001***'
                elif p_value < 0.01:
                    p_text = f'p = {p_value:.3f}**'
                elif p_value < 0.05:
                    p_text = f'p = {p_value:.3f}*'
                else:
                    p_text = f'p = {p_value:.3f}ns'
                
                stats_text = f'R² = {r_squared:.3f}\n{p_text}'
                
                # Position text box
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                                alpha=0.9, edgecolor='black', linewidth=1),
                       verticalalignment='top', fontsize=14, fontweight='bold')
        
        return r_squared, p_value
    
    def _apply_fdr_correction(self) -> Dict[str, float]:
        """Apply FDR correction to all stored p-values."""
        if not self.p_values:
            return {}
        
        # Apply FDR correction using Benjamini-Hochberg
        rejected, p_adjusted, _, _ = multipletests(self.p_values, method='fdr_bh')
        
        # Create dictionary of corrected p-values
        fdr_results = {}
        for i, test_name in enumerate(self.test_names):
            fdr_results[test_name] = {
                'p_value': self.p_values[i],
                'p_adjusted': p_adjusted[i],
                'significant': rejected[i]
            }
        
        return fdr_results
    
    def _save_figure(self, fig: plt.Figure, output_dir: Path, filename: str) -> None:
        """Save a figure to the specified directory and close it properly."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            fig.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            self.logger.debug(f"Saved figure: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save FDR results: {e}")

    def _save_figure(self, fig: plt.Figure, output_dir: Path, filename: str) -> None:
        """Save a high-quality figure."""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            fig.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Saved multi-run figure: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save multi-run figure {filename}: {e}")
        finally:
            plt.close(fig)  # Always close the figure to prevent memory leaks figure {filename}: {e}")
    
    def plot_fitness_evolution(self, simulation, output_path: Path,
                             filename: str = "fitness_evolution.png") -> None:
        """Plot fitness evolution over generations with individual trajectories and statistics."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        try:
            # Plot 1: Individual trajectories with statistical overlays
            self._plot_individual_trajectories(ax1, simulation)
            self._add_statistical_overlays(ax1, simulation)
            
            # Plot 2: Fitness distribution over time (sampled generations)
            self._plot_fitness_distributions(ax2, simulation)
            
            plt.tight_layout(pad=3.0)
            self._save_figure(fig, output_path, filename)
        except Exception as e:
            self.logger.error(f"Failed to plot fitness evolution: {e}")
            plt.close(fig)
    
    def plot_relationship_tree(self, organisms: List[Any], output_path: Path,
                             filename: str = "relationship_tree.png",
                             max_organisms: int = 1000,
                             layout_type: str = "hierarchical") -> None:
        """Create an evolutionary relationship tree visualization."""
        if not organisms:
            self.logger.warning("No organisms provided for relationship tree")
            return
        
        # Limit organisms for performance while maintaining representation
        if len(organisms) > max_organisms:
            organisms = self._sample_organisms_for_tree(organisms, max_organisms)
        
        # Create the network graph
        graph = self._build_relationship_graph(organisms)
        
        if not graph.nodes():
            self.logger.warning("No organisms to plot in relationship tree")
            return
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(18, 12))
        
        try:
            # Choose layout based on type
            if layout_type == "hierarchical":
                pos = self._create_hierarchical_layout(organisms, graph)
            else:
                pos = nx.spring_layout(graph, k=1, iterations=50)
            
            # Draw the tree
            self._draw_relationship_tree(graph, pos, ax)
            
            # Style the plot
            self._style_tree_plot(ax, len(organisms))
            
            # Save the figure
            self._save_figure(fig, output_path, filename)
        except Exception as e:
            self.logger.error(f"Failed to create relationship tree: {e}")
            plt.close(fig)
    
    def _sample_organisms_for_tree(self, organisms: List[Any], max_count: int) -> List[Any]:
        """Intelligently sample organisms for tree visualization."""
        if len(organisms) <= max_count:
            return organisms
        
        # Group by generation
        gen_organisms = defaultdict(list)
        for org in organisms:
            gen_organisms[org.generation].append(org)
        
        # Sample from each generation proportionally
        sampled = []
        total_gens = len(gen_organisms)
        
        for gen, orgs in gen_organisms.items():
            # Take more from recent generations, fewer from early ones
            gen_weight = (gen + 1) / sum(range(1, total_gens + 2))
            sample_size = max(1, int(max_count * gen_weight))
            sample_size = min(sample_size, len(orgs))
            
            # Sample highest and lowest fitness organisms
            orgs_sorted = sorted(orgs, key=lambda x: x.fitness)
            if sample_size >= len(orgs):
                sampled.extend(orgs)
            else:
                # Take from extremes and middle
                indices = np.linspace(0, len(orgs)-1, sample_size, dtype=int)
                sampled.extend([orgs_sorted[i] for i in indices])
        
        return sampled[:max_count]
    
    def _build_relationship_graph(self, organisms: List[Any]) -> nx.DiGraph:
        """Build a directed graph representing parent-child relationships."""
        graph = nx.DiGraph()
        
        # Create a mapping of organism IDs for quick lookup
        org_dict = {org.id: org for org in organisms}
        
        # Add nodes and edges
        for org in organisms:
            # Add node with attributes
            graph.add_node(
                org.id, 
                generation=org.generation, 
                fitness=org.fitness,
                organism=org  # Store reference for later use
            )
            
            # Add edge from parent if parent exists and is in our dataset
            if hasattr(org, 'parent_id') and org.parent_id and org.parent_id in org_dict:
                graph.add_edge(org.parent_id, org.id)
        
        return graph
    
    def _create_hierarchical_layout(self, organisms: List[Any], graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create a hierarchical layout positioning organisms by generation and fitness."""
        pos = {}
        
        # Group organisms by generation
        gen_organisms = defaultdict(list)
        for org in organisms:
            gen_organisms[org.generation].append(org)
        
        # Position nodes generation by generation
        for gen, orgs in gen_organisms.items():
            # Sort by fitness for consistent vertical positioning
            orgs_sorted = sorted(orgs, key=lambda x: x.fitness)
            
            # Calculate positions
            for i, org in enumerate(orgs_sorted):
                # Horizontal position based on generation
                x = gen
                
                # Vertical position based on fitness with some spread
                base_y = org.fitness
                
                # Add vertical offset to prevent overlapping
                if len(orgs_sorted) > 1:
                    # Spread organisms vertically within their fitness range
                    fitness_range = max(o.fitness for o in orgs_sorted) - min(o.fitness for o in orgs_sorted)
                    spread_factor = 0.1 * fitness_range if fitness_range > 0 else 0.1
                    offset = (i - len(orgs_sorted)/2) * spread_factor / len(orgs_sorted)
                    y = base_y + offset
                else:
                    y = base_y
                
                pos[org.id] = (x, y)
        
        return pos
    
    def _draw_relationship_tree(self, graph: nx.DiGraph, pos: Dict[str, Tuple[float, float]], ax) -> None:
        """Draw the relationship tree with proper styling."""
        # Extract fitness values for coloring
        fitness_values = [graph.nodes[node]['fitness'] for node in graph.nodes()]
        
        if not fitness_values:
            return
        
        # Draw edges first (so they appear behind nodes)
        if graph.edges():
            nx.draw_networkx_edges(
                graph, pos, ax=ax,
                arrows=True,
                edge_color=self.tree_colors["edges"],
                alpha=0.4,
                arrowsize=20,
                width=1.2,
                arrowstyle='->'
            )
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=80,
            node_color=fitness_values,
            cmap=plt.cm.get_cmap(self.tree_colors["nodes"]),
            alpha=0.8,
            edgecolors='white',
            linewidths=1
        )
        
        # Add colorbar for fitness
        if nodes:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Fitness', fontsize=16, labelpad=20, fontweight='bold')
            cbar.ax.tick_params(labelsize=14)
    
    def _style_tree_plot(self, ax, num_organisms: int) -> None:
        """Apply consistent styling to the tree plot."""
        ax.set_xlabel('Generation', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel('Fitness', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_title(f'Evolutionary Relationship Tree\n({num_organisms:,} organisms)', 
                    fontsize=20, fontweight='bold', pad=25)
        
        # Style the axes
        ax.tick_params(labelsize=16, width=2, length=8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax.set_facecolor(self.tree_colors["background"])
        
        # Add some padding around the plot
        ax.margins(0.05)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    def _plot_individual_trajectories(self, ax, simulation) -> None:
        """Plot individual organism fitness trajectories."""
        if hasattr(simulation, 'individual_fitness') and simulation.individual_fitness:
            for org_id, fitness_data in simulation.individual_fitness.items():
                if fitness_data:
                    generations, fitness = zip(*fitness_data)
                    ax.plot(generations, fitness, '-', alpha=0.2, linewidth=1.5, color='lightblue')
        
        ax.set_xlabel('Generation', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel('Fitness', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_title('Individual Fitness Trajectories', fontsize=20, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    def _add_statistical_overlays(self, ax, simulation) -> None:
        """Add mean, max, and min fitness lines to the plot."""
        if hasattr(simulation, 'generation_stats') and simulation.generation_stats:
            stats = simulation.generation_stats
            generations = [s['generation'] for s in stats]
            avg_fitness = [s['avg_fitness'] for s in stats]
            max_fitness = [s['max_fitness'] for s in stats]
            min_fitness = [s['min_fitness'] for s in stats]
            
            ax.plot(generations, avg_fitness, 'k-', linewidth=4, label='Average', zorder=10)
            ax.plot(generations, max_fitness, 'g-', linewidth=3, label='Maximum', zorder=9)
            ax.plot(generations, min_fitness, 'r-', linewidth=3, label='Minimum', zorder=9)
            
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=16, 
                     loc='best', framealpha=0.9)
    
    def _plot_fitness_distributions(self, ax, simulation) -> None:
        """Plot fitness distributions for sampled generations."""
        if not hasattr(simulation, 'generation_stats') or not simulation.generation_stats:
            return
            
        max_gen = max(s['generation'] for s in simulation.generation_stats)
        sample_gens = range(1, max_gen + 1, max(1, max_gen // 10))
        
        fitness_data = []
        gen_labels = []
        
        if hasattr(simulation, 'individual_fitness') and simulation.individual_fitness:
            for gen in sample_gens:
                gen_fitness = []
                for org_id, fit_data in simulation.individual_fitness.items():
                    gen_fitness.extend([f for g, f in fit_data if g == gen])
                
                if gen_fitness:  # Only add if we have data
                    fitness_data.append(gen_fitness)
                    gen_labels.append(str(gen))
        
        if fitness_data:
            bp = ax.boxplot(fitness_data, labels=gen_labels, patch_artist=True,
                           widths=0.8, medianprops=dict(linewidth=3))
            
            # Color the boxes with a gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
                patch.set_linewidth(2)
        
        ax.set_xlabel('Generation (sampled)', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel('Fitness Distribution', fontsize=18, fontweight='bold', labelpad=15)
        ax.set_title('Fitness Distribution Over Time', fontsize=20, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    def plot_parent_offspring_relationships(self, data: Union[Dict[str, Any], Dict[str, List]], 
                                          output_dir: Path, mating_strategy: Optional[str] = None) -> None:
        """Plot parent-offspring relationships - improved to handle both single run and multi-run data."""
        try:
            # Detect data format and route accordingly
            if self._is_single_run_data(data):
                if not mating_strategy:
                    self.logger.error("Mating strategy required for individual run plotting")
                    return
                self._plot_individual_run_relationships(data, output_dir, mating_strategy)
            else:
                self._plot_multi_run_relationships(data, output_dir)
        except Exception as e:
            self.logger.error(f"Failed to plot parent-offspring relationships: {e}")

    def _is_single_run_data(self, data: Union[Dict[str, Any], Dict[str, List]]) -> bool:
        """Determine if data is from a single run or aggregated runs."""
        # Check if data has the structure of diploid offspring from a single run
        for key, value in data.items():
            if key in ["dominant", "recessive", "codominant"] and isinstance(value, list):
                return True
        return False

    def _plot_individual_run_relationships(self, diploid_offspring: Dict[str, List], 
                                         output_dir: Path, mating_strategy: str) -> None:
        """Plot relationships for a single run."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f"Parent-Offspring Relationships ({mating_strategy})", 
                        fontsize=24, fontweight='bold', y=0.98)
            
            # Extract data from diploid organisms (objects or dicts)
            extracted_data = self._extract_organism_data(diploid_offspring)
            
            # Plot parent vs offspring fitness
            self._plot_parent_vs_offspring_fitness_individual(axes[0], extracted_data)
            
            # Plot PRS vs offspring fitness
            self._plot_prs_vs_offspring_fitness_individual(axes[1], extracted_data)
            
            # Plot genomic distance vs offspring fitness
            self._plot_genomic_distance_vs_offspring_fitness_individual(axes[2], extracted_data)
            
            # Adjust layout and save
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_dir, f"parent_offspring_relationships_{mating_strategy}.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot individual run relationships: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def _extract_organism_data(self, diploid_offspring: Dict[str, List]) -> Dict[str, Dict[str, List[float]]]:
        """Extract data from diploid organisms (handles both objects and dicts)."""
        extracted_data = {}
        
        for model, organisms in diploid_offspring.items():
            if not organisms:
                continue
                
            extracted_data[model] = {
                'parent_fitness': [],
                'offspring_fitness': [],
                'prs_values': [],
                'genomic_distances': []
            }
            
            for org in organisms:
                try:
                    # Handle both object and dictionary formats
                    if hasattr(org, 'calculate_fitness'):
                        # Object format
                        parent_fit = (org.parent1.calculate_fitness() + org.parent2.calculate_fitness()) / 2
                        offspring_fit = org.calculate_fitness()
                        prs_val = org.calculate_prs()
                        genomic_dist = org.calculate_genomic_distance()
                    elif isinstance(org, dict):
                        # Dictionary format
                        parent_fit = org.get('avg_parent_fitness', 0)
                        offspring_fit = org.get('fitness', 0)
                        prs_val = org.get('prs', 0)
                        genomic_dist = org.get('genomic_distance', 0)
                    else:
                        self.logger.warning(f"Unknown organism format: {type(org)}")
                        continue
                    
                    extracted_data[model]['parent_fitness'].append(parent_fit)
                    extracted_data[model]['offspring_fitness'].append(offspring_fit)
                    extracted_data[model]['prs_values'].append(prs_val)
                    extracted_data[model]['genomic_distances'].append(genomic_dist)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract data from organism: {e}")
                    continue
        
        return extracted_data

    def _plot_parent_vs_offspring_fitness_individual(self, ax: plt.Axes, extracted_data: Dict[str, Dict[str, List[float]]]) -> None:
        """Plot parent vs offspring fitness for individual run."""
        ax.set_title("Parent vs Offspring Fitness", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Average Parent Fitness", fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
        
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        for i, (model, color) in enumerate(zip(["dominant", "recessive", "codominant"], colors)):
            if model in extracted_data and extracted_data[model]['parent_fitness']:
                x_data = np.array(extracted_data[model]['parent_fitness'])
                y_data = np.array(extracted_data[model]['offspring_fitness'])
                
                ax.scatter(x_data, y_data,
                          label=model.capitalize(), color=color, alpha=0.7, s=80, edgecolors='black', linewidths=1)
                
                # Add 2nd degree polynomial fit
                if len(x_data) > 2:
                    self._add_regression_line(ax, x_data, y_data, model, color)
        
        ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    def _plot_prs_vs_offspring_fitness_individual(self, ax: plt.Axes, extracted_data: Dict[str, Dict[str, List[float]]]) -> None:
        """Plot PRS vs offspring fitness for individual run."""
        ax.set_title("PRS vs Offspring Fitness", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Polygenic Risk Score", fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
        
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        for i, (model, color) in enumerate(zip(["dominant", "recessive", "codominant"], colors)):
            if model in extracted_data and extracted_data[model]['prs_values']:
                x_data = np.array(extracted_data[model]['prs_values'])
                y_data = np.array(extracted_data[model]['offspring_fitness'])
                
                ax.scatter(x_data, y_data,
                          label=model.capitalize(), color=color, alpha=0.7, s=80, edgecolors='black', linewidths=1)
                
                # Add 2nd degree polynomial fit
                if len(x_data) > 2:
                    self._add_regression_line(ax, x_data, y_data, model, color)
        
        ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    def _plot_genomic_distance_vs_offspring_fitness_individual(self, ax: plt.Axes, extracted_data: Dict[str, Dict[str, List[float]]]) -> None:
        """Plot genomic distance vs offspring fitness for individual run."""
        ax.set_title("Genomic Distance vs Offspring Fitness", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Genomic Distance", fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
        
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        for i, (model, color) in enumerate(zip(["dominant", "recessive", "codominant"], colors)):
            if model in extracted_data and extracted_data[model]['genomic_distances']:
                x_data = np.array(extracted_data[model]['genomic_distances'])
                y_data = np.array(extracted_data[model]['offspring_fitness'])
                
                ax.scatter(x_data, y_data,
                          label=model.capitalize(), color=color, alpha=0.7, s=80, edgecolors='black', linewidths=1)
                
                # Add 2nd degree polynomial fit
                if len(x_data) > 2:
                    self._add_regression_line(ax, x_data, y_data, model, color)
        
        ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    def _plot_multi_run_relationships(self, data: Dict[str, Any], output_dir: Path) -> None:
        """Plot relationships for multi-run summary."""
        summary = data.get("summary", {})
        if not summary:
            self.logger.error("No summary data found for plotting")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle("Parent-Offspring Relationships Across Multiple Runs", 
                    fontsize=24, fontweight='bold', y=0.98)
        
        try:
            # Plot parent vs offspring fitness
            self._plot_parent_vs_offspring_fitness_summary(axes[0], summary)
            
            # Plot PRS vs offspring fitness
            self._plot_prs_vs_offspring_fitness_summary(axes[1], summary)
            
            # Plot genomic distance vs offspring fitness
            self._plot_genomic_distance_vs_offspring_fitness_summary(axes[2], summary)
            
            # Adjust layout and save
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_dir, "parent_offspring_relationships.png")
        except Exception as e:
            self.logger.error(f"Failed to plot multi-run relationships: {e}")
            plt.close(fig)
    
    def _plot_parent_vs_offspring_fitness_summary(self, ax: plt.Axes, summary: Dict[str, Any]) -> None:
        """Plot parent vs offspring fitness for summary data."""
        ax.set_title("Parent vs Offspring Fitness", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Parent Fitness", fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
        
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        for model, color in zip(["dominant", "recessive", "codominant"], colors):
            model_data = summary.get("diploid_stats", {}).get(model, {}).get("parent_offspring", {})
            if model_data:
                x_mean = model_data.get("x_mean")
                x_std = model_data.get("x_std")
                y_mean = model_data.get("y_mean")
                y_std = model_data.get("y_std")
                
                if all(v is not None for v in [x_mean, x_std, y_mean, y_std]):
                    ax.errorbar(x_mean, y_mean, 
                              xerr=x_std, yerr=y_std,
                              label=model.capitalize(), color=color,
                              fmt='o', alpha=0.8, markersize=12, linewidth=3,
                              capsize=8, capthick=3)
        
        ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    def _plot_prs_vs_offspring_fitness_summary(self, ax: plt.Axes, summary: Dict[str, Any]) -> None:
        """Plot PRS vs offspring fitness for summary data."""
        ax.set_title("PRS vs Offspring Fitness", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Polygenic Risk Score", fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
        
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        for model, color in zip(["dominant", "recessive", "codominant"], colors):
            model_data = summary.get("diploid_stats", {}).get(model, {}).get("prs", {})
            if model_data:
                x_mean = model_data.get("x_mean")
                x_std = model_data.get("x_std")
                y_mean = model_data.get("y_mean")
                y_std = model_data.get("y_std")
                
                if all(v is not None for v in [x_mean, x_std, y_mean, y_std]):
                    ax.errorbar(x_mean, y_mean, 
                              xerr=x_std, yerr=y_std,
                              label=model.capitalize(), color=color,
                              fmt='o', alpha=0.8, markersize=12, linewidth=3,
                              capsize=8, capthick=3)
        
        ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    def _plot_genomic_distance_vs_offspring_fitness_summary(self, ax: plt.Axes, summary: Dict[str, Any]) -> None:
        """Plot genomic distance vs offspring fitness for summary data."""
        ax.set_title("Genomic Distance vs Offspring Fitness", fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Genomic Distance", fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
        
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        for model, color in zip(["dominant", "recessive", "codominant"], colors):
            model_data = summary.get("diploid_stats", {}).get(model, {}).get("genomic_distance", {})
            if model_data:
                x_mean = model_data.get("x_mean")
                x_std = model_data.get("x_std")
                y_mean = model_data.get("y_mean")
                y_std = model_data.get("y_std")
                
                if all(v is not None for v in [x_mean, x_std, y_mean, y_std]):
                    ax.errorbar(x_mean, y_mean, 
                              xerr=x_std, yerr=y_std,
                              label=model.capitalize(), color=color,
                              fmt='o', alpha=0.8, markersize=12, linewidth=3,
                              capsize=8, capthick=3)
        
        ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    def plot_min_max_parent_offspring_fitness(self, diploid_offspring: Dict[str, List],
                                            output_path: Path, mating_strategy: str,
                                            filename_prefix: str = "parent_offspring_fitness") -> None:
        """Plot min/max parent fitness vs offspring fitness."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f"Parent-Offspring Fitness Relationships ({mating_strategy})", 
                        fontsize=24, fontweight='bold', y=0.98)
            
            # Extract data first
            extracted_data = self._extract_organism_data(diploid_offspring)
            
            # Plot for each model
            for i, (model, ax) in enumerate(zip(["dominant", "recessive", "codominant"], axes)):
                if model in extracted_data and extracted_data[model]['parent_fitness']:
                    self._plot_single_model_relationship_improved(ax, extracted_data[model], model)
            
            # Adjust layout and save
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_path, f"{filename_prefix}_{mating_strategy}.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot min/max parent offspring fitness: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def _plot_single_model_relationship_improved(self, ax, model_data: Dict[str, List[float]], model: str) -> None:
        """Plot relationship for a single model - improved version."""
        parent_fitness = np.array(model_data['parent_fitness'])
        offspring_fitness = np.array(model_data['offspring_fitness'])
        
        if len(parent_fitness) == 0:
            ax.text(0.5, 0.5, f'No data for {model}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=18, fontweight='bold')
            ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold')
            return
        
        # Plot parent vs offspring
        color = self.model_colors.get(model, "#1f77b4")
        ax.scatter(parent_fitness, offspring_fitness, 
                  label=f'{model} offspring', color=color, alpha=0.7, s=80, 
                  edgecolors='black', linewidths=1)
        
        # Add 2nd degree polynomial regression line
        if len(parent_fitness) > 2:
            self._add_regression_line(ax, parent_fitness, offspring_fitness, model, color)
        
        # Set labels and title
        ax.set_xlabel("Parent Fitness", fontsize=18, fontweight='bold', labelpad=15)
        ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
        ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold', pad=20)
        
        # Add legend and grid
        ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.tick_params(labelsize=16, width=2, length=8)
        
        # Thicker axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    def plot_genomic_distance_effects(self, diploid_offspring: Dict[str, List], 
                                    output_path: Path, mating_strategy: str) -> None:
        """Plot effects of genomic distance on offspring fitness."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f"Genomic Distance Effects ({mating_strategy})", 
                        fontsize=24, fontweight='bold', y=0.98)
            
            # Extract data first
            extracted_data = self._extract_organism_data(diploid_offspring)
            
            # Plot for each model
            for i, (model, ax) in enumerate(zip(["dominant", "recessive", "codominant"], axes)):
                if model in extracted_data and extracted_data[model]['genomic_distances']:
                    distances = np.array(extracted_data[model]['genomic_distances'])
                    fitness_values = np.array(extracted_data[model]['offspring_fitness'])
                    
                    # Plot scatter
                    color = self.model_colors.get(model, f"C{i}")
                    ax.scatter(distances, fitness_values, 
                             label=model.capitalize(), color=color, alpha=0.7, s=80,
                             edgecolors='black', linewidths=1)
                    
                    # Add 2nd degree polynomial regression lines
                    if len(distances) > 2:
                        self._add_regression_line(ax, distances, fitness_values, model, color)
                    
                    # Set labels and title
                    ax.set_xlabel("Genomic Distance", fontsize=18, fontweight='bold', labelpad=15)
                    ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
                    ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold', pad=20)
                    
                    # Add grid
                    ax.grid(True, alpha=0.3, linewidth=1.5)
                    ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
                    ax.tick_params(labelsize=16, width=2, length=8)
                    
                    # Thicker axis lines
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)
                else:
                    ax.text(0.5, 0.5, f'No data for {model}', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=18, fontweight='bold')
                    ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold')
            
            # Adjust layout and save
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_path, f"genomic_distance_effects_{mating_strategy}.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot genomic distance effects: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def plot_prs_analysis(self, diploid_offspring: Dict[str, List], 
                         output_path: Path, mating_strategy: str) -> None:
        """Plot PRS analysis."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f"PRS Analysis ({mating_strategy})", 
                        fontsize=24, fontweight='bold', y=0.98)
            
            # Extract data first
            extracted_data = self._extract_organism_data(diploid_offspring)
            
            # Plot for each model
            for i, (model, ax) in enumerate(zip(["dominant", "recessive", "codominant"], axes)):
                if model in extracted_data and extracted_data[model]['prs_values']:
                    prs_values = np.array(extracted_data[model]['prs_values'])
                    fitness_values = np.array(extracted_data[model]['offspring_fitness'])
                    
                    # Plot scatter
                    color = self.model_colors.get(model, f"C{i}")
                    ax.scatter(prs_values, fitness_values, 
                             label=model.capitalize(), color=color, alpha=0.7, s=80,
                             edgecolors='black', linewidths=1)
                    
                    # Add 2nd degree polynomial regression lines
                    if len(prs_values) > 2:
                        self._add_regression_line(ax, prs_values, fitness_values, model, color)
                    
                    # Set labels and title
                    ax.set_xlabel("Polygenic Risk Score", fontsize=18, fontweight='bold', labelpad=15)
                    ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
                    ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold', pad=20)
                    
                    # Add grid
                    ax.grid(True, alpha=0.3, linewidth=1.5)
                    ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
                    ax.tick_params(labelsize=16, width=2, length=8)
                    
                    # Thicker axis lines
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)
                else:
                    ax.text(0.5, 0.5, f'No data for {model}', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=18, fontweight='bold')
                    ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold')
            
            # Adjust layout and save
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_path, f"prs_analysis_{mating_strategy}.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot PRS analysis: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def plot_fitness_heatmap(self, diploid_offspring: Dict[str, List], output_path: Path) -> None:
        """Plot parent1 vs parent2 fitness heatmap with offspring intensity."""
        try:
            # Create figure with subplots for each model
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle('Parent Fitness Heatmaps by Model\n(Color = Number of Offspring)', 
                        fontsize=24, fontweight='bold', y=0.98)
            
            models = ["dominant", "recessive", "codominant"]
            
            for i, (model, ax) in enumerate(zip(models, axes)):
                if model in diploid_offspring and diploid_offspring[model]:
                    organisms = diploid_offspring[model]
                    
                    # Extract parent fitnesses and offspring counts
                    parent1_fitness = []
                    parent2_fitness = []
                    offspring_fitness = []
                    
                    for org in organisms:
                        try:
                            if hasattr(org, 'parent1') and hasattr(org, 'parent2'):
                                # Object format
                                p1_fit = org.parent1.calculate_fitness()
                                p2_fit = org.parent2.calculate_fitness()
                                off_fit = org.calculate_fitness()
                            elif isinstance(org, dict):
                                # Dictionary format - need to extract individual parent fitnesses
                                p1_fit = org.get('parent1_fitness', 0)
                                p2_fit = org.get('parent2_fitness', 0)
                                off_fit = org.get('fitness', 0)
                            else:
                                continue
                            
                            parent1_fitness.append(p1_fit)
                            parent2_fitness.append(p2_fit)
                            offspring_fitness.append(off_fit)
                            
                        except Exception as e:
                            self.logger.debug(f"Failed to extract parent fitnesses: {e}")
                            continue
                    
                    if parent1_fitness and parent2_fitness:
                        # Create 2D histogram
                        self._create_parent_fitness_heatmap(ax, parent1_fitness, parent2_fitness, 
                                                          offspring_fitness, model)
                    else:
                        ax.text(0.5, 0.5, f'No data for {model}', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=18, fontweight='bold')
                        ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'No data for {model}', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=18, fontweight='bold')
                    ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold')
            
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_path, "fitness_heatmap.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot fitness heatmap: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def _create_parent_fitness_heatmap(self, ax, parent1_fitness: List[float], 
                                      parent2_fitness: List[float], 
                                      offspring_fitness: List[float], model: str) -> None:
        """Create a heatmap showing parent1 vs parent2 fitness with offspring count intensity."""
        try:
            # Convert to numpy arrays
            p1_fit = np.array(parent1_fitness)
            p2_fit = np.array(parent2_fitness)
            
            # Create 2D histogram
            # Use reasonable number of bins based on data range
            n_bins = min(20, max(5, int(np.sqrt(len(p1_fit)))))
            
            hist, xedges, yedges = np.histogram2d(p1_fit, p2_fit, bins=n_bins)
            
            # Create heatmap
            im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='viridis',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Number of Offspring', fontsize=16, fontweight='bold', labelpad=20)
            cbar.ax.tick_params(labelsize=14)
            
            # Set labels and title
            ax.set_xlabel('Parent 1 Fitness', fontsize=18, fontweight='bold', labelpad=15)
            ax.set_ylabel('Parent 2 Fitness', fontsize=18, fontweight='bold', labelpad=15)
            ax.set_title(f'{model.capitalize()} Model', fontsize=20, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(True, alpha=0.3, linewidth=1.5)
            ax.tick_params(labelsize=16, width=2, length=8)
            
            # Thicker axis lines
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            
        except Exception as e:
            self.logger.error(f"Failed to create heatmap for {model}: {e}")
            ax.text(0.5, 0.5, f'Heatmap failed for {model}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=16, fontweight='bold')

    def save_fdr_results(self, output_dir: Path, filename: str = "fdr_correction_results.txt") -> None:
        """Save FDR correction results to a file."""
        try:
            fdr_results = self._apply_fdr_correction()
            if not fdr_results:
                self.logger.info("No p-values collected for FDR correction")
                return
            
            output_path = Path(output_dir) / filename
            with open(output_path, 'w') as f:
                f.write("FDR Correction Results (Benjamini-Hochberg)\n")
                f.write("=" * 50 + "\n\n")
                
                for test_name, results in fdr_results.items():
                    f.write(f"Test: {test_name}\n")
                    f.write(f"  Original p-value: {results['p_value']:.6f}\n")
                    f.write(f"  FDR-adjusted p-value: {results['p_adjusted']:.6f}\n")
                    f.write(f"  Significant after FDR: {'Yes' if results['significant'] else 'No'}\n")
                    f.write("\n")
                
                # Summary
                total_tests = len(fdr_results)
                significant_tests = sum(1 for r in fdr_results.values() if r['significant'])
                f.write(f"Summary:\n")
                f.write(f"  Total tests: {total_tests}\n")
                f.write(f"  Significant after FDR correction: {significant_tests}\n")
                f.write(f"  Proportion significant: {significant_tests/total_tests:.3f}\n")
            
            self.logger.info(f"FDR results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save FDR results: {e}")


class MultiSimulationVisualizer:
    """Creates visualizations comparing results across multiple simulation runs."""
    
    def __init__(self, style: str = "whitegrid", palette: str = "deep"):
        """Initialize the visualizer with aesthetic settings."""
        # Set up plotting style
        sns.set_style(style)
        sns.set_palette(palette)
        
        # Define consistent colors for different models
        self.model_colors = {
            "dominant": "#2E86AB",
            "recessive": "#A23B72", 
            "codominant": "#F18F01"
        }
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking for FDR correction
        self.p_values = []
        self.test_names = []

    def plot_all_analyses(self, data: Dict[str, Any], output_dir: Path) -> None:
        """Generate all key analysis plots across runs."""
        try:
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Creating multi-run plots in: {output_dir}")
            
            # Debug: Log the actual data structure we received
            self._debug_data_structure(data)
            
            # Extract diploid data from the collected runs
            extracted_data = self._extract_all_diploid_data(data)
            
            # Check if we actually have data
            total_data_points = sum(len(model_data.get('parent_offspring', [])) 
                                   for model_data in extracted_data.values())
            
            if total_data_points == 0:
                self.logger.warning("No diploid offspring data found for plotting!")
                self._create_no_data_plot(output_dir)
                return
            
            self.logger.info(f"Successfully extracted {total_data_points} data points for plotting")
            
            # Generate all key plots with extracted data
            self.plot_fitness_evolution_across_runs(data, output_dir)
            self.plot_prs_analysis_all_runs(extracted_data, output_dir)
            self.plot_genomic_distance_effects_all_runs(extracted_data, output_dir)
            self.plot_parent_offspring_fitness_all_runs(extracted_data, output_dir)
            self.plot_model_comparison_summary(extracted_data, output_dir)
            
            # Save FDR correction results
            self.save_fdr_results(output_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to generate analysis plots: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def _debug_data_structure(self, data: Dict[str, Any]) -> None:
        """Debug the data structure to understand what we're working with."""
        self.logger.debug("=== DATA STRUCTURE DEBUG ===")
        self.logger.debug(f"Top-level keys: {list(data.keys())}")
        
        if "runs" in data:
            runs = data["runs"]
            self.logger.debug(f"Number of runs: {len(runs)}")
            if runs:
                first_run = runs[0]
                self.logger.debug(f"First run keys: {list(first_run.keys())}")
                if "diploid_offspring" in first_run:
                    diploid = first_run["diploid_offspring"]
                    self.logger.debug(f"Diploid offspring keys: {list(diploid.keys())}")
                    for model, organisms in diploid.items():
                        self.logger.debug(f"{model}: {len(organisms)} organisms, type: {type(organisms[0]) if organisms else 'empty'}")
        
        if "collector_data" in data:
            collector = data["collector_data"]
            self.logger.debug(f"Collector data keys: {list(collector.keys())}")
            
        self.logger.debug("=== END DEBUG ===")

    def _extract_all_diploid_data(self, data: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
        """Extract all diploid offspring data from the complete data structure."""
        model_data = {
            "dominant": {"parent_offspring": [], "prs": [], "genomic_distance": []},
            "recessive": {"parent_offspring": [], "prs": [], "genomic_distance": []},
            "codominant": {"parent_offspring": [], "prs": [], "genomic_distance": []}
        }
        
        # Try multiple data sources
        data_sources = []
        
        # Source 1: Direct runs data
        if "runs" in data and data["runs"]:
            data_sources.append(("runs", data["runs"]))
        
        # Source 2: Collector data runs
        if "collector_data" in data and "runs" in data["collector_data"]:
            data_sources.append(("collector_runs", data["collector_data"]["runs"]))
        
        # Source 3: Individual runs (fallback)
        if "individual_runs" in data:
            data_sources.append(("individual_runs", data["individual_runs"]))
        
        for source_name, runs in data_sources:
            self.logger.debug(f"Processing {source_name} with {len(runs)} runs")
            
            for run_idx, run in enumerate(runs):
                diploid_offspring = run.get("diploid_offspring", {})
                
                for model in ["dominant", "recessive", "codominant"]:
                    if model in diploid_offspring:
                        model_orgs = diploid_offspring[model]
                        
                        # Handle different data formats
                        if isinstance(model_orgs, list) and model_orgs:
                            extracted_count = 0
                            for org in model_orgs:
                                try:
                                    if isinstance(org, dict):
                                        # Dictionary format - extract directly
                                        parent_fit = org.get("avg_parent_fitness", 0)
                                        offspring_fit = org.get("fitness", 0)
                                        prs_val = org.get("prs", 0)
                                        genomic_dist = org.get("genomic_distance", 0)
                                    elif hasattr(org, 'calculate_fitness'):
                                        # Object format - use methods
                                        parent_fit = (org.parent1.calculate_fitness() + org.parent2.calculate_fitness()) / 2
                                        offspring_fit = org.calculate_fitness()
                                        prs_val = org.calculate_prs()
                                        genomic_dist = org.calculate_genomic_distance()
                                    else:
                                        self.logger.debug(f"Unknown organism format: {type(org)}")
                                        continue
                                    
                                    model_data[model]["parent_offspring"].append([parent_fit, offspring_fit])
                                    model_data[model]["prs"].append([prs_val, offspring_fit])
                                    model_data[model]["genomic_distance"].append([genomic_dist, offspring_fit])
                                    extracted_count += 1
                                    
                                except Exception as e:
                                    self.logger.debug(f"Failed to extract data from organism: {e}")
                                    continue
                            
                            self.logger.debug(f"Extracted {extracted_count} organisms from {model} in run {run_idx}")
        
        # Log final extraction results
        for model in model_data:
            count = len(model_data[model]["parent_offspring"])
            self.logger.info(f"Total extracted {count} data points for {model} model")
        
        return model_data

    def _create_no_data_plot(self, output_dir: Path) -> None:
        """Create a plot indicating no data was available."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No diploid offspring data available for plotting\nCheck data collection and format', 
               transform=ax.transAxes, ha='center', va='center', fontsize=20, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
        ax.set_title('Multi-Run Analysis - No Data Available', fontsize=24, fontweight='bold')
        ax.axis('off')
        self._save_figure(fig, output_dir, "no_data_available.png")

    def plot_fitness_evolution_across_runs(self, data: Dict[str, Any], output_dir: Path) -> None:
        """Plot fitness evolution across all runs."""
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            runs = data.get("runs", [])
            if not runs:
                self.logger.warning("No runs data for fitness evolution")
                plt.close(fig)
                return
            
            all_fitness_evolution = []
            for run in runs:
                fitness_evolution = run.get("fitness_evolution", [])
                if fitness_evolution:
                    all_fitness_evolution.append(fitness_evolution)
            
            if all_fitness_evolution:
                # Convert to numpy array for easier plotting
                max_len = max(len(fe) for fe in all_fitness_evolution)
                fitness_array = np.full((len(all_fitness_evolution), max_len), np.nan)
                
                for i, fitness_vals in enumerate(all_fitness_evolution):
                    fitness_array[i, :len(fitness_vals)] = fitness_vals
                
                generations = range(max_len)
                
                # Plot individual runs
                for i, fitness_vals in enumerate(all_fitness_evolution):
                    ax.plot(range(len(fitness_vals)), fitness_vals, alpha=0.3, color='lightblue', linewidth=2)
                
                # Plot mean and std (ignoring NaNs)
                mean_fitness = np.nanmean(fitness_array, axis=0)
                std_fitness = np.nanstd(fitness_array, axis=0)
                
                valid_gens = ~np.isnan(mean_fitness)
                ax.plot(np.array(generations)[valid_gens], mean_fitness[valid_gens], 
                       'k-', linewidth=4, label='Mean across runs')
                ax.fill_between(np.array(generations)[valid_gens], 
                               (mean_fitness - std_fitness)[valid_gens], 
                               (mean_fitness + std_fitness)[valid_gens], 
                               alpha=0.3, color='gray', label='±1 std')
                
                ax.set_xlabel('Generation', fontsize=18, fontweight='bold', labelpad=15)
                ax.set_ylabel('Fitness', fontsize=18, fontweight='bold', labelpad=15)
                ax.set_title('Fitness Evolution Across All Runs', fontsize=20, fontweight='bold', pad=20)
                ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.tick_params(labelsize=16, width=2, length=8)
                
                # Thicker axis lines
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
            else:
                ax.text(0.5, 0.5, 'No fitness evolution data available', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=18, fontweight='bold')
                ax.set_title('Fitness Evolution - No Data', fontsize=20, fontweight='bold')
            
            self._save_figure(fig, output_dir, "fitness_evolution_all_runs.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot fitness evolution across runs: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def _calculate_polynomial_fit(self, x_data: np.ndarray, y_data: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, float, float]:
        """Calculate polynomial fit with R² and p-value."""
        if len(x_data) < 3 or len(y_data) < 3:
            return np.array([]), 0.0, 1.0
        
        try:
            # Remove any NaN or infinite values
            mask = np.isfinite(x_data) & np.isfinite(y_data)
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            if len(x_clean) < 3:
                return np.array([]), 0.0, 1.0
            
            # Fit polynomial
            coeffs = np.polyfit(x_clean, y_clean, degree)
            poly_func = np.poly1d(coeffs)
            
            # Generate smooth line for plotting
            x_smooth = np.linspace(np.min(x_clean), np.max(x_clean), 100)
            y_fitted = poly_func(x_smooth)
            
            # Calculate R²
            y_pred = poly_func(x_clean)
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate p-value using F-test
            n = len(x_clean)
            k = degree  # number of parameters (excluding intercept)
            if n > k + 1:
                mse_res = ss_res / (n - k - 1)
                mse_reg = (ss_tot - ss_res) / k
                if mse_res > 0:
                    f_stat = mse_reg / mse_res
                    p_value = 1 - scipy.stats.f.cdf(f_stat, k, n - k - 1)
                else:
                    p_value = 0.0
            else:
                p_value = 1.0
            
            return np.column_stack([x_smooth, y_fitted]), r_squared, p_value
            
        except Exception as e:
            self.logger.debug(f"Polynomial fit failed: {e}")
            return np.array([]), 0.0, 1.0

    def _add_regression_line(self, ax, x_data: np.ndarray, y_data: np.ndarray, 
                           label: str, color: str = 'red') -> Tuple[float, float]:
        """Add 2nd degree polynomial regression with R² and p-values."""
        fitted_data, r_squared, p_value = self._calculate_polynomial_fit(x_data, y_data, degree=2)
        
        if len(fitted_data) > 0:
            # Plot the fitted line
            ax.plot(fitted_data[:, 0], fitted_data[:, 1], '--', 
                   linewidth=4, alpha=0.9, color=color,
                   label=f'{label} (R²={r_squared:.3f})')
            
            # Store p-value for FDR correction
            self.p_values.append(p_value)
            self.test_names.append(f"{label}_fit")
            
            # Add text box with statistics
            if r_squared > 0.01:  # Only show if meaningful relationship
                if p_value < 0.001:
                    p_text = 'p < 0.001***'
                elif p_value < 0.01:
                    p_text = f'p = {p_value:.3f}**'
                elif p_value < 0.05:
                    p_text = f'p = {p_value:.3f}*'
                else:
                    p_text = f'p = {p_value:.3f}ns'
                
                stats_text = f'R² = {r_squared:.3f}\n{p_text}'
                
                # Position text box
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                                alpha=0.9, edgecolor='black', linewidth=1),
                       verticalalignment='top', fontsize=14, fontweight='bold')
        
        return r_squared, p_value

    def plot_prs_analysis_all_runs(self, model_data: Dict[str, Dict[str, List]], output_dir: Path) -> None:
        """Plot PRS analysis across all runs."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle("PRS Analysis Across All Runs", fontsize=24, fontweight='bold', y=0.98)
            
            colors = ["#2E86AB", "#A23B72", "#F18F01"]
            for i, (model, ax) in enumerate(zip(["dominant", "recessive", "codominant"], axes)):
                prs_data = model_data[model]["prs"]
                
                if prs_data:
                    prs_values = [point[0] for point in prs_data if len(point) >= 2]
                    fitness_values = [point[1] for point in prs_data if len(point) >= 2]
                    
                    if prs_values and fitness_values:
                        # Plot scatter
                        ax.scatter(prs_values, fitness_values, 
                                 label=model.capitalize(), color=colors[i], alpha=0.7, s=80,
                                 edgecolors='black', linewidths=1)
                        
                        # Add 2nd degree polynomial regression line
                        if len(prs_values) > 2:
                            self._add_regression_line(ax, np.array(prs_values), np.array(fitness_values), model, colors[i])
                    else:
                        ax.text(0.5, 0.5, f'Invalid data for {model}', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=18, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'No data for {model}', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=18, fontweight='bold')
                
                ax.set_xlabel("Polygenic Risk Score", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.tick_params(labelsize=16, width=2, length=8)
                
                # Thicker axis lines
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    
                if ax.get_children():  # Only add legend if there are plot elements
                    ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
            
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_dir, "prs_analysis_all_runs.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot PRS analysis across runs: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def plot_genomic_distance_effects_all_runs(self, model_data: Dict[str, Dict[str, List]], output_dir: Path) -> None:
        """Plot genomic distance effects across all runs."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle("Genomic Distance Effects Across All Runs", fontsize=24, fontweight='bold', y=0.98)
            
            colors = ["#2E86AB", "#A23B72", "#F18F01"]
            for i, (model, ax) in enumerate(zip(["dominant", "recessive", "codominant"], axes)):
                distance_data = model_data[model]["genomic_distance"]
                
                if distance_data:
                    distances = [point[0] for point in distance_data if len(point) >= 2]
                    fitness_values = [point[1] for point in distance_data if len(point) >= 2]
                    
                    if distances and fitness_values:
                        # Plot scatter
                        ax.scatter(distances, fitness_values, 
                                 label=model.capitalize(), color=colors[i], alpha=0.7, s=80,
                                 edgecolors='black', linewidths=1)
                        
                        # Add 2nd degree polynomial regression line
                        if len(distances) > 2:
                            self._add_regression_line(ax, np.array(distances), np.array(fitness_values), model, colors[i])
                    else:
                        ax.text(0.5, 0.5, f'Invalid data for {model}', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=18, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'No data for {model}', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=18, fontweight='bold')
                
                ax.set_xlabel("Genomic Distance", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.tick_params(labelsize=16, width=2, length=8)
                
                # Thicker axis lines
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    
                if ax.get_children():  # Only add legend if there are plot elements
                    ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
            
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_dir, "genomic_distance_effects_all_runs.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot genomic distance effects across runs: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def plot_parent_offspring_fitness_all_runs(self, model_data: Dict[str, Dict[str, List]], output_dir: Path) -> None:
        """Plot parent-offspring fitness relationships across all runs."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle("Parent-Offspring Fitness Relationships Across All Runs", fontsize=24, fontweight='bold', y=0.98)
            
            colors = ["#2E86AB", "#A23B72", "#F18F01"]
            for i, (model, ax) in enumerate(zip(["dominant", "recessive", "codominant"], axes)):
                parent_offspring_data = model_data[model]["parent_offspring"]
                
                if parent_offspring_data:
                    parent_fitness = [point[0] for point in parent_offspring_data if len(point) >= 2]
                    offspring_fitness = [point[1] for point in parent_offspring_data if len(point) >= 2]
                    
                    if parent_fitness and offspring_fitness:
                        # Plot scatter
                        ax.scatter(parent_fitness, offspring_fitness, 
                                 label=model.capitalize(), color=colors[i], alpha=0.7, s=80,
                                 edgecolors='black', linewidths=1)
                        
                        # Add 2nd degree polynomial regression line
                        if len(parent_fitness) > 2:
                            self._add_regression_line(ax, np.array(parent_fitness), np.array(offspring_fitness), model, colors[i])
                    else:
                        ax.text(0.5, 0.5, f'Invalid data for {model}', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=18, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'No data for {model}', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=18, fontweight='bold')
                
                ax.set_xlabel("Average Parent Fitness", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_ylabel("Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_title(f"{model.capitalize()} Model", fontsize=20, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.tick_params(labelsize=16, width=2, length=8)
                
                # Thicker axis lines
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    
                if ax.get_children():  # Only add legend if there are plot elements
                    ax.legend(fontsize=16, frameon=True, shadow=True, framealpha=0.9)
            
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_dir, "parent_offspring_fitness_all_runs.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot parent-offspring fitness across runs: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def plot_model_comparison_summary(self, model_data: Dict[str, Dict[str, List]], output_dir: Path) -> None:
        """Plot summary comparison of models across runs."""
        try:
            # Calculate summary statistics
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle("Model Comparison Summary", fontsize=24, fontweight='bold', y=0.98)
            
            models = ["dominant", "recessive", "codominant"]
            colors = ["#2E86AB", "#A23B72", "#F18F01"]
            
            # Plot 1: Mean offspring fitness by model
            ax = axes[0, 0]
            mean_fitness = []
            std_fitness = []
            valid_models = []
            
            for model in models:
                parent_offspring_data = model_data[model]["parent_offspring"]
                if parent_offspring_data:
                    offspring_vals = [point[1] for point in parent_offspring_data if len(point) >= 2]
                    if offspring_vals:
                        mean_fitness.append(np.mean(offspring_vals))
                        std_fitness.append(np.std(offspring_vals))
                        valid_models.append(model)
            
            if valid_models:
                bars = ax.bar(valid_models, mean_fitness, yerr=std_fitness, capsize=8, 
                             color=[colors[models.index(m)] for m in valid_models], alpha=0.8,
                             linewidth=2, edgecolor='black')
                ax.set_ylabel("Mean Offspring Fitness", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_title("Mean Offspring Fitness by Model", fontsize=20, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.tick_params(labelsize=16, width=2, length=8)
                
                # Thicker axis lines
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
            else:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=18, fontweight='bold')
            
            # Plot 2: Fitness improvement distribution
            ax = axes[0, 1]
            improvement_data = []
            improvement_labels = []
            
            for model in models:
                parent_offspring_data = model_data[model]["parent_offspring"]
                if parent_offspring_data:
                    improvements = [point[1] - point[0] for point in parent_offspring_data 
                                  if len(point) >= 2]
                    if improvements:
                        improvement_data.append(improvements)
                        improvement_labels.append(model)
            
            if improvement_data:
                bp = ax.boxplot(improvement_data, labels=improvement_labels, patch_artist=True,
                               widths=0.8, medianprops=dict(linewidth=3))
                for patch, model in zip(bp['boxes'], improvement_labels):
                    patch.set_facecolor(colors[models.index(model)])
                    patch.set_alpha(0.8)
                    patch.set_linewidth(2)
                ax.set_ylabel("Fitness Improvement", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_title("Fitness Improvement Distribution", fontsize=20, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.tick_params(labelsize=16, width=2, length=8)
                
                # Thicker axis lines
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
            else:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=18, fontweight='bold')
            
            # Plot 3: PRS correlation
            ax = axes[1, 0]
            correlations = []
            corr_models = []
            
            for model in models:
                prs_data = model_data[model]["prs"]
                if prs_data and len(prs_data) > 1:
                    prs_vals = [point[0] for point in prs_data if len(point) >= 2]
                    fitness_vals = [point[1] for point in prs_data if len(point) >= 2]
                    if len(prs_vals) > 1 and len(fitness_vals) > 1:
                        corr, _ = scipy.stats.pearsonr(prs_vals, fitness_vals)
                        correlations.append(corr)
                        corr_models.append(model)
            
            if corr_models:
                bars = ax.bar(corr_models, correlations, 
                             color=[colors[models.index(m)] for m in corr_models], alpha=0.8,
                             linewidth=2, edgecolor='black')
                ax.set_ylabel("PRS-Fitness Correlation", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_title("PRS-Fitness Correlation by Model", fontsize=20, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.set_ylim(-1, 1)
                ax.tick_params(labelsize=16, width=2, length=8)
                
                # Thicker axis lines
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
            else:
                ax.text(0.5, 0.5, 'No correlation data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=18, fontweight='bold')
            
            # Plot 4: Sample size by model
            ax = axes[1, 1]
            sample_sizes = []
            size_models = []
            
            for model in models:
                size = len(model_data[model]["parent_offspring"])
                if size > 0:
                    sample_sizes.append(size)
                    size_models.append(model)
            
            if size_models:
                bars = ax.bar(size_models, sample_sizes, 
                             color=[colors[models.index(m)] for m in size_models], alpha=0.8,
                             linewidth=2, edgecolor='black')
                ax.set_ylabel("Number of Offspring", fontsize=18, fontweight='bold', labelpad=15)
                ax.set_title("Sample Size by Model", fontsize=20, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linewidth=1.5)
                ax.tick_params(labelsize=16, width=2, length=8)
                
                # Thicker axis lines
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
            else:
                ax.text(0.5, 0.5, 'No sample data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=18, fontweight='bold')
            
            plt.tight_layout(pad=4.0)
            self._save_figure(fig, output_dir, "model_comparison_summary.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot model comparison summary: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def _apply_fdr_correction(self) -> Dict[str, float]:
        """Apply FDR correction to all stored p-values."""
        if not self.p_values:
            return {}
        
        # Apply FDR correction using Benjamini-Hochberg
        rejected, p_adjusted, _, _ = multipletests(self.p_values, method='fdr_bh')
        
        # Create dictionary of corrected p-values
        fdr_results = {}
        for i, test_name in enumerate(self.test_names):
            fdr_results[test_name] = {
                'p_value': self.p_values[i],
                'p_adjusted': p_adjusted[i],
                'significant': rejected[i]
            }
        
        return fdr_results

    def save_fdr_results(self, output_dir: Path, filename: str = "fdr_correction_results.txt") -> None:
        """Save FDR correction results to a file."""
        try:
            fdr_results = self._apply_fdr_correction()
            if not fdr_results:
                self.logger.info("No p-values collected for FDR correction")
                return
            
            output_path = Path(output_dir) / filename
            with open(output_path, 'w') as f:
                f.write("FDR Correction Results (Benjamini-Hochberg)\n")
                f.write("=" * 50 + "\n\n")
                
                for test_name, results in fdr_results.items():
                    f.write(f"Test: {test_name}\n")
                    f.write(f"  Original p-value: {results['p_value']:.6f}\n")
                    f.write(f"  FDR-adjusted p-value: {results['p_adjusted']:.6f}\n")
                    f.write(f"  Significant after FDR: {'Yes' if results['significant'] else 'No'}\n")
                    f.write("\n")
                
                # Summary
                total_tests = len(fdr_results)
                significant_tests = sum(1 for r in fdr_results.values() if r['significant'])
                f.write(f"Summary:\n")
                f.write(f"  Total tests: {total_tests}\n")
                f.write(f"  Significant after FDR correction: {significant_tests}\n")
                f.write(f"  Proportion significant: {significant_tests/total_tests:.3f}\n")
            
            self.logger.info(f"FDR results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save FDR results: {e}")