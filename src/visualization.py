#!/usr/bin/env python3
"""
Visualization tools for evolutionary simulation results.

This module provides clean, informative visualizations of simulation data,
including fitness evolution, parent-offspring relationships, and comparative
analyses across different models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple, Any


from core_models import DiploidOrganism
from analysis_tools import calculate_genomic_distance, calculate_diploid_prs


class SimulationVisualizer:
    """
    Creates publication-quality visualizations of simulation results.
    
    This class handles all plotting functionality with a focus on clarity,
    aesthetics, and scientific communication.
    """
    
    def __init__(self, style: str = "whitegrid", palette: str = "deep"):
        """
        Initialize the visualizer with aesthetic settings.
        
        Args:
            style: Seaborn style to use
            palette: Color palette for plots
        """
        # Set up plotting style
        plt.style.use('default')
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
    
    def plot_fitness_evolution(self, simulation, output_path: Path,
                             filename: str = "fitness_evolution.png") -> None:
        """
        Plot fitness evolution over generations with individual trajectories and statistics.
        
        Args:
            simulation: Simulation object with fitness data
            output_path: Directory to save the plot
            filename: Name of the output file
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Individual trajectories with statistical overlays
        self._plot_individual_trajectories(ax1, simulation)
        self._add_statistical_overlays(ax1, simulation)
        
        # Plot 2: Fitness distribution over time (sampled generations)
        self._plot_fitness_distributions(ax2, simulation)
        
        plt.tight_layout()
        self._save_figure(fig, output_path, filename)
    
    def plot_relationship_tree(self, organisms: List[Any], output_path: Path,
                             filename: str = "relationship_tree.png",
                             max_organisms: int = 1000,
                             layout_type: str = "hierarchical") -> None:
        """
        Create an evolutionary relationship tree visualization.
        
        This method creates a network graph showing parent-child relationships
        between organisms across generations, with nodes colored by fitness.
        
        Args:
            organisms: List of organism objects with id, generation, fitness, and parent_id attributes
            output_path: Directory to save the plot
            filename: Name of the output file
            max_organisms: Maximum number of organisms to plot (for performance)
            layout_type: Type of layout ("hierarchical" or "spring")
        """
        # Limit organisms for performance while maintaining representation
        if len(organisms) > max_organisms:
            organisms = self._sample_organisms_for_tree(organisms, max_organisms)
        
        # Create the network graph
        graph = self._build_relationship_graph(organisms)
        
        if not graph.nodes():
            self.logger.warning("No organisms to plot in relationship tree")
            return
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(16, 10))
        
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
    
    def _sample_organisms_for_tree(self, organisms: List[Any], max_count: int) -> List[Any]:
        """
        Intelligently sample organisms for tree visualization.
        
        This ensures we get a representative sample across generations
        while maintaining parent-child relationships.
        """
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
        """
        Build a directed graph representing parent-child relationships.
        
        Args:
            organisms: List of organism objects
            
        Returns:
            NetworkX directed graph with organism relationships
        """
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
        """
        Create a hierarchical layout positioning organisms by generation and fitness.
        
        This layout places organisms horizontally by generation and vertically by fitness,
        with some jitter to avoid overlapping nodes.
        """
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
        """
        Draw the relationship tree with proper styling.
        
        Args:
            graph: NetworkX graph with organism relationships
            pos: Dictionary mapping node IDs to (x, y) positions
            ax: Matplotlib axis to draw on
        """
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
                arrowsize=15,
                width=0.8,
                arrowstyle='->'
            )
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=50,
            node_color=fitness_values,
            cmap=plt.cm.get_cmap(self.tree_colors["nodes"]),
            alpha=0.8,
            edgecolors='white',
            linewidths=0.5
        )
        
        # Add colorbar for fitness
        if nodes:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Fitness', fontsize=12, labelpad=15)
            cbar.ax.tick_params(labelsize=10)
    
    def _style_tree_plot(self, ax, num_organisms: int) -> None:
        """
        Apply consistent styling to the tree plot.
        
        Args:
            ax: Matplotlib axis to style
            num_organisms: Total number of organisms (for title)
        """
        ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=14, fontweight='bold')
        ax.set_title(f'Evolutionary Relationship Tree\n({num_organisms:,} organisms)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Style the axes
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.tree_colors["background"])
        
        # Add some padding around the plot
        ax.margins(0.05)
    
    def plot_multi_generational_tree(self, organisms: List[Any], output_path: Path,
                                   filename: str = "multi_gen_tree.png",
                                   generations_to_show: Optional[List[int]] = None) -> None:
        """
        Create a focused tree showing specific generations.
        
        This is useful for examining particular evolutionary transitions
        or comparing specific generations.
        
        Args:
            organisms: List of organism objects
            output_path: Directory to save the plot
            filename: Name of the output file
            generations_to_show: List of generation numbers to include (None for all)
        """
        # Filter organisms by generation if specified
        if generations_to_show:
            filtered_organisms = [org for org in organisms if org.generation in generations_to_show]
        else:
            filtered_organisms = organisms
        
        if not filtered_organisms:
            self.logger.warning("No organisms found for specified generations")
            return
        
        # Create a more detailed tree for fewer organisms
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Build graph
        graph = self._build_relationship_graph(filtered_organisms)
        pos = self._create_hierarchical_layout(filtered_organisms, graph)
        
        # Draw with labels for smaller datasets
        self._draw_detailed_tree(graph, pos, ax, show_labels=len(filtered_organisms) < 100)
        
        # Enhanced styling
        gen_str = f" (Generations: {', '.join(map(str, generations_to_show))})" if generations_to_show else ""
        ax.set_title(f'Multi-Generational Relationship Tree{gen_str}', 
                    fontsize=18, fontweight='bold', pad=25)
        ax.set_xlabel('Generation', fontsize=16, fontweight='bold')
        ax.set_ylabel('Fitness', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, output_path, filename)
    
    def _draw_detailed_tree(self, graph: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                           ax, show_labels: bool = False) -> None:
        """
        Draw a more detailed tree with optional labels.
        
        Args:
            graph: NetworkX graph
            pos: Node positions
            ax: Matplotlib axis
            show_labels: Whether to show node labels
        """
        # Draw edges with varied thickness based on fitness difference
        if graph.edges():
            edge_weights = []
            for parent, child in graph.edges():
                parent_fitness = graph.nodes[parent]['fitness']
                child_fitness = graph.nodes[child]['fitness']
                # Thicker edges for bigger fitness improvements
                weight = max(0.5, 3.0 * (child_fitness / parent_fitness) if parent_fitness > 0 else 1.0)
                edge_weights.append(weight)
            
            nx.draw_networkx_edges(
                graph, pos, ax=ax,
                arrows=True,
                edge_color=self.tree_colors["edges"],
                alpha=0.6,
                arrowsize=20,
                width=edge_weights,
                arrowstyle='->'
            )
        
        # Draw nodes with size based on fitness
        fitness_values = [graph.nodes[node]['fitness'] for node in graph.nodes()]
        node_sizes = [max(30, min(200, 50 + f * 20)) for f in fitness_values]
        
        nodes = nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=node_sizes,
            node_color=fitness_values,
            cmap=plt.cm.get_cmap(self.tree_colors["nodes"]),
            alpha=0.8,
            edgecolors='white',
            linewidths=1.0
        )
        
        # Add labels if requested
        if show_labels:
            labels = {node: f"{node}\n({graph.nodes[node]['fitness']:.2f})" 
                     for node in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=8)
        
        # Add colorbar
        if nodes:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
            cbar.set_label('Fitness', fontsize=14, fontweight='bold')

    def _plot_individual_trajectories(self, ax, simulation) -> None:
        """Plot individual organism fitness trajectories."""
        for org_id, fitness_data in simulation.individual_fitness.items():
            generations, fitness = zip(*fitness_data)
            ax.plot(generations, fitness, '-', alpha=0.15, linewidth=0.8, color='lightblue')
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Individual Fitness Trajectories', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
    
    def _add_statistical_overlays(self, ax, simulation) -> None:
        """Add mean, max, and min fitness lines to the plot."""
        stats = simulation.generation_stats
        generations = [s['generation'] for s in stats]
        avg_fitness = [s['avg_fitness'] for s in stats]
        max_fitness = [s['max_fitness'] for s in stats]
        min_fitness = [s['min_fitness'] for s in stats]
        
        ax.plot(generations, avg_fitness, 'k-', linewidth=3, label='Average', zorder=10)
        ax.plot(generations, max_fitness, 'g-', linewidth=2, label='Maximum', zorder=9)
        ax.plot(generations, min_fitness, 'r-', linewidth=2, label='Minimum', zorder=9)
        
        ax.legend(frameon=True, fancybox=True, shadow=True)
    
    def _plot_fitness_distributions(self, ax, simulation) -> None:
        """Plot fitness distributions for sampled generations."""
        max_gen = max(s['generation'] for s in simulation.generation_stats)
        sample_gens = range(1, max_gen + 1, max(1, max_gen // 10))
        
        fitness_data = []
        gen_labels = []
        
        for gen in sample_gens:
            gen_fitness = []
            for org_id, fit_data in simulation.individual_fitness.items():
                gen_fitness.extend([f for g, f in fit_data if g == gen])
            
            if gen_fitness:  # Only add if we have data
                fitness_data.append(gen_fitness)
                gen_labels.append(str(gen))
        
        if fitness_data:
            bp = ax.boxplot(fitness_data, labels=gen_labels, patch_artist=True)
            
            # Color the boxes with a gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_xlabel('Generation (sampled)', fontsize=12)
        ax.set_ylabel('Fitness Distribution', fontsize=12)
        ax.set_title('Fitness Distribution Over Time', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
    
    def plot_parent_offspring_relationships(self, diploid_offspring: Dict[str, List[DiploidOrganism]],
                                          output_path: Path, mating_strategy: str,
                                          filename: str = "parent_offspring_fitness.png",
                                          ylim: Tuple[float, float] = None) -> None:
        """
        Plot parent vs offspring fitness relationships for all models.
        
        Args:
            diploid_offspring: Dictionary of diploid organisms by model
            output_path: Directory to save the plot
            mating_strategy: Mating strategy used
            filename: Name of the output file
            ylim: Optional tuple of (ymin, ymax) to set y-axis limits
        """
        models = ["dominant", "recessive", "codominant"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Find global min and max offspring fitness across all models
        all_offspring_fitness = []
        for model in models:
            organisms = diploid_offspring.get(model, [])
            if organisms:
                all_offspring_fitness.extend([org.fitness for org in organisms])
        
        if ylim is None and all_offspring_fitness:
            min_fitness = min(all_offspring_fitness)
            max_fitness = max(all_offspring_fitness)
            padding = 0.3 * (max_fitness - min_fitness)
            ylim = (min_fitness - padding, max_fitness + padding)
            self.logger.debug(f"Setting y-axis limits to {ylim} based on data range.")
            
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            organisms = diploid_offspring.get(model, [])
            
            self._plot_single_model_relationship(ax, organisms, model)
            ax.set_ylim(ylim)

        fig.suptitle(f'Parent vs Offspring Fitness ({mating_strategy} strategy)', 
                    fontsize=16, y=1.03)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_figure(fig, output_path, f"{filename.split('.')[0]}_{mating_strategy}.png")
    
    def plot_min_max_parent_offspring_fitness(self, diploid_offspring: Dict[str, List[DiploidOrganism]],
                                            output_path: Path, mating_strategy: str,
                                            filename_prefix: str = "parent_offspring_fitness") -> None:
        """
        Plot offspring fitness against min and max parental fitness for all models.
        """
        metrics = {
                "min": ("Min Parental Fitness", lambda org: min(org.parent1_fitness, org.parent2_fitness)),
                "max": ("Max Parental Fitness", lambda org: max(org.parent1_fitness, org.parent2_fitness)),
            }
        models = ["dominant", "recessive", "codominant"]

        for key, (title_prefix, fitness_accessor) in metrics.items():
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Determine global ylim across all models
            all_offspring_fitness = []
            for model in models:
                organisms = diploid_offspring.get(model, [])
                all_offspring_fitness.extend([org.fitness for org in organisms])
            if all_offspring_fitness:
                min_fitness = min(all_offspring_fitness)
                max_fitness = max(all_offspring_fitness)
                padding = 0.3 * (max_fitness - min_fitness)
                ylim = (min_fitness - padding, max_fitness + padding)
            else:
                ylim = None

            for idx, model in enumerate(models):
                ax = axes[idx]
                organisms = diploid_offspring.get(model, [])

                # Override average parent fitness dynamically
                for org in organisms:
                    org.avg_parent_fitness = fitness_accessor(org)  # overwrite temporarily

                self._plot_single_model_relationship(ax, organisms, model)
                if ylim:
                    ax.set_ylim(ylim)

            fig.suptitle(f'{title_prefix} vs Offspring Fitness ({mating_strategy} strategy)', fontsize=16, y=1.03)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            self._save_figure(fig, output_path, f"{filename_prefix}_{key}_{mating_strategy}.png")
        
    def _plot_single_model_relationship(self, ax, organisms: List[DiploidOrganism], model: str) -> None:
        """Plot parent-offspring relationship for a single dominance model."""
        ax.set_title(f'{model.capitalize()} Model\n({len(organisms)} offspring)', 
                    fontsize=12, pad=15)
        ax.set_xlabel('Mean Parent Fitness', fontsize=11)
        ax.set_ylabel('Offspring Fitness', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if not organisms:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # Extract data
        parent_fitness = np.array([org.avg_parent_fitness for org in organisms])
        offspring_fitness = np.array([org.fitness for org in organisms])
        
        # Set focused axis limits
        self._set_focused_limits(ax, parent_fitness, offspring_fitness)
        
        # Check for sufficient variation
        if len(np.unique(parent_fitness)) <= 2:
            self._handle_low_variation(ax, parent_fitness, offspring_fitness, model)
        else:
            self._plot_with_regression(ax, parent_fitness, offspring_fitness, model)
    
    def _set_focused_limits(self, ax, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Set axis limits with appropriate padding based on data range."""
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        
        x_padding = (x_max - x_min) * 0.1 if x_max > x_min else 0.1
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    def _handle_low_variation(self, ax, parent_fitness: np.ndarray, 
                            offspring_fitness: np.ndarray, model: str) -> None:
        """Handle plotting when there's insufficient variation in parent fitness."""
        ax.text(0.5, 0.8, 'Insufficient variation in parent fitness', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
        
        # Still show the data points
        ax.scatter(parent_fitness, offspring_fitness, alpha=0.6, 
                  color=self.model_colors[model], s=40)
        
        # Add mean lines
        ax.axhline(np.mean(offspring_fitness), color='red', linestyle='--', alpha=0.7,
                  label=f'Mean offspring: {np.mean(offspring_fitness):.3f}')
        ax.axvline(np.mean(parent_fitness), color='blue', linestyle='--', alpha=0.7,
                  label=f'Mean parent: {np.mean(parent_fitness):.3f}')
        
        ax.legend(loc='lower right', fontsize=9)
    
    def _plot_with_regression(self, ax, parent_fitness: np.ndarray, 
                            offspring_fitness: np.ndarray, model: str) -> None:
        """Plot data with regression lines and statistics."""
        # Scatter plot
        ax.scatter(parent_fitness, offspring_fitness, alpha=0.6, 
                  color=self.model_colors[model], s=40, 
                  edgecolors='white', linewidth=0.5)
        
        # Create x range for regression lines
        x_range = np.linspace(parent_fitness.min(), parent_fitness.max(), 100)
        
        # Add regression lines and statistics
        stats_text = []
        
        try:
            from scipy import stats
            
            # Linear regression
            slope, intercept, r_value, p_value, _ = stats.linregress(parent_fitness, offspring_fitness)
            y_pred_linear = slope * x_range + intercept
            ax.plot(x_range, y_pred_linear, 'r-', linewidth=2, alpha=0.8,
                   label=f'Linear (R² = {r_value**2:.3f})')
            
            stats_text.extend([
                f'Linear R² = {r_value**2:.3f}',
                f'Slope = {slope:.3f}'
            ])
            
            # Quadratic regression
            if len(np.unique(parent_fitness)) >= 3:
                poly_coeffs = np.polyfit(parent_fitness, offspring_fitness, 2)
                y_pred_quad = np.polyval(poly_coeffs, x_range)
                
                # Calculate R² for quadratic fit
                y_pred_quad_actual = np.polyval(poly_coeffs, parent_fitness)
                ss_res = np.sum((offspring_fitness - y_pred_quad_actual) ** 2)
                ss_tot = np.sum((offspring_fitness - np.mean(offspring_fitness)) ** 2)
                quad_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                ax.plot(x_range, y_pred_quad, 'g--', linewidth=2, alpha=0.8,
                       label=f'Quadratic (R² = {quad_r2:.3f})')
                
                stats_text.append(f'Quad R² = {quad_r2:.3f}')
            
        except Exception as e:
            self.logger.warning(f"Regression failed for {model}: {e}")
        
        # Add diagonal reference line
        data_min = min(parent_fitness.min(), offspring_fitness.min())
        data_max = max(parent_fitness.max(), offspring_fitness.max())
        ax.plot([data_min, data_max], [data_min, data_max], 'k--', alpha=0.5,
               label='No improvement line')
        
        # Add statistics text box
        if stats_text:
            ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
                   fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    def plot_genomic_distance_effects(self, diploid_offspring: Dict[str, List[DiploidOrganism]],
                                    output_path: Path, mating_strategy: str,
                                    filename: str = "genomic_distance_effects.png") -> None:
        """
        Plot the relationship between genomic distance and offspring fitness.
        
        Args:
            diploid_offspring: Dictionary of diploid organisms by model
            output_path: Directory to save the plot
            mating_strategy: Mating strategy used
            filename: Name of the output file
        """
        models = ["dominant", "recessive", "codominant"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            organisms = diploid_offspring.get(model, [])
            
            self._plot_distance_vs_fitness(ax, organisms, model)
        
        fig.suptitle('Parent Genomic Distance vs Offspring Fitness', fontsize=16, y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path, f"{filename.split('.')[0]}_{mating_strategy}.png")
    
    def _plot_distance_vs_fitness(self, ax, organisms: List[DiploidOrganism], model: str) -> None:
        """Plot genomic distance vs fitness for a single model."""
        ax.set_title(f'{model.capitalize()} Model', fontsize=12, pad=15)
        ax.set_xlabel('Genomic Distance Between Parents', fontsize=11)
        ax.set_ylabel('Offspring Fitness', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if not organisms:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        try:
            # Calculate genomic distances (limit to 1000 organisms for performance)
            sample_organisms = organisms[:1000]
            distances = [calculate_genomic_distance(org.allele1, org.allele2) for org in sample_organisms]
            fitness_vals = [org.fitness for org in sample_organisms]
            
            distances = np.array(distances)
            fitness_vals = np.array(fitness_vals)
            
            # Set focused limits
            self._set_focused_limits(ax, distances, fitness_vals)
            
            # Scatter plot
            ax.scatter(distances, fitness_vals, alpha=0.6, color=self.model_colors[model], 
                      s=40, edgecolors='white', linewidth=0.5)
            
            # Add regression if sufficient variation
            if len(np.unique(distances)) > 2:
                self._add_distance_regression(ax, distances, fitness_vals)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
    
    def _add_distance_regression(self, ax, distances: np.ndarray, fitness_vals: np.ndarray) -> None:
        """Add regression lines for distance vs fitness relationship."""
        x_range = np.linspace(distances.min(), distances.max(), 100)
        
        try:
            from scipy import stats
            
            # Linear regression
            slope, intercept, r_value, p_value, _ = stats.linregress(distances, fitness_vals)
            y_pred_linear = slope * x_range + intercept
            ax.plot(x_range, y_pred_linear, 'r-', linewidth=2, alpha=0.8,
                   label=f'Linear (R² = {r_value**2:.3f})')
            
            # Quadratic regression
            poly_coeffs = np.polyfit(distances, fitness_vals, 2)
            y_pred_quad = np.polyval(poly_coeffs, x_range)
            
            # Calculate R² for quadratic
            y_pred_quad_actual = np.polyval(poly_coeffs, distances)
            ss_res = np.sum((fitness_vals - y_pred_quad_actual) ** 2)
            ss_tot = np.sum((fitness_vals - np.mean(fitness_vals)) ** 2)
            quad_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            ax.plot(x_range, y_pred_quad, 'g--', linewidth=2, alpha=0.8,
                   label=f'Quadratic (R² = {quad_r2:.3f})')
            
            # Add statistics
            stats_text = f'Linear: R² = {r_value**2:.3f}, Slope = {slope:.3f}\nQuad: R² = {quad_r2:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.legend(loc='best', fontsize=9)
            
        except Exception as e:
            self.logger.warning(f"Distance regression failed: {e}")
    
    def plot_prs_analysis(self, diploid_offspring: Dict[str, List[DiploidOrganism]],
                         output_path: Path, mating_strategy: str,
                         filename: str = "prs_analysis.png") -> None:
        """
        Plot PRS (Polygenic Risk Score) vs offspring fitness.
        
        Args:
            diploid_offspring: Dictionary of diploid organisms by model
            output_path: Directory to save the plot
            mating_strategy: Mating strategy used
            filename: Name of the output file
        """
        models = ["dominant", "recessive", "codominant"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            organisms = diploid_offspring.get(model, [])
            
            self._plot_prs_vs_fitness(ax, organisms, model)
        
        fig.suptitle('Offspring PRS vs Offspring Fitness', fontsize=16, y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path, f"{filename.split('.')[0]}_{mating_strategy}.png")
    
    def _plot_prs_vs_fitness(self, ax, organisms: List[DiploidOrganism], model: str) -> None:
        """Plot PRS vs fitness for a single model."""
        ax.set_title(f'{model.capitalize()} Model', fontsize=12, pad=15)
        ax.set_xlabel('Offspring PRS Score', fontsize=11)
        ax.set_ylabel('Offspring Fitness', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if not organisms:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        try:
            # Calculate PRS scores (limit for performance)
            sample_organisms = organisms[:1000]
            prs_scores = []
            fitness_vals = []
            
            for org in sample_organisms:
                try:
                    prs = calculate_diploid_prs(org)
                    prs_scores.append(prs)
                    fitness_vals.append(org.fitness)
                except:
                    continue
            
            if not prs_scores:
                ax.text(0.5, 0.5, 'PRS calculation failed', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                return
            
            prs_scores = np.array(prs_scores)
            fitness_vals = np.array(fitness_vals)
            
            # Set focused limits
            self._set_focused_limits(ax, prs_scores, fitness_vals)
            
            # Scatter plot
            ax.scatter(prs_scores, fitness_vals, alpha=0.6, color=self.model_colors[model], 
                      s=40, edgecolors='white', linewidth=0.5)
            
            # Add mean lines
            ax.axhline(np.mean(fitness_vals), color='red', linestyle='--', alpha=0.7,
                      label=f'Mean fitness: {np.mean(fitness_vals):.3f}')
            ax.axvline(np.mean(prs_scores), color='blue', linestyle='--', alpha=0.7,
                      label=f'Mean PRS: {np.mean(prs_scores):.3f}')
            
            # Add correlation if there's variation
            if len(np.unique(prs_scores)) > 2:
                self._add_prs_correlation(ax, prs_scores, fitness_vals)
            
            ax.legend(loc='lower right', fontsize=9)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
    
    def _add_prs_correlation(self, ax, prs_scores: np.ndarray, fitness_vals: np.ndarray) -> None:
        """Add correlation analysis for PRS vs fitness."""
        try:
            from scipy import stats
            
            corr, p_value = stats.pearsonr(prs_scores, fitness_vals)
            
            # Add trend line
            x_range = np.linspace(prs_scores.min(), prs_scores.max(), 100)
            slope, intercept, _, _, _ = stats.linregress(prs_scores, fitness_vals)
            y_pred = slope * x_range + intercept
            
            ax.plot(x_range, y_pred, 'r-', linewidth=2, alpha=0.8,
                   label=f'Linear (R² = {corr**2:.3f})')
            
            ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                   transform=ax.transAxes, fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
        except Exception as e:
            self.logger.warning(f"PRS correlation failed: {e}")
    
    def plot_fitness_heatmap(self, diploid_offspring: Dict[str, List[DiploidOrganism]],
                           output_path: Path, filename: str = "fitness_heatmap.png") -> None:
        """
        Create heatmaps showing offspring fitness based on parent fitness combinations.
        
        Args:
            diploid_offspring: Dictionary of diploid organisms by model
            output_path: Directory to save the plot
            filename: Name of the output file
        """
        models = ["dominant", "recessive", "codominant"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            organisms = diploid_offspring.get(model, [])
            
            self._create_single_heatmap(ax, organisms, model)
        
        fig.suptitle('Parent Fitness Combinations vs Offspring Fitness', fontsize=16, y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path, filename)
    
    def _create_single_heatmap(self, ax, organisms: List[DiploidOrganism], model: str) -> None:
        """Create a heatmap for a single dominance model."""
        ax.set_title(f'{model.capitalize()} Model', fontsize=12)
        ax.set_xlabel('Parent 1 Fitness', fontsize=11)
        ax.set_ylabel('Parent 2 Fitness', fontsize=11)
        
        if not organisms:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            return
        
        # Extract parent fitness values
        parent1_fit = np.array([org.parent1_fitness for org in organisms])
        parent2_fit = np.array([org.parent2_fitness for org in organisms])
        offspring_fit = np.array([org.fitness for org in organisms])
        
        # Create 2D histogram
        bins = min(15, int(np.sqrt(len(organisms))))
        h, xedges, yedges = np.histogram2d(parent1_fit, parent2_fit, 
                                          bins=bins, weights=offspring_fit)
        counts, _, _ = np.histogram2d(parent1_fit, parent2_fit, bins=bins)
        
        # Calculate average fitness per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_fitness = np.divide(h, counts, out=np.zeros_like(h), where=counts>0)
        
        # Create heatmap
        im = ax.imshow(avg_fitness.T, origin='lower', aspect='auto',
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      cmap='viridis', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Average Offspring Fitness', fontsize=10)
    
    def _save_figure(self, fig, output_path: Path, filename: str) -> None:
        """Save figure with consistent settings."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        self.logger.info(f"Saved plot: {filepath}")

