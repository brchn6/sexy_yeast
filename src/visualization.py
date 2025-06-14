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
from typing import Dict, List, Optional, Tuple, Any, Union
import scipy.stats
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set up matplotlib for non-interactive use
plt.switch_backend('Agg')

# Import utility functions from core_models to avoid circular imports
from core_models import calculate_genomic_distance, calculate_diploid_prs


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
    
    def _save_figure(self, fig: plt.Figure, output_dir: Path, filename: str) -> None:
        """Save a figure to the specified directory and close it properly.
        
        Args:
            fig: Figure to save
            output_dir: Directory to save the figure
            filename: Name of the output file
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Saved figure: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save figure {filename}: {e}")
        finally:
            plt.close(fig)  # Always close the figure to prevent memory leaks
    
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
        
        try:
            # Plot 1: Individual trajectories with statistical overlays
            self._plot_individual_trajectories(ax1, simulation)
            self._add_statistical_overlays(ax1, simulation)
            
            # Plot 2: Fitness distribution over time (sampled generations)
            self._plot_fitness_distributions(ax2, simulation)
            
            plt.tight_layout()
            self._save_figure(fig, output_path, filename)
        except Exception as e:
            self.logger.error(f"Failed to plot fitness evolution: {e}")
            plt.close(fig)
    
    def plot_relationship_tree(self, organisms: List[Any], output_path: Path,
                             filename: str = "relationship_tree.png",
                             max_organisms: int = 1000,
                             layout_type: str = "hierarchical") -> None:
        """
        Create an evolutionary relationship tree visualization.
        
        Args:
            organisms: List of organism objects with id, generation, fitness, and parent_id attributes
            output_path: Directory to save the plot
            filename: Name of the output file
            max_organisms: Maximum number of organisms to plot (for performance)
            layout_type: Type of layout ("hierarchical" or "spring")
        """
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
        fig, ax = plt.subplots(figsize=(16, 10))
        
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
    
    def _plot_individual_trajectories(self, ax, simulation) -> None:
        """Plot individual organism fitness trajectories."""
        if hasattr(simulation, 'individual_fitness') and simulation.individual_fitness:
            for org_id, fitness_data in simulation.individual_fitness.items():
                if fitness_data:
                    generations, fitness = zip(*fitness_data)
                    ax.plot(generations, fitness, '-', alpha=0.15, linewidth=0.8, color='lightblue')
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Individual Fitness Trajectories', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
    
    def _add_statistical_overlays(self, ax, simulation) -> None:
        """Add mean, max, and min fitness lines to the plot."""
        if hasattr(simulation, 'generation_stats') and simulation.generation_stats:
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
    
    def plot_parent_offspring_relationships(self, data: Union[Dict[str, Any], Dict[str, List[Dict[str, Any]]]], 
                                          output_dir: Path, mating_strategy: Optional[str] = None) -> None:
        """Plot parent-offspring relationships.
        
        Args:
            data: Dictionary containing simulation data or diploid offspring directly
            output_dir: Directory to save the plots
            mating_strategy: Optional mating strategy for individual run plots
        """
        try:
            # Determine if this is individual run data or multi-run summary data
            if mating_strategy is not None:
                # Individual run case - data should be diploid_offspring dict
                self._plot_individual_run_relationships(data, output_dir, mating_strategy)
            else:
                # Multi-run case - data should have summary structure
                self._plot_multi_run_relationships(data, output_dir)
                
        except Exception as e:
            self.logger.error(f"Error plotting parent-offspring relationships: {e}")
            self.logger.error("Error details:", exc_info=True)
    
    def _plot_individual_run_relationships(self, diploid_offspring: Dict[str, List[Dict[str, Any]]], 
                                         output_dir: Path, mating_strategy: str) -> None:
        """Plot relationships for individual run."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Parent-Offspring Relationships ({mating_strategy})", fontsize=16)
        
        try:
            # Plot parent vs offspring fitness
            self._plot_parent_vs_offspring_fitness_individual(axes[0], diploid_offspring)
            
            # Plot PRS vs offspring fitness
            self._plot_prs_vs_offspring_fitness_individual(axes[1], diploid_offspring)
            
            # Plot genomic distance vs offspring fitness
            self._plot_genomic_distance_vs_offspring_fitness_individual(axes[2], diploid_offspring)
            
            # Adjust layout and save
            plt.tight_layout()
            self._save_figure(fig, output_dir, "parent_offspring_relationships.png")
        except Exception as e:
            self.logger.error(f"Failed to plot individual run relationships: {e}")
            plt.close(fig)
    
    def _plot_multi_run_relationships(self, data: Dict[str, Any], output_dir: Path) -> None:
        """Plot relationships for multi-run summary."""
        summary = data.get("summary", {})
        if not summary:
            self.logger.error("No summary data found for plotting")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Parent-Offspring Relationships Across Multiple Runs", fontsize=16)
        
        try:
            # Plot parent vs offspring fitness
            self._plot_parent_vs_offspring_fitness_summary(axes[0], summary)
            
            # Plot PRS vs offspring fitness
            self._plot_prs_vs_offspring_fitness_summary(axes[1], summary)
            
            # Plot genomic distance vs offspring fitness
            self._plot_genomic_distance_vs_offspring_fitness_summary(axes[2], summary)
            
            # Adjust layout and save
            plt.tight_layout()
            self._save_figure(fig, output_dir, "parent_offspring_relationships.png")
        except Exception as e:
            self.logger.error(f"Failed to plot multi-run relationships: {e}")
            plt.close(fig)
    
    def _plot_parent_vs_offspring_fitness_individual(self, ax: plt.Axes, diploid_offspring: Dict[str, List[Dict[str, Any]]]) -> None:
        """Plot parent vs offspring fitness for individual run."""
        ax.set_title("Parent vs Offspring Fitness")
        ax.set_xlabel("Parent Fitness")
        ax.set_ylabel("Offspring Fitness")
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for i, (model, color) in enumerate(zip(["dominant", "recessive", "codominant"], colors)):
            if model in diploid_offspring and diploid_offspring[model]:
                organisms = diploid_offspring[model]
                parent_fitness = [org["avg_parent_fitness"] for org in organisms]
                offspring_fitness = [org["fitness"] for org in organisms]
                
                ax.scatter(parent_fitness, offspring_fitness, 
                          label=model.capitalize(), color=color, alpha=0.6)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_parent_vs_offspring_fitness_summary(self, ax: plt.Axes, summary: Dict[str, Any]) -> None:
        """Plot parent vs offspring fitness for summary data."""
        ax.set_title("Parent vs Offspring Fitness")
        ax.set_xlabel("Parent Fitness")
        ax.set_ylabel("Offspring Fitness")
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
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
                              fmt='o', alpha=0.7)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_prs_vs_offspring_fitness_individual(self, ax: plt.Axes, diploid_offspring: Dict[str, List[Dict[str, Any]]]) -> None:
        """Plot PRS vs offspring fitness for individual run."""
        ax.set_title("PRS vs Offspring Fitness")
        ax.set_xlabel("Polygenic Risk Score")
        ax.set_ylabel("Offspring Fitness")
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for model, color in zip(["dominant", "recessive", "codominant"], colors):
            if model in diploid_offspring and diploid_offspring[model]:
                organisms = diploid_offspring[model]
                prs_values = []
                offspring_fitness = []
                
                for org in organisms:
                    if "prs" in org:
                        prs_values.append(org["prs"])
                        offspring_fitness.append(org["fitness"])
                
                if prs_values:
                    ax.scatter(prs_values, offspring_fitness, 
                              label=model.capitalize(), color=color, alpha=0.6)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_prs_vs_offspring_fitness_summary(self, ax: plt.Axes, summary: Dict[str, Any]) -> None:
        """Plot PRS vs offspring fitness for summary data."""
        ax.set_title("PRS vs Offspring Fitness")
        ax.set_xlabel("Polygenic Risk Score")
        ax.set_ylabel("Offspring Fitness")
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
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
                              fmt='o', alpha=0.7)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_genomic_distance_vs_offspring_fitness_individual(self, ax: plt.Axes, diploid_offspring: Dict[str, List[Dict[str, Any]]]) -> None:
        """Plot genomic distance vs offspring fitness for individual run."""
        ax.set_title("Genomic Distance vs Offspring Fitness")
        ax.set_xlabel("Genomic Distance")
        ax.set_ylabel("Offspring Fitness")
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for model, color in zip(["dominant", "recessive", "codominant"], colors):
            if model in diploid_offspring and diploid_offspring[model]:
                organisms = diploid_offspring[model]
                distances = []
                offspring_fitness = []
                
                for org in organisms:
                    if "genomic_distance" in org:
                        distances.append(org["genomic_distance"])
                        offspring_fitness.append(org["fitness"])
                
                if distances:
                    ax.scatter(distances, offspring_fitness, 
                              label=model.capitalize(), color=color, alpha=0.6)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_genomic_distance_vs_offspring_fitness_summary(self, ax: plt.Axes, summary: Dict[str, Any]) -> None:
        """Plot genomic distance vs offspring fitness for summary data."""
        ax.set_title("Genomic Distance vs Offspring Fitness")
        ax.set_xlabel("Genomic Distance")
        ax.set_ylabel("Offspring Fitness")
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
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
                              fmt='o', alpha=0.7)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_min_max_parent_offspring_fitness(self, diploid_offspring: Dict[str, List[Dict[str, Any]]],
                                            output_path: Path, mating_strategy: str,
                                            filename_prefix: str = "parent_offspring_fitness") -> None:
        """
        Plot offspring fitness against min and max parental fitness for all models.
        """
        metrics = {
            "min": ("Min Parental Fitness", lambda org: min(org["parent1_fitness"], org["parent2_fitness"])),
            "max": ("Max Parental Fitness", lambda org: max(org["parent1_fitness"], org["parent2_fitness"])),
        }
        models = ["dominant", "recessive", "codominant"]

        for key, (title_prefix, fitness_accessor) in metrics.items():
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            try:
                # Determine global ylim across all models
                all_offspring_fitness = []
                for model in models:
                    organisms = diploid_offspring.get(model, [])
                    all_offspring_fitness.extend([org["fitness"] for org in organisms])
                
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
                        org["avg_parent_fitness"] = fitness_accessor(org)  # overwrite temporarily

                    self._plot_single_model_relationship(ax, organisms, model)
                    if ylim:
                        ax.set_ylim(ylim)

                fig.suptitle(f'{title_prefix} vs Offspring Fitness ({mating_strategy} strategy)', fontsize=16, y=1.03)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                self._save_figure(fig, output_path, f"{filename_prefix}_{key}_{mating_strategy}.png")
            except Exception as e:
                self.logger.error(f"Failed to plot min/max parent offspring fitness: {e}")
                plt.close(fig)
    
    def _plot_single_model_relationship(self, ax, organisms: List[Dict[str, Any]], model: str) -> None:
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
        parent_fitness = np.array([org["avg_parent_fitness"] for org in organisms])
        offspring_fitness = np.array([org["fitness"] for org in organisms])
        
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
                  color=self.model_colors.get(model, 'blue'), s=40)
        
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
                  color=self.model_colors.get(model, 'blue'), s=40, 
                  edgecolors='white', linewidth=0.5)
        
        # Create x range for regression lines
        x_range = np.linspace(parent_fitness.min(), parent_fitness.max(), 100)
        
        # Add regression lines and statistics
        stats_text = []
        
        try:
            # Linear regression
            slope, intercept, r_value, p_value, _ = scipy.stats.linregress(parent_fitness, offspring_fitness)
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
    
    # Additional plotting methods
    def plot_genomic_distance_effects(self, diploid_offspring: Dict[str, List[Dict[str, Any]]], 
                                    output_path: Path, mating_strategy: str) -> None:
        """Plot effects of genomic distance on offspring fitness."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        try:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            for model, color in zip(["dominant", "recessive", "codominant"], colors):
                if model in diploid_offspring and diploid_offspring[model]:
                    organisms = diploid_offspring[model]
                    distances = []
                    fitness_values = []
                    
                    for org in organisms:
                        if "genomic_distance" in org:
                            distances.append(org["genomic_distance"])
                            fitness_values.append(org["fitness"])
                    
                    if distances:
                        ax.scatter(distances, fitness_values, 
                                  label=f'{model.capitalize()} ({len(distances)} points)', 
                                  color=color, alpha=0.6, s=40)
            
            ax.set_xlabel('Genomic Distance (Hamming Distance)', fontsize=12)
            ax.set_ylabel('Offspring Fitness', fontsize=12)
            ax.set_title(f'Genomic Distance vs Offspring Fitness ({mating_strategy} strategy)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self._save_figure(fig, output_path, f"genomic_distance_effects_{mating_strategy}.png")
        except Exception as e:
            self.logger.error(f"Failed to plot genomic distance effects: {e}")
            plt.close(fig)
    
    def plot_prs_analysis(self, diploid_offspring: Dict[str, List[Dict[str, Any]]], 
                         output_path: Path, mating_strategy: str) -> None:
        """Plot PRS analysis for diploid offspring."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        try:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            for model, color in zip(["dominant", "recessive", "codominant"], colors):
                if model in diploid_offspring and diploid_offspring[model]:
                    organisms = diploid_offspring[model]
                    prs_values = []
                    fitness_values = []
                    
                    for org in organisms:
                        if "prs" in org:
                            prs_values.append(org["prs"])
                            fitness_values.append(org["fitness"])
                    
                    if prs_values:
                        ax.scatter(prs_values, fitness_values, 
                                  label=f'{model.capitalize()} ({len(prs_values)} points)', 
                                  color=color, alpha=0.6, s=40)
            
            ax.set_xlabel('Polygenic Risk Score', fontsize=12)
            ax.set_ylabel('Offspring Fitness', fontsize=12)
            ax.set_title(f'PRS vs Offspring Fitness ({mating_strategy} strategy)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self._save_figure(fig, output_path, f"prs_analysis_{mating_strategy}.png")
        except Exception as e:
            self.logger.error(f"Failed to plot PRS analysis: {e}")
            plt.close(fig)
    
    def plot_fitness_heatmap(self, diploid_offspring: Dict[str, List[Dict[str, Any]]], output_path: Path) -> None:
        """Plot fitness heatmap across different models."""
        try:
            # Prepare data for heatmap
            models = ["dominant", "recessive", "codominant"]
            fitness_data = []
            
            for model in models:
                if model in diploid_offspring and diploid_offspring[model]:
                    fitness_values = [org["fitness"] for org in diploid_offspring[model]]
                    fitness_data.append(fitness_values)
                else:
                    fitness_data.append([])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate statistics for each model
            stats_data = []
            for i, (model, values) in enumerate(zip(models, fitness_data)):
                if values:
                    stats_data.append([
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values)
                    ])
                else:
                    stats_data.append([0, 0, 0, 0])
            
            # Create heatmap
            stats_array = np.array(stats_data)
            im = ax.imshow(stats_array, cmap='viridis', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(4))
            ax.set_xticklabels(['Mean', 'Std', 'Min', 'Max'])
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([m.capitalize() for m in models])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Fitness Value')
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(4):
                    text = ax.text(j, i, f'{stats_array[i, j]:.3f}',
                                 ha="center", va="center", color="w", fontweight='bold')
            
            ax.set_title('Fitness Statistics Heatmap by Model', fontsize=14)
            
            self._save_figure(fig, output_path, "fitness_heatmap.png")
        except Exception as e:
            self.logger.error(f"Failed to plot fitness heatmap: {e}")
            plt.close(fig)


class MultiSimulationVisualizer:
    """Creates visualizations comparing results across multiple simulation runs."""
    
    def __init__(self, style: str = "whitegrid", palette: str = "deep"):
        """Initialize the visualizer with aesthetic settings.
        
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
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def _save_figure(self, fig: plt.Figure, output_dir: Path, filename: str) -> None:
        """Save a figure to the specified directory and close it properly.
        
        Args:
            fig: Figure to save
            output_dir: Directory to save the figure
            filename: Name of the output file
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Saved multi-run figure: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save multi-run figure {filename}: {e}")
        finally:
            plt.close(fig)  # Always close the figure to prevent memory leaks
    
    def plot_parent_offspring_relationships(self, data: Dict[str, Any], output_dir: Path) -> None:
        """Plot parent-offspring relationships across multiple runs.
        
        Args:
            data: Dictionary containing simulation data from all runs
            output_dir: Directory to save the plots
        """
        try:
            summary = data.get("summary", {})
            if not summary:
                self.logger.error("No summary data found for plotting")
                return
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle("Parent-Offspring Relationships Across Multiple Runs", fontsize=16)
            
            # Plot parent vs offspring fitness
            self._plot_parent_vs_offspring_fitness(axes[0], summary)
            
            # Plot PRS vs offspring fitness
            self._plot_prs_vs_offspring_fitness(axes[1], summary)
            
            # Plot genomic distance vs offspring fitness
            self._plot_genomic_distance_vs_offspring_fitness(axes[2], summary)
            
            # Adjust layout and save
            plt.tight_layout()
            self._save_figure(fig, output_dir, "multi_run_parent_offspring.png")
            
        except Exception as e:
            self.logger.error(f"Error plotting multi-run parent-offspring relationships: {e}")
            self.logger.error("Error details:", exc_info=True)
    
    def _plot_parent_vs_offspring_fitness(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        """Plot parent vs offspring fitness relationships."""
        ax.set_title("Parent vs Offspring Fitness")
        ax.set_xlabel("Parent Fitness")
        ax.set_ylabel("Offspring Fitness")
        
        for model, color in zip(["dominant", "recessive", "codominant"], 
                              ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            model_data = data.get("diploid_stats", {}).get(model, {}).get("parent_offspring", {})
            if not model_data:
                continue
            
            # Extract means and standard deviations
            x_mean = model_data.get("x_mean")
            x_std = model_data.get("x_std")
            y_mean = model_data.get("y_mean")
            y_std = model_data.get("y_std")
            
            if all(v is not None for v in [x_mean, x_std, y_mean, y_std]):
                ax.errorbar(x_mean, y_mean, 
                          xerr=x_std, yerr=y_std,
                          label=model.capitalize(), color=color,
                          fmt='o', alpha=0.7)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_prs_vs_offspring_fitness(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        """Plot PRS vs offspring fitness relationships."""
        ax.set_title("PRS vs Offspring Fitness")
        ax.set_xlabel("Polygenic Risk Score")
        ax.set_ylabel("Offspring Fitness")
        
        for model, color in zip(["dominant", "recessive", "codominant"], 
                              ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            model_data = data.get("diploid_stats", {}).get(model, {}).get("prs", {})
            if not model_data:
                continue
            
            # Extract means and standard deviations
            x_mean = model_data.get("x_mean")
            x_std = model_data.get("x_std")
            y_mean = model_data.get("y_mean")
            y_std = model_data.get("y_std")
            
            if all(v is not None for v in [x_mean, x_std, y_mean, y_std]):
                ax.errorbar(x_mean, y_mean, 
                          xerr=x_std, yerr=y_std,
                          label=model.capitalize(), color=color,
                          fmt='o', alpha=0.7)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_genomic_distance_vs_offspring_fitness(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        """Plot genomic distance vs offspring fitness relationships."""
        ax.set_title("Genomic Distance vs Offspring Fitness")
        ax.set_xlabel("Genomic Distance")
        ax.set_ylabel("Offspring Fitness")
        
        for model, color in zip(["dominant", "recessive", "codominant"], 
                              ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            model_data = data.get("diploid_stats", {}).get(model, {}).get("genomic_distance", {})
            if not model_data:
                continue
            
            # Extract means and standard deviations
            x_mean = model_data.get("x_mean")
            x_std = model_data.get("x_std")
            y_mean = model_data.get("y_mean")
            y_std = model_data.get("y_std")
            
            if all(v is not None for v in [x_mean, x_std, y_mean, y_std]):
                ax.errorbar(x_mean, y_mean, 
                          xerr=x_std, yerr=y_std,
                          label=model.capitalize(), color=color,
                          fmt='o', alpha=0.7)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_fitness_evolution_comparison(self, data: Dict[str, Any], output_dir: Path) -> None:
        """Plot fitness evolution comparison across runs."""
        try:
            fitness_evolution = data.get("summary", {}).get("fitness_evolution", {})
            if not fitness_evolution:
                self.logger.warning("No fitness evolution data for comparison plot")
                return
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot mean fitness evolution
            if "mean" in fitness_evolution:
                generations = range(len(fitness_evolution["mean"]))
                mean_fitness = fitness_evolution["mean"]
                std_fitness = fitness_evolution.get("std", [0] * len(mean_fitness))
                
                ax.plot(generations, mean_fitness, 'b-', linewidth=2, label='Mean')
                ax.fill_between(generations, 
                               np.array(mean_fitness) - np.array(std_fitness),
                               np.array(mean_fitness) + np.array(std_fitness),
                               alpha=0.3, color='blue')
            
            # Plot min and max if available
            if "min" in fitness_evolution and "max" in fitness_evolution:
                ax.plot(generations, fitness_evolution["min"], 'r--', alpha=0.7, label='Min')
                ax.plot(generations, fitness_evolution["max"], 'g--', alpha=0.7, label='Max')
            
            ax.set_xlabel('Generation', fontsize=12)
            ax.set_ylabel('Fitness', fontsize=12)
            ax.set_title('Fitness Evolution Across Multiple Runs', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self._save_figure(fig, output_dir, "fitness_evolution_comparison.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot fitness evolution comparison: {e}")
            plt.close(fig)
    
    def plot_model_comparison_summary(self, data: Dict[str, Any], output_dir: Path) -> None:
        """Plot summary comparison of different diploid models."""
        try:
            diploid_models = data.get("summary", {}).get("diploid_models", {})
            if not diploid_models:
                self.logger.warning("No diploid model data for summary plot")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Diploid Model Comparison Summary', fontsize=16)
            
            # Prepare data for plotting
            models = ["dominant", "recessive", "codominant"]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            
            # Plot 1: Average offspring fitness
            ax1 = axes[0, 0]
            means = []
            stds = []
            for model in models:
                model_data = diploid_models.get(model, {}).get("avg_offspring_fitness", {})
                means.append(model_data.get("mean", 0))
                stds.append(model_data.get("std", 0))
            
            bars = ax1.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax1.set_title('Average Offspring Fitness')
            ax1.set_ylabel('Fitness')
            
            # Plot 2: Fitness improvement
            ax2 = axes[0, 1]
            improvements = []
            imp_stds = []
            for model in models:
                model_data = diploid_models.get(model, {}).get("fitness_improvement", {})
                improvements.append(model_data.get("mean", 0))
                imp_stds.append(model_data.get("std", 0))
            
            bars = ax2.bar(models, improvements, yerr=imp_stds, capsize=5, color=colors, alpha=0.7)
            ax2.set_title('Fitness Improvement')
            ax2.set_ylabel('Improvement')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Plot 3: Population statistics (if available)
            ax3 = axes[1, 0]
            pop_stats = data.get("summary", {}).get("population_stats", {})
            if pop_stats:
                stats_names = list(pop_stats.keys())
                stats_means = [pop_stats[stat]["mean"] for stat in stats_names]
                bars = ax3.bar(stats_names, stats_means, color='purple', alpha=0.7)
                ax3.set_title('Population Statistics')
                ax3.set_ylabel('Value')
                plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
            else:
                ax3.text(0.5, 0.5, 'No population data', ha='center', va='center', 
                        transform=ax3.transAxes)
                ax3.set_title('Population Statistics')
            
            # Plot 4: Model effectiveness ranking
            ax4 = axes[1, 1]
            if means:
                # Rank models by average offspring fitness
                model_ranking = sorted(zip(models, means), key=lambda x: x[1], reverse=True)
                ranked_models, ranked_fitness = zip(*model_ranking)
                
                bars = ax4.bar(range(len(ranked_models)), ranked_fitness, 
                              color=[colors[models.index(m)] for m in ranked_models], alpha=0.7)
                ax4.set_title('Model Ranking by Fitness')
                ax4.set_ylabel('Average Fitness')
                ax4.set_xticks(range(len(ranked_models)))
                ax4.set_xticklabels([m.capitalize() for m in ranked_models])
            else:
                ax4.text(0.5, 0.5, 'No fitness data', ha='center', va='center', 
                        transform=ax4.transAxes)
                ax4.set_title('Model Ranking')
            
            plt.tight_layout()
            self._save_figure(fig, output_dir, "model_comparison_summary.png")
            
        except Exception as e:
            self.logger.error(f"Failed to plot model comparison summary: {e}")
            plt.close(fig)