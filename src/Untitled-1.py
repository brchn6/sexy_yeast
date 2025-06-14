#!/usr/bin/env python3
"""
Debug script to analyze the data structure being passed to visualizers.
"""

def analyze_data_structure(data, level=0, max_level=3):
    """Recursively analyze data structure."""
    indent = "  " * level
    
    if level > max_level:
        return f"{indent}... (max depth reached)"
    
    if isinstance(data, dict):
        result = f"{indent}Dict with keys: {list(data.keys())}\n"
        for key, value in data.items():
            if key in ['runs', 'summary', 'diploid_offspring', 'collector_data']:
                result += f"{indent}{key}:\n"
                result += analyze_data_structure(value, level + 1, max_level)
            elif isinstance(value, (list, dict)) and len(str(value)) > 100:
                result += f"{indent}{key}: {type(value).__name__} (size: {len(value) if hasattr(value, '__len__') else 'unknown'})\n"
            else:
                result += f"{indent}{key}: {type(value).__name__} = {str(value)[:100]}\n"
        return result
    elif isinstance(data, list):
        if not data:
            return f"{indent}Empty list\n"
        result = f"{indent}List with {len(data)} items\n"
        if data:
            result += f"{indent}First item type: {type(data[0]).__name__}\n"
            result += analyze_data_structure(data[0], level + 1, max_level)
        return result
    else:
        return f"{indent}{type(data).__name__}: {str(data)[:100]}\n"

# Let's trace where the data comes from:
print("Data Flow Analysis:")
print("1. SimulationDataCollector.add_run_data() receives run_data")
print("2. SimulationDataCollector.get_all_data() returns data")
print("3. MultiSimulationVisualizer.plot_all_analyses() receives this data")
print("4. Need to check what's actually in run_data['diploid_offspring']")


