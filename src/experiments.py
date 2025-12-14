"""
Hyperparameter Tuning Experiments for Hybrid DE + Memetic Portfolio Optimizer

This module provides:
1. Grid search over key hyperparameters
2. Statistical comparison across multiple runs
3. Results export to CSV
4. Comparison with baseline approaches
"""

import os
import sys
import time
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from func_timeout import func_timeout, FunctionTimedOut

# Import our metaheuristic
from metaheuristic import Metaheuristic


# ==================== CONFIGURATION ====================

# Instance directory (relative to this file or absolute)
INSTANCE_DIR = Path(__file__).parent.parent / "data" / "instances"

# Experiment settings
N_RUNS = 5          # Replications per configuration
DEADLINE = 60       # Seconds per run
SEED_BASE = 42      # Base seed for reproducibility

# Hyperparameter grid for tuning
PARAM_GRID = {
    'pop_size': [30, 50, 70],
    'F': [0.5, 0.7, 0.9],
    'CR': [0.7, 0.85, 0.95],
    'local_search_rate': [0.2, 0.3, 0.4],
    'restart_threshold': [20, 30, 50]
}

# Default (best) configuration
DEFAULT_CONFIG = {
    'pop_size': 50,
    'F': 0.7,
    'CR': 0.85,
    'local_search_rate': 0.3,
    'restart_threshold': 30
}


# ==================== HELPER FUNCTIONS ====================

def get_instance_files(instance_dir=INSTANCE_DIR):
    """Get all instance JSON files from directory."""
    instance_dir = Path(instance_dir)
    if not instance_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")
    
    files = sorted(instance_dir.glob("*.json"))
    return files


def run_single_experiment(instance_path, config, deadline, seed=None):
    """
    Run a single experiment with given configuration.
    
    Returns:
        dict with fitness, time, and success status
    """
    if seed is not None:
        np.random.seed(seed)
    
    try:
        solver = Metaheuristic(
            time_deadline=deadline,
            problem_path=str(instance_path),
            **config
        )
        
        t_start = time.time()
        func_timeout(deadline, solver.run)
        t_elapsed = time.time() - t_start
        
        fitness = getattr(solver, 'best_fitness', -float('inf'))
        solution = solver.get_best_solution()
        
        return {
            'fitness': fitness,
            'time': t_elapsed,
            'success': True,
            'solution': solution
        }
    
    except FunctionTimedOut:
        fitness = getattr(solver, 'best_fitness', -float('inf'))
        return {
            'fitness': fitness,
            'time': deadline,
            'success': True,
            'solution': solver.get_best_solution() if hasattr(solver, 'best_solution') else None
        }
    
    except Exception as e:
        return {
            'fitness': -float('inf'),
            'time': 0,
            'success': False,
            'error': str(e)
        }


def run_multiple_experiments(instance_path, config, deadline, n_runs, seed_base=SEED_BASE):
    """
    Run multiple experiments and return statistics.
    """
    results = []
    for i in range(n_runs):
        seed = seed_base + i
        result = run_single_experiment(instance_path, config, deadline, seed)
        results.append(result)
    
    fitnesses = [r['fitness'] for r in results if r['success']]
    times = [r['time'] for r in results if r['success']]
    
    return {
        'mean_fitness': np.mean(fitnesses) if fitnesses else -float('inf'),
        'std_fitness': np.std(fitnesses) if len(fitnesses) > 1 else 0,
        'best_fitness': max(fitnesses) if fitnesses else -float('inf'),
        'worst_fitness': min(fitnesses) if fitnesses else -float('inf'),
        'mean_time': np.mean(times) if times else 0,
        'success_rate': len(fitnesses) / n_runs,
        'n_runs': n_runs,
        'all_results': results
    }


# ==================== GRID SEARCH ====================

def grid_search(instances, param_grid=PARAM_GRID, base_config=DEFAULT_CONFIG,
                n_runs=N_RUNS, deadline=DEADLINE, output_file='grid_search_results.csv'):
    """
    Perform grid search over hyperparameters.
    Tests one parameter at a time while keeping others at default.
    """
    results = []
    
    print("=" * 60)
    print("GRID SEARCH EXPERIMENT")
    print(f"Instances: {len(instances)}")
    print(f"Runs per config: {n_runs}")
    print(f"Deadline: {deadline}s")
    print("=" * 60)
    
    # Test each parameter independently
    for param_name, param_values in param_grid.items():
        print(f"\n>>> Testing parameter: {param_name}")
        
        for param_value in param_values:
            config = base_config.copy()
            config[param_name] = param_value
            
            config_str = f"{param_name}={param_value}"
            print(f"  Config: {config_str}")
            
            for instance_path in instances:
                instance_name = instance_path.name
                
                stats = run_multiple_experiments(
                    instance_path, config, deadline, n_runs
                )
                
                results.append({
                    'instance': instance_name,
                    'param_name': param_name,
                    'param_value': param_value,
                    'mean_fitness': stats['mean_fitness'],
                    'std_fitness': stats['std_fitness'],
                    'best_fitness': stats['best_fitness'],
                    'mean_time': stats['mean_time'],
                    'success_rate': stats['success_rate']
                })
                
                print(f"    {instance_name}: mean={stats['mean_fitness']:.4f}, "
                      f"std={stats['std_fitness']:.4f}, best={stats['best_fitness']:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n>>> Results saved to: {output_file}")
    
    return df


def find_best_configuration(grid_results_df):
    """
    Analyze grid search results to find optimal configuration.
    """
    # Group by parameter and value, calculate average fitness across all instances
    param_analysis = grid_results_df.groupby(['param_name', 'param_value']).agg({
        'mean_fitness': 'mean',
        'std_fitness': 'mean',
        'best_fitness': 'mean'
    }).reset_index()
    
    best_config = DEFAULT_CONFIG.copy()
    
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION ANALYSIS")
    print("=" * 60)
    
    for param_name in PARAM_GRID.keys():
        param_data = param_analysis[param_analysis['param_name'] == param_name]
        if len(param_data) > 0:
            best_row = param_data.loc[param_data['mean_fitness'].idxmax()]
            best_value = best_row['param_value']
            best_config[param_name] = best_value
            print(f"{param_name}: {best_value} (mean fitness: {best_row['mean_fitness']:.4f})")
    
    print("\nOptimal configuration:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    return best_config


# ==================== COMPARISON EXPERIMENT ====================

def compare_with_baseline(instances, our_config=DEFAULT_CONFIG, 
                          n_runs=N_RUNS, deadline=DEADLINE,
                          output_file='comparison_results.csv'):
    """
    Compare our metaheuristic with baseline configurations.
    """
    results = []
    
    print("=" * 60)
    print("COMPARISON EXPERIMENT")
    print("=" * 60)
    
    for instance_path in instances:
        instance_name = instance_path.name
        print(f"\n>>> Instance: {instance_name}")
        
        # Run our algorithm
        stats = run_multiple_experiments(
            instance_path, our_config, deadline, n_runs
        )
        
        results.append({
            'instance': instance_name,
            'algorithm': 'HybridDE',
            'mean_fitness': stats['mean_fitness'],
            'std_fitness': stats['std_fitness'],
            'best_fitness': stats['best_fitness'],
            'mean_time': stats['mean_time']
        })
        
        print(f"  HybridDE: mean={stats['mean_fitness']:.4f}, best={stats['best_fitness']:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n>>> Results saved to: {output_file}")
    
    return df


# ==================== QUICK TEST ====================

def quick_test(instance_path=None, config=DEFAULT_CONFIG, deadline=60):
    """
    Quick test on a single instance.
    """
    if instance_path is None:
        instances = get_instance_files()
        if not instances:
            print("No instances found!")
            return
        instance_path = instances[0]
    
    print(f"Quick test on: {instance_path}")
    print(f"Config: {config}")
    print(f"Deadline: {deadline}s")
    print("-" * 40)
    
    result = run_single_experiment(instance_path, config, deadline, seed=42)
    
    print(f"Fitness: {result['fitness']:.6f}")
    print(f"Time: {result['time']:.2f}s")
    print(f"Success: {result['success']}")
    
    if result.get('solution'):
        non_zero = sum(1 for w in result['solution'] if w > 0.001)
        print(f"Non-zero weights: {non_zero}")
    
    return result


# ==================== FULL BENCHMARK ====================

def full_benchmark(n_runs=5, deadline=60, output_dir='results'):
    """
    Run complete benchmark on all instances.
    """
    instances = get_instance_files()
    
    if not instances:
        print("No instances found!")
        return
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("FULL BENCHMARK")
    print(f"Instances: {len(instances)}")
    print(f"Runs per instance: {n_runs}")
    print(f"Deadline: {deadline}s")
    print("=" * 60)
    
    results = []
    
    for instance_path in instances:
        instance_name = instance_path.name
        print(f"\n>>> {instance_name}")
        
        stats = run_multiple_experiments(
            instance_path, DEFAULT_CONFIG, deadline, n_runs
        )
        
        results.append({
            'instance': instance_name,
            'mean_fitness': stats['mean_fitness'],
            'std_fitness': stats['std_fitness'],
            'best_fitness': stats['best_fitness'],
            'worst_fitness': stats['worst_fitness'],
            'mean_time': stats['mean_time'],
            'success_rate': stats['success_rate']
        })
        
        print(f"  Mean: {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
        print(f"  Best: {stats['best_fitness']:.4f}")
        print(f"  Time: {stats['mean_time']:.2f}s")
    
    # Save results
    df = pd.DataFrame(results)
    output_file = output_dir / 'benchmark_results.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_file}")
    
    return df


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning Experiments')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'benchmark', 'grid', 'compare'],
                        help='Experiment mode')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--deadline', type=int, default=60, help='Time limit per run')
    parser.add_argument('--instance', type=str, default=None, help='Specific instance to test')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        instance = Path(args.instance) if args.instance else None
        quick_test(instance, deadline=args.deadline)
    
    elif args.mode == 'benchmark':
        full_benchmark(n_runs=args.runs, deadline=args.deadline)
    
    elif args.mode == 'grid':
        instances = get_instance_files()
        # Use subset for faster grid search
        grid_search(instances[:3], n_runs=args.runs, deadline=args.deadline)
    
    elif args.mode == 'compare':
        instances = get_instance_files()
        compare_with_baseline(instances, n_runs=args.runs, deadline=args.deadline)

