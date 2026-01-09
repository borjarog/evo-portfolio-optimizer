"""
Experiments Module for Enhanced Portfolio Optimizer

This module provides the experimental framework required for the course project:
1. Hyperparameter optimization through grid search
2. Statistical validation with multiple runs
3. Performance comparison and analysis
4. Results export and documentation

Usage:
    python src/experiments.py
"""

import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

from metaheuristic import Metaheuristic

# ==================== CONFIGURATION ====================

# Experimental settings
N_RUNS = 5          # Replications per configuration (for final evaluation)
N_RUNS_OPTIMIZATION = 3  # Replications for hyperparameter optimization (faster)
DEADLINE = 60       # Seconds per run (tournament specification)
SEED_BASE = 42      # Base seed for reproducibility

# Instance directory
INSTANCE_DIR = Path(__file__).parent.parent / "data" / "instances"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Enhanced hyperparameter grid
PARAM_GRID = {
    'vns_probability': [0.2, 0.3, 0.4],
    'local_search_interval': [8, 10, 12],
    'restart_threshold_factor': [0.8, 1.0, 1.2]
}

# Fast mode: reduced grid for quick testing
PARAM_GRID_FAST = {
    'vns_probability': [0.2, 0.4],  # Only extremes + middle
    'local_search_interval': [8, 12],
    'restart_threshold_factor': [0.8, 1.2]
}

# Representative instances for optimization (small, medium, large)
# Format: (n, k) tuples to match instances
REPRESENTATIVE_INSTANCES = [
    'instance_n50_k2_1.json',      # Small
    'instance_n100_k10_7.json',    # Medium-small
    'instance_n500_k25_9.json',    # Medium-large
    'instance_n1000_k50_14.json'   # Large
]

# Default optimized configuration
DEFAULT_CONFIG = {
    'vns_probability': 0.3,
    'local_search_interval': 10,
    'restart_threshold_factor': 1.0
}

# ==================== HELPER FUNCTIONS ====================

def get_instance_files(instance_dir=INSTANCE_DIR):
    """Get all instance JSON files from directory."""
    instance_dir = Path(instance_dir)
    if not instance_dir.exists():
        raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

    files = sorted(instance_dir.glob("*.json"))
    return files

def run_single_experiment(instance_path, config=None, deadline=DEADLINE, seed=None):
    """
    Run a single experiment with given configuration.

    Args:
        instance_path: Path to instance file
        config: Algorithm configuration parameters
        deadline: Time limit in seconds
        seed: Random seed for reproducibility

    Returns:
        dict with fitness, time, and success status
    """
    if seed is not None:
        np.random.seed(seed)

    if config is None:
        config = DEFAULT_CONFIG

    try:
        solver = Metaheuristic(
            time_deadline=deadline,
            problem_path=str(instance_path),
            **config
        )

        t_start = time.time()
        solver.run()
        t_elapsed = time.time() - t_start

        fitness = solver.best_fitness
        solution = solver.get_best_solution()

        return {
            'fitness': fitness,
            'time': t_elapsed,
            'success': True,
            'solution': solution
        }

    except Exception as e:
        return {
            'fitness': -float('inf'),
            'time': deadline,
            'success': False,
            'error': str(e)
        }

def run_multiple_experiments(instance_path, config=None, n_runs=N_RUNS, deadline=DEADLINE, seed_base=SEED_BASE):
    """
    Run multiple experiments and return statistics.

    Args:
        instance_path: Path to instance file
        config: Algorithm configuration
        n_runs: Number of replications
        deadline: Time limit per run
        seed_base: Base seed for reproducibility

    Returns:
        Dictionary with statistical results
    """
    results = []
    for i in range(n_runs):
        seed = seed_base + i
        result = run_single_experiment(instance_path, config, deadline, seed)
        results.append(result)

    # Calculate statistics
    fitnesses = [r['fitness'] for r in results if r['success'] and r['fitness'] > -float('inf')]
    times = [r['time'] for r in results if r['success']]

    return {
        'mean_fitness': np.mean(fitnesses) if fitnesses else -float('inf'),
        'std_fitness': np.std(fitnesses) if len(fitnesses) > 1 else 0,
        'best_fitness': max(fitnesses) if fitnesses else -float('inf'),
        'worst_fitness': min(fitnesses) if fitnesses else -float('inf'),
        'median_fitness': np.median(fitnesses) if fitnesses else -float('inf'),
        'mean_time': np.mean(times) if times else deadline,
        'success_rate': len(fitnesses) / n_runs,
        'n_runs': n_runs,
        'all_results': results
    }

# ==================== HYPERPARAMETER OPTIMIZATION ====================

def hyperparameter_optimization(instances=None, output_file='hyperparameter_results.csv', 
                                fast_mode=False, param_grid=None, n_runs_opt=None):
    """
    Perform grid search over hyperparameters.

    Args:
        instances: List of instance files (None for all or representative subset)
        output_file: Output CSV file name
        fast_mode: If True, use reduced grid and representative instances
        param_grid: Custom parameter grid (None for default)
        n_runs_opt: Number of runs for optimization (None for default)

    Returns:
        DataFrame with results and best configuration
    """
    if param_grid is None:
        param_grid = PARAM_GRID_FAST if fast_mode else PARAM_GRID
    
    if n_runs_opt is None:
        n_runs_opt = N_RUNS_OPTIMIZATION if fast_mode else N_RUNS
    
    if instances is None:
        if fast_mode:
            # Use representative subset for faster optimization
            all_instances = get_instance_files()
            instances = [inst for inst in all_instances 
                        if inst.name in REPRESENTATIVE_INSTANCES]
            if not instances:
                # Fallback: use first 4 instances
                instances = all_instances[:4]
        else:
            # Use all available instances for comprehensive optimization
            # as recommended in project statement: "employ different instances 
            # when optimizing the configuration of the metaheuristic"
            instances = get_instance_files()

    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION")
    if fast_mode:
        print("⚡ FAST MODE ENABLED ⚡")
    print(f"Parameters: {list(param_grid.keys())}")
    print(f"Instances: {len(instances)}")
    print(f"Runs per config: {n_runs_opt}")
    print("=" * 70)

    results = []

    # Test each parameter independently
    for param_name, param_values in param_grid.items():
        print(f"\n>>> Optimizing parameter: {param_name}")

        for param_value in param_values:
            config = DEFAULT_CONFIG.copy()
            config[param_name] = param_value

            config_str = f"{param_name}={param_value}"
            print(f"  Testing: {config_str}")

            param_results = []

            for instance_path in instances:
                instance_name = instance_path.name

                stats = run_multiple_experiments(instance_path, config, n_runs=n_runs_opt)
                param_results.append(stats['mean_fitness'])

                results.append({
                    'parameter': param_name,
                    'value': param_value,
                    'instance': instance_name,
                    'mean_fitness': stats['mean_fitness'],
                    'std_fitness': stats['std_fitness'],
                    'best_fitness': stats['best_fitness'],
                    'success_rate': stats['success_rate']
                })

            avg_fitness = np.mean([f for f in param_results if f > -float('inf')])
            print(f"    Average fitness: {avg_fitness:.6f}")

    # Save results
    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / output_file
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    # Find best configuration
    print("\n" + "=" * 50)
    print("OPTIMAL CONFIGURATION")
    print("=" * 50)

    best_config = DEFAULT_CONFIG.copy()

    for param_name in PARAM_GRID.keys():
        param_data = df[df['parameter'] == param_name].groupby('value')['mean_fitness'].mean()
        if len(param_data) > 0:
            best_value = param_data.idxmax()
            best_fitness = param_data.max()
            print(f"{param_name}: {best_value} (avg fitness: {best_fitness:.6f})")
            best_config[param_name] = best_value

    print(f"\n>>> Results saved to: {output_path}")
    return df, best_config

# ==================== PERFORMANCE EVALUATION ====================

def performance_evaluation(instances=None, config=None, output_file='performance_results.csv'):
    """
    Evaluate algorithm performance on all instances.

    Args:
        instances: List of instance files (None for all)
        config: Algorithm configuration (None for default)
        output_file: Output CSV file name

    Returns:
        DataFrame with performance results
    """
    if instances is None:
        instances = get_instance_files()

    if config is None:
        config = DEFAULT_CONFIG

    print("=" * 70)
    print("PERFORMANCE EVALUATION")
    print(f"Configuration: {config}")
    print(f"Instances: {len(instances)}")
    print(f"Runs per instance: {N_RUNS}")
    print("=" * 70)

    results = []

    for i, instance_path in enumerate(instances, 1):
        instance_name = instance_path.name

        # Get instance info
        with open(instance_path, 'r') as f:
            data = json.load(f)
            n, k = data['n'], data['k']

        print(f"\n[{i}/{len(instances)}] {instance_name} (n={n}, k={k})")

        stats = run_multiple_experiments(instance_path, config)

        results.append({
            'instance': instance_name,
            'n': n,
            'k': k,
            'mean_fitness': stats['mean_fitness'],
            'std_fitness': stats['std_fitness'],
            'best_fitness': stats['best_fitness'],
            'worst_fitness': stats['worst_fitness'],
            'median_fitness': stats['median_fitness'],
            'mean_time': stats['mean_time'],
            'success_rate': stats['success_rate']
        })

        print(f"  Mean: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
        print(f"  Best: {stats['best_fitness']:.6f}")
        print(f"  Time: {stats['mean_time']:.1f}s")

    # Save results
    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / output_file
    df.to_csv(output_path, index=False)

    # Summary statistics
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)

    valid_results = df[df['mean_fitness'] > -float('inf')]

    print(f"Successful instances: {len(valid_results)}/{len(df)}")
    print(f"Average mean fitness: {valid_results['mean_fitness'].mean():.6f}")
    print(f"Average best fitness: {valid_results['best_fitness'].mean():.6f}")
    print(f"Average computation time: {valid_results['mean_time'].mean():.1f}s")
    print(f"Average success rate: {valid_results['success_rate'].mean():.1%}")

    print(f"\n>>> Results saved to: {output_path}")
    return df

# ==================== STATISTICAL ANALYSIS ====================

def statistical_analysis(results_df):
    """
    Perform statistical analysis of results.

    Args:
        results_df: DataFrame with performance results
        comparison_file: Optional CSV file with previous results for comparison
    """
    print("\n" + "=" * 50)
    print("STATISTICAL ANALYSIS")
    print("=" * 50)

    # Basic statistics
    valid_results = results_df[results_df['mean_fitness'] > -float('inf')]

    print(f"Sample size: {len(valid_results)}")
    print(f"Mean fitness: {valid_results['mean_fitness'].mean():.6f}")
    print(f"Standard deviation: {valid_results['mean_fitness'].std():.6f}")
    print(f"Median fitness: {valid_results['mean_fitness'].median():.6f}")
    print(f"Min fitness: {valid_results['mean_fitness'].min():.6f}")
    print(f"Max fitness: {valid_results['mean_fitness'].max():.6f}")

    # Statistical tests for significance
    if len(valid_results) >= 5:
        try:
            from scipy.stats import normaltest
            # Test for normality
            stat, p_norm = normaltest(valid_results['mean_fitness'])
            normal_dist = p_norm > 0.05

            print(f"Normality test p-value: {p_norm:.6f} (normal: {normal_dist})")

            # Confidence interval for mean
            from scipy.stats import t
            mean_fit = valid_results['mean_fitness'].mean()
            std_fit = valid_results['mean_fitness'].std()
            n = len(valid_results)

            confidence = 0.95
            dof = n - 1
            t_critical = t.ppf((1 + confidence) / 2, dof)
            margin_error = t_critical * (std_fit / np.sqrt(n))

            print(f"95% confidence interval: [{mean_fit - margin_error:.6f}, {mean_fit + margin_error:.6f}]")

        except:
            print("Advanced statistical tests not available")

# ==================== MAIN EXECUTION ====================

def main(fast_mode=False):
    """
    Main experimental execution.
    
    Args:
        fast_mode: If True, use reduced configuration for faster execution
    """
    print("Enhanced Portfolio Optimizer - Experimental Framework")
    if fast_mode:
        print("⚡ FAST MODE: Reduced instances and runs for quicker results ⚡")
    print("=" * 70)

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Get instances
    all_instances = get_instance_files()
    print(f"Found {len(all_instances)} total instances")

    # 1. Hyperparameter optimization
    if fast_mode:
        print("\n1. HYPERPARAMETER OPTIMIZATION (FAST MODE)")
        print("   Using representative instances and reduced grid")
        param_df, best_config = hyperparameter_optimization(
            instances=None, 
            fast_mode=True,
            output_file='hyperparameter_results_fast.csv'
        )
    else:
        print("\n1. HYPERPARAMETER OPTIMIZATION")
        print("   Using all instances as recommended: 'employ different instances")
        print("   when optimizing the configuration of the metaheuristic'")
        param_df, best_config = hyperparameter_optimization(
            instances=all_instances,
            fast_mode=False
        )

    # 2. Performance evaluation with best configuration (always use all instances)
    print("\n2. PERFORMANCE EVALUATION")
    print("   Evaluating on ALL instances with best configuration")
    perf_df = performance_evaluation(all_instances, best_config)

    # 3. Statistical analysis
    print("\n3. STATISTICAL ANALYSIS")
    statistical_analysis(perf_df)

    print("\n" + "=" * 70)
    print("EXPERIMENTAL FRAMEWORK COMPLETED")
    print("Check the 'results/' directory for detailed outputs")
    print("=" * 70)

def main_fast():
    """Convenience function for fast mode execution."""
    main(fast_mode=True)

if __name__ == "__main__":
    import sys
    
    # Check for fast mode argument
    fast_mode = "--fast" in sys.argv or "-f" in sys.argv
    
    try:
        import pandas as pd
        if fast_mode:
            print("🚀 Starting experiments in FAST MODE...")
            main_fast()
        else:
            print("🔬 Starting experiments in FULL MODE...")
            print("   (Use --fast or -f flag for faster execution)")
            main()
    except ImportError as e:
        print(f"Error: Required package not installed: {e}")
        print("Install dependencies with: pip install pandas numpy scipy")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()