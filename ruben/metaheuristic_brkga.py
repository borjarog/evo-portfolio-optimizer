import json
import random
import time
import numpy as np
from scipy.optimize import minimize
import warnings

class Metaheuristic:
    """
    Biased Random-Key Genetic Algorithm (BRKGA) with SLSQP.
    Features:
    - Representation: Vector of n random keys [0, 1].
    - Decoder: Sort keys -> Select Top-k -> SLSQP.
    - Elitism.
    - Parametric Uniform Crossover.
    - Random Immigrants.
    """

    def read_problem_instance(self, problem_path):
        with open(problem_path, 'r') as f:
            data = json.load(f)
        
        self.n = data['n']
        self.k_constraints = data['k']
        self.r = np.array(data['r'])
        self.R = data['R']
        self.dij = np.array(data['dij'])
        self.delta = data.get('delta', 0.01)

    def get_best_solution(self):
        return self.best_solution

    def calculate_fitness(self, solution):
        # Verification helper (same logic as before)
        w = np.array(solution)
        if np.dot(self.r, w) < self.R - 1e-6: return -float('inf')
        for val in w:
             if val > 1e-6 and val < self.delta - 1e-6: return -float('inf')
        z = 0.0
        for i in range(self.n - 1):
             if w[i] > 1e-6:
                for j in range(i + 1, self.n):
                    if w[j] > 1e-6: z += w[i] * w[j] * self.dij[i][j]
        return z

    def optimize_weights_from_keys(self, keys):
        """
        Decoder: Keys -> Indices -> Weights (SLSQP) -> Fitness
        """
        # Select top k indices based on keys (largest keys first)
        # np.argsort returns indices that sort the array, so we want the last k
        sorted_indices = np.argsort(keys)
        selected_indices = sorted(sorted_indices[-self.k_constraints:])
        
        # Optimization (SLSQP)
        k = len(selected_indices)
        r_sub = self.r[selected_indices]
        D_sub = np.zeros((k, k))
        for ia, idx_a in enumerate(selected_indices):
            for ib, idx_b in enumerate(selected_indices):
                D_sub[ia, ib] = self.dij[idx_a][idx_b]
        
        def objective(w_sub):
            return -np.sum(np.triu(np.outer(w_sub, w_sub) * D_sub, 1))

        def constraint_return(w_sub):
            return np.dot(r_sub, w_sub) - self.R

        def constraint_budget(w_sub):
            return np.sum(w_sub) - 1.0

        cons = [
            {'type': 'eq', 'fun': constraint_budget},
            {'type': 'ineq', 'fun': constraint_return}
        ]
        bounds = [(self.delta, 1.0) for _ in range(k)]
        w0 = np.array([max(self.delta, 1.0/k)] * k); w0 = w0/w0.sum()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-4, options={'disp': False, 'maxiter': 50}) # Reduced maxiter for speed
            
            if res.success:
                best_w_sub = res.x
                full_sol = [0.0] * self.n
                for i, idx in enumerate(selected_indices):
                    full_sol[idx] = best_w_sub[i]
                return -res.fun, full_sol
            else:
                return -float('inf'), [0.0] * self.n
        except:
             return -float('inf'), [0.0] * self.n

    def run(self):
        self.read_problem_instance(self.problem_path)
        self.start_time = time.time()
        self.best_fitness = -float('inf')
        self.best_solution = [0.0] * self.n
        
        # BRKGA Configuration
        pop_size = max(50, self.n) # Population size
        pe = 0.20 # Elite portion
        pm = 0.20 # Mutant (immigrant) portion
        rho = 0.70 # Inheritance probability from elite
        
        num_elite = int(pop_size * pe)
        num_mutants = int(pop_size * pm)
        num_crossover = pop_size - num_elite - num_mutants
        
        # Initialize Population (random keys)
        population = [np.random.rand(self.n) for _ in range(pop_size)]
        fitness_values = []
        
        # Initial evaluation
        for chrom in population:
            fit, sol = self.optimize_weights_from_keys(chrom)
            fitness_values.append((fit, sol, chrom)) # Store full tuple
        
        # Sort by fitness (descending)
        fitness_values.sort(key=lambda x: x[0], reverse=True)
        
        self.best_fitness = fitness_values[0][0]
        self.best_solution = fitness_values[0][1]
        
        while time.time() - self.start_time < self.time_deadline:
            
            # Classification
            elites = fitness_values[:num_elite]
            
            # Next Generation
            next_population_data = []
            
            # 1. Elitism: Copy elites
            next_population_data.extend(elites)
            
            # 2. Mutants: Generate random keys
            for _ in range(num_mutants):
                chrom = np.random.rand(self.n)
                # Assess later or now? Assess now to keep structure consistent
                fit, sol = self.optimize_weights_from_keys(chrom)
                next_population_data.append((fit, sol, chrom))
            
            # 3. Crossover
            for _ in range(num_crossover):
                # Select one elite and one non-elite
                elite_parent = random.choice(elites)[2]
                # Randomly pick non-elite? Standard BRKGA picks from entire pool or non-elite pool.
                # Standard: elite parent + random parent from population.
                other_parent = random.choice(fitness_values)[2]
                
                # Parametric Uniform Crossover
                child = np.zeros(self.n)
                for gene_idx in range(self.n):
                    if random.random() < rho:
                        child[gene_idx] = elite_parent[gene_idx]
                    else:
                        child[gene_idx] = other_parent[gene_idx]
                
                fit, sol = self.optimize_weights_from_keys(child)
                next_population_data.append((fit, sol, child))
            
            # Update Population
            fitness_values = next_population_data
            fitness_values.sort(key=lambda x: x[0], reverse=True)
            
            # Update Global Best
            if fitness_values[0][0] > self.best_fitness:
                self.best_fitness = fitness_values[0][0]
                self.best_solution = fitness_values[0][1]

    def __init__(self, time_deadline, problem_path, **kwargs):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
