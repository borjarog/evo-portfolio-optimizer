import json
import random
import time
import numpy as np
from scipy.optimize import minimize
import warnings

class Metaheuristic:
    """
    BRKGA v2 with SLSQP.
    Features:
    - Standard bounds (0, 1).
    - Warning suppression.
    - Robust logic.
    """

    def read_problem_instance(self, problem_path):
        with open(problem_path, 'r') as f:
            data = json.load(f)
        
        self.n = data['n']
        self.k_constraints = data['k']
        self.r = np.array(data['r'])
        self.R = data['R']
        self.dij = np.array(data['dij'])

    def get_best_solution(self):
        return self.best_solution

    def calculate_fitness(self, solution):
        w = np.array(solution)
        if np.dot(self.r, w) < self.R - 1e-6: return -float('inf')
        
        z = 0.0
        for i in range(self.n - 1):
             if w[i] > 1e-6:
                for j in range(i + 1, self.n):
                    if w[j] > 1e-6: z += w[i] * w[j] * self.dij[i][j]
        return z

    def optimize_weights_from_keys(self, keys):
        sorted_indices = np.argsort(keys)
        selected_indices = sorted(sorted_indices[-self.k_constraints:])
        
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
        # Standard bounds
        bounds = [(0.0, 1.0) for _ in range(k)]
        w0 = np.array([1.0/k] * k)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    objective, 
                    w0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=cons, 
                    tol=1e-4, 
                    options={'disp': False, 'maxiter': 50}
                )
            
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
        
        pop_size = max(50, self.n)
        pe = 0.20
        pm = 0.20
        rho = 0.70
        
        num_elite = int(pop_size * pe)
        num_mutants = int(pop_size * pm)
        num_crossover = pop_size - num_elite - num_mutants
        
        population = [np.random.rand(self.n) for _ in range(pop_size)]
        fitness_values = []
        
        for chrom in population:
            fit, sol = self.optimize_weights_from_keys(chrom)
            fitness_values.append((fit, sol, chrom))
        
        fitness_values.sort(key=lambda x: x[0], reverse=True)
        
        self.best_fitness = fitness_values[0][0]
        self.best_solution = fitness_values[0][1]
        
        while time.time() - self.start_time < self.time_deadline:
            
            elites = fitness_values[:num_elite]
            next_population_data = []
            
            next_population_data.extend(elites)
            
            for _ in range(num_mutants):
                chrom = np.random.rand(self.n)
                fit, sol = self.optimize_weights_from_keys(chrom)
                next_population_data.append((fit, sol, chrom))
            
            for _ in range(num_crossover):
                elite_parent = random.choice(elites)[2]
                other_parent = random.choice(fitness_values)[2]
                
                child = np.zeros(self.n)
                # Vectorized crossover if possible, but loop is fine
                for gene_idx in range(self.n):
                    if random.random() < rho:
                        child[gene_idx] = elite_parent[gene_idx]
                    else:
                        child[gene_idx] = other_parent[gene_idx]
                
                fit, sol = self.optimize_weights_from_keys(child)
                next_population_data.append((fit, sol, child))
            
            fitness_values = next_population_data
            fitness_values.sort(key=lambda x: x[0], reverse=True)
            
            if fitness_values[0][0] > self.best_fitness:
                self.best_fitness = fitness_values[0][0]
                self.best_solution = fitness_values[0][1]

    def __init__(self, time_deadline, problem_path, **kwargs):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
