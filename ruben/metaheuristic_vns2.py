import json
import random
import time
import numpy as np
from scipy.optimize import minimize
import warnings

class Metaheuristic:
    """
    VNS v2 (Variable Neighborhood Search) with SLSQP.
    Features:
    - Standard bounds (0, 1).
    - Warning suppression.
    - Robust neighbor generation.
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
        if np.dot(self.r, w) < self.R - 1e-6:
            return -float('inf')
        
        z = 0.0
        for i in range(self.n - 1):
             if w[i] > 1e-6:
                for j in range(i + 1, self.n):
                    if w[j] > 1e-6:
                        z += w[i] * w[j] * self.dij[i][j]
        return z

    def optimize_weights(self, selected_indices):
        k = len(selected_indices)
        if k == 0:
            return -float('inf'), [0.0] * self.n

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
        except Exception:
             return -float('inf'), [0.0] * self.n

    def run(self):
        self.read_problem_instance(self.problem_path)
        self.start_time = time.time()
        self.best_fitness = -float('inf')
        self.best_solution = [0.0] * self.n
        
        # Initial Solution
        current_indices = sorted(random.sample(range(self.n), self.k_constraints))
        current_fitness, current_sol = self.optimize_weights(current_indices)
        
        self.best_fitness = current_fitness
        self.best_solution = current_sol
        
        max_k_neigh = 3
        
        while time.time() - self.start_time < self.time_deadline:
            k_neigh = 1
            while k_neigh <= max_k_neigh and (time.time() - self.start_time < self.time_deadline):
                
                # 1. Shaking
                candidate_indices = list(current_indices)
                non_selected = [i for i in range(self.n) if i not in current_indices]
                
                num_swap = min(k_neigh, len(candidate_indices), len(non_selected))
                to_remove = random.sample(candidate_indices, num_swap)
                to_add = random.sample(non_selected, num_swap)
                
                shaken_indices = sorted([i for i in candidate_indices if i not in to_remove] + to_add)
                shaken_fit, shaken_sol = self.optimize_weights(shaken_indices)
                
                # 2. Local Search (First Improvement)
                best_local_indices = shaken_indices
                best_local_fit = shaken_fit
                best_local_sol = shaken_sol
                
                # Only run local search if we have time
                if time.time() - self.start_time < self.time_deadline:
                    non_selected_local = [i for i in range(self.n) if i not in best_local_indices]
                    
                    movement_space = []
                    for idx_out in best_local_indices:
                        for idx_in in non_selected_local:
                            movement_space.append((idx_out, idx_in))
                    
                    # Check up to 30 neighbors or 30% to be fast
                    sample_size = min(len(movement_space), max(30, int(len(movement_space) * 0.3)))
                    moves_to_check = random.sample(movement_space, sample_size)
                    
                    for idx_out, idx_in in moves_to_check:
                        neighbor_indices = sorted([i for i in best_local_indices if i != idx_out] + [idx_in])
                        fit, sol = self.optimize_weights(neighbor_indices)
                        
                        if fit > best_local_fit:
                            best_local_fit = fit
                            best_local_indices = neighbor_indices
                            best_local_sol = sol
                            # First Improvement break
                            break
                
                # 3. Move or Next Neighborhood
                if best_local_fit > current_fitness:
                    current_fitness = best_local_fit
                    current_indices = best_local_indices
                    current_sol = best_local_sol
                    k_neigh = 1
                    
                    if current_fitness > self.best_fitness:
                        self.best_fitness = current_fitness
                        self.best_solution = current_sol
                else:
                    k_neigh += 1

    def __init__(self, time_deadline, problem_path, **kwargs):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
