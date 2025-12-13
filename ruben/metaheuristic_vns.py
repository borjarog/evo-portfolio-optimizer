import json
import random
import time
import numpy as np
from scipy.optimize import minimize
import warnings

class Metaheuristic:
    """
    Variable Neighborhood Search (VNS) metaheuristic with SLSQP.
    Features:
    - Neighborhoods: Swap k assets (k=1, 2, 3).
    - Local Search: Best Improvement in k=1 neighborhood.
    - Shaking: Random Swap in k neighborhood.
    - SLSQP for weight optimization.
    - Delta constraint handling.
    """

    def read_problem_instance(self, problem_path):
        with open(problem_path, 'r') as f:
            data = json.load(f)
        
        self.n = data['n']
        self.k_constraints = data['k'] # Rename to avoid confusion with VNS k
        self.r = np.array(data['r'])
        self.R = data['R']
        self.dij = np.array(data['dij'])
        self.delta = data.get('delta', 0.01)

    def get_best_solution(self):
        return self.best_solution

    def calculate_fitness(self, solution):
        """
        Calculates fitness for a full solution vector w.
        """
        w = np.array(solution)
        expected_return = np.dot(self.r, w)
        if expected_return < self.R - 1e-6:
            return -float('inf')
        
        for val in w:
            if val > 1e-6 and val < self.delta - 1e-6:
                return -float('inf')

        z = 0.0
        # z = sum_{i<j} w_i w_j d_ij
        for i in range(self.n - 1):
             if w[i] > 1e-6:
                for j in range(i + 1, self.n):
                    if w[j] > 1e-6:
                        z += w[i] * w[j] * self.dij[i][j]
        return z

    def optimize_weights(self, selected_indices):
        """
        Optimizes weights for the selected asset indices using SLSQP.
        """
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
        
        bounds = [(self.delta, 1.0) for _ in range(k)]
        
        w0 = np.array([max(self.delta, 1.0/k)] * k)
        w0 = w0 / w0.sum()
        
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
                    options={'disp': False, 'maxiter': 100}
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
        
        # VNS Variables
        max_k_neigh = 3 # Max neighborhood size (1-swap, 2-swap, 3-swap)
        
        while time.time() - self.start_time < self.time_deadline:
            k_neigh = 1
            while k_neigh <= max_k_neigh and (time.time() - self.start_time < self.time_deadline):
                # 1. Shaking
                # Generate a random neighbor in N_k(current)
                # Swap k_neigh items
                candidate_indices = list(current_indices)
                non_selected = [i for i in range(self.n) if i not in current_indices]
                
                # Ensure we have enough items to swap
                num_swap = min(k_neigh, len(candidate_indices), len(non_selected))
                
                to_remove = random.sample(candidate_indices, num_swap)
                to_add = random.sample(non_selected, num_swap)
                
                shaken_indices = sorted([i for i in candidate_indices if i not in to_remove] + to_add)
                
                # Evaluate shaken solution
                shaken_fit, shaken_sol = self.optimize_weights(shaken_indices)
                
                # 2. Local Search (Best Improvement in 1-swap neighborhood of shaken solution)
                # We perform a quick local search starting from shaken solution
                # Limit local search iterations or neighborhood check to save time
                
                best_local_indices = shaken_indices
                best_local_fit = shaken_fit
                best_local_sol = shaken_sol
                
                improved_local = True
                while improved_local and (time.time() - self.start_time < self.time_deadline):
                    improved_local = False
                    
                    non_selected_local = [i for i in range(self.n) if i not in best_local_indices]
                    
                    # Sample neighborhood for speed (Adaptive)
                    movement_space = []
                    for idx_out in best_local_indices:
                        for idx_in in non_selected_local:
                            movement_space.append((idx_out, idx_in))
                    
                    sample_size = min(len(movement_space), max(20, int(len(movement_space) * 0.2)))
                    moves_to_check = random.sample(movement_space, sample_size)
                    
                    for idx_out, idx_in in moves_to_check:
                        neighbor_indices = sorted([i for i in best_local_indices if i != idx_out] + [idx_in])
                        fit, sol = self.optimize_weights(neighbor_indices)
                        
                        if fit > best_local_fit:
                            best_local_fit = fit
                            best_local_indices = neighbor_indices
                            best_local_sol = sol
                            improved_local = True
                            # First Improvement strategy for speed inside VNS
                            break 
                
                # 3. Move or Next Neighborhood
                if best_local_fit > current_fitness:
                    current_fitness = best_local_fit
                    current_indices = best_local_indices
                    current_sol = best_local_sol
                    k_neigh = 1 # Improvements found, reset to smallest neighborhood
                    
                    # Update Global Best
                    if current_fitness > self.best_fitness:
                        self.best_fitness = current_fitness
                        self.best_solution = current_sol
                else:
                    k_neigh += 1 # No improvement, look in larger neighborhood

    def __init__(self, time_deadline, problem_path, **kwargs):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
