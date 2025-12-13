import json
import random
import time
import numpy as np
from scipy.optimize import minimize
import warnings

class Metaheuristic:
    """
    Improved Tabu Search metaheuristic with SLSQP.
    Features:
    - Minimum investment constraint (delta).
    - Aspiration criteria.
    - Adaptive neighborhood sampling.
    """

    def read_problem_instance(self, problem_path):
        with open(problem_path, 'r') as f:
            data = json.load(f)
        
        self.n = data['n']
        self.k = data['k']
        self.r = np.array(data['r'])
        self.R = data['R']
        self.dij = np.array(data['dij'])
        # Default delta to 0.01 if not present
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
        
        # Check delta constraint
        for val in w:
            if val > 1e-6 and val < self.delta - 1e-6:
                return -float('inf')

        z = 0.0
        # Efficient calculation if d_ij is symmetric/full
        # We assume dij is list of lists
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
        
        # Precompute matrix for objective to speed up? 
        # For small k (2-10), double loop in python is bearable, but numpy is better.
        # Let's extract D_sub
        D_sub = np.zeros((k, k))
        for ia, idx_a in enumerate(selected_indices):
            for ib, idx_b in enumerate(selected_indices):
                D_sub[ia, ib] = self.dij[idx_a][idx_b]
        
        def objective(w_sub):
            # Maximize 0.5 * w.T * D * w (if symmetric)
            # The problem is sum_{i<j} w_i w_j d_{ij} which is 0.5 * w.T * D * w if D is symmetric with 0 diag
            # Let's stick to the summation form for correctness with provided dij
            # z = sum_{i<j} w_i w_j d_{ij}
            # Using matrix op: triu excluding diag?
            # z = np.sum(np.triu(np.outer(w_sub, w_sub) * D_sub, 1))
            # using outer product:
            return -np.sum(np.triu(np.outer(w_sub, w_sub) * D_sub, 1))

        def constraint_return(w_sub):
            return np.dot(r_sub, w_sub) - self.R

        def constraint_budget(w_sub):
            return np.sum(w_sub) - 1.0

        cons = [
            {'type': 'eq', 'fun': constraint_budget},
            {'type': 'ineq', 'fun': constraint_return}
        ]
        
        # Bounds: delta <= w_i <= 1.0
        bounds = [(self.delta, 1.0) for _ in range(k)]
        
        # Initial guess
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
        current_indices = sorted(random.sample(range(self.n), self.k))
        current_fitness, current_sol = self.optimize_weights(current_indices)
        
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_solution = current_sol
        
        tabu_list = [] # Store indices that are FORBIDDEN to be ADDED (recently dropped)
        tabu_tenure = 7 # Fixed tenure or adaptive
        no_improve_count = 0
        max_no_improve = 100 
        
        while time.time() - self.start_time < self.time_deadline:
            
            non_selected = [i for i in range(self.n) if i not in current_indices]
            
            best_neighbor_fitness = -float('inf')
            best_neighbor_indices = None
            best_neighbor_sol = None
            best_move_dropped_node = None
            
            movement_space = []
            for idx_out in current_indices:
                for idx_in in non_selected:
                    movement_space.append((idx_out, idx_in))

            # Adaptive Sampling
            total_moves = len(movement_space)
            sample_size = min(total_moves, max(100, int(total_moves * 0.5)))
            
            if sample_size < total_moves:
                moves_to_evaluate = random.sample(movement_space, sample_size)
            else:
                moves_to_evaluate = movement_space
            
            move_found_in_iter = False
            
            for idx_out, idx_in in moves_to_evaluate:
                
                # Setup candidate
                candidate_indices = sorted([i for i in current_indices if i != idx_out] + [idx_in])
                
                # Check Tabu (is 'idx_in' forbidden?)
                is_tabu = idx_in in tabu_list
                
                # Optimized Aspiration Check:
                # If tabu, we normally skip. BUT we must check if it's "Worth it".
                # To check if it matches global best, we MUST solve LP.
                
                # Optimization
                fit, sol = self.optimize_weights(candidate_indices)
                
                if is_tabu:
                    # Aspiration Criteria:
                    if fit > self.best_fitness:
                        # Allow move!
                        pass
                    else:
                        # Still tabu
                        continue
                
                # Update best neighbor
                if fit > best_neighbor_fitness:
                    best_neighbor_fitness = fit
                    best_neighbor_indices = candidate_indices
                    best_neighbor_sol = sol
                    best_move_dropped_node = idx_out
                    move_found_in_iter = True
            
            if not move_found_in_iter:
                # Restart
                current_indices = sorted(random.sample(range(self.n), self.k))
                tabu_list = []
                # Don't reset global best
                continue
            
            # Execute Move
            current_indices = best_neighbor_indices
            
            # Update Tabu List
            # Ban the node we just dropped (idx_out) from coming back too soon
            tabu_list.append(best_move_dropped_node)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

            # Update Global Best
            if best_neighbor_fitness > self.best_fitness:
                self.best_fitness = best_neighbor_fitness
                self.best_solution = best_neighbor_sol
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                 # Restart
                 current_indices = sorted(random.sample(range(self.n), self.k))
                 tabu_list = []
                 no_improve_count = 0
    
    def __init__(self, time_deadline, problem_path, **kwargs):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
