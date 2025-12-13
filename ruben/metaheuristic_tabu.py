import json
import random
import time
import numpy as np
from scipy.optimize import minimize

class Metaheuristic:
    """
    Tabu Search metaheuristic with SLSQP for weight optimization.
    """

    def read_problem_instance(self, problem_path):
        with open(problem_path, 'r') as f:
            data = json.load(f)
        
        self.n = data['n']
        self.k = data['k']
        self.r = np.array(data['r'])
        self.R = data['R']
        self.dij = np.array(data['dij'])

    def get_best_solution(self):
        return self.best_solution

    def calculate_fitness(self, solution):
        """
        Calculates fitness for a full solution vector w.
        """
        # Efficient calculation using numpy
        w = np.array(solution)
        expected_return = np.dot(self.r, w)
        if expected_return < self.R - 1e-6: # Tolerance for float comparison
            return -float('inf')
        
        # Diversity: sum(w_i * w_j * d_ij) for i < j
        # Full matrix formulation: 0.5 * w^T * D * w (if D is symmetric and 0 diagonal)
        # However, self.dij might be upper triangular or full. The problem says sum_{i<j}.
        # Let's stick to the loop or a verified matrix op for safety given the data format.
        # Data format: data['dij'] is usually a list of lists.
        
        z = 0.0
        # Manual summation to be safe with the exact definition
        # Z = sum_{i=1}^{n-1} sum_{j=i+1}^{n} w_i w_j d_{ij}
        # In python list of lists: d_ij might assume i < j access.
        
        for i in range(self.n - 1):
             if w[i] > 1e-6:
                for j in range(i + 1, self.n):
                    if w[j] > 1e-6:
                        z += w[i] * w[j] * self.dij[i][j]
        return z

    def optimize_weights(self, selected_indices):
        """
        Optimizes weights for the selected asset indices using SLSQP.
        Returns (fitness, full_solution_vector)
        """
        k = len(selected_indices)
        if k == 0:
            return -float('inf'), [0.0] * self.n

        # Extract sub-problem data
        r_sub = self.r[selected_indices]
        # Create sub-matrix for D. 
        # The original D is likely list-of-lists. Let's make it a flexible lookup.
        # Objective: Maximize 0.5 * w_sub.T * D_sub * w_sub ? 
        # The objective is sum_{i<j} w_i w_j d_{ij}.
        
        def objective(w_sub):
            # We want to maximize, so minimize negative
            z = 0.0
            for ia, idx_a in enumerate(selected_indices):
                for ib, idx_b in enumerate(selected_indices):
                    if idx_a < idx_b:
                        z += w_sub[ia] * w_sub[ib] * self.dij[idx_a][idx_b]
            return -z

        def constraint_return(w_sub):
            return np.dot(r_sub, w_sub) - self.R

        def constraint_budget(w_sub):
            return np.sum(w_sub) - 1.0

        # Constraints
        cons = [
            {'type': 'eq', 'fun': constraint_budget},
            {'type': 'ineq', 'fun': constraint_return}
        ]
        
        # Bounds: 0 <= w_i <= 1
        bounds = [(0.0, 1.0) for _ in range(k)]
        
        # Initial guess: equal weights
        w0 = np.array([1.0/k] * k)
        
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-4)
            
            if res.success:
                best_w_sub = res.x
                # Construct full solution
                full_sol = [0.0] * self.n
                for i, idx in enumerate(selected_indices):
                    full_sol[idx] = best_w_sub[i]
                
                # Double check fitness explicitly? No, res.fun is -fitness
                fitness = -res.fun
                return fitness, full_sol
            else:
                return -float('inf'), [0.0] * self.n
        except Exception:
             return -float('inf'), [0.0] * self.n


    def run(self):
        self.read_problem_instance(self.problem_path)
        self.start_time = time.time()
        self.best_fitness = -float('inf')
        self.best_solution = [0.0] * self.n
        
        # Initial Solution: Random k assets
        current_indices = sorted(random.sample(range(self.n), self.k))
        current_fitness, current_sol = self.optimize_weights(current_indices)
        
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.best_solution = current_sol
        
        # Tabu Search Parameters
        tabu_list = []
        tabu_tenure = min(10, self.n // 2)
        max_no_improve = 20
        no_improve_count = 0
        
        while time.time() - self.start_time < self.time_deadline:
            # Generate neighborhood
            # Swap 1 in, 1 out
            # To save time, we might sample the neighborhood if n is large
            
            non_selected = [i for i in range(self.n) if i not in current_indices]
            
            best_neighbor_indices = None
            best_neighbor_fitness = -float('inf')
            best_neighbor_sol = None
            
            # Simple neighborhood exploration (random sampling if too large)
            num_neighbors = len(current_indices) * len(non_selected)
            if num_neighbors > 20:
                 # Sample 20 moves randomly to check
                 moves = []
                 for _ in range(20):
                     idx_out = random.choice(current_indices)
                     idx_in = random.choice(non_selected)
                     moves.append((idx_out, idx_in))
            else:
                moves = []
                for idx_out in current_indices:
                    for idx_in in non_selected:
                        moves.append((idx_out, idx_in))
            
            move_found = False
            
            for idx_out, idx_in in moves:
                # Check Tabu
                # Tabu if (idx_in) was recently dropped? Or (idx_out) recently added?
                # Simple tabu: store (idx_in) as tabu to be added? 
                # Let's store the move signature or just strict tabu on added nodes.
                
                is_tabu = (idx_in in tabu_list) # Prevent adding back something recently dropped?
                # Usually: Tabu list stores attributes of moves. 
                # E.g. Drop i -> i is tabu for X iters. Add j -> j is tabu for X iters?
                # Let's assume tabu list stores indices that are "forbidden to be flipped".
                
                # Create candidate
                candidate_indices = sorted([i for i in current_indices if i != idx_out] + [idx_in])
                
                # Aspiration criteria: evaluate anyway if we can't tell, 
                # generally we verify fitness later. 
                # Wait, optimizing weights is expensive. We only want to optimize non-tabu moves OR aspiration.
                
                if is_tabu:
                    # Only check if it might be better than global best (Aspiration)
                    # We don't know fitness yet. Hard to assume.
                    # Skip tabu for now to save compute.
                    continue
                
                fit, sol = self.optimize_weights(candidate_indices)
                
                if fit > best_neighbor_fitness:
                    best_neighbor_fitness = fit
                    best_neighbor_indices = candidate_indices
                    best_neighbor_sol = sol
                    move_made = (idx_out, idx_in) # Dropped out, Added in
                    move_found = True

            # If no non-tabu move found (or all were bad), break or restart?
            if not move_found:
                 # Restart with random
                 current_indices = sorted(random.sample(range(self.n), self.k))
                 tabu_list = []
                 continue
            
            # Move to best neighbor
            current_indices = best_neighbor_indices
            
            # Update Global Best
            if best_neighbor_fitness > self.best_fitness:
                self.best_fitness = best_neighbor_fitness
                self.best_solution = best_neighbor_sol
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Update Tabu List
            # Forbidden to DROP idx_in immediately? No, forbidden to ADD idx_out immediately.
            # Usually: If we dropped X, X is tabu (cannot be added back) for T iters.
            dropped_node = move_made[0]
            tabu_list.append(dropped_node)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)

            # Check termination
            if no_improve_count > max_no_improve:
                 # Local optima trap -> Restart
                 current_indices = sorted(random.sample(range(self.n), self.k))
                 no_improve_count = 0
                 tabu_list = []

    def __init__(self, time_deadline, problem_path, **kwargs):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
