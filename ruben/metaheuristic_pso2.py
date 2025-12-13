import json
import random
import time
import numpy as np
from scipy.optimize import minimize
import warnings

class Metaheuristic:
    """
    PSO v2 (Advanced) with SLSQP.
    Features:
    - Adaptive Inertia Weight (0.9 -> 0.4).
    - Velocity Clamping (Vmax).
    - Diversity Maintenance (Re-initialization).
    - Standard bounds & warning suppression.
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
        # Fallback fitness for internal logic if needed, usually optimize_weights does the heavy lifting
        w = np.array(solution)
        if np.dot(self.r, w) < self.R - 1e-6: return -float('inf')
        
        z = 0.0
        for i in range(self.n - 1):
             if w[i] > 1e-6:
                for j in range(i + 1, self.n):
                    if w[j] > 1e-6: z += w[i] * w[j] * self.dij[i][j]
        return z

    def optimize_weights_from_position(self, position):
        sorted_indices = np.argsort(position)
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
        
        # PSO Parameters
        pop_size = 40
        c1 = 1.496
        c2 = 1.496
        v_max = 0.2
        
        # Initialize Swarm
        positions = np.random.rand(pop_size, self.n)
        velocities = np.random.uniform(-0.1, 0.1, (pop_size, self.n))
        
        pbest_pos = positions.copy()
        pbest_score = np.full(pop_size, -float('inf'))
        
        gbest_pos = np.zeros(self.n)
        gbest_score = -float('inf')
        
        # Initial Evaluation
        for i in range(pop_size):
            fit, sol = self.optimize_weights_from_position(positions[i])
            
            pbest_score[i] = fit
            pbest_pos[i] = positions[i].copy()
            
            if fit > gbest_score:
                gbest_score = fit
                gbest_pos = positions[i].copy()
                self.best_fitness = fit
                self.best_solution = sol
        
        while time.time() - self.start_time < self.time_deadline:
            
            # Adaptive Inertia Weight: Linear Decay from 0.9 to 0.4
            elapsed = time.time() - self.start_time
            w_inertia = 0.9 - (0.5 * (elapsed / self.time_deadline))
            if w_inertia < 0.4: w_inertia = 0.4
            
            # Check for Stagnation (Diversity Loss)
            # If variance of positions is too low, re-initialize part of swarm
            pos_std = np.std(positions)
            if pos_std < 0.05 and elapsed > self.time_deadline * 0.2:
                # Re-initialize 80% of worst particles
                num_to_reset = int(pop_size * 0.8)
                idxs = np.argsort(pbest_score)[:num_to_reset]
                for idx in idxs:
                    positions[idx] = np.random.rand(self.n)
                    velocities[idx] = np.random.uniform(-0.1, 0.1, self.n)
                    pbest_score[idx] = -float('inf') # Reset memory
            
            for i in range(pop_size):
                r1 = np.random.rand(self.n)
                r2 = np.random.rand(self.n)
                
                velocities[i] = (w_inertia * velocities[i] + 
                                 c1 * r1 * (pbest_pos[i] - positions[i]) + 
                                 c2 * r2 * (gbest_pos - positions[i]))
                
                # Velocity Clamping
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
                
                fit, sol = self.optimize_weights_from_position(positions[i])
                
                if fit > pbest_score[i]:
                    pbest_score[i] = fit
                    pbest_pos[i] = positions[i].copy()
                    
                    if fit > gbest_score:
                        gbest_score = fit
                        gbest_pos = positions[i].copy()
                        self.best_fitness = fit
                        self.best_solution = sol

    def __init__(self, time_deadline, problem_path, **kwargs):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
