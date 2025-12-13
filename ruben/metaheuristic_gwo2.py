import json
import random
import time
import numpy as np
from scipy.optimize import minimize
import warnings

class Metaheuristic:
    """
    GWO v2 with SLSQP.
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
        
        pop_size = 30
        positions = np.random.rand(pop_size, self.n)
        
        Alpha_pos = np.zeros(self.n)
        Alpha_score = -float('inf')
        
        Beta_pos = np.zeros(self.n)
        Beta_score = -float('inf')
        
        Delta_pos = np.zeros(self.n)
        Delta_score = -float('inf')
        
        for i in range(pop_size):
            fit, sol = self.optimize_weights_from_position(positions[i])
            if fit > Alpha_score:
                Alpha_score = fit; Alpha_pos = positions[i].copy(); self.best_solution = sol; self.best_fitness = fit
            elif fit > Beta_score:
                Beta_score = fit; Beta_pos = positions[i].copy()
            elif fit > Delta_score:
                Delta_score = fit; Delta_pos = positions[i].copy()
        
        while time.time() - self.start_time < self.time_deadline:
            
            elapsed = time.time() - self.start_time
            a = 2 - 2 * (elapsed / self.time_deadline)
            if a < 0: a = 0
            
            for i in range(pop_size):
                for j in range(self.n):
                    r1 = random.random(); r2 = random.random()
                    A1 = 2*a*r1 - a; C1 = 2*r2
                    D_alpha = abs(C1 * Alpha_pos[j] - positions[i, j])
                    X1 = Alpha_pos[j] - A1 * D_alpha
                    
                    r1 = random.random(); r2 = random.random()
                    A2 = 2*a*r1 - a; C2 = 2*r2
                    D_beta = abs(C2 * Beta_pos[j] - positions[i, j])
                    X2 = Beta_pos[j] - A2 * D_beta
                    
                    r1 = random.random(); r2 = random.random()
                    A3 = 2*a*r1 - a; C3 = 2*r2
                    D_delta = abs(C3 * Delta_pos[j] - positions[i, j])
                    X3 = Delta_pos[j] - A3 * D_delta
                    
                    positions[i, j] = (X1 + X2 + X3) / 3
                
                positions[i] = np.clip(positions[i], 0, 1)
            
            for i in range(pop_size):
                fit, sol = self.optimize_weights_from_position(positions[i])
                
                if fit > Alpha_score:
                    Alpha_score = fit; Alpha_pos = positions[i].copy()
                    self.best_fitness = Alpha_score; self.best_solution = sol
                elif fit > Beta_score:
                    Beta_score = fit; Beta_pos = positions[i].copy()
                elif fit > Delta_score:
                    Delta_score = fit; Delta_pos = positions[i].copy()

    def __init__(self, time_deadline, problem_path, **kwargs):
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline
