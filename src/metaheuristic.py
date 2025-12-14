"""
Hybrid Adaptive Portfolio Optimizer
Automatically selects best strategy based on problem size:
- DE + Local Search for small/medium instances (n < 500)
- PSO for large instances (n >= 500)
"""

import json
import time
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class Metaheuristic:
    """
    Hybrid Adaptive Portfolio Optimizer.
    Combines DE for small instances and PSO for large instances.
    """

    def read_problem_instance(self, problem_path):
        """Read problem instance from JSON file."""
        with open(problem_path, 'r') as f:
            data = json.load(f)
        
        self.n = data['n']
        self.k = data['k']
        self.r = np.array(data['r'], dtype=np.float64)
        self.R = data['R']
        self.dij = np.array(data['dij'], dtype=np.float64)
        self.delta = data.get('delta', 0.01)

    def get_best_solution(self):
        """Return best solution in tournament format."""
        if self.best_solution is None:
            return [0.0] * self.n
        if isinstance(self.best_solution, np.ndarray):
            return self.best_solution.tolist()
        return list(self.best_solution)

    # ==================== SHARED UTILITIES ====================
    
    def _precompute(self):
        """Precompute common data structures."""
        self.all_indices = np.arange(self.n)
        self.top_return_indices = np.argsort(self.r)[-self.k:]
        
        avg_div = np.mean(self.dij, axis=1)
        r_norm = (self.r - self.r.min()) / (self.r.max() - self.r.min() + 1e-10)
        d_norm = (avg_div - avg_div.min()) / (avg_div.max() - avg_div.min() + 1e-10)
        self.asset_scores = r_norm * d_norm + np.where(self.r >= self.R, 0.3, 0)
        self.top_diversity_indices = np.argsort(avg_div)[-self.k:]

    def _decode_indices(self, keys):
        """Decode keys to asset indices."""
        if self.n > 100:
            part = np.argpartition(keys, -self.k)[-self.k:]
            return tuple(sorted(part))
        return tuple(sorted(np.argsort(keys)[-self.k:]))

    def _optimize_weights(self, selected_indices, use_delta=True):
        """Optimize weights for selected assets."""
        if selected_indices in self.cache:
            return self.cache[selected_indices]
        
        indices_arr = np.array(selected_indices)
        r_sub = self.r[indices_arr]
        D_sub = self.dij[np.ix_(indices_arr, indices_arr)]
        
        if r_sub.max() < self.R:
            result = (-np.inf, np.zeros(self.n))
            self._cache(selected_indices, result)
            return result
        
        k = self.k
        D_triu = np.triu(D_sub, 1)
        D_sym = D_triu + D_triu.T
        
        def objective(w):
            return -np.sum(np.outer(w, w) * D_triu)
        
        def objective_jac(w):
            return -D_sym @ w
        
        lb = self.delta if use_delta else 0.0
        bounds = [(lb, 1.0)] * k
        ones_k = np.ones(k)
        
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: ones_k},
            {'type': 'ineq', 'fun': lambda w: np.dot(r_sub, w) - self.R, 'jac': lambda w: r_sub}
        ]
        
        # Try multiple starting points for large k
        w0_list = [np.full(k, 1.0/k)]
        if np.dot(r_sub, w0_list[0]) < self.R:
            w0_ret = r_sub / r_sub.sum()
            w0_ret = np.maximum(w0_ret, lb)
            w0_ret /= w0_ret.sum()
            w0_list = [w0_ret]
        
        if k >= 50:
            div_scores = np.sum(D_sub, axis=1)
            w0_div = div_scores / div_scores.sum() if div_scores.sum() > 0 else w0_list[0]
            w0_div = np.maximum(w0_div, lb)
            w0_div /= w0_div.sum()
            w0_list.append(w0_div)
        
        maxiter = 30 if self.n >= 500 else 50
        best_result = None
        
        for w0 in w0_list:
            try:
                res = minimize(
                    objective, w0, method='SLSQP', jac=objective_jac,
                    bounds=bounds, constraints=cons,
                    options={'disp': False, 'maxiter': maxiter, 'ftol': 1e-4}
                )
                
                if res.success or res.fun < 0:
                    w_opt = np.clip(res.x, lb, 1.0)
                    w_opt /= w_opt.sum()
                    
                    if np.dot(r_sub, w_opt) >= self.R - 1e-6:
                        fitness = -res.fun
                        if best_result is None or fitness > best_result[0]:
                            full_sol = np.zeros(self.n)
                            full_sol[indices_arr] = w_opt
                            best_result = (fitness, full_sol)
            except:
                pass
        
        if best_result is None:
            # Greedy fallback
            scores = np.sum(D_sub, axis=1) * r_sub
            w = scores / scores.sum() if scores.sum() > 0 else np.ones(k) / k
            w = np.maximum(w, lb)
            w /= w.sum()
            if np.dot(r_sub, w) >= self.R:
                fitness = np.sum(np.outer(w, w) * D_triu)
                full_sol = np.zeros(self.n)
                full_sol[indices_arr] = w
                best_result = (fitness, full_sol)
            else:
                best_result = (-np.inf, np.zeros(self.n))
        
        self._cache(selected_indices, best_result)
        return best_result

    def _cache(self, key, result):
        if len(self.cache) < self.cache_limit:
            self.cache[key] = result

    # ==================== PSO STRATEGY (Large Instances) ====================
    
    def _run_pso(self):
        """PSO strategy optimized for large instances."""
        pop_size = 30
        w_inertia = 0.7
        c1, c2 = 1.5, 1.5
        
        # Initialize
        positions = np.random.rand(pop_size, self.n)
        
        # Bias some toward high-score assets
        for i in range(pop_size // 2):
            bias = self.asset_scores * 0.5 * np.random.rand(self.n)
            positions[i] = np.clip(positions[i] + bias, 0, 1)
        
        velocities = np.random.uniform(-0.1, 0.1, (pop_size, self.n))
        
        pbest_pos = positions.copy()
        pbest_score = np.full(pop_size, -np.inf)
        gbest_pos = np.zeros(self.n)
        gbest_score = -np.inf
        
        # Initial evaluation
        for i in range(pop_size):
            indices = self._decode_indices(positions[i])
            fit, sol = self._optimize_weights(indices, use_delta=False)
            
            pbest_score[i] = fit
            pbest_pos[i] = positions[i].copy()
            
            if fit > gbest_score:
                gbest_score = fit
                gbest_pos = positions[i].copy()
                self.best_fitness = fit
                self.best_solution = sol.copy()
        
        # Main loop
        while time.time() - self.start_time < self.time_deadline:
            for i in range(pop_size):
                r1 = np.random.rand(self.n)
                r2 = np.random.rand(self.n)
                
                velocities[i] = (w_inertia * velocities[i] + 
                                c1 * r1 * (pbest_pos[i] - positions[i]) + 
                                c2 * r2 * (gbest_pos - positions[i]))
                
                positions[i] = np.clip(positions[i] + velocities[i], 0, 1)
                
                indices = self._decode_indices(positions[i])
                fit, sol = self._optimize_weights(indices, use_delta=False)
                
                if fit > pbest_score[i]:
                    pbest_score[i] = fit
                    pbest_pos[i] = positions[i].copy()
                    
                    if fit > gbest_score:
                        gbest_score = fit
                        gbest_pos = positions[i].copy()
                        self.best_fitness = fit
                        self.best_solution = sol.copy()

    # ==================== DE STRATEGY (Small/Medium Instances) ====================
    
    def _run_de(self):
        """DE strategy for small/medium instances."""
        pop_size = 40 if self.n >= 100 else 50
        F, CR = 0.7, 0.85
        local_rate = 0.15 if self.n >= 100 else 0.3
        restart_thresh = 20 if self.n >= 100 else 30
        
        # Initialize population
        population = np.random.rand(pop_size, self.n)
        num_biased = int(pop_size * 0.6)
        for i in range(num_biased):
            bias = self.asset_scores * 0.7 * np.random.rand(self.n)
            population[i] = np.clip(population[i] + bias, 0, 1)
        
        # Evaluate
        fitness_data = []
        for i in range(pop_size):
            indices = self._decode_indices(population[i])
            fit, sol = self._optimize_weights(indices)
            fitness_data.append([fit, sol, indices, population[i]])
            
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best_solution = sol.copy()
        
        fitness_data.sort(key=lambda x: x[0], reverse=True)
        no_improve = 0
        
        while time.time() - self.start_time < self.time_deadline:
            new_fitness_data = []
            
            for i in range(pop_size):
                target = fitness_data[i][3]
                target_fit = fitness_data[i][0]
                
                # Mutation
                available = list(range(pop_size))
                available.remove(i)
                r1, r2, r3 = np.random.choice(available, 3, replace=False)
                mutant = fitness_data[r1][3] + F * (fitness_data[r2][3] - fitness_data[r3][3])
                mutant = np.clip(mutant, 0, 1)
                
                # Crossover
                trial = target.copy()
                mask = np.random.rand(self.n) < CR
                mask[np.random.randint(self.n)] = True
                trial[mask] = mutant[mask]
                
                # Evaluate trial
                trial_indices = self._decode_indices(trial)
                trial_fit, trial_sol = self._optimize_weights(trial_indices)
                
                if trial_fit >= target_fit:
                    new_fitness_data.append([trial_fit, trial_sol, trial_indices, trial])
                else:
                    new_fitness_data.append(fitness_data[i])
                
                if trial_fit > self.best_fitness:
                    self.best_fitness = trial_fit
                    self.best_solution = trial_sol.copy()
                    no_improve = 0
            
            new_fitness_data.sort(key=lambda x: x[0], reverse=True)
            
            # Local search on elite
            if time.time() - self.start_time < self.time_deadline - 2:
                num_elite = max(1, int(pop_size * local_rate))
                max_neighbors = 25 if self.n <= 100 else 15
                
                for i in range(min(num_elite, 3)):
                    if time.time() - self.start_time >= self.time_deadline - 1:
                        break
                    
                    curr_indices = new_fitness_data[i][2]
                    curr_fit = new_fitness_data[i][0]
                    
                    indices_list = list(curr_indices)
                    non_selected = [j for j in range(self.n) if j not in curr_indices]
                    
                    if not non_selected:
                        continue
                    
                    neighbors = [(idx_out, idx_in) 
                                for idx_out in indices_list 
                                for idx_in in non_selected]
                    
                    if len(neighbors) > max_neighbors:
                        neighbors = [neighbors[j] for j in np.random.choice(len(neighbors), max_neighbors, replace=False)]
                    
                    best_imp = None
                    for idx_out, idx_in in neighbors:
                        new_idx = tuple(sorted([j for j in indices_list if j != idx_out] + [idx_in]))
                        new_fit, new_sol = self._optimize_weights(new_idx)
                        
                        if new_fit > curr_fit and (best_imp is None or new_fit > best_imp[1]):
                            best_imp = (new_idx, new_fit, new_sol)
                    
                    if best_imp:
                        new_keys = np.random.rand(self.n) * 0.4
                        for idx in best_imp[0]:
                            new_keys[idx] = 0.6 + np.random.rand() * 0.4
                        new_fitness_data[i] = [best_imp[1], best_imp[2], best_imp[0], new_keys]
                        
                        if best_imp[1] > self.best_fitness:
                            self.best_fitness = best_imp[1]
                            self.best_solution = best_imp[2].copy()
                            no_improve = 0
            
            no_improve += 1
            
            if no_improve >= restart_thresh:
                num_restart = max(1, int(pop_size * 0.2))
                for i in range(pop_size - num_restart, pop_size):
                    new_keys = np.random.rand(self.n)
                    new_idx = self._decode_indices(new_keys)
                    new_fit, new_sol = self._optimize_weights(new_idx)
                    new_fitness_data[i] = [new_fit, new_sol, new_idx, new_keys]
                    
                    if new_fit > self.best_fitness:
                        self.best_fitness = new_fit
                        self.best_solution = new_sol.copy()
                no_improve = 0
            
            new_fitness_data.sort(key=lambda x: x[0], reverse=True)
            fitness_data = new_fitness_data

    # ==================== MAIN ====================
    
    def run(self):
        """Main execution - selects best strategy based on problem size."""
        self.read_problem_instance(self.problem_path)
        self.start_time = time.time()
        self.cache = {}
        self.cache_limit = 15000
        self._precompute()
        
        # Select strategy based on problem size
        if self.n >= 500:
            self._run_pso()
        else:
            self._run_de()

    def __init__(self, time_deadline, problem_path, **kwargs):
        """Initialize the metaheuristic."""
        self.problem_path = problem_path
        self.time_deadline = time_deadline
        self.best_solution = None
        self.best_fitness = -np.inf
        self.n = None
        self.k = None
        self.cache = {}
        self.cache_limit = 15000
