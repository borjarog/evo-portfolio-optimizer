import json
import random
import time
import math

class Metaheuristic:
    """
    In this class you should implement your metaheuristic proposal. The code that you submit for the tournament should be 
    included in this class. Please, bear in mind that the current template includes all the mandatory methods, but you can implement any
    other method that you need to. In fact, you are highly encouraged to make a good software design a decompose the behavior of your algorithm
    into several iindependent components or methods.

    The HEADERS for the provided methods CANNOT be modified. Failing to do so will result in your algorithm not participating in the tournament.
    """

    def read_problem_instance(self,problem_path):
        """
        TODO: This method is MANDATORY. The goal of this method is reading a hard drive path that contains a text file with a problem instance.
        The method should read all of the information in the problem instance and store it inside attributes of the Metaheuristic object. 
        This method SHOULD NOT SEARCH nor carry out tasks that indirectly contribute to searching. Typically, you will prepare
        data structures to hold relevant information from the problem instance
        Args:
            problem_path: Text file that contains information about a problem instance
        """
        with open(problem_path, 'r') as f:
            data = json.load(f)
        
        self.n = data['n']
        self.k = data['k']
        self.r = data['r']
        self.R = data['R']
        self.dij = data['dij']

    def get_best_solution(self):
        """
        This method is used to return EXTERNALLY the best solution found so far in the metaheuristic. The solution should be returned in a very
        specific format. For that, you are addressed to the project specification. Please, bear in mind that, INTERNALLY, you can represent
        solutions in any format that you see fit. However, externally, solutions should always be returned in the same way in order to participate in the tournament.
        If you follow this template, self.best_solution should contain the best solution found so far and you should return that solution encoded in the specified format.
        If the returned solution does not follow the format specified in the project specification, you will be disqualified from the tournament.
        """
        return self.best_solution

    def calculate_fitness(self, solution):
        """
        Calculates the fitness of a solution.
        Args:
            solution: A list of n floats representing the weights.
        Returns:
            The objective value Z if feasible, else -infinity.
        """
        # Check constraints
        # 1. Sum of weights = 1 (Assumed to be handled by generation, but good to be aware)
        # 2. Exactly k assets selected (Assumed handled by generation)
        # 3. Expected return >= R
        
        expected_return = sum(self.r[i] * solution[i] for i in range(self.n))
        if expected_return < self.R:
            return -float('inf')
        
        # Calculate Objective Z
        # Z = sum_{i=1}^{n-1} sum_{j=i+1}^{n} w_i * w_j * d_{i,j}
        z = 0
        for i in range(self.n - 1):
            if solution[i] > 0:
                for j in range(i + 1, self.n):
                    if solution[j] > 0:
                        z += solution[i] * solution[j] * self.dij[i][j]
        return z

    def run_random_search(self):
        while time.time() - self.start_time < self.time_deadline:
            # Select k random assets
            selected_indices = random.sample(range(self.n), self.k)
            
            # Generate k uniform numbers
            weights = [random.random() for _ in range(self.k)]
            total_weight = sum(weights)
            
            # Normalize
            normalized_weights = [w / total_weight for w in weights]
            
            # Construct full solution vector
            solution = [0.0] * self.n
            for idx, w in zip(selected_indices, normalized_weights):
                solution[idx] = w
            
            # Evaluate
            fitness = self.calculate_fitness(solution)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution

    def run_greedy_heuristic(self):
        # Precompute metrics
        # D_i: Average difference to other assets
        D = []
        for i in range(self.n):
            avg_diff = sum(self.dij[i]) / self.n # dij is symmetric, but stored as full matrix? 
            # Wait, dij in JSON example is a list of lists, likely full matrix.
            # Let's verify if dij is full matrix or upper triangular. 
            # The example shows full matrix with 0 diagonal.
            D.append(avg_diff)
            
        # Normalize D and r to 0-1
        min_D, max_D = min(D), max(D)
        norm_D = [(d - min_D) / (max_D - min_D) if max_D > min_D else 0 for d in D]
        
        min_r, max_r = min(self.r), max(self.r)
        norm_r = [(r - min_r) / (max_r - min_r) if max_r > min_r else 0 for r in self.r]
        
        # Metric M_i = r_i * D_i (using normalized values)
        M = [nr * nd for nr, nd in zip(norm_r, norm_D)]
        
        # Handle case where sum(M) is 0 to avoid division by zero
        total_M = sum(M)
        if total_M == 0:
             probs = [1.0/self.n] * self.n
        else:
             probs = [m / total_M for m in M]

        while time.time() - self.start_time < self.time_deadline:
            # Select k assets using fitness proportional selection (roulette wheel)
            # We can use random.choices (available in Python 3.6+)
            selected_indices = []
            while len(set(selected_indices)) < self.k:
                 # Draw more than needed to ensure uniqueness, or draw one by one
                 # Drawing one by one with replacement until k unique
                 # This is a bit inefficient but strictly follows "fitness proportional selection"
                 # A better way for "select k assets" might be without replacement, but standard roulette is with replacement.
                 # However, we need k *distinct* assets.
                 # Let's use numpy-like choice if possible, but we can't use numpy.
                 # We'll just sample with weights.
                 
                 # To ensure k unique, we can sample one, remove from pool, re-normalize, repeat.
                 # Or just sample with replacement and discard duplicates until k unique.
                 candidates = random.choices(range(self.n), weights=probs, k=self.k * 2)
                 unique_candidates = list(set(candidates))
                 if len(unique_candidates) >= self.k:
                     selected_indices = unique_candidates[:self.k]
                 else:
                     # Fallback if not enough unique
                     remaining = list(set(range(self.n)) - set(unique_candidates))
                     needed = self.k - len(unique_candidates)
                     selected_indices = unique_candidates + random.sample(remaining, needed)
            
            # Generate k uniform numbers
            weights = [random.random() for _ in range(self.k)]
            total_weight = sum(weights)
            
            # Normalize
            normalized_weights = [w / total_weight for w in weights]
            
            # Construct solution
            solution = [0.0] * self.n
            for idx, w in zip(selected_indices, normalized_weights):
                solution[idx] = w
                
            # Evaluate
            fitness = self.calculate_fitness(solution)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution

    def run_genetic_algorithm(self):
        # Parameters
        pop_size = 100
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            # Represents a solution as a list of n real values between 0 and 1
            # Decoding: select k assets with highest weight, normalize to sum 1
            individual = [random.random() for _ in range(self.n)]
            population.append(individual)
            
        while time.time() - self.start_time < self.time_deadline:
            # Evaluate fitness
            fitnesses = []
            decoded_solutions = []
            for ind in population:
                # Decode
                # Get indices of k largest elements
                # We can use sort
                indexed_ind = list(enumerate(ind))
                indexed_ind.sort(key=lambda x: x[1], reverse=True)
                top_k_indices = [x[0] for x in indexed_ind[:self.k]]
                
                # Normalize weights of these k assets
                top_k_weights = [x[1] for x in indexed_ind[:self.k]]
                total_w = sum(top_k_weights)
                
                solution = [0.0] * self.n
                if total_w > 0:
                    for idx, w in zip(top_k_indices, top_k_weights):
                        solution[idx] = w / total_w
                else:
                    # Edge case: all zero? shouldn't happen with random 0-1
                    pass
                
                decoded_solutions.append(solution)
                fitnesses.append(self.calculate_fitness(solution))
            
            # Update best solution found so far
            max_fit = max(fitnesses)
            if max_fit > self.best_fitness:
                self.best_fitness = max_fit
                best_idx = fitnesses.index(max_fit)
                self.best_solution = decoded_solutions[best_idx]
            
            # Selection: Binary tournament to select 30% parents
            num_parents = int(0.3 * pop_size)
            parents = []
            for _ in range(num_parents):
                c1, c2 = random.sample(range(pop_size), 2)
                if fitnesses[c1] > fitnesses[c2]:
                    parents.append(population[c1])
                else:
                    parents.append(population[c2])
            
            # Crossover: One-point crossover with 80% prob
            children = []
            # We need to generate enough children to fill the population?
            # "The top individuals from the population and generated children are used as the next generation."
            # Usually this means we generate some children and then select the best from (pop + children) to be the new pop.
            # Let's generate children from parents. How many? 
            # The description says "selected parents". It doesn't specify how they pair up.
            # Usually parents are paired.
            # Let's pair them up randomly or sequentially.
            
            # Let's assume we generate children from the selected parents.
            # If we have 30 parents, we can make 15 pairs -> 30 children? Or just make as many children as possible?
            # "The top individuals from the population and generated children are used as the next generation."
            # This implies a steady state or elitist approach.
            # Let's generate children by pairing parents.
            
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    p1 = parents[i]
                    p2 = parents[i+1]
                    
                    if random.random() < 0.8:
                        # One-point crossover
                        point = random.randint(1, self.n - 1)
                        c1 = p1[:point] + p2[point:]
                        c2 = p2[:point] + p1[point:]
                        children.append(c1)
                        children.append(c2)
                    else:
                        children.append(p1[:])
                        children.append(p2[:])
            
            # Mutation: Swap mutation with 20% prob
            for child in children:
                if random.random() < 0.2:
                    # Swap two genes
                    idx1, idx2 = random.sample(range(self.n), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
            
            # Next generation
            # Combine population and children
            pool = population + children
            # Evaluate pool (we already have fitness for population, need for children)
            pool_fitnesses = fitnesses[:] # Copy existing fitnesses
            
            for child in children:
                # Decode and evaluate child
                indexed_ind = list(enumerate(child))
                indexed_ind.sort(key=lambda x: x[1], reverse=True)
                top_k_indices = [x[0] for x in indexed_ind[:self.k]]
                top_k_weights = [x[1] for x in indexed_ind[:self.k]]
                total_w = sum(top_k_weights)
                solution = [0.0] * self.n
                if total_w > 0:
                    for idx, w in zip(top_k_indices, top_k_weights):
                        solution[idx] = w / total_w
                
                fit = self.calculate_fitness(solution)
                pool_fitnesses.append(fit)
                
                # Check if child is best
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = solution
            
            # Select top 100
            # Sort pool by fitness
            combined = list(zip(pool, pool_fitnesses))
            combined.sort(key=lambda x: x[1], reverse=True)
            
            population = [x[0] for x in combined[:pop_size]]


    def run(self):
        """
        This method is in charge of reading the problem instance from a file and then executing the whole logic of the metaheuristic, including initialization
        and the main search procedure.
        TODO: You should implement from the pass statement.
        """
        self.read_problem_instance(self.problem_path) #You should keep this line. Otherwise, disqualified from the tournament
        
        self.start_time = time.time()
        self.best_fitness = -float('inf')
        
        if self.strategy == 'random':
            self.run_random_search()
        elif self.strategy == 'greedy':
            self.run_greedy_heuristic()
        elif self.strategy == 'genetic':
            self.run_genetic_algorithm()
        else:
            # Default to random if unknown
            self.run_random_search()

    def __init__(self,time_deadline,problem_path,strategy='genetic',**kwargs):
        """
        Class initializer. It takes as an argument the maximum computation time (in seconds), controlled externally, and the path that contains the problem instance to be solved
        YOU CAN MODIFY THE HEADER TO INCLUDE OPTIONAL PARAMETERS WITH DEFAULT VALUES ( e.g., __init__(self, time_deadline, problem_path, mut_prob=0.5) )
        You should configure the algorithm before its execution in this method (i.e., hyperparameter values, data structure initialization, etc.)
        Args:
            problem_path: String that contains the path to the file that describes the problem instance
            time_deadline: Computation time limit for the metaheuristic
            kwargs: Other arguments can be passed to the algorithm using key-value pairs. For instance, Metaheuristic(20, 'instance1.txt', mut_prob=0.3) would call the initializer with 20 seconds, for reading the instance1.txt file and passing an optional parameter of mut_prob=0.3
        """
        self.problem_path = problem_path # This attribute is meant to contain the path to the problem instance
        self.best_solution = None #This attribute is meant to hold, at any time, the best solution found by the algorithm so far. Hence, you should update it accordingly. The solution enconding does not matter.
        self.time_deadline = time_deadline # Computation limit (in seconds) for the metaheuristic 
        self.strategy = strategy
        #TODO: Configure the metaheuristic (e.g., selection operator, crossover, mutation, hyperparameter values, etc.)



    

