import json
import random
import time

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
        expected_return = sum(self.r[i] * solution[i] for i in range(self.n))
        if expected_return < self.R:
            return -float('inf')
        
        z = 0
        for i in range(self.n - 1):
            if solution[i] > 0:
                for j in range(i + 1, self.n):
                    if solution[j] > 0:
                        z += solution[i] * solution[j] * self.dij[i][j]
        return z

    def run(self):
        """
        This method is in charge of reading the problem instance from a file and then executing the whole logic of the metaheuristic, including initialization
        and the main search procedure.
        TODO: You should implement from the pass statement.
        """
        self.read_problem_instance(self.problem_path) #You should keep this line. Otherwise, disqualified from the tournament
        
        self.start_time = time.time()
        self.best_fitness = -float('inf')
        
        # Precompute metrics
        # D_i: Average difference to other assets
        D = []
        for i in range(self.n):
            avg_diff = sum(self.dij[i]) / self.n 
            D.append(avg_diff)
            
        # Normalize D and r to 0-1
        min_D, max_D = min(D), max(D)
        norm_D = [(d - min_D) / (max_D - min_D) if max_D > min_D else 0 for d in D]
        
        min_r, max_r = min(self.r), max(self.r)
        norm_r = [(r - min_r) / (max_r - min_r) if max_r > min_r else 0 for r in self.r]
        
        # Metric M_i = r_i * D_i (using normalized values)
        M = [nr * nd for nr, nd in zip(norm_r, norm_D)]
        
        # Handle case where sum(M) is 0
        total_M = sum(M)
        if total_M == 0:
             probs = [1.0/self.n] * self.n
        else:
             probs = [m / total_M for m in M]

        while time.time() - self.start_time < self.time_deadline:
            # Select k assets using fitness proportional selection (roulette wheel)
            selected_indices = []
            while len(set(selected_indices)) < self.k:
                 candidates = random.choices(range(self.n), weights=probs, k=self.k * 2)
                 unique_candidates = list(set(candidates))
                 if len(unique_candidates) >= self.k:
                     selected_indices = unique_candidates[:self.k]
                 else:
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

    def __init__(self,time_deadline,problem_path,**kwargs):
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
