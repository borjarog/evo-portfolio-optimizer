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
        
        # Parameters
        pop_size = 100
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            # Represents a solution as a list of n real values between 0 and 1
            individual = [random.random() for _ in range(self.n)]
            population.append(individual)
            
        while time.time() - self.start_time < self.time_deadline:
            # Evaluate fitness
            fitnesses = []
            decoded_solutions = []
            for ind in population:
                # Decode
                indexed_ind = list(enumerate(ind))
                indexed_ind.sort(key=lambda x: x[1], reverse=True)
                top_k_indices = [x[0] for x in indexed_ind[:self.k]]
                
                top_k_weights = [x[1] for x in indexed_ind[:self.k]]
                total_w = sum(top_k_weights)
                
                solution = [0.0] * self.n
                if total_w > 0:
                    for idx, w in zip(top_k_indices, top_k_weights):
                        solution[idx] = w / total_w
                
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
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    p1 = parents[i]
                    p2 = parents[i+1]
                    
                    if random.random() < 0.8:
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
                    idx1, idx2 = random.sample(range(self.n), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
            
            # Next generation
            pool = population + children
            pool_fitnesses = fitnesses[:] 
            
            for child in children:
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
                
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = solution
            
            # Select top 100
            combined = list(zip(pool, pool_fitnesses))
            combined.sort(key=lambda x: x[1], reverse=True)
            
            population = [x[0] for x in combined[:pop_size]]

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
