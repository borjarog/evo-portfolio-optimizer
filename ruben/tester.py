"""
This file contains an example of how you can run your metaheuristic with a bound of n seconds and providing a specific problem instance. Your final submitted algorithm should work with this file
with no issue. Otherwise, it may not be able to participate in the tournament. 

Example of how to call this file from the terminal:
python tester.py -d 60 -i instance01.txt

OR

python tester.py --deadline 60 --instance instance01.txt
"""

import click #May need to pip install click
import importlib
# from metaheuristic import Metaheuristic # Removed static import
from func_timeout import func_timeout, FunctionTimedOut #Requires pip install func_timeout
import time

@click.command()
@click.option(
    "-d",
    "--deadline",
    type=int,
    default=60,
    help="Execution deadline"
)
@click.option(
    "-i",
    "--instance",
    type=click.Path(exists=True),
    required=True,
    help="Path to the problem instance to be solved"
)
@click.option(
    "-s",
    "--strategy",
    type=str,
    default="genetic",
    help="Strategy to run: random, greedy, genetic, tabu, tabu_improved, tabu2, vns, brkga, gwo, vns2, brkga2, gwo2, pso, pso2"
)
def run_metaheuristic(deadline, instance, strategy):
    module_name = f"metaheuristic_{strategy}"
    try:
        module = importlib.import_module(module_name)
        Metaheuristic = module.Metaheuristic
    except ImportError:
        print(f"Error: Could not import strategy '{strategy}'. Make sure {module_name}.py exists.")
        return

    met = Metaheuristic(deadline, instance)
    total_time = None
    try:
        t1 = time.time()
        func_timeout(deadline, met.run)
        total_time = time.time() - t1
    except FunctionTimedOut:
        total_time = deadline
    #TODO: Whatever you want to do after executing your metaheuristic
    print(f"Execution time: {total_time}")
    print(f"Best fitness: {getattr(met, 'best_fitness', 'N/A')}")
    print(f"Best solution: {met.get_best_solution()}")
       
    
if __name__ == "__main__":
    run_metaheuristic()
    
    