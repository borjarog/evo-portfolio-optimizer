import os
import time
import importlib
import pandas as pd
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURACIÓN DE USUARIO
# ==========================================
STRATEGIES = ['pso2']
N_RUNS = 5      # Número de intentos por instancia (Recomendado: 5 o 10)
DEADLINE = 60   # Segundos LÍMITE por CADA intento individual

# ==========================================
# SISTEMA
# ==========================================
INSTANCE_PREFIX = 'instance_'
INSTANCE_EXTENSION = '.json'

def run_single_instance(instance_path, strategy_name, timeout, n_runs):
    """
    Ejecuta la estrategia N veces y devuelve la MEJOR solución encontrada.
    """
    module_name = f"metaheuristic_{strategy_name}"
    
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        Metaheuristic = module.Metaheuristic
    except ImportError:
        return None, None, f"Error Import: {module_name}"
    except AttributeError:
        return None, None, f"Error Clase: {module_name}"

    total_time_accumulated = 0
    total_fitness_accumulated = 0
    errors = []

    # --- BUCLE DE INTENTOS (REPLICAS) ---
    for i in range(n_runs):
        solver = Metaheuristic(timeout, instance_path)
        
        try:
            t_start = time.time()
            # Ejecutamos con límite de tiempo por intento
            func_timeout(timeout, solver.run)
            t_run = time.time() - t_start
        except FunctionTimedOut:
            t_run = timeout
        except Exception as e:
            t_run = 0
            errors.append(str(e))
        
        # Recuperamos el fitness de este intento
        fit = getattr(solver, 'best_fitness', -float('inf'))
        
        total_time_accumulated += t_run
        total_fitness_accumulated += fit

    # Devolvemos el fitness promedio de los 5 intentos y el tiempo promedio
    avg_time = total_time_accumulated / n_runs if n_runs > 0 else 0
    avg_fitness = total_fitness_accumulated / n_runs if n_runs > 0 else 0
    error_msg = "; ".join(errors) if errors else None
    
    return avg_fitness, avg_time, error_msg

def main():
    instance_files = [f for f in os.listdir('.') if f.startswith(INSTANCE_PREFIX) and f.endswith(INSTANCE_EXTENSION)]
    instance_files.sort()

    if not instance_files:
        print("Error: No hay instancias.")
        return

    print(f"--- EXPERIMENTO ROBUSTO ({N_RUNS} Intentos por Instancia) ---")
    print(f"Estrategias: {STRATEGIES}")
    print(f"CPUs: {os.cpu_count()}")
    print("-" * 50)

    for strategy in STRATEGIES:
        print(f"\n>>> Estrategia: {strategy.upper()}")
        results = []
        
        with ProcessPoolExecutor() as executor:
            # Enviamos n_runs como argumento extra
            future_to_inst = {
                executor.submit(run_single_instance, inst, strategy, DEADLINE, N_RUNS): inst
                for inst in instance_files
            }

            for future in as_completed(future_to_inst):
                inst_name = future_to_inst[future]
                try:
                    # Aquí recibimos el MEJOR de los 5 intentos
                    fitness, avg_time, error = future.result()
                    
                    if error:
                        print(f"   [X] {inst_name}: {error}")
                    else:
                        print(f"   [OK] {inst_name} -> Best Fit: {fitness:.4f} (Avg Time: {avg_time:.2f}s)")
                        
                        if fitness is not None:
                            results.append({
                                'Instance': inst_name,
                                'Strategy': strategy,
                                'Fitness': fitness, # Guardamos solo el mejor
                                'Avg_Time': avg_time
                            })
                except Exception as exc:
                    print(f"   [!] Fallo crítico en {inst_name}: {exc}")

        if results:
            results.sort(key=lambda x: x['Instance'])
            filename = f"resultados_{strategy}.csv"
            pd.DataFrame(results).to_csv(filename, index=False)
            print(f">>> Guardado: {filename}")

if __name__ == "__main__":
    main()