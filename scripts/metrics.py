import time
from codecarbon import EmissionsTracker

def measure_energy(func, *args, **kwargs):
    """Mesure la consommation énergétique d'une fonction."""
    tracker = EmissionsTracker()
    tracker.start()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    elapsed_time = time.time() - start_time
    emissions = tracker.stop()
    return result, emissions, elapsed_time

def print_results(name, emissions, time_elapsed):
    """Affiche les résultats des benchmarks."""
    print(f"=== {name} ===")
    print(f"Energy Consumption: {emissions:.4f} kWh")
    print(f"Time Elapsed: {time_elapsed:.2f} seconds")
    print("=================")
    
def print_aggregate_results(name, avg_emissions, std_emissions, avg_time, std_time, output_file=None):
    print(f"=== {name} ===")
    print(f"Energy Consumption: {avg_emissions:.4f} ± {std_emissions:.4f} kWh")
    print(f"Time Elapsed: {avg_time:.2f} ± {std_time:.2f} seconds")
    print("=================")
    
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name},{avg_emissions},{std_emissions},{avg_time},{std_time}\n")

