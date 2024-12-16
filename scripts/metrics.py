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
