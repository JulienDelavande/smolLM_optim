"""
Utility functions for logging and saving results.
"""
import csv

def write_results(output_file: str, model_name: str, strategy: str, backend: str, 
                  runtime: float, energy_kwh: float, co2eq: float, phrases: list, predictions: list):
    """
    Write benchmark results to a CSV file.
    Each line: model_name, strategy, backend, runtime, energy_kWh, co2eq, input, output
    """
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["model_name","strategy","backend","runtime(s)","energy(kWh)","co2eq(kg)","input","prediction"])
        for (ph, pred) in zip(phrases, predictions):
            writer.writerow([model_name, strategy, backend, runtime, energy_kwh, co2eq, ph, pred])
