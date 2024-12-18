"""
Run the benchmark: load model, run inference on dataset samples, measure energy and record results.
"""
from codecarbon import EmissionsTracker
from .dataset_loader import load_dataset_samples
from .model_loader import load_model
from .utils import write_results
import time

def run_benchmark(model_name: str, strategy: str, backend: str, dataset_name: str, 
                  dataset_config: str, dataset_split: str, max_samples: int, output_file: str):
    """
    Run the benchmark for a text generation model with given strategy and backend.
    """
    phrases = load_dataset_samples(dataset_name, dataset_config, dataset_split, max_samples)

    tracker = EmissionsTracker(log_level="error")
    tracker.start()
    pipe = load_model(model_name, strategy, backend)
    start_time = time.time()
    results = []
    # Generate text for each phrase - we can define a fixed max_length to ensure consistent outputs
    for phrase in phrases:
        out = pipe(phrase, max_length=50, num_return_sequences=1)
        # out is a list of dicts with 'generated_text'
        results.append((phrase, out[0]["generated_text"] if out else ""))
    runtime = time.time() - start_time
    emissions: dict = tracker.stop()

    energy_kwh = emissions.get("energy_consumed (kWh)", None)
    co2eq = emissions.get("emissions (kg)", None)

    write_results(
        output_file=output_file,
        model_name=model_name,
        strategy=strategy,
        backend=backend,
        runtime=runtime,
        energy_kwh=energy_kwh,
        co2eq=co2eq,
        phrases=phrases,
        predictions=results
    )
