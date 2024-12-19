"""
Run the benchmark: load model, run inference on dataset samples, measure energy and record results.
"""
from codecarbon import EmissionsTracker
from .dataset_loader import load_dataset_samples
from .model_loader import load_model
from .utils import write_results
import time

def run_benchmark(model_name: str, strategy: str, backend: str, dataset_name: str, 
                 dataset_split: str, max_samples: int, output_file: str):
    """
    Run the benchmark for a text generation model with given strategy and backend.
    """
    phrases = load_dataset_samples(dataset_name, dataset_split, max_samples)
    pipe, tokenizer = load_model(model_name, strategy, backend)
    results = []
    question = "Tell a story about Philipp?"
    context = "My name is Philipp and I live in Nuremberg."
    prompt = f'{context} {question}'

    # Generate text for each phrase - we can define a fixed max_length to ensure consistent outputs
    for phrase in phrases:
        tracker = EmissionsTracker(log_level="error")
        tracker.start()
        out = pipe(prompt)
        emissions = tracker.stop()
        print(tracker.final_emissions_data)
        print(f"phrase: {phrase}")
        print(f"prediction: {out[0]['generated_text']}")
        
        generated_text = out[0]["generated_text"]
        generated_tokens = tokenizer(generated_text)["input_ids"]
        token_count = len(generated_tokens)
        results.append({
            "phrase": phrase,
            "prediction": out[0]["generated_text"],
            "emissions": emissions,
            "token_count": token_count,
            "cpu_energy": tracker.final_emissions_data.cpu_energy,
            "gpu_energy": tracker.final_emissions_data.gpu_energy,
            "ram_energy": tracker.final_emissions_data.ram_energy,
            "energy_consumed": tracker.final_emissions_data.energy_consumed,
            "duration": tracker.final_emissions_data.duration
        })
    
    total_tokens = sum([r["token_count"] for r in results])
    total_energy = sum([r["energy_consumed"] for r in results])
    energy_per_token = total_energy / total_tokens
    print(f"Energy per token: {energy_per_token*1000} Wh")
    # energy_kwh = tracker.get("energy_consumed (kWh)", None)
    # co2eq = tracker.get("emissions (kg)", None)
    

    # write_results(
    #     output_file=output_file,
    #     model_name=model_name,
    #     strategy=strategy,
    #     backend=backend,
    #     runtime=runtime,
    #     energy_kwh='energy_kwh',
    #     co2eq='co2eq',
    #     phrases=phrases,
    #     predictions=results
    # )
