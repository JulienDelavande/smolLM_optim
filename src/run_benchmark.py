from codecarbon import EmissionsTracker
from .dataset_loader import load_dataset_samples
from .model_loader import load_model

def run_benchmark(model_name: str, strategy: str, backend: str, dataset_name: str, 
                 dataset_split: str, samples: int):
    """
    Run the benchmark for a text generation model with given strategy and backend.
    """
    phrases = load_dataset_samples(dataset_name, dataset_split, samples)
    pipe, tokenizer = load_model(model_name, strategy, backend)
    results = []

    total_tokens = 0
    total_energy = 0
    total_duration = 0
    total_cpu_energy = 0
    total_gpu_energy = 0
    total_ram_energy = 0

    print("Starting benchmark...")

    for phrase in phrases:
        tracker = EmissionsTracker()
        tracker.start()
        out = pipe(phrase, max_new_tokens=50)
        emissions = tracker.stop()

        generated_text = out[0]["generated_text"][len(phrase):]
        generated_tokens = tokenizer(generated_text)["input_ids"]
        token_count = len(generated_tokens)

        energy_data = tracker.final_emissions_data
        total_tokens += token_count
        total_energy += energy_data.energy_consumed
        total_duration += energy_data.duration
        total_cpu_energy += energy_data.cpu_energy
        total_gpu_energy += energy_data.gpu_energy
        total_ram_energy += energy_data.ram_energy
        total_co2_emissions += emissions

        results.append({
            "phrase": phrase,
            "prediction": out[0]["generated_text"],
            "emissions": emissions,
            "token_count": token_count,
            "cpu_energy": energy_data.cpu_energy,
            "gpu_energy": energy_data.gpu_energy,
            "ram_energy": energy_data.ram_energy,
            "energy_consumed": energy_data.energy_consumed,
            "duration": energy_data.duration
        })

    # Calculate final metrics
    energy_per_token = total_energy / total_tokens if total_tokens > 0 else 0
    duration_per_token = total_duration / total_tokens if total_tokens > 0 else 0
    cpu_energy_per_token = total_cpu_energy / total_tokens if total_tokens > 0 else 0
    gpu_energy_per_token = total_gpu_energy / total_tokens if total_tokens > 0 else 0
    ram_energy_per_token = total_ram_energy / total_tokens if total_tokens > 0 else 0
    co2_emissions_per_token = total_co2_emissions / total_tokens if total_tokens > 0 else 0
    
    # Print summary
    print("\nBenchmark Results Summary")
    print("========================")
    print(f"Model: {model_name}")
    print(f"Strategy: {strategy}")
    print(f"Backend: {backend}")
    print(f"Dataset: {dataset_name} ({dataset_split})")
    print(f"Samples processed: {len(phrases)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total energy consumed: {total_energy:.6f} kWh")
    print(f"Total duration: {total_duration} seconds")
    print(f"Energy per token: {energy_per_token*1000} Wh/token")
    print(f"Duration per token: {duration_per_token:.4f} seconds/token")
    print(f"CPU energy per token: {cpu_energy_per_token*1000} Wh/token")
    print(f"GPU energy per token: {gpu_energy_per_token*1000} Wh/token")
    print(f"RAM energy per token: {ram_energy_per_token*1000} Wh/token")
    print(f"Equivalent CO2 emissions per token: {co2_emissions_per_token} kg eqCO2/token")
