from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from .metrics import measure_energy
from datasets import load_dataset

def load_model(model_name="HuggingFaceTB/SmolLM-135M", device="cpu"):
    """Charge le modèle et le tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def run_benchmark(model, tokenizer, input_text, device="cpu"):
    """Exécute un benchmark sur un texte d'entrée."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
    return outputs

def load_data_wikitext(num_samples=100):
    """Charge un dataset pour le benchmark et filtre les échantillons vides."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    filtered_samples = [sample for sample in dataset["text"] if sample.strip()]
    return filtered_samples[:num_samples]

def run_benchmark_on_dataset(model, tokenizer, samples, device="cpu"):
    """Exécute un benchmark sur plusieurs échantillons du dataset."""
    emissions_list = []
    times_list = []
    
    for sample in samples:
        _, emissions, time_elapsed = measure_energy(run_benchmark, model, tokenizer, sample, device)
        emissions_list.append(emissions)
        times_list.append(time_elapsed)
    
    # Moyennes et écart-types
    avg_emissions = np.mean(emissions_list)
    std_emissions = np.std(emissions_list)
    avg_time = np.mean(times_list)
    std_time = np.std(times_list)
    
    return avg_emissions, std_emissions, avg_time, std_time