from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from .metrics import measure_energy
from datasets import load_dataset

def load_model(model_name="HuggingFaceTB/SmolLM-135M"):
    """Charge le modèle et le tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def run_benchmark(model, tokenizer, input_text):
    """Exécute un benchmark sur un texte d'entrée."""
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    return outputs

def load_data():
    """Charge un dataset pour le benchmark."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return dataset["text"]

def run_benchmark_on_dataset(model, tokenizer, dataset, num_samples=100):
    """Exécute un benchmark sur plusieurs échantillons du dataset."""
    samples = dataset[:num_samples]
    emissions_list = []
    times_list = []
    
    for sample in samples:
        _, emissions, time_elapsed = measure_energy(run_benchmark, model, tokenizer, sample)
        emissions_list.append(emissions)
        times_list.append(time_elapsed)
    
    # Moyennes et écart-types
    avg_emissions = np.mean(emissions_list)
    std_emissions = np.std(emissions_list)
    avg_time = np.mean(times_list)
    std_time = np.std(times_list)
    
    return avg_emissions, std_emissions, avg_time, std_time