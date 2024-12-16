import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.utils import measure_energy, print_results

# Charger le modèle SmolLM
def load_model(model_name="HuggingFaceTB/SmolLM-135M"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def run_benchmark(model, tokenizer, input_text):
    """Exécute un benchmark sur un texte d'entrée."""
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    return outputs

if __name__ == "__main__":
    # Charger le modèle et les données
    model, tokenizer = load_model()
    input_text = "Hello, how are you?"

    # Benchmark initial
    _, emissions, time_elapsed = measure_energy(run_benchmark, model, tokenizer, input_text)
    print_results("Baseline Benchmark", emissions, time_elapsed)
