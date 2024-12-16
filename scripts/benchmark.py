import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.metrics import measure_energy, print_results
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
    model, tokenizer = load_model()
    input_text = "Hello, how are you?"
    _, emissions, time_elapsed = measure_energy(run_benchmark, model, tokenizer, input_text)
    print_results("Baseline Benchmark", emissions, time_elapsed)
