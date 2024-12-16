import torch
from transformers import AutoModelForCausalLM
from scripts.metrics import measure_energy, print_results

def apply_quantization(model):
    """Applique la quantization au modèle."""
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

if __name__ == "__main__":
    from scripts.benchmark import load_model, run_benchmark
    model, tokenizer = load_model()

    quantized_model = apply_quantization(model)

    input_text = "Hello, how are you?"
    _, emissions, time_elapsed = measure_energy(run_benchmark, quantized_model, tokenizer, input_text)
    print_results("Quantized Benchmark", emissions, time_elapsed)
