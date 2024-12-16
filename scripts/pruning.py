import torch
import torch.nn.utils.prune as prune
from scripts.metrics import measure_energy, print_results

def apply_pruning(model, amount=0.3):
    """Applique un pruning aux couches linéaires du modèle."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

if __name__ == "__main__":
    from scripts.benchmark import load_model, run_benchmark
    model, tokenizer = load_model()

    pruned_model = apply_pruning(model)

    input_text = "Hello, how are you?"
    _, emissions, time_elapsed = measure_energy(run_benchmark, pruned_model, tokenizer, input_text)
    print_results("Pruned Benchmark", emissions, time_elapsed)
