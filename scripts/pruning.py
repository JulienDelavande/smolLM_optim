import torch
import torch.nn.utils.prune as prune
from scripts.utils import measure_energy, print_results

def apply_pruning(model, amount=0.3):
    """Applique un pruning aux couches linéaires du modèle."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Pour rendre le pruning permanent
    return model

if __name__ == "__main__":
    # Charger le modèle original
    from scripts.benchmark import load_model, run_benchmark
    model, tokenizer = load_model()

    # Appliquer le pruning
    pruned_model = apply_pruning(model)

    # Benchmark après pruning
    input_text = "Hello, how are you?"
    _, emissions, time_elapsed = measure_energy(run_benchmark, pruned_model, tokenizer, input_text)
    print_results("Pruned Benchmark", emissions, time_elapsed)
