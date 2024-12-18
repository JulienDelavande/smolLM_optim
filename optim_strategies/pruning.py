from transformers import AutoModelForCausalLM
from optimum.intel.neural_compressor import INCModelForCausalLM
from optimum.intel import IncQuantizer, IncPruner


def apply_pruning(model, amount=0.2, backend="default"):
    """
    Applique une technique de pruning optimisée au modèle en utilisant Hugging Face Optimum.

    Args:
        model: Le modèle Hugging Face à optimiser.
        amount: Proportion de poids à pruner dans chaque couche (float entre 0 et 1).
        backend: Backend utilisé pour le pruning ("default", "nvidia", "onednn").

    Returns:
        model: Le modèle optimisé après pruning.
    """
    print("Starting pruning with Hugging Face Optimum...")

    if backend not in ["default", "nvidia", "onednn"]:
        raise ValueError(f"Backend '{backend}' not supported. Choose from 'default', 'nvidia', or 'onednn'.")

    pruner = IncPruner(model)
    if backend == "nvidia":
        pruner.set_backend("nvidia")
        print("Using NVIDIA backend for optimized pruning.")
    elif backend == "onednn":
        pruner.set_backend("onednn")
        print("Using oneDNN backend for optimized pruning.")
    else:
        print("Using default backend for pruning.")

    model = pruner.prune(amount=amount)

    print("Pruning completed.")
    return model
