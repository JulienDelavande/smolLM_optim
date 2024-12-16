import torch
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.3):
    """Applique un pruning L1 sur les couches linéaires."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model
