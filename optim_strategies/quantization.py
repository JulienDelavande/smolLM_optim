import torch

def apply_quantization(model):
    """Applique la quantization dynamique."""
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
