"""
Load and prepare models with specified optimization strategy and backend for text generation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.intel.neural_compressor import INCQuantizer
from optimum.exporters.onnx import export
from transformers.onnx import FeaturesManager
import os

def load_model(model_name: str, strategy: str, backend: str):
    """
    Load and return a pipeline for text-generation according to strategy and backend.
    backend: "onnx_cpu", "onnx_gpu", "hf_cpu", "hf_gpu"
    strategy: "none", "quantization", "pruning"
    """
    device = "cpu"
    if backend.endswith("gpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if backend.startswith("onnx"):
        # ONNX backend
        onnx_model_path = f"{model_name}-onnx"
        if not os.path.exists(onnx_model_path):
            os.makedirs(onnx_model_path)

        # Load ORT model
        model = ORTModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M",from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if strategy == "quantization":
            quantizer = ORTQuantizer.from_pretrained(model)
            model = quantizer.quantize(save_directory=onnx_model_path, quantization_config="dynamic")

        # Pruning note: pruning for ONNX should be done before export, so this is a placeholder.
        # If pruning is desired, it should be integrated before exporting.

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)
        return pipe

    else:
        # HF backend
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if strategy == "quantization":
            # PyTorch dynamic quantization
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        # Pruning would be done here if integrated. 
        # Example: use `optimum` pruning utilities before final loading.

        model.to(device)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)
        return pipe
