"""
Load and prepare models with specified optimization strategy and backend for text generation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import os

def load_model(model_name: str, strategy: str, backend: str):
    """
    Load and return a pipeline for text-generation according to strategy and backend.
    backend: "onnx_cpu", "onnx_gpu", "base_cpu", "base_gpu"
    strategy: "none", "quantization", "pruning"
    """
    device = "cpu"
    if backend.endswith("gpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if backend.startswith("base"):
        # HF backend
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if strategy == 'none':
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)
            
        if strategy == "quantization8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
        if strategy == "quantization16bit":
            quantization_config = BitsAndBytesConfig(load_in_16bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
        return pipe
            
            

        # Pruning would be done here if integrated. 
        # Example: use `optimum` pruning utilities before final loading.
    
    if backend.startswith("onnx"):
        # ONNX backend
        onnx_model_path = f"{model_name}-onnx"
        if not os.path.exists(onnx_model_path):
            os.makedirs(onnx_model_path)

        # Load ORT model
        model = ORTModelForCausalLM.from_pretrained(model_name, from_transformers=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if strategy == "none":
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)

        if strategy == "quantization":
            quantizer = ORTQuantizer.from_pretrained(model)
            dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            model = quantizer.quantize(
                save_dir=onnx_model_path,
                quantization_config=dqconfig,
            )
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device=="cuda" else -1)

        # Pruning note: pruning for ONNX should be done before export, so this is a placeholder.
        # If pruning is desired, it should be integrated before exporting.


        return pipe



        
        return pipe
