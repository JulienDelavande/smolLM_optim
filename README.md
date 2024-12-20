# Benchmarking SmolLM-135M: Energy Efficiency and Performance Analysis

## Overview
This repository benchmarks the [SmolLM-135M language model](https://huggingface.co/HuggingFaceTB/SmolLM-135M) to understand the trade-offs between performance and energy consumption. Using a subset of the [SQuAD dataset](https://huggingface.co/datasets/rajpurkar/squad) (test split, 50 samples), we measure how different hardware configurations, backends, and optimization methods affect inference speed, energy usage, and CO₂ emissions.

## Key Features
- **Model:** [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- **Hardware:** 
  - CPU (Intel Xeon)
  - GPU (NVIDIA T4)
- **Backends:** 
  - Hugging Face Transformers
  - ONNX Runtime
- **Optimization Strategies:**
  - No optimization (baseline)
  - Static 8-bit quantization (GPU)
  - Static 16-bit quantization (GPU)
  - Dynamic 16-bit quantization (CPU)
- **Metrics:**
  - Per-token energy consumption (CPU, GPU, RAM)
  - Per-token inference time
  - Per-token CO₂ emissions (calculated with [CodeCarbon](https://github.com/mlco2/codecarbon))

## Methodology
1. **Data Loading:** A subset of SQuAD test samples is loaded and preprocessed.
2. **Model Inference:** The samples are passed through the SmolLM-135M model with various backend and optimization settings.
3. **Energy & Emissions Tracking:** 
   - Use CodeCarbon to record CPU/GPU power draw and RAM usage.
   - Convert energy consumption into CO₂ emissions based on local energy grid carbon intensity.

## Usage
To reproduce the benchmark results:
```bash
python main.py --model_name HuggingFaceTB/SmolLM-135M \
    --strategy quantization8bit --backend base_gpu \
    --dataset_name squad --dataset_split test --samples 50
```

**Backends:** `base_gpu`, `base_cpu`, `onnx_cpu`  
**Strategies:** `none`, `quantization8bit`, `quantization16bit`, `quantization` (dynamic 16-bit)

## References
- [SmolLM-135M Model](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- [SQuAD Dataset](https://huggingface.co/datasets/rajpurkar/squad)
- [Hugging Face Transformers](https://huggingface.co)
- [ONNX Runtime](https://onnxruntime.ai)
- [CodeCarbon](https://github.com/mlco2/codecarbon)
