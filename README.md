# Benchmarking SmolLM-135M: Energy Optimization and Performance Analysis

## Overview
This project evaluates the energy consumption and performance of the [SmolLM-135M language model](https://huggingface.co/HuggingFaceTB/SmolLM-135M) using various hardware configurations, backends, and optimization strategies. The study is conducted on the [SQuAD dataset](https://huggingface.co/datasets/rajpurkar/squad) (test split, 10 samples) and focuses on understanding the trade-offs between energy efficiency and model performance.

## Features
- **Hardware**: Benchmarked on CPU (Intel Xeon) and GPU (NVIDIA T4).
- **Backends**: Hugging Face Transformers and ONNX.
- **Optimization Strategies**:
  - Static 8-bit and 16-bit Quantization (GPU).
  - Dynamic 16-bit Quantization (CPU).
  - No optimization (baseline).
- **Metrics**:
  - Energy consumption (CPU, GPU, RAM) per token generated.
  - Inference time per token.
  - CO₂ emissions (calculated using CodeCarbon) per token generated.

## Methodology
1. **Dataset Loading**: Samples are preprocessed from the SQuAD dataset.
2. **Model Inference**: Predictions are generated using the specified backend and optimization strategy.
3. **Energy Tracking**:
   - CodeCarbon library measures energy consumed by CPU, GPU, and RAM.
   - CO₂ emissions are calculated based on carbon intensity of the local energy grid.

## Key Findings
- Static 8-bit quantization on GPU achieves the best energy efficiency with minimal performance loss.
- The ONNX backend offers competitive performance for CPU-based inference.
- Dynamic quantization balances flexibility and efficiency, particularly for CPU-bound tasks.

## Usage
To reproduce the benchmarks, use the following script:
```bash
python main.py --model_name HuggingFaceTB/SmolLM-135M \
    --strategy quantization8bit --backend base_gpu \
    --dataset_name squad --dataset_split test --samples 50
```
backend available options: `base_gpu`, `base_cpu`, `onnx_cpu`
strategy available options: `none`, `quantization8bit`, `quantization16bit`, `quantization` (dynamic 16 bit)

## Recommendations
- Use static 8-bit quantization for GPU deployments to minimize energy consumption.
- Employ the ONNX backend for CPU-bound inference tasks to improve efficiency.

## Limitations
- Results are based on only quantization optimisations and may vary with other strategies.
- Benchmarks are specific to NVIDIA T4 GPUs and Intel Xeon CPUs.

## Future Work
Further research with other optimisation startégies such as pruning and diverse hardware configurations is recommended to generalize the findings.

## References
- [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- [SQuAD Dataset](https://huggingface.co/datasets/rajpurkar/squad)
- [Hugging Face Transformers](https://huggingface.co)
- [ONNX Runtime](https://onnxruntime.ai)
- [CodeCarbon](https://github.com/mlco2/codecarbon)
