import argparse
from src.run_benchmark import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimization benchmarks on a text generation model.")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M", 
                        help="Hugging Face model name or path.")
    parser.add_argument("--strategy", type=str, choices=["none", "quantization", "quantization8bit", "quantization16bit"], 
                        default="none", help="Optimization strategy.")
    parser.add_argument("--backend", type=str, choices=["base_cpu", "base_gpu", "onnx_cpu", "onnx_gpu"], 
                        default="onnx_cpu", help="Backend and hardware.")
    parser.add_argument("--dataset_name", type=str, default="squad", help="Name of the Hugging Face dataset.")
    parser.add_argument("--dataset_split", type=str, default="test", 
                        help="Split of the dataset to load.")
    parser.add_argument("--samples", type=int, default=10, 
                        help="Number of samples to test from the dataset.")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model_name,
        strategy=args.strategy,
        backend=args.backend,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        samples=args.samples,
    )
