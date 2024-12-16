import argparse
from scripts.metrics import measure_energy, print_results
from scripts.benchmark_model import load_model, run_benchmark
from optim_strategies.quantization import apply_quantization
from optim_strategies.pruning import apply_pruning

def main(strategy):
    model, tokenizer = load_model()
    input_text = "Hello, how are you?"

    if strategy == "baseline":
        print("Running baseline benchmark...")
        _, emissions, time_elapsed = measure_energy(run_benchmark, model, tokenizer, input_text)
        print_results("Baseline Benchmark", emissions, time_elapsed)

    elif strategy == "quantization":
        print("Applying quantization...")
        quantized_model = apply_quantization(model)
        _, emissions, time_elapsed = measure_energy(run_benchmark, quantized_model, tokenizer, input_text)
        print_results("Quantized Benchmark", emissions, time_elapsed)

    elif strategy == "pruning":
        print("Applying pruning...")
        pruned_model = apply_pruning(model)
        _, emissions, time_elapsed = measure_energy(run_benchmark, pruned_model, tokenizer, input_text)
        print_results("Pruned Benchmark", emissions, time_elapsed)

    else:
        print("Unknown strategy. Please choose between 'baseline', 'quantization', or 'pruning'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SmolLM model with different optimization strategies.")
    parser.add_argument("--strategy", type=str, required=True, help="Optimization strategy: baseline, quantization, or pruning")
    args = parser.parse_args()

    main(args.strategy)
