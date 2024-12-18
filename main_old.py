import argparse
from scripts.metrics import measure_energy, print_results, print_aggregate_results
from scripts.benchmark_model import load_model, run_benchmark_on_dataset, load_data_wikitext
from optim_strategies.quantization import apply_quantization
from optim_strategies.pruning import apply_pruning

def main(strategy, model_name, num_samples, device, backend, output_file):
    # Charger le modèle et les données
    model, tokenizer = load_model(model_name, device)
    dataset = load_data_wikitext(num_samples)

    if strategy == "baseline":
        print("Running baseline benchmark...")
        avg_emissions, std_emissions, avg_time, std_time = run_benchmark_on_dataset(
            model, tokenizer, dataset, device
        )
        print_aggregate_results("Baseline Benchmark", avg_emissions, std_emissions, avg_time, std_time)

    elif strategy == "quantization":
        print("Applying quantization...")
        quantized_model = apply_quantization(model)
        avg_emissions, std_emissions, avg_time, std_time = run_benchmark_on_dataset(
            quantized_model, tokenizer, dataset, device
        )
        print_aggregate_results("Quantized Benchmark", avg_emissions, std_emissions, avg_time, std_time)

    elif strategy == "pruning":
        print("Applying pruning...")
        amount_pruning = 0.4
        pruned_model = apply_pruning(model, amount=amount_pruning, backend=backend)
        avg_emissions, std_emissions, avg_time, std_time = run_benchmark_on_dataset(
            pruned_model, tokenizer, dataset, device
        )
        print_aggregate_results("Pruned Benchmark", avg_emissions, std_emissions, avg_time, std_time)

    else:
        print("Unknown strategy. Please choose between 'baseline', 'quantization', or 'pruning'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SmolLM model with different optimization strategies.")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M", help="Model name to load")
    parser.add_argument("--strategy", type=str, required=True, help="Optimization strategy: baseline, quantization, or pruning")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to run the benchmark on")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the benchmark on")
    parser.add_argument("--output_file", type=str, default="results.csv", help="Output file to save the results")
    parser.add_argument("--backend", type=str, default="default", help="Backend for pruning (default, nvidia, onednn)")
    args = parser.parse_args()

    main(args.strategy, args.model, args.num_samples, args.device, args.backend, args.output_file)
