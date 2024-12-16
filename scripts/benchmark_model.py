from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="HuggingFaceTB/SmolLM-135M"):
    """Charge le modèle et le tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def run_benchmark(model, tokenizer, input_text):
    """Exécute un benchmark sur un texte d'entrée."""
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    return outputs
