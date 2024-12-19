"""
Load a Hugging Face dataset.
"""
from datasets import load_dataset

def load_dataset_samples(dataset_name: str, dataset_split: str, max_samples: int):
    """
    Load a dataset from Hugging Face Datasets and return max_samples texts.
    """
    ds = load_dataset(dataset_name, split=dataset_split)
    # Assume ds is a text dataset with a 'text' field. Adjust if needed.
    samples = ds[:max_samples]["question"]
    return [s for s in samples if s.strip()]
