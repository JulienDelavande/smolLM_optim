"""
Load a Hugging Face dataset.
"""
from datasets import load_dataset

def load_dataset_samples(dataset_name: str, dataset_split: str, samples: int):
    """
    Load a dataset from Hugging Face Datasets and return max_samples texts.
    """
    ds = load_dataset(dataset_name, split=dataset_split)
    samples = ds[:samples]["question"]
    return [s for s in samples if s.strip()]
