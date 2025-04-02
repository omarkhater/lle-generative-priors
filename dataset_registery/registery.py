import mlflow
from .loader import load_huggingface_dataset

class DatasetRegistry:
    """
    A simple registry to keep track of datasets.
    It loads a dataset, logs key metadata using MLflow, and stores the dataset for later use.
    """
    def __init__(self):
        self.datasets = {}
    
    def register_dataset(self, name: str, dataset_id: str, source: str):
        """
        Load a Hugging Face dataset and register it with the given name.
        Metadata (e.g., dataset ID and sample count) is logged via MLflow.
        """
        dataset = load_huggingface_dataset(dataset_id)
        self.datasets[name] = dataset
        metadata = {"dataset_id": dataset_id}
        try:
            metadata["n_samples"] = len(dataset)
        except Exception:
            metadata["n_samples"] = "Unknown"
        
        for key, value in metadata.items():
            mlflow.log_param(f"{name}_{key}", value)
    
    def get_dataset(self, name: str):
        """Return the registered dataset by name."""
        return self.datasets.get(name)
    
    def list_datasets(self):
        """List all registered dataset names."""
        return list(self.datasets.keys())
