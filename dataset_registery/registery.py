import mlflow
from .loader import load_huggingface_dataset
from torch.utils.data import DataLoader

class DatasetRegistry:
    """
    A registry for managing datasets by name, split, and type.
    Supports both paired and unpaired datasets and logs metadata via MLflow.
    """
    def __init__(self):
        # Structure: { dataset_name: { split_name: PyTorchDataset, ... }, ... }
        self.datasets = {}

    def register_dataset(self, name: str, dataset_id: str, splits: list = ['train'], dataset_type: str = "paired"):
        """
        Load a Hugging Face dataset and register it under the given name for specified splits.
        
        Parameters:
            name (str): Identifier for the dataset.
            dataset_id (str): Hugging Face dataset ID.
            splits (list): List of split names to register (e.g. ["train", "validation", "test"]).
            dataset_type (str): Either "paired" or "unpaired".
        """
        hf_ds = load_huggingface_dataset(dataset_id)
        self.datasets[name] = {}

        for split in splits:
            if split in hf_ds:
                ds_split = hf_ds[split]
                if dataset_type == "paired":
                    # Lazy import of custom paired dataset wrapper.
                    from .paired_dataset import PairedImageDataset
                    wrapped_ds = PairedImageDataset(ds_split)
                else:
                    # For unpaired datasets.
                    from .unpaired_dataset import UnpairedImageDataset
                    wrapped_ds = UnpairedImageDataset(ds_split)
                    
                self.datasets[name][split] = wrapped_ds

                # Log metadata
                mlflow.log_param(f"{name}_{split}_n_samples", len(wrapped_ds))
            else:
                print(f"Warning: Split '{split}' not found in dataset '{dataset_id}'")
        mlflow.log_param(f"{name}_dataset_id", dataset_id)

    def get_dataset(self, name: str, split: str):
        """Retrieve the registered dataset for a given name and split."""
        return self.datasets.get(name, {}).get(split)

    def list_datasets(self):
        """List all registered dataset names and their splits."""
        return {name: list(splits.keys()) for name, splits in self.datasets.items()}

    def get_dataloader(self, name: str, split: str, batch_size: int = 32, shuffle: bool = True):
        """
        Convenience method to return a DataLoader for a registered dataset split.
        """
        ds = self.get_dataset(name, split)
        if ds is None:
            raise ValueError(f"Dataset {name} with split {split} not registered.")
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
