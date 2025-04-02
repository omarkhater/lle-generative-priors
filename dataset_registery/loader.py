import mlflow.data.huggingface_dataset as hf_dataset
from torch.utils.data import DataLoader, Dataset
from typing import Any
from datasets import load_dataset
import torch
import torchvision.transforms as transforms
from PIL import Image

def load_huggingface_dataset(dataset_id: str) -> Any:
    """
    Load a Hugging Face dataset using MLflow's API.
    
    Parameters:
    -----------
    dataset_id : str
        The Hugging Face dataset ID (e.g., "okhater/NTIRE2025")
    """

    hf_ds = load_dataset(dataset_id)
    
    if hasattr(hf_ds, "keys"):
        first_key = next(iter(hf_ds.keys()))
        hf_ds = hf_ds[first_key]
    
    return hf_dataset.from_huggingface(
        ds=hf_ds,
        path=dataset_id,
    )

class HuggingFaceTorchDataset(Dataset):
    """
    A simple wrapper to convert an MLflow Hugging Face dataset into a PyTorch Dataset.
    Automatically converts PIL images to tensors.
    """
    def __init__(self, hf_dataset_obj):
        self.hf_dataset = hf_dataset_obj
        self.raw_dataset = self.hf_dataset.source.load()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        
        # Convert PIL images to tensors if present
        processed_sample = {}
        for key, value in sample.items():
            if isinstance(value, Image.Image):
                processed_sample[key] = self.transform(value)
            else:
                processed_sample[key] = value
        
        return processed_sample

def to_pytorch_dataloader(hf_dataset_obj, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Convert a Hugging Face dataset into a PyTorch DataLoader.
    """
    torch_dataset = HuggingFaceTorchDataset(hf_dataset_obj)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)
