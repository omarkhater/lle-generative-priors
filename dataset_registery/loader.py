from datasets import load_dataset
from torch.utils.data import DataLoader

def load_huggingface_dataset(dataset_id: str, hf_cache_dir: str = None):
    """
    Load a Hugging Face dataset and return it in a dictionary format.
    
    If the dataset object has multiple splits, return a dict with split names as keys.
    Otherwise, wrap it in a dict with a default key.
    
    Parameters:
        dataset_id (str): Hugging Face dataset ID.
        hf_cache_dir (str): Optional directory to store the dataset locally.
    """
    ds = load_dataset(dataset_id, cache_dir=hf_cache_dir)
    
    # If ds is a dict (with splits) then return it; otherwise, assume a single split.
    if hasattr(ds, "keys"):
        return ds
    else:
        return {"default": ds}

def to_pytorch_dataloader(
        torch_dataset, 
        batch_size: int = 32, 
        shuffle: bool = True,
        collate_fn: callable = None
        ) -> DataLoader:
    """
    Convert a PyTorch Dataset to a DataLoader.

    Parameters:
        torch_dataset (torch.utils.data.Dataset): The PyTorch dataset to convert.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        collate_fn (callable): Function to merge a list of samples into a batch.
    """
    
    return DataLoader(
        torch_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn
    )