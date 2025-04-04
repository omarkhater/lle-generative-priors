from datasets import load_dataset

def load_huggingface_dataset(dataset_id: str, data_dir: str = None):
    """
    Load a Hugging Face dataset and return it in a dictionary format.
    
    If the dataset object has multiple splits, return a dict with split names as keys.
    Otherwise, wrap it in a dict with a default key.
    
    Parameters:
        dataset_id (str): Hugging Face dataset ID.
        data_dir (str): Optional directory to store the dataset locally.
    """
    ds = load_dataset(dataset_id, cache_dir=data_dir)
    
    # If ds is a dict (with splits) then return it; otherwise, assume a single split.
    if hasattr(ds, "keys"):
        return ds
    else:
        return {"default": ds}

# Optionally, you can include a helper to convert to a DataLoader if needed.
def to_pytorch_dataloader(torch_dataset, batch_size: int = 32, shuffle: bool = True):
    from torch.utils.data import DataLoader
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)
