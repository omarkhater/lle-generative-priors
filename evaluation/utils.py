import torch
from torch.utils.data import DataLoader

def validate_dataloader(dataloader: DataLoader, expected_dims: int, paired: bool = True) -> bool:
    """
    Validate that the dataloader yields items with the expected dimensions.
    
    For example, for Stage1 and E2E evaluation we expect the input x to have shape 
    [B, m, 3, H, W] (with m >= 1) and, if paired, the ground truth y to have shape [B, 3, H, W].
    
    Args:
        dataloader (DataLoader): The dataloader to validate.
        expected_dims (int): Expected number of dimensions for x (e.g. 5).
        paired (bool): Whether y (ground truth) is provided.
        
    Returns:
        True if the dataloader structure is valid; otherwise, raises ValueError.
    """
    try:
        sample = next(iter(dataloader))
    except StopIteration:
        raise ValueError("Dataloader is empty.")
    
    if paired:
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError("When paired, dataloader must yield (x, y).")
        x, y = sample
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            raise ValueError("Both x and y must be torch.Tensors.")
        if x.dim() != expected_dims:
            raise ValueError(f"Expected x with {expected_dims} dimensions, but got {x.shape}.")
        if y.dim() != 4:
            raise ValueError(f"Expected y with 4 dimensions ([B, 3, H, W]), but got {y.shape}.")
        if x.size(2) != 3 or y.size(1) != 3:
            raise ValueError("Expected channel dimension 3 for both x and y.")
    else:
        if not isinstance(sample, (tuple, list)) or len(sample) < 1:
            raise ValueError("When unpaired, dataloader must yield at least (x,).")
        x = sample[0]
        if not isinstance(x, torch.Tensor):
            raise ValueError("x must be a torch.Tensor.")
        if x.dim() != expected_dims:
            raise ValueError(f"Expected x with {expected_dims} dimensions, but got {x.shape}.")
    return True


def aggregate_metrics(metrics_list: list) -> dict:
    """
    Aggregate a list of metric dictionaries by averaging their values.
    
    Args:
        metrics_list (list): List of dictionaries with metric values.
    
    Returns:
        dict: Dictionary of aggregated (averaged) metrics.
    """
    if not metrics_list:
        return {}
    agg = {}
    keys = metrics_list[0].keys()
    for key in keys:
        agg[key] = sum(m.get(key, 0) for m in metrics_list) / len(metrics_list)
    return agg
