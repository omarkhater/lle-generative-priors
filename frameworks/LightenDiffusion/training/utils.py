import torch

def get_optimizer(
        model: torch.nn.Module, 
        lr:float = 1e-3, 
        weight_decay:float =1e-5
    )-> torch.optim.Optimizer:
    """
    Get Adam optimizer for the model.

    Args:
        model (torch.nn.Module): The model for which to get the optimizer.
        lr (float), Optional (default=1e-3): The learning rate for the optimizer.
        weight_decay (float), Optional (default=1e-5): The weight decay for the optimizer.

    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(
        optimizer: torch.optim.Optimizer, 
        gamma:float=0.9
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:

    """
    Get learning rate scheduler for the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to get the scheduler.
        gamma (float), Optional (default=0.9): The factor by which to reduce the learning rate.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        factor=gamma, 
        patience=5
    )