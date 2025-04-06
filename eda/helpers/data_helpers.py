import torch
from typing import Tuple
from torch.utils.data import DataLoader
from dataset_registery.custom_dataset import variable_low_images_collate

def split_dataloader(
        dataloader: torch.utils.data.DataLoader, 
        split_ratio: float = .2
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Split a PyTorch DataLoader into two separate DataLoaders based on a given ratio.

    Args:
        dataloader (torch.utils.data.DataLoader):
            The DataLoader to split.
        split_ratio (float), Optional (default=.2):
            The ratio to split the DataLoader. 
    Returns:
        (torch.utils.data.DataLoader, torch.utils.data.DataLoader): Two separate DataLoaders.
        The first DataLoader will contain (1 - split_ratio) of the data.
        The second DataLoader will contain split_ratio of the data.
    """
    total_size = len(dataloader.dataset)
    split_size = int(total_size * split_ratio)
    train_size = total_size - split_size

    train_data, val_data = torch.utils.data.random_split(dataloader.dataset, [train_size, split_size])
    train_loader = DataLoader(
        train_data, 
        batch_size=dataloader.batch_size, 
        shuffle=True, 
        collate_fn=variable_low_images_collate
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=dataloader.batch_size, 
        shuffle=True, 
        collate_fn=variable_low_images_collate
    )

    return train_loader, val_loader
