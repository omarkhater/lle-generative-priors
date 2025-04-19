import yaml
import torch
from typing import Dict, Any
from dataset_registery.registery import DatasetManager
from eda.helpers.data_helpers import split_dataloader

class ExperimentConfig:
    """Helper class for loading and managing experiment configurations."""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Initialize experiment configuration.
        
        Args:
            config_path: Path to YAML config file
            config_dict: Dictionary with configuration (alternative to config_path)
        """
        self.config_path = config_path
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
            
        self._set_defaults()
        
    def _set_defaults(self):
        """Set default values for missing configuration fields."""
        defaults = {
            'experiment_name': 'default_experiment',
            'run_name': 'default_run',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'log_artifacts': True,
            'val_split_ratio': 0.2
        }
        
        for key, default_value in defaults.items():
            if key not in self.config:
                self.config[key] = default_value
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary syntax."""
        return self.config[key]
    
    def keys(self):
        """Get all configuration keys."""
        return self.config.keys()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config
        

def setup_dataloaders(config: ExperimentConfig):
    """
    Set up dataloaders based on experiment configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        dict: Dictionary containing train, val, and test dataloaders
    """
    dataset_name = config.get('dataset_name')
    dataset_id = config.get('dataset_id')
    batch_size = config.get('batch_size', 16)
    val_split_ratio = config.get('val_split_ratio', 0.2)
    registry = DatasetManager()
    registry.initialize_dataset(
        name=dataset_name,
        dataset_id=dataset_id,
        splits=["train", "test"],
        dataset_type="paired",
        hf_cache_dir=f"../../datasets/{dataset_name}",
    )
    train_loader = registry.get_dataloader(dataset_name, "train", batch_size=batch_size, shuffle=True)
    test_loader = registry.get_dataloader(dataset_name, "test", batch_size=batch_size, shuffle=False)
    train_loader, val_loader = split_dataloader(train_loader, split_ratio=val_split_ratio)
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }