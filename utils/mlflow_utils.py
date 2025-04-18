import os
import mlflow
from typing import Dict, Any, Optional
import subprocess
import torch
import psutil
import logging


def log_hardware_info():
    mlflow.log_param("num_gpus", torch.cuda.device_count())
    if torch.cuda.is_available():
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
        mlflow.log_param("gpu_mem_total_MB", torch.cuda.get_device_properties(0).total_memory // (1024**2))
    mlflow.log_param("cpu_count", psutil.cpu_count())
    mlflow.log_param("ram_total_GB", round(psutil.virtual_memory().total / (1024**3), 2))

def setup_mlflow_tracking(tracking_uri: Optional[str] = None) -> None:
    """
    Configure MLflow tracking URI.
    
    Args:
        tracking_uri: URI for MLflow tracking. If None, uses default local directory.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logging.info(f"MLflow tracking URI set to: {tracking_uri}")
    else:
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        logging.info("MLflow tracking URI set to local directory: ./mlruns")
        
def create_experiment_group(group_name: str, description: str = "") -> str:
    """
    Create or get an MLflow experiment group.
    
    Args:
        group_name: Name for the experiment group
        description: Optional description of the experiment group
        
    Returns:
        experiment_id: The ID of the created or existing experiment
    """
    experiment = mlflow.get_experiment_by_name(group_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=group_name,
            tags={"description": description}
        )
        print(f"Created new experiment '{group_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{group_name}' with ID: {experiment_id}")
    
    return experiment_id

def log_dict_as_params(params_dict: Dict[str, Any]) -> None:
    """
    Log a dictionary of parameters to the current MLflow run.
    
    Args:
        params_dict: Dictionary of parameters to log
    """
    for key, value in params_dict.items():
        if not isinstance(value, (str, int, float, bool)):
            mlflow.log_param(key, str(value))
        else:
            mlflow.log_param(key, value)

def log_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("git_branch", branch)
    except Exception as e:
        mlflow.set_tag("git_info_error", str(e))
