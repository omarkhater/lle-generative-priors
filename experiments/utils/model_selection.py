import os
import mlflow
import pandas as pd
import logging
from typing import List, Dict, Any
import torch

def get_top_models_by_metric(
    experiment_name: str,
    metric_name: str,
    top_k: int = 3,
    mlflow_tracking_uri: str = None,
    higher_is_better: bool = True
) -> List[Dict[str, Any]]:
    """
    Get the top k models from an experiment based on a specific metric.
    
    Args:
        experiment_name: Name of the experiment to search in
        metric_name: Metric to sort by (e.g., 'psnr', 'ssim')
        top_k: Number of top models to return
        mlflow_tracking_uri: MLflow tracking URI
        higher_is_better: If True, sort in descending order; otherwise, ascending
    
    Returns:
        List of dictionaries containing run_id, metric value, and params
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Get all runs for the experiment without filtering by metric initially
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
    )
    
    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    # Try different metric key formats
    metric_formats = [
        f"metrics.higher_is_better/test/{metric_name}" if higher_is_better else f"metrics.lower_is_better/test/{metric_name}",
        f"metrics.test/{metric_name}",
        f"metrics.{metric_name}"
    ]
    
    # Find the first format that exists in the DataFrame
    valid_metric_column = None
    for format in metric_formats:
        if format in runs.columns and not runs[format].isna().all():
            valid_metric_column = format
            break
    
    if valid_metric_column is None:
        raise ValueError(f"No runs found with metric '{metric_name}' in experiment '{experiment_name}'")
    
    # Extract the actual metric key from the column name (removing 'metrics.' prefix)
    metric_key = valid_metric_column.replace("metrics.", "")
    
    # Filter out rows where the metric is NULL
    filtered_runs = runs[~runs[valid_metric_column].isna()]
    
    if filtered_runs.empty:
        raise ValueError(f"No runs found with non-NULL values for '{metric_name}' in experiment '{experiment_name}'")
    
    # Sort based on higher_is_better flag
    sorted_runs = filtered_runs.sort_values(
        by=valid_metric_column, 
        ascending=not higher_is_better
    )
    
    # Get top k runs
    top_runs = sorted_runs.head(top_k)
    
    # Extract relevant information
    results = []
    for _, run in top_runs.iterrows():
        result = {
            "run_id": run["run_id"],
            "params": {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.")},
            "artifact_uri": run["artifact_uri"],
            "metric_value": run[valid_metric_column],  # Use the valid column we already identified
            "metric_name": metric_key
        }
        
        results.append(result)
    
    return results

def download_model_from_run(run_id: str, local_path: str = None, mlflow_tracking_uri: str = None) -> str:
    """
    Download a model artifact from an MLflow run and return path to the model weights file.
    
    Args:
        run_id: MLflow run ID
        local_path: Path to save the model (if None, uses a temp directory)
        mlflow_tracking_uri: MLflow tracking URI
    
    Returns:
        Path to downloaded model weights file
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    if local_path is None:
        import tempfile
        local_path = tempfile.mkdtemp()
    
    # Download the model
    model_path = os.path.join(local_path, run_id)
    os.makedirs(model_path, exist_ok=True)
    
    # Possible model file paths to check
    model_file_paths = []
    
    try:
        # Try to download from "model" artifact
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=model_path
        )
        # Check common model file paths
        model_file_paths = [
            os.path.join(artifact_path, "data", "model.pth"),
            os.path.join(artifact_path, "model.pth"),
            os.path.join(artifact_path, "weights.pth")
        ]
    except Exception as e:
        logging.warning(f"Could not download 'model' artifact: {e}")
        # Try to download from "pytorch_model" artifact
        try:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="pytorch_model",
                dst_path=model_path
            )
            # Check common model file paths
            model_file_paths = [
                os.path.join(artifact_path, "model.pth"),
                artifact_path if os.path.isfile(artifact_path) else None
            ]
        except Exception as e2:
            logging.error(f"Failed to download model artifacts: {e2}")
            # Try to load directly as a PyTorch model
            try:
                loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
                artifact_path = os.path.join(model_path, "model.pth")
                torch.save(loaded_model.state_dict(), artifact_path)
                model_file_paths = [artifact_path]
            except Exception as e3:
                logging.error(f"All attempts to download model failed: {e3}")
                raise ValueError(f"Could not download model for run {run_id}")
    
    # Find the first valid model file path
    model_file_path = None
    for path in model_file_paths:
        if path and os.path.isfile(path):
            model_file_path = path
            break
    
    if model_file_path is None:
        # If we couldn't find a model file but have an artifact path, try searching for .pth files
        if os.path.exists(artifact_path):
            for root, _, files in os.walk(artifact_path):
                for file in files:
                    if file.endswith(".pth"):
                        model_file_path = os.path.join(root, file)
                        break
                if model_file_path:
                    break
    
    if model_file_path is None:
        raise ValueError(f"Could not find a model weights file in the downloaded artifacts for run {run_id}")
    
    logging.info(f"Downloaded model from run {run_id}, model file at: {model_file_path}")
    return model_file_path