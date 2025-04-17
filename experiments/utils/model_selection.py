import os
import mlflow
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple

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
    
    # Get all runs for the experiment
    metric_key = f"higher_is_better/test/{metric_name}" if higher_is_better else f"lower_is_better/test/{metric_name}"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric_key} IS NOT NULL",
        order_by=[f"metrics.{metric_key} {'DESC' if higher_is_better else 'ASC'}"]
    )
    
    if runs.empty:
        # Try without higher_is_better prefix
        metric_key = f"test/{metric_name}"
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"metrics.{metric_key} IS NOT NULL",
            order_by=[f"metrics.{metric_key} {'DESC' if higher_is_better else 'ASC'}"]
        )
    
    if runs.empty:
        raise ValueError(f"No runs found with metric '{metric_name}' in experiment '{experiment_name}'")
    
    # Get top k runs
    top_runs = runs.head(top_k)
    
    # Extract relevant information
    results = []
    for _, run in top_runs.iterrows():
        result = {
            "run_id": run["run_id"],
            "params": {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.")},
            "artifact_uri": run["artifact_uri"]
        }
        
        # Find the metric value, checking different possible formats
        for key in [metric_key, f"higher_is_better/test/{metric_name}", 
                   f"lower_is_better/test/{metric_name}", f"test/{metric_name}", metric_name]:
            col_name = f"metrics.{key}"
            if col_name in run and not pd.isna(run[col_name]):
                result["metric_value"] = run[col_name]
                result["metric_name"] = key
                break
        
        if "metric_value" not in result:
            logging.warning(f"Could not find metric value for run {run['run_id']}")
            result["metric_value"] = None
            
        results.append(result)
    
    return results

def download_model_from_run(run_id: str, local_path: str = None, mlflow_tracking_uri: str = None) -> str:
    """
    Download a model artifact from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        local_path: Path to save the model (if None, uses a temp directory)
        mlflow_tracking_uri: MLflow tracking URI
    
    Returns:
        Path to downloaded model
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    if local_path is None:
        import tempfile
        local_path = tempfile.mkdtemp()
    
    # Download the model
    model_path = os.path.join(local_path, run_id)
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # Try to download from "model" artifact
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=model_path
        )
    except Exception as e:
        logging.warning(f"Could not download 'model' artifact: {e}")
        # Try to download from "pytorch_model" artifact
        try:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="pytorch_model",
                dst_path=model_path
            )
        except Exception as e2:
            logging.error(f"Failed to download model artifacts: {e2}")
            # Try to load directly as a PyTorch model
            try:
                loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
                artifact_path = os.path.join(model_path, "model.pth")
                torch.save(loaded_model.state_dict(), artifact_path)
            except Exception as e3:
                logging.error(f"All attempts to download model failed: {e3}")
                raise ValueError(f"Could not download model for run {run_id}")
    
    logging.info(f"Downloaded model from run {run_id} to {artifact_path}")
    return artifact_path