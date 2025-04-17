import os
import argparse
import yaml
import torch
import mlflow
from dotenv import load_dotenv, find_dotenv
from experiments.utils.model_selection import get_top_models_by_metric, download_model_from_run
from utils.mlflow_utils import setup_mlflow_tracking

load_dotenv(find_dotenv())
tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")

def select_top_models(config_path: str) -> list:
    """
    Select top Stage1 models based on a metric.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of paths to downloaded models
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    base_config = config["base"]
    selection_config = base_config["stage1_model_selection"]
    
    # Setup MLflow tracking
    setup_mlflow_tracking(tracking_uri)
    
    # Get top models
    top_models = get_top_models_by_metric(
        experiment_name=selection_config["experiment_name"],
        metric_name=selection_config["metric"],
        top_k=selection_config["top_k"],
        mlflow_tracking_uri=tracking_uri,
        higher_is_better=selection_config.get("higher_is_better", True)
    )
    
    # Download models
    model_paths = []
    for i, model_info in enumerate(top_models):
        print(f"\nFound model {i+1}/{len(top_models)}:")
        print(f"  Run ID: {model_info['run_id']}")
        print(f"  {selection_config['metric']}: {model_info['metric_value']}")
        print(f"  Parameters: {model_info['params']}")
        
        model_dir = os.path.join("models", "stage1", f"model_{i}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = download_model_from_run(
            run_id=model_info["run_id"],
            local_path=model_dir,
            mlflow_tracking_uri=tracking_uri
        )
        model_paths.append({
            "path": model_path,
            "run_id": model_info["run_id"],
            "metric_value": model_info["metric_value"],
            "params": model_info["params"]
        })
        
    return model_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select top Stage1 models")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config file")
    args = parser.parse_args()
    
    model_paths = select_top_models(args.config)
    
    print("\nSelected models:")
    for i, model_info in enumerate(model_paths):
        print(f"{i+1}. Path: {model_info['path']}")
        print(f"   Run ID: {model_info['run_id'][:8]}...")