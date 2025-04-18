import os
import torch
import argparse
import yaml
import mlflow
from dataset_registery.registery import DatasetManager
from utils.mlflow_utils import setup_mlflow_tracking, create_experiment_group, log_dict_as_params
from evaluation.classic_enhancement import evaluate_classic_enhancement, visualize_classic_results
from frameworks.ClassicEnhancement.models.enhancement import ClassicImageEnhancer
import json

def run_experiment(config_path):
    """
    Run classic enhancement experiment with MLflow tracking.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup experiment tracking
    tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")
    setup_mlflow_tracking(tracking_uri=tracking_uri)
    create_experiment_group(
        config['experiment_name'], 
        description=config.get('description', '')
    )
    mlflow.set_experiment(config.get('experiment_name'))
    run_name = f"classic_enhancement_b{config['brightness']}_c{config['contrast']}_g{config['gamma']}"
    mlflow.start_run(run_name=run_name)
    log_dict_as_params(config)
    device = torch.device(config.get('device', 'cuda') 
                         if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup data loaders
    registry = DatasetManager()
    registry.initialize_dataset(
        name=config['data_name'],
        dataset_id=config['dataset_id'],
        splits=["train", "test"],
        dataset_type="paired",
        hf_cache_dir=f"../../datasets/{config['data_name']}",
    )
    
    train_loader = registry.get_dataloader(
        config['data_name'], 
        "train", 
        batch_size=config['batch_size'], 
        shuffle=config['shuffle']
    )
    test_loader = registry.get_dataloader(
        config['data_name'], 
        "test", 
        batch_size=config['batch_size'], 
        shuffle=config['shuffle']
    )

    enhancer = ClassicImageEnhancer(
        brightness=config.get('brightness', 2.0),
        contrast=config.get('contrast', 1.5),
        sharpness=config.get('sharpness', 1.2),
        saturation=config.get('saturation', 1.1),
        gamma=config.get('gamma', 0.8),
        use_clahe=config.get('use_clahe', False),
        clahe_clip_limit=config.get('clahe_clip_limit', 2.0),
        clahe_grid_size=config.get('clahe_grid_size', 8)
    )
    enhancer.to(device)
    log_dict_as_params(config)
    mlflow.log_param('train_size', len(train_loader.dataset))
    mlflow.log_param('val_size', 0)
    mlflow.log_param('test_size', len(test_loader.dataset))
    
    # Evaluate on validation set
    higher = {"psnr", "ssim"}
    lower  = {"tv_illumination", "pi", "niqe", "lpips"}
    print("Evaluating on training set...")
    train_metrics = evaluate_classic_enhancement(enhancer, train_loader, device=device)
    
    for name, val in train_metrics.items():
        if name in higher:
            key = f"higher_is_better/train/{name}"
        elif name in lower:
            key = f"lower_is_better/train/{name}"
        else:
            key = f"train/{name}"
        mlflow.log_metric(key, float(val))
        
    print("Evaluating on test set...")
    test_metrics = evaluate_classic_enhancement(enhancer, test_loader, device=device)
    for name, val in test_metrics.items():
        if name in higher:
            key = f"higher_is_better/test/{name}"
        elif name in lower:
            key = f"lower_is_better/test/{name}"
        else:
            key = f"test/{name}"
        mlflow.log_metric(key, float(val))

    # Visualize results
    print("Generating visualizations...")
    vis_dir = os.path.join("outputs", "visualizations", run_name)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize train samples
    train_fig = visualize_classic_results(
        enhancer, 
        train_loader, 
        num_samples=config.get('num_visualization_samples', 4),
        device=device
    )
    train_fig_path = os.path.join(vis_dir, "train_samples.png")
    train_fig.savefig(train_fig_path)
    mlflow.log_artifact(train_fig_path, "visualizations")
    
    # Visualize test samples
    test_fig = visualize_classic_results(
        enhancer, 
        test_loader, 
        num_samples=config.get('num_visualization_samples', 4),
        device=device
    )
    test_fig_path = os.path.join(vis_dir, "test_samples.png")
    test_fig.savefig(test_fig_path)
    mlflow.log_artifact(test_fig_path, "visualizations")
    model_params = {
        'brightness': enhancer.brightness,
        'contrast': enhancer.contrast,
        'sharpness': enhancer.sharpness,
        'saturation': enhancer.saturation,
        'gamma': enhancer.gamma,
        'use_clahe': enhancer.use_clahe,
        'clahe_clip_limit': enhancer.clahe_clip_limit,
        'clahe_grid_size': enhancer.clahe_grid_size
    }
    model_dir = os.path.join("outputs", "models", run_name)
    os.makedirs(model_dir, exist_ok=True)
    params_path = os.path.join(model_dir, "enhancer_params.json")
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    mlflow.log_artifact(params_path, "model")
    print(f"Experiment completed successfully. Results saved to {vis_dir}")
    print(f"Test metrics: {test_metrics}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classic enhancement experiment with MLflow tracking')
    parser.add_argument(
        '--config', 
        type=str,
        default='experiments/configs/classic_enhancement.yaml',
        help='Path to experiment config file'
    )
    args = parser.parse_args()
    
    run_experiment(args.config)