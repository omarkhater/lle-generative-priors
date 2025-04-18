import os
import yaml
import torch
import mlflow
import itertools
import argparse
from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv

from frameworks.LightenDiffusion.models.LightenDiffusion import Stage1, Stage2, LightenDiffusionPipeline
from frameworks.LightenDiffusion.models.unet import DiffusionUNet
from frameworks.LightenDiffusion.models.decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from frameworks.LightenDiffusion.training.stage2 import Stage2Trainer
from frameworks.LightenDiffusion.visualization.visualize_stage2 import visualize_stage2_results
from evaluation.lighten_diffusion_stage2 import evaluate_stage2_metrics_avgfirst
from experiments.scripts.select_top_stage1_models import select_top_models
from experiments.utils.general_utils import setup_dataloaders
from eda.helpers.training_helpers import get_optimizer, get_scheduler
from utils.mlflow_utils import setup_mlflow_tracking, log_dict_as_params
from experiments.utils.s3_utils import upload_file_to_s3, upload_directory_to_s3

# Load environment variables
load_dotenv(find_dotenv())
tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")
s3_bucket = os.environ.get("S3_BUCKET")

def visualize_and_save_samples(model, dataloader, output_dir, num_samples=3):
    """
    Visualize and save model outputs for MLflow artifacts.
    
    Args:
        model: Model to visualize
        dataloader: Dataloader containing samples
        output_dir: Directory to save output images
        num_samples: Number of samples to visualize
        
    Returns:
        str: Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    plt_interactive = plt.isinteractive()
    plt.ioff()
    
    visualize_stage2_results(model, dataloader, num_samples=num_samples)
    
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig.savefig(os.path.join(output_dir, f'sample_{i}.png'))
        plt.close(fig)
    
    # Restore interactive mode if it was on
    if plt_interactive:
        plt.ion()
        
    return output_dir

def load_trained_model(weights_path, inital_model, device = torch.device('cuda')):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"File not found: {weights_path}")
    model_weights = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = model_weights if isinstance(model_weights, dict) else model_weights.state_dict()
    inital_model.load_state_dict(state_dict)
    inital_model.to(device)
    inital_model.eval()
    print(f"Loaded model weights from {weights_path}")
    return inital_model


def run_experiment_with_model(
    stage1_model_path: str,
    stage1_model_id: str,
    config: Dict[str, Any],
    diffusion_config: Dict[str, Any],
    sweep_params: Dict[str, Any],
    dataloaders: Dict[str, torch.utils.data.DataLoader]
):
    """
    Run Stage2 experiment with a specific Stage1 model and sweep parameters.
    
    Args:
        stage1_model_path: Path to the Stage1 model
        stage1_model_id: Identifier for the Stage1 model
        config: Base configuration
        sweep_params: Parameters to sweep
        dataloaders: Dictionary of dataloaders
    """
    device = torch.device(config['device'])
    latent_space_dim = config['latent_space_dim']
    stage1_model = Stage1(
        encoder=ImageEncoder(latent_space_dim),
        decoder=ImageDecoder(latent_space_dim),
        decomposer=RetinexDecomposition(),
    )
    
    stage1_model = load_trained_model(stage1_model_path, stage1_model, device=device)
    diffusion_config["ch_mult"] = tuple(diffusion_config["ch_mult"])
    diffusion_net = DiffusionUNet(**diffusion_config)
    stage2_model = Stage2(diffusion_unet=diffusion_net)
    model = LightenDiffusionPipeline(stage1=stage1_model, stage2=stage2_model)
    model.to(device)
    run_name = f"stage1_{stage1_model_id}_lambda{sweep_params['lambda_scc']}_gamma{sweep_params['gamma']}"
    mlflow.set_experiment(config['experiment_name'])
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        params = {**config, **sweep_params, 'stage1_model_id': stage1_model_id}
        log_dict_as_params(params)
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(
            stage2_model,
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )
        
        scheduler = None
        if config.get('use_scheduler', False):
            scheduler = get_scheduler(optimizer, gamma=config['scheduler_gamma'])
        
        # Create trainer
        num_diffusion_steps = config['num_diffusion_timesteps']
        trainer = Stage2Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config['num_epochs'],
            val_frequency=config['val_frequency'],
            patience=config['patience'],
            lambda_scc=sweep_params['lambda_scc'],
            betas=torch.linspace(0.0001, 0.02, steps=num_diffusion_steps),
            num_diffusion_timesteps=num_diffusion_steps,
            num_sampling_timesteps=config['num_sampling_timesteps'],
            gamma=sweep_params['gamma'],
            save_visualization_dir=os.path.join("outputs", "visualizations", run_name),
        )
        best_model, metrics = trainer.train()
        
        # Log metrics
        mlflow.log_metric("best_val_loss", metrics['best_loss'])
        mlflow.log_metric("best_epoch", metrics['best_epoch'])
        
        for i, loss in enumerate(metrics['train_losses']):
            mlflow.log_metric("losses/train", float(loss), step=i)

        val_steps = [j * config['val_frequency'] for j in range(len(metrics['val_losses']))]
        for step, vloss in zip(val_steps, metrics['val_losses']):
            mlflow.log_metric("losses/val", float(vloss), step=step)

        higher = {"psnr", "ssim"}
        lower  = {"tv_illumination", "pi", "niqe", "lpips"}
        if hasattr(trainer, "all_val_metrics"):
            for epoch_idx, vm in enumerate(trainer.all_val_metrics):
                for name, val in vm.items():
                    if name in higher:
                        key = f"higher_is_better/val/{name}"
                    elif name in lower:
                        key = f"lower_is_better/val/{name}"
                    else:
                        key = f"val/{name}"
                    mlflow.log_metric(key, float(val), step=epoch_idx)


        # Evaluate on test set
        test_metrics = evaluate_stage2_metrics_avgfirst(best_model, dataloaders['test'])
        for name, val in test_metrics.items():
            if name in higher:
                key = f"higher_is_better/test/{name}"
            elif name in lower:
                key = f"lower_is_better/test/{name}"
            else:
                key = f"test/{name}"
            mlflow.log_metric(key, float(val))
        
        # Visualize results
        vis_dir = os.path.join("outputs", "visualizations", run_name)
        visualize_and_save_samples(best_model, dataloaders['val'], vis_dir)
        mlflow.log_artifacts(vis_dir, "visualizations")
        
        # Save model
        model_dir = os.path.join("outputs", "models", run_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "lighten_diffusion_model.pth")
        torch.save(best_model.state_dict(), model_path)
        mlflow.log_artifact(model_path, "model")
        
        # Upload to S3 if configured
        if s3_bucket:
            s3_prefix = f"experiments/{config['experiment_name']}/{run_name}"
            upload_file_to_s3(model_path, s3_bucket, f"{s3_prefix}/model.pth")
            upload_directory_to_s3(vis_dir, s3_bucket, f"{s3_prefix}/visualizations")
            
        # Log PyTorch model
        mlflow.pytorch.log_model(best_model, "pytorch_model")

def main():
    parser = argparse.ArgumentParser(description="Run Stage2 experiment with loss weight sweep")
    parser.add_argument(
        "--config", 
        type=str, 
        default="experiments/configs/sweep_stage2_loss_weights.yaml", 
        help="Path to experiment config file"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    base_config = config["base"]
    diffusion_config = config["diffusion_unet"]
    setup_mlflow_tracking(tracking_uri)
    
    # Get top Stage1 models
    top_model_infos = select_top_models(args.config)
    model_paths = [(info["path"], info["run_id"]) for info in top_model_infos]
    
    # Setup dataloaders
    dataloaders = setup_dataloaders(base_config)
    
    # Generate parameter combinations
    sweep_config = config["sweep"]
    keys, values = zip(*sweep_config.items())
    
    # Limit to maximum 10 runs
    max_runs = 10
    total_combinations = len(model_paths) * len(list(itertools.product(*values)))
    
    # Strategy: If too many combinations, prioritize best models with fewer parameter combinations
    if total_combinations > max_runs:
        # Sort models by metric value (already sorted by get_top_models_by_metric)
        # Take just enough parameter combinations to stay under max_runs
        params_per_model = max(1, max_runs // len(model_paths))
        param_combinations = list(itertools.product(*values))[:params_per_model]
    else:
        param_combinations = list(itertools.product(*values))
    
    # Run experiments
    run_count = 0
    for model_path, model_id in model_paths:
        for param_values in param_combinations:
            sweep_params = dict(zip(keys, param_values))
            run_experiment_with_model(
                stage1_model_path=model_path,
                stage1_model_id=model_id[:8],  # Short ID for naming
                config=base_config,
                diffusion_config=diffusion_config,
                sweep_params=sweep_params,
                dataloaders=dataloaders
            )
            run_count += 1
            if run_count >= max_runs:
                print(f"Reached maximum number of runs ({max_runs}). Stopping.")
                return

if __name__ == "__main__":
    main()