import os
import yaml
import torch
import mlflow
import argparse
from dotenv import load_dotenv, find_dotenv
import matplotlib
matplotlib.use('Agg')
from frameworks.LightenDiffusion.models.LightenDiffusion import Stage2, LightenDiffusionPipeline
from frameworks.LightenDiffusion.models.unet import DiffusionUNet
from frameworks.LightenDiffusion.training.stage2 import Stage2Trainer
from frameworks.LightenDiffusion.losses import stage2_loss_wrapper
from eda.helpers.training_helpers import get_optimizer, get_scheduler
from experiments.utils.general_utils import setup_dataloaders, ExperimentConfig
from utils.mlflow_utils import setup_mlflow_tracking, create_experiment_group, log_dict_as_params
from evaluation.lighten_diffusion_stage2 import evaluate_stage2_metrics_avgfirst
from experiments.scripts.stage2_loss_weights import visualize_and_save_samples, load_stage1_model
from experiments.utils.s3_utils import upload_file_to_s3, upload_directory_to_s3
from experiments.scripts.select_top_stage1_models import select_top_models

# Load environment variables
load_dotenv(find_dotenv())
tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")
s3_bucket = os.environ.get("S3_BUCKET")

def run_experiment(config_path, model_index=0):
    """Run Stage2 experiment with MLflow tracking using the top Stage1 model by default."""
    config = ExperimentConfig(config_path)
    setup_mlflow_tracking(tracking_uri=tracking_uri)
    create_experiment_group(
        config.get('experiment_name'), 
        description=config.get('description', '')
    )
    
    # Get top Stage1 models
    top_model_infos = select_top_models(config_path)
    if not top_model_infos:
        raise ValueError("No Stage1 models found. Check your configuration.")
    
    if model_index >= len(top_model_infos):
        print(f"Warning: Requested model index {model_index} exceeds available models. Using best model (index 0) instead.")
        model_index = 0
    
    selected_model = top_model_infos[model_index]
    stage1_model_path = selected_model["path"]
    stage1_model_id = selected_model["run_id"][:8]  # Short ID for naming
    
    # Set run name based on selected model and configuration
    run_name = config.get('run_name', f"stage2_model{model_index}_{stage1_model_id}")
    
    # Setup dataloaders
    dataloaders = setup_dataloaders(config)
    
    # Load Stage1 model
    device = torch.device(config.get('device'))
    latent_space_dim = config.get('latent_space_dim')
    stage1_model = load_stage1_model(stage1_model_path, device, latent_space_dim)
    
    # Create Stage2 model
    diffusion_net = DiffusionUNet(
        in_channels=latent_space_dim*2,
        out_channels=latent_space_dim,
        ch=config.get('diffusion_ch', 64),
        ch_mult=config.get('diffusion_ch_mult', (1, 2, 3, 4)),
        num_res_blocks=config.get('diffusion_num_res_blocks', 2),
        dropout=config.get('diffusion_dropout', 0.0),
        conditional=True,
        resamp_with_conv=True,
    )
    stage2_model = Stage2(diffusion_unet=diffusion_net)
    
    # Create full pipeline
    model = LightenDiffusionPipeline(stage1=stage1_model, stage2=stage2_model)
    model.to(device)
    
    # Define metrics categorization
    higher_metrics = {"psnr", "ssim"}
    lower_metrics = {"lpips", "niqe", "pi"}
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        log_dict_as_params(config.to_dict())
        mlflow.log_param('train_size', len(dataloaders['train'].dataset))
        mlflow.log_param('val_size', len(dataloaders['val'].dataset))
        mlflow.log_param('test_size', len(dataloaders['test'].dataset))
        mlflow.log_param('stage1_model_id', stage1_model_id)
        mlflow.log_param('stage1_model_path', stage1_model_path)
        
        # Create optimizer and scheduler
        optimizer = get_optimizer(
            stage2_model,
            lr=float(config.get('learning_rate')),
            weight_decay=float(config.get('weight_decay'))
        )
        
        scheduler = None
        if config.get('use_scheduler', False):
            scheduler = get_scheduler(optimizer, gamma=config.get('scheduler_gamma'))
        
        # Create trainer
        num_diffusion_steps = config.get('num_diffusion_timesteps')
        trainer = Stage2Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            criterion=stage2_loss_wrapper,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config.get('num_epochs'),
            val_frequency=config.get('val_frequency'),
            patience=config.get('patience'),
            lambda_scc=config.get('lambda_scc', 0.01),
            betas=torch.linspace(0.0001, 0.02, steps=num_diffusion_steps),
            num_diffusion_timesteps=num_diffusion_steps,
            num_sampling_timesteps=config.get('num_sampling_timesteps', 50),
            gamma=config.get('gamma', 0.2)
        )
        
        # Train model
        best_model, metrics = trainer.train()
        
        # Log metrics
        mlflow.log_metric("best_val_loss", metrics['best_loss'])
        mlflow.log_metric("best_epoch", metrics['best_epoch'])
        
        # Log per-epoch validation metrics history (if available)
        if hasattr(trainer, "all_val_metrics"):
            for epoch_idx, val_metrics in enumerate(trainer.all_val_metrics):
                for metric_name, value in val_metrics.items():
                    if metric_name in higher_metrics:
                        mlflow.log_metric(f"higher_is_better/val/{metric_name}", float(value), step=epoch_idx)
                    elif metric_name in lower_metrics:
                        mlflow.log_metric(f"lower_is_better/val/{metric_name}", float(value), step=epoch_idx)
                    else:
                        mlflow.log_metric(f"val/{metric_name}", float(value), step=epoch_idx)
        
        # Evaluate on test set
        test_metrics = evaluate_stage2_metrics_avgfirst(best_model, dataloaders['test'])
        for name, value in test_metrics.items():
            if name in higher_metrics:
                mlflow.log_metric(f"higher_is_better/test/{name}", float(value))
            elif name in lower_metrics:
                mlflow.log_metric(f"lower_is_better/test/{name}", float(value))
            else:
                mlflow.log_metric(f"test/{name}", float(value))
        
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
            s3_prefix = f"experiments/{config.get('experiment_name')}/{run_name}"
            upload_file_to_s3(model_path, s3_bucket, f"{s3_prefix}/model.pth")
            upload_directory_to_s3(vis_dir, s3_bucket, f"{s3_prefix}/visualizations")
            
        # Log PyTorch model
        mlflow.pytorch.log_model(best_model, "pytorch_model")
        
        return best_model, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Stage2 experiment with MLflow tracking')
    parser.add_argument(
        '--config', 
        type=str,
        required=True,
        help='Path to experiment config file'
    )
    parser.add_argument(
        '--model-index',
        type=int,
        default=0,
        help='Index of Stage1 model to use (default: 0, which is the best model)'
    )
    args = parser.parse_args()
    
    run_experiment(args.config, args.model_index)