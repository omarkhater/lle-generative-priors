import os
import sys
import argparse
import torch
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from frameworks.LightenDiffusion.models.LightenDiffusion import Stage1
from frameworks.LightenDiffusion.models.decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from frameworks.LightenDiffusion.training.stage1 import Stage1Trainer
from evaluation.lighten_diffusion_stage1 import evaluate_stage1_metrics_individual
from evaluation.utils import aggregate_metrics
from frameworks.LightenDiffusion.visualization.visualize_stage1 import visualize_stage1_results
from eda.helpers.training_helpers import get_optimizer, get_scheduler
from utils.mlflow_utils import setup_mlflow_tracking, create_experiment_group, log_dict_as_params
from experiments.utils.general_utils import ExperimentConfig, setup_dataloaders
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")


def visualize_and_save_samples(model, dataloader, output_dir, num_samples=3):
    """
    Visualize and save model outputs for MLflow artifacts.
    
    Args:
        model: The model to visualize
        dataloader: Dataloader containing samples
        output_dir: Directory to save output images
        num_samples: Number of samples to visualize
        
    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Temporarily disable interactive mode
    plt_interactive = plt.isinteractive()
    plt.ioff()
    visualize_stage1_results(model, dataloader, num_samples=num_samples)
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig.savefig(os.path.join(output_dir, f'sample_{i}.png'))
        plt.close(fig)
    
    # Restore interactive mode if it was on
    if plt_interactive:
        plt.ion()
        
    return output_dir

def run_experiment(config_path):
    """Run Stage1 experiment with MLflow tracking."""
    config = ExperimentConfig(config_path)
    setup_mlflow_tracking(tracking_uri=tracking_uri)
    create_experiment_group(
        config.get('experiment_name'), 
        description=config.get('description', '')
    )
    mlflow.set_experiment(config.get('experiment_name'))
    dataloaders = setup_dataloaders(config)
    with mlflow.start_run(run_name=config.get('run_name')):
        log_dict_as_params(config.to_dict())
        mlflow.log_param('train_size', len(dataloaders['train'].dataset))
        mlflow.log_param('val_size', len(dataloaders['val'].dataset))
        mlflow.log_param('test_size', len(dataloaders['test'].dataset))
        stage1_model = Stage1(
            encoder=ImageEncoder(config.get('latent_space_dim')),
            decoder=ImageDecoder(config.get('latent_space_dim')),
            decomposer=RetinexDecomposition(
                channels=config.get('latent_space_dim'),
                num_cross_attention_heads=config.get('num_cross_attn_heads'),
                num_self_attention_heads=config.get('num_self_attn_heads')
            ),
        )
        device = torch.device(config.get('device'))
        stage1_model.to(device)
        optimizer = get_optimizer(
            stage1_model,
            lr=float(config.get('learning_rate')),
            weight_decay=float(config.get('weight_decay'))
        )
        
        scheduler = None
        if config.get('use_scheduler', False):
            scheduler = get_scheduler(
                optimizer, 
                gamma=config.get('scheduler_gamma', 0.8)
            )

        trainer = Stage1Trainer(
            model=stage1_model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            num_epochs=config.get('num_epochs'),
            val_frequency=config.get('val_frequency'),
            patience=config.get('patience'),
            weight_cont=config.get('weight_cont'),
            weight_rec=config.get('weight_rec'),
            weight_ref=config.get('weight_ref'),
            weight_ill=config.get('weight_ill'),
            lambda_g=config.get('lambda_g'),
            after_validate=True
        )
        best_model, metrics = trainer.train()
        mlflow.log_metric('best_loss', metrics['best_loss'])
        mlflow.log_metric('best_epoch', metrics['best_epoch'])
        
        # Log training losses under "losses/train"
        for i, loss in enumerate(metrics['train_losses']):
            mlflow.log_metric("losses/train", loss, step=i)
        
        # Log validation losses under "losses/val"
        for i, val_idx in enumerate([j * config.get('val_frequency') for j in range(len(metrics['val_losses']))]):
            mlflow.log_metric("losses/val", metrics['val_losses'][i], step=val_idx)
        
        # Log the validation metrics history computed in after_validation (if available)
        higher_metrics = {"psnr", "ssim"}
        lower_metrics = {"tv_illumination", "pi", "niqe", "lpips"}
        if hasattr(trainer, "all_val_metrics"):
            for idx, val_metrics in enumerate(trainer.all_val_metrics):
                for metric_name, value in val_metrics.items():
                    if metric_name in higher_metrics:
                        mlflow.log_metric(f"higher_is_better/val/{metric_name}", value, step=idx)
                    elif metric_name in lower_metrics:
                        mlflow.log_metric(f"lower_is_better/val/{metric_name}", value, step=idx)
                    else:
                        mlflow.log_metric(f"val/{metric_name}", value, step=idx)
        
        test_metrics = evaluate_stage1_metrics_individual(
            best_model, 
            dataloaders['test'], 
            device
        )
        
        for metric_name, value in test_metrics.items():
            if metric_name in higher_metrics:
                mlflow.log_metric(f"higher_is_better/test/{metric_name}", value)
            elif metric_name in lower_metrics:
                mlflow.log_metric(f"lower_is_better/test/{metric_name}", value)
            else:
                # For any metric that does not fall in the above groups, you may log it directly
                mlflow.log_metric(f"test/{metric_name}", value)
        
        if config.get('save_model', True):
            model_dir = config.get('model_save_dir', 'trained_models/stage1')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(
                model_dir, 
                f"{config.get('experiment_name')}_{config.get('run_name')}.pth"
            )
            torch.save(best_model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
        
        mlflow.pytorch.log_model(best_model, "model")
        if config.get('log_artifacts', True):
            vis_dir = os.path.join(
                'outputs/visualizations', 
                config.get('experiment_name'),
                config.get('run_name')
            )
            visualize_and_save_samples(
                best_model, 
                dataloaders['test'],
                vis_dir, 
                num_samples=config.get('num_vis_samples', 3)
            )
            mlflow.log_artifacts(vis_dir, "visualizations")
        
        return best_model, metrics, test_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Stage1 experiment with MLflow tracking')
    parser.add_argument(
        '--config', 
        type=str,
        required=True, 
        help='Path to experiment config file'
    )
    args = parser.parse_args()
    
    run_experiment(args.config)