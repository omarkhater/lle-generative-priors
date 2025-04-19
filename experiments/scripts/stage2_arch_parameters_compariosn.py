"""
Stage 2 parameter sweep experiment.

This script runs a parameter sweep for Stage 2 model architectures
using the top performing Stage 1 models from a previous experiment.
"""

import os
import argparse
import logging
import torch
import mlflow
import itertools
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import numpy as np
from experiments.base import ExperimentBase
from frameworks.LightenDiffusion.models.unet import DiffusionUNet
from frameworks.LightenDiffusion.training.stage2 import Stage2Trainer
from evaluation.lighten_diffusion_stage2 import evaluate_stage2_metrics_avgfirst
from frameworks.LightenDiffusion.visualization.visualize_stage2 import visualize_stage2_results
from experiments.utils.general_utils import setup_dataloaders as default_setup_dataloaders
from experiments.utils.model_selection import get_top_models_by_metric, download_model_from_run
from experiments.utils.model_loading import load_trained_model
from frameworks.LightenDiffusion.models.decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from frameworks.LightenDiffusion.models.stage1 import Stage1
from frameworks.LightenDiffusion.models.LightenDiffusion import Stage2, LightenDiffusionPipeline
from eda.helpers.training_helpers import get_optimizer, get_scheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Stage2ParameterSweepExperiment(ExperimentBase):
    """
    Experiment for sweeping Stage 2 architecture parameters using top Stage 1 models.
    
    This experiment:
    1. Identifies the top-performing Stage 1 models based on given validation metric.
    2. Creates parameter combinations for the DiffusionUNet architecture
    3. Trains multiple Stage 2 models with different configurations
    4. Distributes training across available GPUs
    """
        
    def setup_dataloaders(self) -> Dict[str, Any]:
        """Sets up data loaders using the default setup function."""
        logging.info(f"Setting up data loaders with config: {self.config.config}")
        return default_setup_dataloaders(self.config)
    
    def get_tracking_uri(self) -> str:
        """
        Returns the MLflow tracking URI.
        
        Returns:
            str: MLflow tracking URI
        """
        if os.environ.get("MLFLOW_SERVER"):
            return os.environ["MLFLOW_SERVER"]
        else:
            logger.error("MLFLOW_SERVER environment variable not set")
        return None
    def find_top_stage1_models(self) -> List[Dict]:
        """
        Find top-performing Stage 1 models from the source experiment using MLflow.
        
        Returns:
            List of dictionaries containing model information and paths
        """
        source_experiment = self.config.get('source_experiment')
        metric = self.config.get('model_selection', {}).get('metric')
        top_k = self.config.get('model_selection', {}).get('top_k', 3)
        mlflow_tracking_uri = self.get_tracking_uri()
        higher_is_better = self.config.get('model_selection', {}).get('higher_is_better', True)

        top_mlflow_models = get_top_models_by_metric(
                experiment_name=source_experiment,
                metric_name=metric.split('/')[-1],  # Extract the base metric name
                top_k=top_k,
                mlflow_tracking_uri=mlflow_tracking_uri,
                higher_is_better=higher_is_better
            )
        
        if not top_mlflow_models:
            logger.error(f"No top models found for experiment '{source_experiment}' with metric '{metric}'")
            return []
        
        model_results = []
        for model_info in top_mlflow_models:
            model_path = download_model_from_run(
                run_id=model_info['run_id'],
                mlflow_tracking_uri=mlflow_tracking_uri
            )
            
            checkpoint = torch.load(model_path, map_location='cpu')
            encoded_channels = checkpoint.get('encoded_channels',
                                            checkpoint.get('model_config', {}).get('encoded_channels', 64))
            
            model_results.append({
                'run_id': model_info['run_id'],
                'model_path': model_path,
                'metric_value': model_info['metric_value'],
                'encoded_channels': encoded_channels
            })
            logger.info(f"Downloaded model {model_info['run_id']} with metric {model_info['metric_value']}")

        return model_results
    
    def _load_stage1_model(
            self, 
            info: dict) -> Stage1:
        latent_space_dim = info.get('encoded_channels')
        stage1_model = Stage1(
            encoder=ImageEncoder(latent_space_dim),
            decoder=ImageDecoder(latent_space_dim),
            decomposer=RetinexDecomposition(),
        )
        stage1_model = load_trained_model(
            info.get('model_path'), 
            stage1_model,
            device=self.device
            )
        return stage1_model 
    
    def _init_stage2_model(self,diffusion_config: Dict) -> None:
        """
        
        Initialize the Stage 2 model with the given diffusion configuration.

        Args:
            diffusion_config: Dictionary containing diffusion model configuration
        Returns:
            Stage2 model instance        
        """

        if type(diffusion_config.get("ch_mult")) != tuple:
            try:
                diffusion_config["ch_mult"] = tuple(diffusion_config.get("ch_mult"))
            except Exception as e:
                logger.error(f"Error converting ch_mult to tuple: {e}")
                raise ValueError("ch_mult must be a tuple of integers.")

        config_keys = [
            "in_channels",
            "out_channels",
            "ch",
            "ch_mult",
            "num_res_blocks",
            "dropout",
            "conditional",
            "resamp_with_conv"
        ]
        config_diffusion_verified = {k: diffusion_config[k] for k in config_keys if k in diffusion_config}
        diffusion_net = DiffusionUNet(
            **config_diffusion_verified)
        stage2_model = Stage2(diffusion_unet=diffusion_net)
        
        return stage2_model
    
    def _init_pipeline_mode(self, stage1_model_info, diffusion_params) -> None:
        """
        Initialize the pipeline mode for the experiment.
        """
        stage1_model = self._load_stage1_model(stage1_model_info)
        diffusion_params.update({
            "in_channels": stage1_model.encoder.base_channels * 2,
            "out_channels": stage1_model.encoder.base_channels,
            "ch": stage1_model.encoder.base_channels
        }
        )
        stage2_model = self._init_stage2_model(diffusion_params)
        
        model = LightenDiffusionPipeline(
            stage1 = stage1_model,
            stage2 = stage2_model,
        )
        return model

    def generate_parameter_combinations(self) -> List[Dict]:
        """
        Generate all parameter combinations for the DiffusionUNet.
        
        Returns:
            List of dictionaries containing parameter sets
        """
        params = self.config.get('diffusion_unet_params', {})
        
        keys = list(params.keys())
        values = list(params.values())
        
        all_combinations = list(itertools.product(*values))
        
        param_dicts = []
        for combo in all_combinations:
            param_dict = {keys[i]: combo[i] for i in range(len(keys))}
            param_dicts.append(param_dict)
        
        # Limit number of experiments if specified
        max_exps = self.config.get('max_experiments', -1)
        if max_exps > 0 and max_exps < len(param_dicts):
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(param_dicts), max_exps, replace=False)
            param_dicts = [param_dicts[i] for i in indices]
        
        logger.info(f"Generated {len(param_dicts)} parameter combinations")
        return param_dicts
    
    def build_model(self, stage1_model_info: Dict, diffusion_params: Dict) -> DiffusionUNet:
        """
        Builds a DiffusionUNet model with the given parameters.
        
        Args:
            stage1_model_info: Dictionary containing Stage 1 model information
            diffusion_params: Dictionary containing DiffusionUNet parameters
            
        Returns:
            Configured DiffusionUNet model
        """
        model = self._init_pipeline_mode(
            stage1_model_info, 
            diffusion_params
        )
        return model
    
    def build_trainer(self, 
                      model: LightenDiffusionPipeline,
                      dataloaders: Dict[str, Any],
                      diffusion_params: Dict,
        ) -> Stage2Trainer:
        """
        Builds the Stage2Trainer for the current parameter set.
        
        Args:
            model: LightenDiffusionPipeline model
            dataloaders: Dictionary containing train, val, and test dataloaders
            diffusion_params: Dictionary containing DiffusionUNet parameters
            
        Returns:
            Configured Stage2Trainer instance
        """
        loss_params = self.config.get('stage2_loss_params', {})
        training_params = self.config.get('training', {})
        optimizer = get_optimizer(
            model,
            lr=float(training_params.get('learning_rate')),
            weight_decay=float(training_params.get('weight_decay'))
        )
        
        if training_params.get("use_scheduler") == True:
            scheduler = get_scheduler(
                optimizer, 
                gamma=training_params.get('scheduler_gamma')
            )
        else:
            scheduler = None
        num_diffusion_timesteps=diffusion_params.get('num_diffusion_timesteps')
        trainer = Stage2Trainer(
            model = model, 
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            num_epochs=training_params.get('epochs'),
            val_frequency=training_params.get('val_frequency'),
            patience=training_params.get('patience'),
            lambda_scc=loss_params.get('lambda_scc', 0.001),
            betas=torch.linspace(0.0001, 0.02, steps=num_diffusion_timesteps),
            num_diffusion_timesteps=num_diffusion_timesteps,
            num_sampling_timesteps=diffusion_params.get('num_sampling_timesteps'),
            gamma=loss_params.get('gamma'),
            save_visualization_dir=self.vis_dir,
        )
        return trainer
    
    def train(self, trainer: Stage2Trainer) -> Tuple[DiffusionUNet, Dict[str, Any]]:
        """
        Executes the training process for one variation.
        
        Args:
            trainer: Configured Stage2Trainer instance
        
        Returns:
            A tuple containing the best trained model and training metrics
        """
        best_model, metrics = trainer.train()
        return best_model, metrics
    
    def evaluate(
            self, 
            model: LightenDiffusionPipeline, 
            dataloader: torch.utils.data.DataLoader,
        ) -> Dict[str, float]:
        """
        Evaluates the Stage2 model.
        
        Args:
            model: Trained LightenDiffusionPipeline model
            dataloader: DataLoader for the test set
        Returns:
            Dictionary of evaluation metrics
        """
        if dataloader is None:
            logger.warning("Test dataloader is None, skipping evaluation.")
            return {}
        
        logger.info("Evaluating model on test set...")
        return evaluate_stage2_metrics_avgfirst(model, dataloader, self.device)
    
    def visualize(
            self, 
            model: LightenDiffusionPipeline,
            dataloader: torch.utils.data.DataLoader,
        ) -> None:
        """
        Generates and saves visualizations for the current model.
        
        Args:
            model: Trained LightenDiffusionPipeline model
            dataloader: DataLoader to use for visualization

        Returns:
            None
        """
        if dataloader:
            num_samples = self.config.get('num_visualizations', 4)
            logger.info(f"Generating {num_samples} visualizations using validation set...")
            
            try:
                visualize_stage2_results(
                    pipeline=model,
                    dataloader=dataloader,
                    num_samples=num_samples,
                    save_dir=self.vis_dir,
                )
                logger.info(f"Visualizations saved to {self.vis_dir}")
            except Exception as e:
                logger.error(f"Error during visualization: {e}", exc_info=True)
        else:
            logger.warning("Skipping visualization: Validation dataloader not available.")
    
    def _generate_variation_name(self, stage1_model: Dict, diffusion_params: Dict) -> str:
        """
        Generate a unique name for the current variation.
        
        Args:
            stage1_model: Dictionary containing Stage 1 model information
            diffusion_params: Dictionary containing DiffusionUNet parameters
            
        Returns:
            Unique variation identifier string
        """
        s1_id = stage1_model['run_id'].split('_')[-1]
        ch_mult_str = "-".join(str(x) for x in diffusion_params['ch_mult'])
        
        return (f"s1_{s1_id}_"
                f"res{diffusion_params['num_res_blocks']}_"
                f"ch{ch_mult_str}_"
                f"drop{diffusion_params['dropout']}_"
                f"resconv{int(diffusion_params['resamp_with_conv'])}_"
                f"t{diffusion_params['num_diffusion_timesteps']}")
    

    def _pick_variations(self, variations: List[Tuple], max_variations: int) -> List[Tuple]:
        """
        Pick a subset of variations based on the max_variations parameter.
        
        Args:
            variations: List of all variations
            max_variations: Maximum number of variations to select
            
        Returns:
            Subset of variations
        """
        if max_variations > 0 and max_variations < len(variations):
            np.random.seed(42)
            indices = np.random.choice(len(variations), max_variations, replace=False)
            variations = [variations[i] for i in indices]
        return variations
    
    def run(self) -> None:
        """
        Overrides the base `run` method to handle Stage 2 parameter variations.

        Iterates through top Stage 1 models and diffusion parameter combinations,
        setting up and executing a separate MLflow run for each.
        """
        self._setup_tracking()
        dataloaders = self.setup_dataloaders()
        top_models = self.find_top_stage1_models()
        if not top_models:
            logger.error("No Stage 1 models found. Exiting.")
            return

        param_combinations = self.generate_parameter_combinations()
        if not param_combinations:
            logger.error("No parameter combinations generated. Exiting.")
            return
        
        variations = []
        for model in top_models:
            for params in param_combinations:
                variations.append((model, params))
        
        variations = self._pick_variations(variations, self.config.get('max_variations', -1))
        logger.info(f"Created {len(variations)} experiment variations")
        original_run_name_base = self.config.get('run_name', self.config.get('experiment_name'))
        use_parallel = self.config.get('gpu', {}).get('parallel', False)
        device_ids = self.config.get('gpu', {}).get('device_ids', "auto")
        
        if use_parallel and device_ids == "auto":
            device_ids = list(range(torch.cuda.device_count()))
        elif use_parallel and isinstance(device_ids, list):
            device_ids = device_ids
        else:
            device_ids = [self.device_id]  # Use the default device

        for i, (stage1_model, diffusion_params) in enumerate(tqdm(variations, desc="Processing variations")):
            if use_parallel and len(device_ids) > 1:
                current_device_id = device_ids[i % len(device_ids)]
                self.device = torch.device(f"cuda:{current_device_id}" if torch.cuda.is_available() else "cpu")

            variation_name = self._generate_variation_name(stage1_model, diffusion_params)
            current_run_name = f"{original_run_name_base}_{variation_name}"
            original_instance_run_name = self.run_name
            original_output_dir = self.output_dir
            original_model_dir = self.model_dir
            original_vis_dir = self.vis_dir
            self.run_name = current_run_name
            self.output_dir = os.path.join("outputs", self.config.get('experiment_name'), self.run_name)
            self.model_dir = os.path.join(self.output_dir, "models")
            self.vis_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)
            try:
                with mlflow.start_run(run_name=self.run_name):
                    logger.info(f"Starting MLflow run for variation: {variation_name}")
                    self._log_params(dataloaders)
                    mlflow.log_params({f"s1_model_{k}": v for k, v in stage1_model.items() 
                                      if k != 'model_path'})
                    mlflow.log_params({f"diff_{k}": v for k, v in diffusion_params.items()})
                    mlflow.log_artifact(self.config.config_path)
                    model = self.build_model(stage1_model, diffusion_params)
                    mlflow.log_param("total_parameters", sum(p.numel() for p in model.parameters()))
                    trainer = self.build_trainer(
                        model=model,
                        diffusion_params=diffusion_params,
                        dataloaders=dataloaders
                    )
                    logger.info(f"Training variation: {variation_name}...")
                    best_model, train_metrics = self.train(trainer)
                    logger.info(f"Training finished for variation: {variation_name}.")
                    
                    # Log training metrics
                    if 'best_loss' in train_metrics:
                        mlflow.log_metric('best_val_loss', train_metrics['best_loss'])
                    if 'best_epoch' in train_metrics:
                        mlflow.log_metric('best_epoch', train_metrics['best_epoch'])
                    if 'train_losses' in train_metrics:
                        for i, loss in enumerate(train_metrics['train_losses']):
                            mlflow.log_metric("losses/train", loss, step=i)
                    if 'val_losses' in train_metrics:
                        val_freq = self.config.get('training', {}).get('val_frequency', 1)
                        for i, loss in enumerate(train_metrics['val_losses']):
                            mlflow.log_metric("losses/val", loss, step=i * val_freq)
                    
                    if hasattr(trainer, "all_val_metrics"):
                        self._log_epoch_metrics(trainer.all_val_metrics, prefix="val")
                    
                    test_metrics = self.evaluate(
                        model=best_model,
                        dataloader=dataloaders.get('test'),
                    )
                    
                    if test_metrics:
                        self._log_final_metrics(test_metrics, prefix="test")
                        logger.info(f"Test metrics for {variation_name}: {test_metrics}")
                    model_path = self._save_model(best_model)
                    if model_path:
                        self._log_model_artifact(model_path)
                        self._log_pytorch_model(best_model)

                    self.visualize(
                        model=best_model,
                        dataloaders=dataloaders.get('test'),
                    )
                    self._log_visualization_artifacts()
                    
                    if self.config.get("upload_to_s3", False):
                        self._upload_to_s3()
                    
                    logger.info(f"Finished MLflow run for variation: {variation_name}")
                
            except Exception as e:
                logger.error(f"Error processing variation {variation_name}: {e}", exc_info=True)
                # If a run was started, mark it as failed
                if mlflow.active_run():
                    mlflow.set_tag("status", "FAILED")
                    mlflow.log_param("error", str(e))
                    mlflow.end_run(status="FAILED")
            finally:
                # Restore original instance variables
                self.run_name = original_instance_run_name
                self.output_dir = original_output_dir
                self.model_dir = original_model_dir
                self.vis_dir = original_vis_dir
                # Ensure any active run is ended
                if mlflow.active_run():
                    mlflow.end_run()
        
        logger.info("All variations processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage2 Parameter Sweep Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/stage2_arch_parameters_sweep.yaml",
        help="Path to the experiment configuration YAML file."
    )
    args = parser.parse_args()
    experiment = Stage2ParameterSweepExperiment(config_path=args.config)
    experiment.run()