import os
import argparse
import logging
import mlflow
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from experiments.base import ExperimentBase
from experiments.utils.general_utils import setup_dataloaders as default_setup_dataloaders
from frameworks.LightenDiffusion.models.decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from frameworks.LightenDiffusion.models.stage1 import Stage1
from frameworks.LightenDiffusion.training.stage1 import Stage1Trainer
from frameworks.LightenDiffusion.visualization.visualize_stage1 import visualize_stage1_results
from evaluation.lighten_diffusion_stage1 import evaluate_stage1_metrics_individual
from eda.helpers.training_helpers import get_optimizer, get_scheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Stage1ArchComparisonExperiment(ExperimentBase):
    """
    Experiment class for comparing different Stage1 architectures.

    This class overrides the standard `run` method to iterate through
    architecture variations defined in the configuration file. Each variation
    is executed as a separate, top-level MLflow run.
    """

    def setup_dataloaders(self) -> Dict[str, Any]:
        """Sets up data loaders using the default setup function."""

        return default_setup_dataloaders(self.config)

    # The following methods are designed to work for a SINGLE variation
    # They will be called within the loop in the overridden `run` method.

    def build_model(self, variation_config: Dict[str, Any]) -> Stage1:
        """
        Builds the Stage1 model for a specific architecture variation.

        Args:
            variation_config: Dictionary containing architecture parameters for this variation.

        Returns:
            An instance of the Stage1 model.
        """
        encoder = ImageEncoder(
            base_channels=variation_config['base_channels'],
            channel_factors=tuple(variation_config['channel_factors']), # Ensure tuple
            in_channels=3, # Assuming 3 input channels (RGB)
            encoded_channels=variation_config['encoded_channels']
        )

        decomposer = RetinexDecomposition(
            channels=variation_config['encoded_channels'],
            num_cross_attention_heads=variation_config['num_cross_attention_heads'],
            num_self_attention_heads=variation_config['num_self_attention_heads']
        )

        decoder = ImageDecoder(
            base_channels=variation_config['base_channels'],
            channel_factors=tuple(variation_config['channel_factors']), # Ensure tuple
            out_channels=3, # Assuming 3 output channels (RGB)
            encoded_channels=variation_config['encoded_channels']
        )

        model = Stage1(encoder, decomposer, decoder)
        model.to(self.device)
        return model

    def build_trainer(self, model: Stage1, dataloaders: Dict[str, Any], variation_config: Dict[str, Any]) -> Stage1Trainer:
        """
        Builds the Stage1Trainer for a specific architecture variation.

        Args:
            model: The Stage1 model instance for the current variation.
            dataloaders: Dictionary of data loaders.
            variation_config: Dictionary containing architecture parameters (not used here but kept for consistency).

        Returns:
            An instance of Stage1Trainer configured with base parameters.
        """
        optimizer = get_optimizer(
            model,
            lr=float(self.config.get('learning_rate')), # Use base learning rate
            weight_decay=float(self.config.get('weight_decay', 0.0))
        )

        scheduler = None
        if self.config.get('use_scheduler', False):
             scheduler = get_scheduler(
                 optimizer, gamma=self.config.get('scheduler_gamma', 0.8)
             )

        trainer = Stage1Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler,
            num_epochs=self.config.get('num_epochs'),
            val_frequency=self.config.get('val_frequency'),
            patience=self.config.get('patience'),
            weight_cont=self.config.get('weight_cont'),
            weight_rec=self.config.get('weight_rec'),
            weight_ref=self.config.get('weight_ref'),
            weight_ill=self.config.get('weight_ill'),
            lambda_g=self.config.get('lambda_g'),
            after_validate=True
        )
        return trainer

    def train(self, trainer: Stage1Trainer) -> Tuple[Stage1, Dict[str, Any]]:
        """
        Executes the training process for one variation.

        Args:
            trainer: The configured Stage1Trainer instance.

        Returns:
            A tuple containing the best trained model and training metrics.
        """
        best_model, metrics = trainer.train()
        return best_model, metrics

    def evaluate(self, model: Stage1, dataloader: Any) -> Dict[str, float]:
        """
        Evaluates the Stage1 model for one variation.

        Args:
            model: The trained Stage1 model for the current variation.
            dataloader: The dataloader for evaluation (typically the test set).

        Returns:
            A dictionary of evaluation metrics.
        """
        if dataloader is None:
            logger.warning("Test dataloader is None, skipping evaluation.")
            return {}
        logger.info(f"Evaluating model on test set...")
        return evaluate_stage1_metrics_individual(model, dataloader, self.device)

    def visualize(self, model: Stage1, dataloaders: Dict[str, Any], variation_config: Dict[str, Any]) -> None:
        """
        Generates and saves visualizations for one variation.

        Args:
            model: The trained Stage1 model for the current variation.
            dataloaders: Dictionary of data loaders.
            variation_config: Dictionary containing architecture parameters (used for logging).
        """
        vis_loader = dataloaders.get('val')
        if vis_loader:
            num_samples = self.config.get('num_visualizations', 4)
            logger.info(f"Generating {num_samples} visualizations using validation set and saving to {self.vis_dir}...")
            plt_interactive = plt.isinteractive()
            if plt_interactive:
                plt.ioff()

            try:
                visualize_stage1_results(
                    model,
                    vis_loader,
                    num_samples=num_samples,
                    show_plot=False, # Ensure plots are not displayed interactively
                    device=self.device,
                    save_dir=self.vis_dir # Pass the directory for saving
                )
                logger.info(f"Visualizations saved to {self.vis_dir}")

            except Exception as e:
                 logger.error(f"Error during visualization for variation {variation_config.get('name', 'unknown')}: {e}", exc_info=True)
            finally:
                # Restore interactive mode if it was on
                if plt_interactive:
                    plt.ion()
        else:
            logger.warning("Skipping visualization: Validation dataloader not available.")

    def run(self) -> None:
        """
        Overrides the base `run` method to handle architecture variations.

        Iterates through each variation defined in the config, setting up
        and executing a separate, top-level MLflow run for each.
        """
        self._setup_tracking() 
        dataloaders = self.setup_dataloaders()

        variations = self.config.get("variations", [])
        if not variations:
            logger.error("No variations found in the configuration file. Exiting.")
            return

        logger.info(f"Found {len(variations)} architecture variations to process.")

        # Store original run name and directories from base config for reference
        original_run_name_base = self.config.get('run_name', self.config.get('experiment_name'))

        for variation_config in tqdm(variations, desc="Processing Variations"):
            variation_name = variation_config.get("name", "unnamed_variation")

            # --- Per-Variation Setup ---
            # 1. Define the run name for this specific variation
            current_run_name = f"{original_run_name_base}_{variation_name}"

            # 2. Temporarily update instance variables related to the run name and output paths
            #    This ensures base class methods (_save_model, _save_visualizations, etc.)
            #    use the correct paths for the current variation's run.
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
            # --- End Per-Variation Setup ---

            # Start a *new*, *top-level* MLflow run for this variation
            try:
                with mlflow.start_run(run_name=self.run_name): # Use the updated self.run_name
                    logger.info(f"Starting MLflow run for variation: {variation_name} (Run Name: {self.run_name})")
                    self._log_params(dataloaders)
                    mlflow.log_params({f"var_{k}": v for k, v in variation_config.items()})
                    mlflow.log_artifact(self.config.config_path) 
                    model = self.build_model(variation_config)
                    mlflow.log_param("total_parameters", sum(p.numel() for p in model.parameters()))
                    trainer = self.build_trainer(model, dataloaders, variation_config)
                    logger.info(f"Training variation: {variation_name}...")
                    best_model, train_metrics = self.train(trainer)
                    logger.info(f"Training finished for variation: {variation_name}.")
                    if 'best_loss' in train_metrics:
                        mlflow.log_metric('best_val_loss', train_metrics['best_loss'])
                    if 'best_epoch' in train_metrics:
                        mlflow.log_metric('best_epoch', train_metrics['best_epoch'])
                    if 'train_losses' in train_metrics:
                         for i, loss in enumerate(train_metrics['train_losses']):
                             mlflow.log_metric("losses/train", loss, step=i)
                    if 'val_losses' in train_metrics:
                         val_freq = self.config.get('val_frequency', 1)
                         for i, loss in enumerate(train_metrics['val_losses']):
                             mlflow.log_metric("losses/val", loss, step=i * val_freq)
                    if hasattr(trainer, "all_val_metrics"):
                        self._log_epoch_metrics(trainer.all_val_metrics, prefix="val")

                    test_metrics = self.evaluate(best_model, dataloaders.get('test'))
                    if test_metrics: # Only log if evaluation happened
                        self._log_final_metrics(test_metrics, prefix="test")
                        logger.info(f"Test metrics for {variation_name}: {test_metrics}")

                    # Save and Log Model (uses updated self.model_dir)
                    model_path = self._save_model(best_model) # Saves to self.model_dir / "model.pth"
                    if model_path:
                        self._log_model_artifact(model_path) # Logs model.pth as 'model' artifact
                        self._log_pytorch_model(best_model) # Logs as 'pytorch_model'

                    # Visualization (uses updated self.vis_dir)
                    self.visualize(best_model, dataloaders, variation_config)
                    self._log_visualization_artifacts() # Logs content of self.vis_dir as 'visualizations'

                    # Optional S3 Upload (uses updated self.run_name, self.model_dir, self.vis_dir)
                    if self.config.get("upload_to_s3", False): # Check config flag
                         self._upload_to_s3()

                    logger.info(f"Finished MLflow run for variation: {variation_name}")

            except Exception as e:
                logger.error(f"Error processing variation {variation_name}: {e}", exc_info=True)
                # If a run was started, mark it as failed
                if mlflow.active_run():
                    mlflow.set_tag("status", "FAILED")
                    mlflow.log_param("error", str(e))
                    mlflow.end_run(status="FAILED") # Explicitly end failed run
            finally:
                 # Restore original instance variables for the next iteration (or end)
                 self.run_name = original_instance_run_name
                 self.output_dir = original_output_dir
                 self.model_dir = original_model_dir
                 self.vis_dir = original_vis_dir
                 # Ensure any active run is ended if an error occurred before the 'with' block finished
                 if mlflow.active_run():
                     mlflow.end_run()


        logger.info("All variations processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage1 Architecture Comparison Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/stage1_architecture_parameters_comparison.yaml",
        help="Path to the experiment configuration YAML file."
    )
    args = parser.parse_args()
    experiment = Stage1ArchComparisonExperiment(config_path=args.config)
    experiment.run()