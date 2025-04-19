import os
import yaml
import torch
import mlflow
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List, Union
from dotenv import load_dotenv, find_dotenv
from utils.mlflow_utils import setup_mlflow_tracking, create_experiment_group, log_dict_as_params
from experiments.utils.general_utils import ExperimentConfig
load_dotenv(find_dotenv())

class ExperimentBase(ABC):
    """
    Abstract base class for running machine learning experiments with MLflow tracking.

    Handles common tasks like configuration loading, MLflow setup, device management,
    parameter logging, metric logging, artifact logging, and model saving.

    Subclasses must implement abstract methods for experiment-specific logic like
    data loading, model building, training, evaluation, and visualization.
    """
    def __init__(self, config_path: str):
        """
        Initializes the experiment base.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")
        self.s3_bucket = os.environ.get("S3_BUCKET")
        self.device = self._setup_device()
        self.run_name = self._generate_run_name()
        self.output_dir = os.path.join("outputs", self.config.get('experiment_name'), self.run_name)
        self.model_dir = os.path.join(self.output_dir, "models")
        self.vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # Define metric categories (can be overridden by subclasses if needed)
        self.higher_metrics = {"psnr", "ssim"}
        self.lower_metrics = {"tv_illumination", "pi", "niqe", "lpips"}


    def _load_config(self, config_path: str) -> ExperimentConfig:
        """Loads the experiment configuration from a YAML file."""
        return ExperimentConfig(config_path)

    def _setup_device(self) -> torch.device:
        """Sets up the computation device (CPU or CUDA)."""
        device_name = self.config.get('device', 'cuda')
        device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        return device

    def _setup_tracking(self) -> None:
        """Sets up MLflow tracking and experiment."""
        setup_mlflow_tracking(tracking_uri=self.tracking_uri)
        create_experiment_group(
            self.config.get('experiment_name'),
            description=self.config.get('description', '')
        )
        mlflow.set_experiment(self.config.get('experiment_name'))

    def _generate_run_name(self) -> str:
        """Generates a default run name. Can be overridden by subclasses."""
        return self.config.get('run_name', f"{self.config.get('experiment_name')}_run")

    def _log_params(self, dataloaders: Dict[str, Any]) -> None:
        """Logs configuration parameters and dataset sizes to MLflow."""
        log_dict_as_params(self.config.to_dict())
        if 'train' in dataloaders and hasattr(dataloaders['train'], 'dataset'):
            mlflow.log_param('train_size', len(dataloaders['train'].dataset))
        if 'val' in dataloaders and hasattr(dataloaders['val'], 'dataset'):
            mlflow.log_param('val_size', len(dataloaders['val'].dataset))
        if 'test' in dataloaders and hasattr(dataloaders['test'], 'dataset'):
            mlflow.log_param('test_size', len(dataloaders['test'].dataset))

    def _log_metrics_categorized(self, metrics: Dict[str, float], prefix: str, step: Optional[int] = None) -> None:
        """
        Logs metrics to MLflow, categorizing them based on predefined sets.

        Args:
            metrics: Dictionary of metric names to values.
            prefix: Prefix for the metric key (e.g., 'val', 'test').
            step: Optional step number for logging time-series metrics.
        """
        for name, value in metrics.items():
            if name in self.higher_metrics:
                key = f"higher_is_better/{prefix}/{name}"
            elif name in self.lower_metrics:
                key = f"lower_is_better/{prefix}/{name}"
            else:
                key = f"{prefix}/{name}"

            mlflow.log_metric(key, float(value), step=step)

    def _log_epoch_metrics(self, all_metrics: List[Dict[str, float]], prefix: str) -> None:
        """Logs metrics recorded per epoch during training/validation."""
        for epoch_idx, epoch_metrics in enumerate(all_metrics):
            self._log_metrics_categorized(epoch_metrics, prefix, step=epoch_idx)

    def _log_final_metrics(self, metrics: Dict[str, float], prefix: str) -> None:
        """Logs final evaluation metrics."""
        self._log_metrics_categorized(metrics, prefix)

    def _save_model(self, model: Any, model_name: str = "model.pth") -> str:
        """
        Saves the model state dictionary locally.

        Args:
            model: The model to save.
            model_name: The filename for the saved model.

        Returns:
            The path to the saved model file.
        """
        model_path = os.path.join(self.model_dir, model_name)
        if hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            print(f"Warning: Model type {type(model)} may not be serializable with torch.save.")
            # Potentially add other saving mechanisms (e.g., pickle) if needed
        return model_path

    def _log_model_artifact(self, model_path: str, artifact_path: str = "model") -> None:
        """Logs the saved model file as an MLflow artifact."""
        mlflow.log_artifact(model_path, artifact_path=artifact_path)

    def _log_pytorch_model(self, model: Any, artifact_path: str = "pytorch_model") -> None:
        """Logs the model using mlflow.pytorch.log_model."""
        if hasattr(model, 'state_dict'): # Basic check if it's a PyTorch model
             mlflow.pytorch.log_model(model, artifact_path)
        else:
            print(f"Skipping mlflow.pytorch.log_model for non-PyTorch model type {type(model)}")


    def _save_visualizations(self, figs: Union[plt.Figure, List[plt.Figure]], filenames: Union[str, List[str]]) -> List[str]:
        """
        Saves matplotlib figures locally.

        Args:
            figs: A single figure or a list of figures.
            filenames: A single filename or a list of filenames corresponding to the figures.

        Returns:
            A list of paths to the saved visualization files.
        """
        if not isinstance(figs, list):
            figs = [figs]
        if not isinstance(filenames, list):
            filenames = [filenames]

        saved_paths = []
        for fig, filename in zip(figs, filenames):
            if fig is None:
                print(f"Warning: Skipping None figure for filename {filename}")
                continue
            path = os.path.join(self.vis_dir, filename)
            try:
                fig.savefig(path)
                plt.close(fig) # Close figure to free memory
                saved_paths.append(path)
                print(f"Visualization saved to {path}")
            except Exception as e:
                print(f"Error saving visualization {filename}: {e}")
        return saved_paths

    def _log_visualization_artifacts(self, artifact_path: str = "visualizations") -> None:
        """Logs the entire visualization directory as MLflow artifacts."""
        if os.path.exists(self.vis_dir) and os.listdir(self.vis_dir):
            mlflow.log_artifacts(self.vis_dir, artifact_path=artifact_path)
        else:
            print("Visualization directory is empty or does not exist. Skipping artifact logging.")

    def _upload_to_s3(self) -> None:
        """Uploads model and visualization artifacts to S3 if configured."""
        if self.s3_bucket:
            from experiments.utils.s3_utils import upload_directory_to_s3 # Lazy import
            s3_prefix = f"experiments/{self.config.get('experiment_name')}/{self.run_name}"
            print(f"Uploading artifacts to s3://{self.s3_bucket}/{s3_prefix}")
            if os.path.exists(self.model_dir) and os.listdir(self.model_dir):
                upload_directory_to_s3(self.model_dir, self.s3_bucket, f"{s3_prefix}/models")
            if os.path.exists(self.vis_dir) and os.listdir(self.vis_dir):
                upload_directory_to_s3(self.vis_dir, self.s3_bucket, f"{s3_prefix}/visualizations")

    @abstractmethod
    def setup_dataloaders(self) -> Dict[str, Any]:
        """
        Sets up and returns data loaders for train, validation, and test sets.

        Returns:
            A dictionary containing 'train', 'val', and 'test' data loaders.
        """
        pass

    @abstractmethod
    def build_model(self) -> Any:
        """
        Builds and returns the model(s) or pipeline for the experiment.

        Returns:
            The constructed model or pipeline.
        """
        pass

    @abstractmethod
    def build_trainer(self, model: Any, dataloaders: Dict[str, Any]) -> Any:
        """
        Builds and returns the trainer instance responsible for the training loop.

        Args:
            model: The model to be trained.
            dataloaders: The data loaders.

        Returns:
            The trainer instance.
        """
        pass

    @abstractmethod
    def train(self, trainer: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes the training process using the provided trainer.

        Args:
            trainer: The trainer instance.

        Returns:
            A tuple containing:
                - The best trained model.
                - A dictionary of training metrics (e.g., best loss, epoch history).
        """
        pass

    @abstractmethod
    def evaluate(self, model: Any, dataloader: Any) -> Dict[str, float]:
        """
        Evaluates the model on the given dataloader.

        Args:
            model: The model to evaluate.
            dataloader: The dataloader for evaluation (e.g., test set).

        Returns:
            A dictionary of evaluation metrics.
        """
        pass

    @abstractmethod
    def visualize(self, model: Any, dataloaders: Dict[str, Any]) -> None:
        """
        Generates and saves visualizations of the model's performance or outputs.
        This method should call _save_visualizations internally.
        """
        pass

    def run(self) -> None:
        """
        Runs the complete experiment lifecycle: setup, train, evaluate, log, save.
        """
        self._setup_tracking()

        with mlflow.start_run(run_name=self.run_name):
            print(f"Starting MLflow run: {self.run_name}")

            # Setup
            dataloaders = self.setup_dataloaders()
            self._log_params(dataloaders) # Log params early
            model = self.build_model()

            # Training (if applicable)
            trainer = self.build_trainer(model, dataloaders)
            if trainer:
                print("Starting training...")
                best_model, train_metrics = self.train(trainer)
                print("Training finished.")

                # Log training-related metrics
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

                # Log detailed validation metrics per epoch if available from trainer
                if hasattr(trainer, "all_val_metrics"):
                    self._log_epoch_metrics(trainer.all_val_metrics, prefix="val")

            else:
                print("Skipping training phase.")
                best_model = model # Use the initial model if no training occurs
                train_metrics = {}

            # Evaluation
            if 'test' in dataloaders and dataloaders['test'] is not None:
                print("Starting evaluation on test set...")
                test_metrics = self.evaluate(best_model, dataloaders['test'])
                self._log_final_metrics(test_metrics, prefix="test")
                print(f"Test metrics: {test_metrics}")
            else:
                print("Skipping evaluation on test set (no test dataloader found).")
                test_metrics = {}

            # Save and Log Model
            model_path = self._save_model(best_model)
            self._log_model_artifact(model_path)
            self._log_pytorch_model(best_model) # Log PyTorch specific format

            # Visualization
            print("Generating visualizations...")
            self.visualize(best_model, dataloaders)
            self._log_visualization_artifacts()

            # Optional S3 Upload
            self._upload_to_s3()

            print(f"Experiment run {self.run_name} completed successfully.")
            return best_model, train_metrics, test_metrics