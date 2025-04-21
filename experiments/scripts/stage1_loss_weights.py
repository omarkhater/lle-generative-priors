import os
import yaml
import torch
import mlflow
from frameworks.LightenDiffusion.models.stage1 import Stage1
from frameworks.LightenDiffusion.models.decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from frameworks.LightenDiffusion.training.stage1 import Stage1Trainer
from frameworks.LightenDiffusion.training.utils import get_optimizer, get_scheduler
from experiments.utils.general_utils import setup_dataloaders
from utils.mlflow_utils import log_dict_as_params, setup_mlflow_tracking, log_hardware_info, log_git_info
from dotenv import load_dotenv, find_dotenv
from evaluation.lighten_diffusion_stage1 import evaluate_stage1_metrics_individual
from frameworks.LightenDiffusion.visualization.visualize_stage1 import visualize_stage1_results
import optuna
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv(find_dotenv())
tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")


def run_single_experiment(
        exp_cfg: dict,
        trial_params: dict,
        trial_number: int,
        objective_metric: str = 'psnr',
        ):
    current_exp_cfg = exp_cfg.copy()
    current_exp_cfg.update(trial_params)
    log_dict_as_params(trial_params)
    final_metric_value = None
    logging.info(f"Starting Trial {trial_number} - Run ID: {mlflow.active_run().info.run_id}")
    log_dict_as_params(exp_cfg)
    mlflow.log_param("trial_number", trial_number)
    base_params_to_log = {k: v for k, v in current_exp_cfg.items() if k not in trial_params}
    log_dict_as_params(base_params_to_log)
    device = torch.device(current_exp_cfg['device'])
    dataloaders = setup_dataloaders(current_exp_cfg)
    encoder    = ImageEncoder(**current_exp_cfg['encoder'])
    decoder    = ImageDecoder(**current_exp_cfg['decoder'])
    decomposer = RetinexDecomposition(**current_exp_cfg['decomposer'])
    model = Stage1(encoder, decomposer, decoder).to(device)
    optimizer = get_optimizer(
                            model,
                            lr = float(current_exp_cfg['learning_rate']),
                            weight_decay = float(current_exp_cfg.get('weight_decay'))
                            )
    scheduler = None
    if current_exp_cfg.get('use_scheduler', False):
        scheduler = get_scheduler(optimizer, gamma=current_exp_cfg.get('scheduler_gamma'))

    save_dir = os.path.join(
            current_exp_cfg.get('local_save_directory', 'experiments/artifacts'),
            current_exp_cfg['experiment_name'],
            mlflow.active_run().info.run_id
        ) 
    os.makedirs(save_dir, exist_ok=True)
    trainer = Stage1Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        optimizer=optimizer,
        device=device,
        val_frequency=current_exp_cfg['val_frequency'],
        patience=current_exp_cfg['patience'],
        num_epochs=current_exp_cfg['num_epochs'],
        scheduler=scheduler,
        weight_rec=current_exp_cfg['weight_rec'],
        weight_ref=current_exp_cfg['weight_ref'],
        weight_ill=current_exp_cfg['weight_ill'],
        lambda_g=current_exp_cfg['lambda_g'],
        num_visualizations=2,
        show_plot=False,
        save_dir=save_dir,
        pretrain_content_ratio=current_exp_cfg.get('pretrain_content_ratio'),
        pretrain_ctdn_ratio=current_exp_cfg.get('pretrain_ctdn_ratio'),
    )

    _, metrics = trainer.train()
    best_epoch = metrics.get('best_epoch', -1)

    if trainer.all_val_metrics and best_epoch != -1 and best_epoch < len(trainer.all_val_metrics):
        epoch_metrics = trainer.all_val_metrics[best_epoch]
        final_metric_value = epoch_metrics.get(objective_metric)

        if final_metric_value is not None:
            mlflow.log_metric(f"final_{objective_metric}", final_metric_value)
            logging.info(f"Trial {trial_number} final {objective_metric}: {final_metric_value}")
        else:
            logging.warning(f"Trial {trial_number}: Objective metric '{objective_metric}' not found in validation metrics for best epoch {best_epoch}.")
            final_metric_value = float('-inf') if objective_metric in Stage1Trainer.higher_is_better else float('inf')
    else:
        logging.warning(f"Trial {trial_number}: Could not retrieve final objective metric value. Metrics dict: {epoch_metrics}, all_val_metrics empty or best_epoch invalid.")
        final_metric_value = float('-inf') if objective_metric in Stage1Trainer.higher_is_better else float('inf')

    if os.path.exists(save_dir) and any(fname.endswith('.png') for fname in os.listdir(save_dir)):
            mlflow.log_artifacts(save_dir, artifact_path="visualizations")

    if final_metric_value is None:
        logging.error(f"Trial {trial_number}: final_metric_value was None before returning. Assigning default poor value.")
        final_metric_value = float('-inf') if objective_metric in Stage1Trainer.higher_is_better else float('inf')

    test_loader = dataloaders['test']

    test_metrics = evaluate_stage1_metrics_individual(
            model, 
            dataloaders['test'], 
            device
        )
    trainer._log_evaluation_metrics(test_metrics, prefix ="test")
    visualize_stage1_results(
            model, 
            test_loader, 
            num_samples=4,
            save_dir=f"{save_dir}/test_visualizations"
        )

    return float(final_metric_value)


def objective(trial, base_cfg, sweep_cfg, model_cfgs, objective_metric):
    """
    Objective function for Optuna. Suggests parameters and calls run_single_experiment.
    """
    
    exp_cfg = base_cfg.copy()
    exp_cfg.update(model_cfgs)
    parameters_list = ["weight_rec", "weight_ref", "weight_ill", "lambda_g"]
    trial_params = {}
    for param_name in parameters_list:
        param_config = sweep_cfg.get(param_name)
        start = param_config["start"]
        end = param_config["end"]
        scale = param_config.get("scale", "linear")
        
        if scale == "log":
            trial_params[param_name] = trial.suggest_float(param_name, start, end, log=True)
        else: 
            trial_params[param_name] = trial.suggest_float(param_name, start, end)
    
    run_name = f"trial_{trial.number}"
    with mlflow.start_run(run_name=run_name, nested=True):
        final_metric = run_single_experiment(
            exp_cfg=exp_cfg,
            trial_params=trial_params,
            trial_number=trial.number,
            objective_metric=objective_metric
        )
    return final_metric


def get_run_name_for_optuna():
    run_name = "default_run_name"
    return run_name

def run_optuna_sweep(full_cfg, n_trials, objective_metric):
    """
    Runs the Optuna sweep for hyperparameter optimization.
    """
    setup_mlflow_tracking(tracking_uri)
    mlflow.set_experiment(full_cfg['base']['experiment_name'])
    base_cfg = full_cfg.get("base")
    sweep_cfg = full_cfg.get("sweep")
    model_cfgs = {k: v for k, v in full_cfg.items() if k in ['encoder', 'decoder', 'decomposer']}
    run_name = base_cfg.get("run_name", get_run_name_for_optuna())
    description = base_cfg.get("description", "Optuna sweep for Stage 1 loss weights")
    with mlflow.start_run(run_name= run_name):
        parent_run_id = mlflow.active_run().info.run_id
        logging.info(f"Starting Optuna sweep parent run with ID: {parent_run_id}")
        log_dict_as_params(base_cfg)
        log_dict_as_params({"optuna_n_trials": n_trials, "optuna_objective_metric": objective_metric})
        log_dict_as_params({f"sweep_{k}": str(v) for k, v in sweep_cfg.items()})
        log_hardware_info()
        log_git_info()
        mlflow.log_artifact("experiments/configs/sweep_stage1_loss_weights.yaml", artifact_path="config")
        direction = "minimize"
        if objective_metric in Stage1Trainer.higher_is_better:
            direction = "maximize"
        
        study = optuna.create_study(
            study_name=base_cfg.get("experiment_name", "stage1_loss_weights_sweep"),
            direction=direction
        )
        study.optimize(
            lambda trial: objective(trial, base_cfg, sweep_cfg, model_cfgs, objective_metric),
            n_trials=n_trials,
            n_jobs=1
        )
        logging.info("Optuna sweep finished.")
        if study.best_trial:
            logging.info(f"Best trial: {study.best_trial.number}")
            logging.info(f"Best value ({objective_metric}): {study.best_value}")
            logging.info(f"Best params: {study.best_trial.params}")
            mlflow.log_metric(f"best_trial_{objective_metric}", study.best_value)
            mlflow.log_param("best_trial_number", study.best_trial.number)
            log_dict_as_params({f"best_{k}": v for k, v in study.best_trial.params.items()})
            try:
                fig = optuna.visualization.plot_optimization_history(study)
                mlflow.log_figure(fig, "optuna_optimization_history.png")
                fig = optuna.visualization.plot_param_importances(study)
                mlflow.log_figure(fig, "optuna_param_importances.png")
            except Exception as e:
                logging.warning(f"Could not log Optuna visualizations: {e}")
        else:
            logging.warning("Optuna study finished without finding a best trial.")
    return study.best_trial if study.best_trial else None



def main():

    with open("experiments/configs/sweep_stage1_loss_weights.yaml") as f:
        full_cfg = yaml.safe_load(f)

    base_cfg = full_cfg.get("base")
    objective_metric = base_cfg.get("objective_metric", "psnr")
    n_trials = base_cfg.get("n_trials", 10)
    best_trial = run_optuna_sweep(full_cfg, n_trials, objective_metric)

    if best_trial:
        logging.info(f"Sweep complete. Best trial #{best_trial.number} achieved {objective_metric}: {best_trial.value} with params: {best_trial.params}")
    else:
        logging.info("Sweep complete, but no successful trials were recorded.")

if __name__ == "__main__":
    main()