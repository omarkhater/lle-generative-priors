import os
import yaml
import itertools
import torch
import mlflow
import mlflow.pytorch
from frameworks.LightenDiffusion.models.stage1 import Stage1
from frameworks.LightenDiffusion.models.decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from frameworks.LightenDiffusion.training.stage1 import Stage1Trainer
from experiments.utils import setup_dataloaders
from utils.mlflow_utils import log_dict_as_params, setup_mlflow_tracking
from evaluation.lighten_diffusion_stage1 import evaluate_stage1_metrics_individual
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")


def run_single_experiment(exp_cfg: dict, sweep_id: int):
    # 1) Setup MLflow
    setup_mlflow_tracking(tracking_uri)                     
    mlflow.set_experiment(exp_cfg['experiment_name'])
    run_name = (
        f"sweep_{sweep_id}"
        f"_cont{exp_cfg['weight_cont']}"
        f"_rec{exp_cfg['weight_rec']}"
        f"_ref{exp_cfg['weight_ref']}"
        f"_ill{exp_cfg['weight_ill']}"
        f"_g{exp_cfg['lambda_g']}"
    )
    mlflow.start_run(run_name=run_name)
    log_dict_as_params(exp_cfg)

    # 2) Device + data
    device = torch.device(exp_cfg['device'])
    dataloaders = setup_dataloaders(exp_cfg)

    # 3) Model
    encoder    = ImageEncoder(**exp_cfg['encoder'])
    decoder    = ImageDecoder(**exp_cfg['decoder'])
    decomposer = RetinexDecomposition(**exp_cfg['decomposer'])
    model = Stage1(encoder, decomposer, decoder).to(device)

    # 4) Optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(exp_cfg['learning_rate']),
        weight_decay=float(exp_cfg.get('weight_decay', 0.0))
    )
    scheduler = None
    if exp_cfg.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=exp_cfg.get('scheduler_gamma', 0.8)
        )

    # 5) Trainer
    trainer = Stage1Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=exp_cfg['num_epochs'],
        val_frequency=exp_cfg['val_frequency'],
        patience=exp_cfg['patience'],
        weight_cont=exp_cfg['weight_cont'],
        weight_rec=exp_cfg['weight_rec'],
        weight_ref=exp_cfg['weight_ref'],
        weight_ill=exp_cfg['weight_ill'],
        lambda_g=exp_cfg['lambda_g'],
        after_validate=True
    )

    # 6) Train + capture metrics
    best_model, metrics = trainer.train()

    # 7) Log “best” metrics
    mlflow.log_metric('best_loss',  metrics['best_loss'])
    mlflow.log_metric('best_epoch', metrics['best_epoch'])

    # 8) Log training & validation losses
    for i, loss in enumerate(metrics['train_losses']):
        mlflow.log_metric("losses/train", float(loss), step=i)

    val_steps = [j * exp_cfg['val_frequency'] for j in range(len(metrics['val_losses']))]
    for step, vloss in zip(val_steps, metrics['val_losses']):
        mlflow.log_metric("losses/val", float(vloss), step=step)

    # 9) Log per-epoch validation metrics (psnr/ssim etc.)
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

    # 10) Evaluate & log test metrics
    test_metrics = evaluate_stage1_metrics_individual(best_model, dataloaders['test'], device)
    for name, val in test_metrics.items():
        if name in higher:
            key = f"higher_is_better/test/{name}"
        elif name in lower:
            key = f"lower_is_better/test/{name}"
        else:
            key = f"test/{name}"
        mlflow.log_metric(key, float(val))

    # 11) Save + log model artifact
    if exp_cfg.get('save_model', True):
        mdir = exp_cfg.get('model_save_dir', 'trained_models/stage1')
        os.makedirs(mdir, exist_ok=True)
        path = os.path.join(mdir, f"{exp_cfg['experiment_name']}_sweep_{sweep_id}.pth")
        torch.save(best_model.state_dict(), path)
        mlflow.log_artifact(path)

    # 12) Log full PyTorch model for easier load back
    mlflow.pytorch.log_model(best_model, "model")
    mlflow.end_run()


def main():
    with open("experiments/configs/sweep_stage1_loss_weights.yaml") as f:
        sweep_cfg = yaml.safe_load(f)

    # load base, then override only loss-weights
    base = sweep_cfg['base']
    keys, values = zip(*sweep_cfg['sweep'].items())
    for i, combo in enumerate(itertools.product(*values)):
        exp_cfg = base.copy()
        exp_cfg.update(dict(zip(keys, combo)))
        exp_cfg['encoder']    = sweep_cfg['encoder']
        exp_cfg['decoder']    = sweep_cfg['decoder']
        exp_cfg['decomposer'] = sweep_cfg['decomposer']
        run_single_experiment(exp_cfg, i)

if __name__ == "__main__":
    main()