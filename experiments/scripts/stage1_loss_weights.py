import os
import yaml
import itertools
import torch
import mlflow
from frameworks.LightenDiffusion.models.stage1 import Stage1
from frameworks.LightenDiffusion.models.decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from frameworks.LightenDiffusion.training.stage1 import Stage1Trainer
from experiments.utils.general_utils import setup_dataloaders
from utils.mlflow_utils import log_dict_as_params, setup_mlflow_tracking
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
tracking_uri = os.environ.get("MLFLOW_SERVER", "file:./mlruns")


def run_single_experiment(exp_cfg: dict, sweep_id: int):
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
    device = torch.device(exp_cfg['device'])
    dataloaders = setup_dataloaders(exp_cfg)
    encoder    = ImageEncoder(**exp_cfg['encoder'])
    decoder    = ImageDecoder(**exp_cfg['decoder'])
    decomposer = RetinexDecomposition(**exp_cfg['decomposer'])
    model = Stage1(encoder, decomposer, decoder).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(exp_cfg['learning_rate']),
        weight_decay=float(exp_cfg.get('weight_decay'))
    )
    scheduler = None
    if exp_cfg.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=exp_cfg.get('scheduler_gamma')
        )
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
        show_plot=False,
        save_dir=f"{exp_cfg.get('local_save_directory')}/{exp_cfg['experiment_name']}/{run_name}",
        pretrain_content_ratio=exp_cfg.get('pretrain_content_ratio'),
        pretrain_ctdn_ratio=exp_cfg.get('pretrain_ctdn_ratio'),
    )

    trainer.train()
    mlflow.end_run()


def main():
    with open("experiments/configs/sweep_stage1_loss_weights.yaml") as f:
        sweep_cfg = yaml.safe_load(f)
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