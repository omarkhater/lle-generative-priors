import os
import yaml
import itertools
import torch
from frameworks.LightenDiffusion.models.stage1 import Stage1
from frameworks.LightenDiffusion.models.decom import ImageEncoder, ImageDecoder, RetinexDecomposition
from frameworks.LightenDiffusion.training.stage1 import Stage1Trainer
from experiments.utils import setup_dataloaders
from utils.mlflow_utils import log_dict_as_params
import mlflow

def run_single_experiment(exp_cfg, sweep_id):
    # Setup dataloaders
    dataloaders = setup_dataloaders(exp_cfg)
    # Build model
    encoder = ImageEncoder(**exp_cfg['encoder'])
    decoder = ImageDecoder(**exp_cfg['decoder'])
    decomposer = RetinexDecomposition(**exp_cfg['decomposer'])
    model = Stage1(encoder, decomposer, decoder)
    model.to(exp_cfg['device'])
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=exp_cfg['learning_rate'])
    # Trainer
    trainer = Stage1Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        optimizer=optimizer,
        device=exp_cfg['device'],
        num_epochs=exp_cfg['num_epochs'],
        weight_cont=exp_cfg['weight_cont'],
        weight_rec=exp_cfg['weight_rec'],
        weight_ref=exp_cfg['weight_ref'],
        weight_ill=exp_cfg['weight_ill'],
        lambda_g=exp_cfg['lambda_g'],
    )
    # MLflow logging
    mlflow.start_run(run_name=f"sweep_{sweep_id}")
    log_dict_as_params(exp_cfg)
    best_model, metrics = trainer.train()
    for k, v in metrics.items():
        mlflow.log_metric(k, v if isinstance(v, float) else float(v[-1]))
    mlflow.end_run()

def main():
    with open("experiments/configs/sweep_stage1_loss_weights.yaml") as f:
        sweep_cfg = yaml.safe_load(f)
    # Generate all combinations
    keys, values = zip(*sweep_cfg['sweep'].items())
    for i, combo in enumerate(itertools.product(*values)):
        exp_cfg = sweep_cfg['base'].copy()
        exp_cfg.update(dict(zip(keys, combo)))
        exp_cfg['encoder'] = sweep_cfg['encoder']
        exp_cfg['decoder'] = sweep_cfg['decoder']
        exp_cfg['decomposer'] = sweep_cfg['decomposer']
        exp_cfg['device'] = sweep_cfg['device']
        exp_cfg['learning_rate'] = sweep_cfg['learning_rate']
        exp_cfg['num_epochs'] = sweep_cfg['num_epochs']
        run_single_experiment(exp_cfg, i)

if __name__ == "__main__":
    main()