# LLE Generative Priors

A PyTorch-based implementation of Low-Light Enhancement using Generative Priors with diffusion models.

## Project Structure

```
├── dataset_registery/      # Dataset handling and loading
├── docs/                   # Documentation and reports
├── eda/                    # Exploratory Data Analysis
├── evaluation/            # Model evaluation scripts
├── experiments/           # Experiment configurations and runners
├── frameworks/            # Core model implementations
│   └── LightenDiffusion/
│       ├── models/       # Model architectures
│       └── training/     # Training loops and utilities
└── utils/                # Utility functions
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install the package and dependencies:
```bash
pip install -r requirements.in
pip install -e .
```

## Key Components

- **Two-Stage Architecture**:
  - Stage 1: Retinex decomposition and image enhancement
  - Stage 2: Diffusion-based refinement

- **experiment**:
    * YAML Based Configuration files for each experiment categoery.
    * Python scripts to run each experiment.

## Training

The project uses MLflow for experiment tracking. Training can be launched using the experiment scripts in `experiments/scripts/`.

Example:
```bash
python experiments/scripts/stage2_loss_weights.py --config experiments/configs/sweep_stage2_loss_weights.yaml
```

## Configuration

- Model configurations are stored in `experiments/configs/`
- Environment variables should be set in `.env`

Please ensure the following keys are present: 

```bash
PYTHONPATH=.
AWS_ACCESS_KEY_ID=<YOUR AWS KEY ID>
AWS_SECRET_ACCESS_KEY=<YOUR AWS SECRETE ACCESS KEY>
AWS_REGION=<YOUR AWS REGION>
AWS_S3_BUCKET=<YOUR AWS BUCKET>
MLFLOW_BACKEND_STORE_URI=<YOUR POSTGRE Database URL>
MLFLOW_ARTIFACT_STORE_URI=<S3 BUCKET URL>
MLFLOW_SERVER=<YOUR MLFLOW SERVER>
```
- MLflow tracking URI can be configured through environment variables

## Model Artifacts

Models and checkpoints are saved to:
- Local: `trained_models/` directory
- MLflow: Tracked under the configured MLflow server

## Development

- Python 3.x
- PyTorch
- MLflow for experiment tracking
- Docker support available