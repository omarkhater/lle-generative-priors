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
2. Create fresh python environment.
```bash
python -m venv ./.venv 
```
3. Activate the environemnt. 

On Windows 

```bash
./.venv/Scripts/activate
```

For linux:

```bash
source ./.venv/bin/activate
```
4. Install the package and dependencies:
```bash
pip install -r requirements.in
pip install -e .
```

### MLFlow setup
    - We created Amazon RDS database (postgresql), s3 bucket to store results.
    - We deployed Lightsail EC2 instance to host our MLFlow server for easier access from different devices.

## Datasets

The dataset used in this project is published [here](https://huggingface.co/datasets/okhater/SICE_paired). 

## Key Components

- **Data set Registery**:
    - Enable Easy loading to the data from a HuggingFace repo to a Pytorch data loaders. 

- **Histogram Equalization Based approach**:
    - Under `frameworks/ClassicEnhancement`

- **Two-Stage Architecture**:
  - Stage 1: Retinex decomposition and image enhancement (`frameworks\LightenDiffusion\models\stage1.py`)
  - Stage 2: Diffusion-based refinement (`frameworks\LightenDiffusion\models\LightenDiffusion.py`)

- **experiment**:
    * YAML Based Configuration files for each experiment categoery.
    * Python scripts to run each experiment.

## Key EDA Scripts:
- `eda\evaluate_stage1.ipynb`: Early results on stage 1 training.
- `eda\evaluate_stage2.ipynb`: Early results on stage 2 training.
- `eda\Stage1_Training copy.ipynb`: Early training script for stage 1.
- `eda\Stage2_Training copy.ipynb`: Early training script for stage 2.
- `eda\reproduce_results.ipynb`: Evaluate paper checkpoint on our dataloaders. Note that you might need to fork original paper codes and get their checkpoint from their [repo](https://github.com/JianghaiSCU/LightenDiffusion/tree/main).

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