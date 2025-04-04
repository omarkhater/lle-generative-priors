import argparse
import os
import yaml

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config(mode="training"):
    """
    Parse command line arguments and configuration file.
    
    Args:
        mode: Either "training" or "evaluation" to determine specific arguments
        
    Returns:
        args: Command line arguments
        config: Configuration namespace
    """
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models')
    parser.add_argument("--config", default='unsupervised.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--mode', type=str, default=mode, help='training or evaluation')
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored images")
    
    # Add mode-specific arguments
    if mode == "training":
        parser.add_argument('--seed', default=230, type=int, metavar='N',
                            help='Seed for initializing training (default: 230)')
    elif mode == "evaluation":
        parser.add_argument("--paired", action="store_true", 
                            help="Set if the dataset is paired (supervised)")
        # Override default resume path for evaluation
        parser.set_defaults(resume='ckpt/stage2/stage2_weight.pth.tar')
    
    args = parser.parse_args()

    # Load config from YAML file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config
