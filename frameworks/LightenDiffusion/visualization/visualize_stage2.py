import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from .visualize_stage1 import tensor_to_image, get_visualization_batch

def _get_stage2_data(pipeline: torch.nn.Module, data_loader: DataLoader, is_paired: bool, seed: int = 42) -> Tuple[dict, Optional[torch.Tensor]]:
    torch.manual_seed(seed)
    pipeline.eval()
    device = next(pipeline.parameters()).device
    sample_batch, gt_batch = get_visualization_batch(data_loader, is_paired, device)
    with torch.no_grad():
        outputs = pipeline(sample_batch, gt_batch) if gt_batch is not None else pipeline(sample_batch)
        stage2_out = outputs.get("stage2", None)
        if stage2_out is None:
            raise ValueError("Stage2 outputs not found in pipeline results.")
    return stage2_out, sample_batch, gt_batch

def visualize_stage2_results_aggregate(pipeline: torch.nn.Module,
                                       data_loader: DataLoader,
                                       num_samples: int = 8,
                                       is_paired: bool = True,
                                       seed: int = 42) -> None:
    """
    Visualize aggregated outputs from the Stage2 diffusion process by averaging 
    over a batch of samples. The following outputs are aggregated and then visualized:
      - x0, x_t, noise, noise_pred, noise difference, reference feature,
        R_low, L_high, and Ground Truth High (if available).
    All latent outputs are mapped to RGB.

    Args:
        pipeline (torch.nn.Module): The full LightenDiffusion pipeline containing Stage2 and map_to_rgb.
        data_loader (DataLoader): DataLoader yielding paired (x, y) batches or unpaired.
        num_samples (int): (Unused here but kept for interface consistency) Number of samples for visualization.
        is_paired (bool): Whether ground truth high images are provided.
        seed (int): Random seed for reproducibility.
    """
    stage2_out, _, gt_batch= _get_stage2_data(pipeline, data_loader, is_paired, seed)

    # Extract outputs.
    x0 = stage2_out["x0"]
    x_t = stage2_out["x_t"]
    noise = stage2_out["noise"]
    noise_pred = stage2_out["noise_pred"]
    reference_feature = stage2_out["reference_feature"]
    R_low = stage2_out["R_low"]
    L_high = stage2_out["L_high"]

    # Aggregate by averaging over the batch dimension.
    x0_avg = torch.mean(x0, dim=0, keepdim=True)
    x_t_avg = torch.mean(x_t, dim=0, keepdim=True)
    noise_avg = torch.mean(noise, dim=0, keepdim=True)
    noise_pred_avg = torch.mean(noise_pred, dim=0, keepdim=True)
    noise_diff_avg = torch.mean(torch.abs(noise - noise_pred), dim=0, keepdim=True)
    reference_avg = torch.mean(reference_feature, dim=0, keepdim=True)
    R_low_avg = torch.mean(R_low, dim=0, keepdim=True)
    L_high_avg = torch.mean(L_high, dim=0, keepdim=True)
    gt_avg = torch.mean(gt_batch, dim=0, keepdim=True) if (is_paired and gt_batch is not None) else None

    # Map aggregated outputs to RGB.
    x0_rgb = pipeline.map_to_rgb(x0_avg)
    x_t_rgb = pipeline.map_to_rgb(x_t_avg)
    noise_rgb = pipeline.map_to_rgb(noise_avg)
    noise_pred_rgb = pipeline.map_to_rgb(noise_pred_avg)
    noise_diff_rgb = pipeline.map_to_rgb(noise_diff_avg)
    reference_rgb = pipeline.map_to_rgb(reference_avg)
    R_low_rgb = pipeline.map_to_rgb(R_low_avg)
    L_high_rgb = pipeline.map_to_rgb(L_high_avg)
    if gt_avg is not None:
        # Assuming ground truth high images are already in RGB space.
        gt_rgb = gt_avg

    num_cols = 9 if is_paired else 8
    fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3))
    col = 0
    axes[col].imshow(tensor_to_image(x0_rgb[0]))
    axes[col].set_title("x0 (Composite)")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(tensor_to_image(x_t_rgb[0]))
    axes[col].set_title("x_t (Noised)")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(tensor_to_image(noise_rgb[0]))
    axes[col].set_title("Noise")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(tensor_to_image(noise_pred_rgb[0]))
    axes[col].set_title("Noise Pred")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(tensor_to_image(noise_diff_rgb[0]))
    axes[col].set_title("Noise Diff")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(tensor_to_image(reference_rgb[0]))
    axes[col].set_title("Reference")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(tensor_to_image(R_low_rgb[0]))
    axes[col].set_title("R_low")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(tensor_to_image(L_high_rgb[0]))
    axes[col].set_title("L_high")
    axes[col].axis("off")
    col += 1

    if is_paired:
        axes[col].imshow(tensor_to_image(gt_rgb[0]))
        axes[col].set_title("GT High")
        axes[col].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_stage2_results(pipeline: torch.nn.Module,
                             data_loader: DataLoader,
                             num_samples: int = 8,
                             is_paired: bool = True,
                             approach: str = "aggregate") -> None:
    """
    Top-level visualization function for Stage2 diffusion outputs.
    Supports two approaches:

        - 'aggregate': Visualizes the average (aggregated) output over a batch.
    
    Args:
        pipeline (torch.nn.Module): The full LightenDiffusion pipeline (with Stage2 and map_to_rgb).
        data_loader (DataLoader): DataLoader yielding (low, high) batches in paired mode or (low,) otherwise.
        num_samples (int): Number of samples (rows) to visualize for the individual mode.
        is_paired (bool): Whether ground truth high images are provided.
        approach (str): Either "individual" or "aggregate" visualization mode.
    """

    if approach.lower() == "aggregate":
        visualize_stage2_results_aggregate(pipeline, data_loader, num_samples, is_paired)
    else:
        raise ValueError("Invalid approach. Use 'aggregate' for aggregated visualization.")