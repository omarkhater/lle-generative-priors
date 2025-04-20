import torch
import matplotlib.pyplot as plt
import matplotlib  # added to inspect backend
from torch.utils.data import DataLoader
from .visualization_utils import map_to_rgb, tensor_to_image, get_visualization_batch
from typing import Optional, Tuple, List, Dict, Any
import os
import mlflow

def extract_low_quality_images(sample_batch: torch.Tensor) -> torch.Tensor:
    """
    Returns the low-quality images.
    If there are exactly 2 sub-images, returns both.
    Otherwise, returns only the first.
    """
    B, m, _, _, _ = sample_batch.shape
    if m == 2:
        return sample_batch  # shape: [B, 2, 3, H, W]
    else:
        return sample_batch[:, 0:1, :, :, :]  # shape: [B, 1, 3, H, W]

def extract_encoded_features(outputs: List[Dict[str, Any]], m: int) -> torch.Tensor:
    """
    Extracts and stacks encoded features from the Stage1 outputs.
    If there are exactly 2 outputs, returns both; otherwise, returns only the first.
    """
    if m == 2:
        return torch.stack([out["f"] for out in outputs], dim=1)  # shape: [B, 2, C, H_f, W_f]
    else:
        return torch.stack([outputs[0]["f"]], dim=1)  # shape: [B, 1, C, H_f, W_f]

def extract_decomposed_components(outputs: List[Dict[str, Any]], m: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts reflectance (R) and illumination (L) components.
    If exactly 2 sub-images are available, returns them individually;
    otherwise, returns only the first.
    """
    if m == 2:
        R_all = torch.stack([out["R"] for out in outputs], dim=1)  # shape: [B, 2, C, H_r, W_r]
        L_all = torch.stack([out["L"] for out in outputs], dim=1)  # shape: [B, 2, C, H_r, W_r]
        return R_all, L_all
    else:
        R_all = torch.stack([outputs[0]["R"]], dim=1)  # shape: [B, 1, C, H_r, W_r]
        L_all = torch.stack([outputs[0]["L"]], dim=1)  # shape: [B, 1, C, H_r, W_r]
        return R_all.squeeze(1), L_all.squeeze(1)

def extract_reconstruction(outputs: List[Dict[str, Any]], m: int) -> torch.Tensor:
    """
    Extracts reconstruction images from outputs.
    If exactly 2 sub-images exist, returns both reconstructions;
    otherwise, returns the first.
    """
    if m == 2:
        recon_all = torch.stack([out["recon"] for out in outputs], dim=1)  # shape: [B, 2, 3, H, W]
        return recon_all
    else:
        return outputs[0]["recon"]

def compute_rl_product(R: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Computes elementwise product R * L.
    Works for both individual outputs (if m != 2) and for a stack (if m == 2).
    """
    return R * L

def plot_stage1_figures(low_quality: torch.Tensor, encoded_feats: torch.Tensor,
                        R: torch.Tensor, L: torch.Tensor, RL_product: torch.Tensor,
                        reconstruction: torch.Tensor,
                        gt_batch: Optional[torch.Tensor],
                        num_samples: int, m: int, is_paired: bool) -> List[plt.Figure]:
    """
    Plots the Stage1 outputs.
    
    If m == 2, it creates one separate figure per sample with two rows—each row showing:
      - Low-quality image,
      - Encoded feature,
      - Reflectance,
      - Illumination,
      - R x L product,
      - Reconstruction,
      - Ground truth (if available).
    
    The figure title indicates the sample number.
    
    Otherwise, for m != 2, it creates a single grid figure with one row per sample.
    """
    figs = []
    cols_single = 1 + 1 + 1 + 1 + 1 + 1 + (1 if is_paired else 0)  # = 6 or 7
    
    if m == 2:
        # For each sample, create a separate figure with 2 rows.
        for i in range(min(num_samples, low_quality.shape[0])):
            fig, axes = plt.subplots(2, cols_single, figsize=(3*cols_single, 6))
            fig.suptitle(f"Sample {i}", fontsize=16)
            for row in range(2):
                col = 0
                # Low Quality image
                axes[row, col].imshow(tensor_to_image(low_quality[i, row]))
                axes[row, col].set_title(f"Low Q {row+1}")
                axes[row, col].axis("off")
                col += 1
                # Encoded Feature
                feat = encoded_feats[i, row]
                feat_disp = feat if feat.shape[0] == 3 else map_to_rgb(feat.unsqueeze(0)).squeeze(0)
                axes[row, col].imshow(tensor_to_image(feat_disp))
                axes[row, col].set_title(f"Encoded {row+1}")
                axes[row, col].axis("off")
                col += 1
                # Reflectance
                R_disp = R[i, row] if R[i, row].shape[0] == 3 else map_to_rgb(R[i, row].unsqueeze(0)).squeeze(0)
                axes[row, col].imshow(tensor_to_image(R_disp))
                axes[row, col].set_title(f"Reflectance {row+1}")
                axes[row, col].axis("off")
                col += 1
                # Illumination
                L_disp = L[i, row] if L[i, row].shape[0] == 3 else map_to_rgb(L[i, row].unsqueeze(0)).squeeze(0)
                axes[row, col].imshow(tensor_to_image(L_disp))
                axes[row, col].set_title(f"Illumination {row+1}")
                axes[row, col].axis("off")
                col += 1
                # R x L Product
                rl_disp = RL_product[i, row] if RL_product[i, row].shape[0] == 3 else map_to_rgb(RL_product[i, row].unsqueeze(0)).squeeze(0)
                axes[row, col].imshow(tensor_to_image(rl_disp))
                axes[row, col].set_title(f"R x L {row+1}")
                axes[row, col].axis("off")
                col += 1
                # Reconstruction
                axes[row, col].imshow(tensor_to_image(reconstruction[i, row]))
                axes[row, col].set_title(f"Reconstruction {row+1}")
                axes[row, col].axis("off")
                col += 1
                # Ground Truth (optional); for simplicity, we display it on each row
                if is_paired and gt_batch is not None:
                    axes[row, col].imshow(tensor_to_image(gt_batch[i]))
                    axes[row, col].set_title("Ground Truth")
                    axes[row, col].axis("off")
            plt.tight_layout()
            figs.append(fig)
    else:
        # For m != 2, show one row per sample in a single grid.
        num_rows = min(num_samples, low_quality.shape[0])
        fig, axes = plt.subplots(num_rows, cols_single, figsize=(3*cols_single, 3*num_rows))
        if num_rows == 1:
            axes = axes[None, :]
        for i in range(num_rows):
            col = 0
            axes[i, col].imshow(tensor_to_image(low_quality[i, 0]))
            axes[i, col].set_title("Low Quality")
            axes[i, col].axis("off")
            col += 1
            feat = encoded_feats[i, 0]
            feat_disp = feat if feat.shape[0] == 3 else map_to_rgb(feat.unsqueeze(0)).squeeze(0)
            axes[i, col].imshow(tensor_to_image(feat_disp))
            axes[i, col].set_title("Encoded")
            axes[i, col].axis("off")
            col += 1
            R_disp = R[i] if R[i].shape[0] == 3 else map_to_rgb(R[i].unsqueeze(0)).squeeze(0)
            axes[i, col].imshow(tensor_to_image(R_disp))
            axes[i, col].set_title("Reflectance")
            axes[i, col].axis("off")
            col += 1
            L_disp = L[i] if L[i].shape[0] == 3 else map_to_rgb(L[i].unsqueeze(0)).squeeze(0)
            axes[i, col].imshow(tensor_to_image(L_disp))
            axes[i, col].set_title("Illumination")
            axes[i, col].axis("off")
            col += 1
            rl_disp = RL_product[i] if RL_product[i].shape[0] == 3 else map_to_rgb(RL_product[i].unsqueeze(0)).squeeze(0)
            axes[i, col].imshow(tensor_to_image(rl_disp))
            axes[i, col].set_title("R x L")
            axes[i, col].axis("off")
            col += 1
            axes[i, col].imshow(tensor_to_image(reconstruction[i]))
            axes[i, col].set_title("Reconstruction")
            axes[i, col].axis("off")
            col += 1
            if is_paired and gt_batch is not None:
                axes[i, col].imshow(tensor_to_image(gt_batch[i]))
                axes[i, col].set_title("Ground Truth")
                axes[i, col].axis("off")
        plt.tight_layout()
        figs.append(fig)
    return figs

def visualize_stage1_results(
        stage1: torch.nn.Module,
        data_loader: DataLoader,
        num_samples: int = 8,
        is_paired: bool = True,
        seed: int = 42,
        save_dir: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
    """
    Top-level function to visualize Stage1 outputs.
    
    - Obtains one batch from the DataLoader.
    - Runs the Stage1 model.
    - Extracts low-quality images, encoded features, decomposed components,
      and reconstruction outputs.
    
    When m == 2, each sample is displayed in a separate figure with 2 rows
    (one per low-quality sub-image) and an overall title indicating the sample number.
    Otherwise, it selects the first image and displays one row per sample in a grid.
    """
    torch.manual_seed(seed)
    stage1.eval()
    device = next(stage1.parameters()).device

    if num_samples < 1:
        return
    if num_samples > len(data_loader.dataset):
        num_samples = len(data_loader.dataset)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if show_plot:
        plt.ion()
    else:
        plt.ioff()

    sample_batch, gt_batch = get_visualization_batch(data_loader, is_paired, device)
    B, m, _, _, _ = sample_batch.shape

    with torch.no_grad():
        outputs = stage1(sample_batch)

    low_quality = extract_low_quality_images(sample_batch)
    encoded_feats = extract_encoded_features(outputs, m)
    R, L = extract_decomposed_components(outputs, m)
    reconstruction = extract_reconstruction(outputs, m)
    RL_product = compute_rl_product(R, L)

    figs = plot_stage1_figures(low_quality, encoded_feats, R, L, RL_product,
                               reconstruction, gt_batch, num_samples, m, is_paired)
    
    if mlflow.active_run():
        for idx, fig in enumerate(figs):
            mlflow.log_figure(fig, f"sample_{idx}.png")
            plt.close(fig)
    
    if save_dir:
        for idx, fig in enumerate(figs):
            fig_path = os.path.join(save_dir, f"sample_{idx}.png")
            fig.savefig(fig_path)
            plt.close(fig)
    
    if show_plot:
        plt.show()


