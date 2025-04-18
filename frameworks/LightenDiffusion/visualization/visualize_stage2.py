import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional, List
import os
from .visualize_stage1 import tensor_to_image
from .visualization_utils import map_to_rgb

def select_visualization_indices(batch_size: int, num_samples: int, seed: int) -> List[int]:
    """
    Selects a list of random indices from a batch given a seed.
    
    Args:
        batch_size (int): Number of samples in the batch.
        num_samples (int): Number of sample indices to select.
        seed (int): Random seed for reproducibility.
    
    Returns:
        List[int]: List of selected indices.
    """
    import random
    random.seed(seed)
    indices = list(range(batch_size))
    random.shuffle(indices)
    return indices[:min(num_samples, batch_size)]

def visualize_stage2_results_aggregate(
    pipeline: torch.nn.Module,
    data_loader: DataLoader,
    num_samples: int = 1,
    random_seed: int = 42,
    save_dir: Optional[str] = None,
) -> None:
    """
    Visualizes aggregated outputs from the Stage2 diffusion process along with the final
    enhanced image. The visualization shows:
    
      Row 1: 
         - \(I_{\mathrm{low}}\): the input low-light image(s). If m == 2, both sub-images are concatenated;
           otherwise, the first one is shown.
         - \(I_{\mathrm{high}}\): the ground-truth high-light image.
         - \(F_{\mathrm{low}}\) and \(F_{\mathrm{high}}\): encoded features (projected to RGB using PCA).
    
      Row 2: 
         - \(R_{\mathrm{low}}, R_{\mathrm{high}}, L_{\mathrm{low}}, L_{\mathrm{high}}\): decomposition outputs (RGB).
    
      Row 3: 
         - \(x_{0}, x_{t}, \hat{x}_{t}, x_{0}(F_{\mathrm{low}}^{\hat{}})\): the key latent terms in the diffusion model.
    
      Row 4: 
         - \(I_{\mathrm{low}}^{\hat{}}\): the final enhanced output image.
    
    All latent outputs (which are of low resolution, e.g. 32×32) are processed in batch
    through map_to_rgb.
    
    Args:
        pipeline (torch.nn.Module): The full LightenDiffusionPipeline model.
        data_loader (DataLoader): Dataloader yielding paired (x, y) batches (x: [B, m, 3, H, W]; y: [B, 3, H, W]).
        num_samples (int): Number of samples from the batch to visualize.
        random_seed (int): Seed for random selection of samples.
    """
    torch.manual_seed(random_seed)
    batch_iter = iter(data_loader)
    x, y = next(batch_iter)
    device = next(pipeline.parameters()).device
    x = x.to(device)  # low-light images: [B, m, 3, H, W]
    y = y.to(device)  # high-light images: [B, 3, H, W]
    B, m, _, H, W = x.shape

    # Select indices.
    sel_indices = select_visualization_indices(B, num_samples, random_seed)
    sel_idx_tensor = torch.tensor(sel_indices, device=device)

    # Forward pass through the pipeline.
    with torch.no_grad():
        outputs = pipeline(x, y)
        stage1_low = outputs["stage1_low"]      # Aggregated low decomposition.
        stage1_high = outputs["stage1_high"]    # High decomposition (from a singleton input).
        stage2_out  = outputs["stage2"]

    # Unpack Stage1 outputs (latent variables: sizes such as [B, 3, 32, 32]).
    # For each tensor, use advanced indexing to select a batch of selected samples.
    R_low  = stage1_low["R"][sel_idx_tensor]  
    L_low  = stage1_low["L"][sel_idx_tensor]
    f_low  = stage1_low["f"][sel_idx_tensor]
    # For high, we get from stage1_high (assumed shape [B, 3, h, w]).
    R_high = stage1_high["R"][sel_idx_tensor]
    L_high = stage1_high["L"][sel_idx_tensor]
    f_high = stage1_high["f"][sel_idx_tensor]

    # Unpack Stage2 outputs (latent outputs).
    x0    = stage2_out["x0"][sel_idx_tensor]                  # Composite: R_low * L_high.
    x_t   = stage2_out["x_t"][sel_idx_tensor]                  # Noised version.
    ref_f = stage2_out["reference_feature"][sel_idx_tensor]    # R_low * (L_low^gamma).

    # Reverse diffusion on f_low yields x̂_t.
    with torch.no_grad():
        x_hat_t = pipeline.sample_reverse(f_low)  # Processes batch: shape [nvis, 3, h, w].

    # Final enhanced image (decoded from restored latent features); shape [B, 3, H, W].
    I_hat_low = pipeline.predict(x)[sel_idx_tensor]

    # For the original I_low images:
    # If m == 2, concatenate the two sub-images horizontally; otherwise, use the first sub-image.
    if m == 2:
        I_low_all = torch.cat([x[:,0], x[:,1]], dim=-1)  # Now shape: [B, 3, H, 2*W].
    else:
        I_low_all = x[:,0]  # shape: [B, 3, H, W].
    selected_I_low = I_low_all[sel_idx_tensor]
    # The high-light input remains as y (shape [B, 3, H, W]).
    selected_I_high = y[sel_idx_tensor]

    # Process latent outputs in batch via map_to_rgb.
    # Each call below returns a tensor of shape [nvis, 3, h, w].
    mapped_f_low   = map_to_rgb(f_low)
    mapped_f_high  = map_to_rgb(f_high)
    mapped_R_low   = map_to_rgb(R_low)
    mapped_R_high  = map_to_rgb(R_high)
    mapped_L_low   = map_to_rgb(L_low)
    mapped_L_high  = map_to_rgb(L_high)
    mapped_x0      = map_to_rgb(x0)
    mapped_x_t     = map_to_rgb(x_t)
    mapped_ref_f   = map_to_rgb(ref_f)

    def to_cpu(t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu()

    selected_I_low  = to_cpu(selected_I_low)
    selected_I_high = to_cpu(selected_I_high)
    mapped_f_low    = to_cpu(mapped_f_low)
    mapped_f_high   = to_cpu(mapped_f_high)
    mapped_R_low    = to_cpu(mapped_R_low)
    mapped_R_high   = to_cpu(mapped_R_high)
    mapped_L_low    = to_cpu(mapped_L_low)
    mapped_L_high   = to_cpu(mapped_L_high)
    mapped_x0       = to_cpu(mapped_x0)
    mapped_x_t      = to_cpu(mapped_x_t)
    mapped_ref_f    = to_cpu(mapped_ref_f)
    selected_I_hat  = to_cpu(I_hat_low)

    # Create a grid of subplots.
    num_cols = 6
    num_rows = 2
    for i in range(len(sel_indices)):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 3*num_rows))
        fig.suptitle(f"Sample {sel_indices[i]}", fontsize=16)

        # Row 1: I_low, I_high, F_low, F_high, R_low, R_high
        axes[0, 0].imshow(tensor_to_image(selected_I_low[i]))
        axes[0, 0].set_title(r"$I_{\mathrm{low}}$")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(tensor_to_image(selected_I_high[i]))
        axes[0, 1].set_title(r"$I_{\mathrm{high}}$")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(tensor_to_image(mapped_f_low[i]))
        axes[0, 2].set_title(r"$F_{\mathrm{low}}$")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(tensor_to_image(mapped_f_high[i]))
        axes[0, 3].set_title(r"$F_{\mathrm{high}}$")
        axes[0, 3].axis("off")


        axes[0, 4].imshow(tensor_to_image(mapped_R_low[i]))
        axes[0, 4].set_title(r"$R_{\mathrm{low}}$")
        axes[0, 4].axis("off")

        axes[0, 5].imshow(tensor_to_image(mapped_R_high[i]))
        axes[0, 5].set_title(r"$R_{\mathrm{high}}$")
        axes[0, 5].axis("off")

        # Row 2: L_low, L_high, x0, x_t, x0(F_low^hat), I^hat

        axes[1, 0].imshow(tensor_to_image(mapped_L_low[i]))
        axes[1, 0].set_title(r"$L_{\mathrm{low}}$")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(tensor_to_image(mapped_L_high[i]))
        axes[1, 1].set_title(r"$L_{\mathrm{high}}$")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(tensor_to_image(mapped_x0[i]))
        axes[1, 2].set_title(r"$x_{0}$")
        axes[1, 2].axis("off")

        axes[1, 3].imshow(tensor_to_image(mapped_x_t[i]))
        axes[1, 3].set_title(r"$x_{t}$")
        axes[1, 3].axis("off")


        axes[1, 4].imshow(tensor_to_image(mapped_ref_f[i]))
        axes[1, 4].set_title(r"$x_{0}(F_{\mathrm{low}}^{\hat{}})$")
        axes[1, 4].axis("off")

        axes[1, 5].imshow(tensor_to_image(selected_I_hat[i]))
        axes[1, 5].set_title(r"$I_{\mathrm{low}}^{\hat{}}$")
        axes[1, 5].axis("off")
        
    if save_dir:
        for i, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)
            fig_path = os.path.join(save_dir, f"sample_{i}.png")
            fig.savefig(fig_path)
            plt.close(fig)
    else:
        plt.show()


def visualize_stage2_results(
    pipeline: torch.nn.Module,
    data_loader: DataLoader,
    num_samples: int = 1,
    random_seed: int = 42,
    save_dir: Optional[str] = None,
) -> None:
    """
    Top-level visualization function for Stage2 results using aggregated outputs.
    
    Args:
        pipeline (torch.nn.Module): The full LightenDiffusionPipeline model.
        data_loader (DataLoader): Dataloader yielding (x, y) batches (x: [B, m, 3, H, W]; y: [B, 3, H, W]).
        num_samples (int): Number of samples to display.
        random_seed (int): Seed for random selection.
    """
    visualize_stage2_results_aggregate(
        pipeline, 
        data_loader, 
        num_samples, 
        random_seed,
        save_dir)
