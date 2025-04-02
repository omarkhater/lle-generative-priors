import torch
from .metrics import to_numpy, compute_psnr, compute_ssim, compute_lpips, compute_niqe, compute_pi

def validate_dataloader(dataloader):
    """
    Validate that the dataloader yields tuples of (x, y) with x of shape [B, 6, H, W].
    Raise an error if not.
    """
    try:
        sample = next(iter(dataloader))
    except StopIteration:
        raise ValueError("Dataloader is empty.")
    
    if not isinstance(sample, (list, tuple)) or len(sample) < 2:
        raise ValueError("Dataloader must yield a tuple (x, y).")
    
    x = sample[0]
    if not isinstance(x, torch.Tensor):
        raise ValueError("The first element of the dataloader batch must be a torch.Tensor.")
    
    if x.dim() != 4 or x.size(1) != 6:
        raise ValueError("Expected input tensor shape [B, 6, H, W], but got {}".format(x.shape))
    
    return True

def compute_batch_metrics(batch: tuple, pred: torch.Tensor, is_paired: bool) -> dict:
    """
    Compute evaluation metrics for a single batch.
    
    Args:
        batch: Tuple containing input data, with first element being a tensor of shape [B, 6, H, W]
        pred: Model prediction tensor of shape [B, 3, H, W]
        is_paired: Whether ground-truth images are available for comparison
        
    Returns:
        Dictionary of computed metrics for the first sample in the batch
    """
    metrics = {}
    if is_paired:
        gt = batch[0][:, 3:, :, :].to(pred.device)
        gt_np = to_numpy(gt[0])
        pred_np = to_numpy(pred[0])
        metrics["psnr"] = compute_psnr(gt_np, pred_np)
        metrics["ssim"] = compute_ssim(gt_np, pred_np)
        metrics["lpips"] = compute_lpips(gt[0], pred[0])
    else:
        low = batch[0][:, :3, :, :].to(pred.device)
        metrics["lpips"] = compute_lpips(low[0], pred[0])
    
    metrics["niqe"] = compute_niqe(pred[0])
    metrics["pi"] = compute_pi(metrics["lpips"], metrics["niqe"])
    return metrics

def aggregate_metrics(metrics_list: list, is_paired: bool) -> dict:
    """
    Aggregate metrics from multiple batches by computing their average.
    
    Args:
        metrics_list: List of metric dictionaries from individual batches
        is_paired: Whether evaluation was performed on paired data
        
    Returns:
        Dictionary of averaged metrics across all batches
    """
    agg = {}
    if not metrics_list:
        return agg
    keys = metrics_list[0].keys()
    for key in keys:
        agg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    if not is_paired:
        agg.pop("psnr", None)
        agg.pop("ssim", None)
    return agg

def evaluate_model(dataloader, model, device, is_paired=True):
    """
    Evaluate a low-light enhancement model using a dataloader.
    
    Args:
        dataloader (torch.utils.data.DataLoader): Yields batches of (x, y) where x has shape [B, 6, H, W].
        model (torch.nn.Module): The enhancement model that accepts an input tensor and returns enhanced images.
        device (torch.device): Device on which evaluation is performed.
        is_paired (bool): Whether ground-truth images are provided (paired data).
    
    Returns:
        dict: Average evaluation metrics (PSNR, SSIM, LPIPS, NIQE, PI).
    """
    validate_dataloader(dataloader)
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for batch in dataloader:
            # Assume batch is a tuple (x, y); x: [B, 6, H, W]
            x = batch[0].to(device)
            # Forward pass: let model handle input appropriately.
            pred = model(x)
            batch_metrics = compute_batch_metrics(batch, pred, is_paired)
            metrics_list.append(batch_metrics)
    return aggregate_metrics(metrics_list, is_paired)
