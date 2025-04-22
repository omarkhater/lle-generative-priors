import random
import os
from typing import Tuple, List, Dict
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class CustomDataset(Dataset):
    """
    Custom Dataset for paired low-light image enhancement tasks.

    This dataset groups samples by their "label" field (acting as a sample identifier)
    and separates images into low-quality images and a target image. The target image is 
    identified by checking the PIL image's filename for keywords such as "label" or "gt". 
    All other images are considered low-quality.

    At retrieval time:
      - If num_low is a positive integer, that many low images are randomly selected.
      - If num_low is -1, then all available low images are selected.
    All images are resized to target_size and converted to tensors.

    Each sample returns:
      - A list of transformed low-quality image tensors.
      - A transformed target image tensor.
    """
    def __init__(
        self, 
        hf_dataset, 
        num_low: int = 2, 
        seed: int = 42,
        target_size: Tuple[int, int] = (256, 256)
    ):
        """
        Parameters:
            hf_dataset (datasets.Dataset): Hugging Face dataset object.
                Each record is expected to be a dict with keys:
                    - "image": a PIL.Image (or array convertible to PIL.Image) with a .filename attribute.
                    - "label": a string used as the sample identifier (e.g., "sample_1").
            num_low (int): Number of low-quality images to randomly select per sample.
                           If set to -1, all available low images are selected.
            seed (int): Random seed for reproducibility.
            target_size (Tuple[int, int]): Size to which images will be resized.
        """
        self.hf_dataset = hf_dataset
        self.num_low = num_low
        self.rng = random.Random(seed)
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict[str, List[int]]]:
        samples: List[Dict[str, List[int]]] = []
        for indices in self._group_by_sample().values():
            low_idxs, lbl_idx = self._separate_sample(indices)
            if lbl_idx is not None and low_idxs:
                samples.append({"low_idxs": low_idxs, "label_idx": lbl_idx})
        return samples

    def _group_by_sample(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for idx, record in enumerate(self.hf_dataset):
            sample_id = record["label"]
            groups.setdefault(sample_id, []).append(idx)
        return groups

    def _separate_sample(self, indices: List[int]) -> Tuple[List[int], int]:
        low_indices: List[int] = []
        label_idx: int | None = None
        for i in indices:
            img_obj = self.hf_dataset[i]["image"]
            fname = getattr(img_obj, "filename", "")
            base = os.path.basename(fname).lower() if fname else ""
            if "gt" in base or "label" in base:
                label_idx = i
            else:
                low_indices.append(i)
        return low_indices, label_idx 
    
    def _prepare_samples(self) -> List[Dict[str, List[int]]]:
        samples: List[Dict[str, List[int]]] = []
        for idxs in self._group_by_sample().values():
            lows, lbl = self._separate_sample(idxs)
            if lbl is not None and lows:
                samples.append({"low_idxs": lows, "label_idx": lbl})
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        grp = self.samples[idx]
        low_idxs = grp["low_idxs"]
        lbl_idx = grp["label_idx"]

        # select up to num_low
        if self.num_low != -1 and len(low_idxs) > self.num_low:
            chosen = self.rng.sample(low_idxs, self.num_low)
        else:
            chosen = low_idxs

        # load + transform low images
        lows: List[torch.Tensor] = []
        for i in chosen:
            img = self.hf_dataset[i]["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            lows.append(self.transform(img))

        # load + transform label image
        lbl_img = self.hf_dataset[lbl_idx]["image"]
        if not isinstance(lbl_img, Image.Image):
            lbl_img = Image.fromarray(lbl_img)
        label = self.transform(lbl_img)

        return lows, label

def variable_low_images_collate(batch: List[Tuple[List[torch.Tensor], torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable numbers of low images per sample.
    Pads each sample's low images list to the maximum number found in the batch.

    Returns:
        Tuple of:
            - Tensor of shape (batch_size, max_n, C, H, W) for low images.
            - Tensor of shape (batch_size, C, H, W) for label images.
    """
    low_imgs_list, labels = zip(*batch)
    labels = torch.stack(labels, dim=0)
    max_n = max(len(low_list) for low_list in low_imgs_list)
    padded_low = []
    for low_list in low_imgs_list:
        if len(low_list) < max_n:
            pad_tensor = torch.zeros_like(low_list[0])
            padded = low_list + [pad_tensor] * (max_n - len(low_list))
        else:
            padded = low_list
        padded_low.append(torch.stack(padded, dim=0))
    low_imgs_batch = torch.stack(padded_low, dim=0)
    return low_imgs_batch, labels