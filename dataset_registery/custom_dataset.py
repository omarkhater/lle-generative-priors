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

    def _prepare_samples(self) -> List[Dict]:
        groups = self._group_by_sample()
        samples = []
        for _, items in groups.items():
            low_imgs, label_img = self._separate_sample(items)
            if label_img is not None and low_imgs:
                samples.append({
                    "low_images": low_imgs,
                    "label": label_img,
                })
        return samples

    def _group_by_sample(self) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = {}
        for item in self.hf_dataset:
            sample_id = item["label"]  # Using the "label" field as the sample identifier.
            groups.setdefault(sample_id, []).append(item)
        return groups

    def _separate_sample(self, items: List[Dict]) -> Tuple[List[Image.Image], Image.Image]:
        low_images: List[Image.Image] = []
        label_image: Image.Image | None = None
        for record in items:
            image_obj = record["image"]
            if not isinstance(image_obj, Image.Image):
                image_obj = Image.fromarray(image_obj)
            filename = getattr(image_obj, "filename", None)
            if filename:
                basename = os.path.basename(filename).lower()
                if "label" in basename or "gt" in basename:
                    label_image = image_obj
                else:
                    low_images.append(image_obj)
            else:
                low_images.append(image_obj)
        return low_images, label_image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        sample = self.samples[idx]
        low_images = sample["low_images"]
        label_image = sample["label"]

        if self.num_low == -1:
            selected_low = low_images
        elif len(low_images) > self.num_low:
            selected_low = self.rng.sample(low_images, self.num_low)
        else:
            selected_low = low_images

        transformed_low = [self.transform(img) for img in selected_low]
        transformed_label = self.transform(label_image)

        return transformed_low, transformed_label

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