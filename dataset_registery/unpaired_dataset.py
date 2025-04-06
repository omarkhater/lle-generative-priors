import random
from typing import Tuple, List, Union
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


class UnpairedImageDataset(Dataset):
    """
    A PyTorch Dataset for unpaired low-light image enhancement tasks.

    Each sample is expected to be a dictionary with a key (default "image")
    that contains either a single image or a list of images.
    When a sample contains a list of images, a fixed number of images are selected:
      - If num_images > 0, randomly select that many images.
      - If num_images == -1, use all available images.
    All images are resized to the specified target size and converted to tensors.
    """
    def __init__(
        self,
        hf_dataset: Dataset,
        num_images: int = 1,
        seed: int = 42,
        target_size: Tuple[int, int] = (256, 256),
        image_key: str = "image"
    ) -> None:
        """
        Parameters:
            hf_dataset (Dataset): Hugging Face dataset object where each sample is a dictionary.
            num_images (int): Number of images to select per sample.
                              If set to -1, all available images are selected.
            seed (int): Random seed for reproducibility.
            target_size (Tuple[int, int]): Desired image size after resizing.
            image_key (str): Key for the image(s) in each sample.
        """
        self.hf_dataset = hf_dataset
        self.num_images = num_images
        self.rng = random.Random(seed)
        self.image_key = image_key
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, List[torch.Tensor]]:
        sample = self.hf_dataset[idx]
        images = sample[self.image_key]
        if not isinstance(images, list):
            images = [images]
        if self.num_images == -1:
            selected_images = images
        elif len(images) > self.num_images:
            selected_images = self.rng.sample(images, self.num_images)
        else:
            selected_images = images
        transformed_images = []
        for img in selected_images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            transformed_images.append(self.transform(img))
        if len(transformed_images) == 1:
            return transformed_images[0]
        return transformed_images
