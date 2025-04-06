import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PairedImageDataset(Dataset):
    """
    A PyTorch Dataset for paired low-light image enhancement tasks.

    Each sample is expected to be a dictionary with two keys:
      - low_key: a list (or single instance) of low-quality images.
      - target_key: the corresponding ground truth image.

    At retrieval time, if num_low is set to a positive integer, that many low images are randomly
    selected from the available ones; if num_low is -1, then all available low images are used.
    All images are resized to target_size and converted to tensors.
    """
    def __init__(self, hf_dataset, num_low: int = 2, seed: int = 42, 
                 target_size: tuple = (256, 256), 
                 low_key: str = "low_images", target_key: str = "target"):
        """
        Parameters:
            hf_dataset (datasets.Dataset): Hugging Face dataset object where each sample is a dict.
            num_low (int): Number of low images to randomly select per sample.
                           If set to -1, all available low images are selected.
            seed (int): Random seed for reproducibility.
            target_size (tuple): Size to which images will be resized.
            low_key (str): Key for low-quality images in each sample.
            target_key (str): Key for the target (ground truth) image in each sample.
        """
        self.hf_dataset = hf_dataset
        self.num_low = num_low
        self.rng = random.Random(seed)
        self.low_key = low_key
        self.target_key = target_key
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        low_images = sample[self.low_key]
        target_image = sample[self.target_key]
        if not isinstance(low_images, list):
            low_images = [low_images]
        if self.num_low == -1:
            selected_low = low_images
        elif len(low_images) > self.num_low:
            selected_low = self.rng.sample(low_images, self.num_low)
        else:
            selected_low = low_images
        transformed_low = []
        for img in selected_low:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            transformed_low.append(self.transform(img))

        if not isinstance(target_image, Image.Image):
            target_image = Image.fromarray(target_image)
        transformed_target = self.transform(target_image)

        return transformed_low, transformed_target
