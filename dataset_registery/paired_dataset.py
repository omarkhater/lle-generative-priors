from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PairedImageDataset(Dataset):
    """
    A PyTorch Dataset for paired image datasets.
    
    Assumes that for each sample:
      - The ground truth image is at index 2 * idx.
      - The corresponding input image is at index 2 * idx + 1.
    """
    def __init__(self, hf_dataset, input_key="image", gt_key="image"):
        self.dataset = hf_dataset
        self.input_key = input_key
        self.gt_key = gt_key
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset) // 2

    def __getitem__(self, idx):
        # Retrieve paired samples.
        input_sample = self.dataset[2 * idx + 1]
        gt_sample = self.dataset[2 * idx]
        
        input_image = input_sample.get(self.input_key, input_sample.get("image"))
        gt_image = gt_sample.get(self.gt_key, gt_sample.get("image"))

        if isinstance(input_image, Image.Image):
            input_image = self.transform(input_image)
        if isinstance(gt_image, Image.Image):
            gt_image = self.transform(gt_image)

        return input_image, gt_image
