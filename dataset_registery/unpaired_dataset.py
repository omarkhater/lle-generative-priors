from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class UnpairedImageDataset(Dataset):
    """
    A PyTorch Dataset for unpaired image datasets.
    
    Returns a single image per sample.
    """
    def __init__(self, hf_dataset, input_key="image"):
        self.dataset = hf_dataset
        self.input_key = input_key
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample.get(self.input_key, sample.get("image"))
        if isinstance(image, Image.Image):
            image = self.transform(image)
        return image
