from .loader import load_huggingface_dataset, to_pytorch_dataloader
from .custom_dataset import variable_low_images_collate

class DatasetManager:
    """
    A registry for managing paired and unpaired LLIE datasets by name and split.
    """
    def __init__(self) -> None:
        self.datasets: dict[str, dict] = {}

    def initialize_dataset(
        self,
        name: str,
        dataset_id: str,
        splits: list[str] = ['train'],
        dataset_type: str = "paired",
        hf_cache_dir: str | None = None,
        num_low: int = 2,
        seed: int = 42,
        target_size: tuple[int, int] = (256, 256),
        image_key: str = "image"
    ) -> None:
        """
        Loads a Hugging Face dataset and registers it for the specified splits.

        For paired datasets, each sample must include:
          - low_key: a collection (list or single instance) of low-quality images.
          - target_key: the corresponding target image.
        For unpaired datasets, each sample must include:
          - image_key: an image or list of images.

        Parameters:
            name: Dataset identifier.
            dataset_id: Hugging Face dataset ID.
            splits: List of splits to register.
            dataset_type: "paired" or "unpaired".
            hf_cache_dir: Optional cache directory for hugging face dataset downloader.
            num_low: For paired datasets, number of low images to select per sample (-1 for all).
                     For unpaired datasets, number of images to select (-1 for all).
            seed: Random seed.
            target_size: Desired image size.
            image_key: Key for images (unpaired).
        """
        hf_ds = load_huggingface_dataset(dataset_id, hf_cache_dir=hf_cache_dir)
        self.datasets[name] = {}
        for split in splits:
            if split not in hf_ds:
                print(f"Warning: Split '{split}' not found in dataset '{dataset_id}'")
                continue
            ds_split = hf_ds[split]
            if dataset_type == "paired":
                from .custom_dataset import CustomDataset
                wrapped_ds = CustomDataset(
                    ds_split,
                    num_low=num_low,
                    seed=seed,
                    target_size=target_size
                )
            elif dataset_type == "unpaired":
                from .unpaired_dataset import UnpairedImageDataset
                wrapped_ds = UnpairedImageDataset(
                    ds_split,
                    num_images=num_low,
                    seed=seed,
                    target_size=target_size,
                    image_key=image_key
                )
            else:
                raise ValueError("dataset_type must be either 'paired' or 'unpaired'")
            self.datasets[name][split] = wrapped_ds

    def get_dataset(self, name: str, split: str) -> object:
        """
        Retrieves the registered dataset for the given name and split.
        """
        return self.datasets[name][split]

    def list_datasets(self) -> dict[str, list[str]]:
        """
        Returns a dictionary mapping dataset names to their registered splits.
        """
        return {name: list(splits.keys()) for name, splits in self.datasets.items()}
    
    def get_dataloader(
        self,
        name: str,
        split: str,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> object:
        """
        Converts the registered dataset for the given name and split into a DataLoader.
        """
        dataset = self.get_dataset(name, split)
        return to_pytorch_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=variable_low_images_collate
        )
    