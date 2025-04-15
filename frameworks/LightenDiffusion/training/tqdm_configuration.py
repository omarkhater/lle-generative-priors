from tqdm.auto import tqdm


class TqdmManager:
    def __init__(self, total: int, desc: str, leave: bool = True, **kwargs):
        """
        Initialize a tqdm progress bar with centralized configuration.

        Args:
            total (int): Total number of iterations.
            desc (str): Description for the progress bar.
            leave (bool): Whether to leave the progress bar after completion.
            **kwargs: Additional keyword arguments to pass to tqdm.
        """
        default_kwargs = {
            "ncols": 100,
            "dynamic_ncols": True,
            "mininterval": 0.1,
            "smoothing": 0.1,
            "leave": leave
        }
        default_kwargs.update(kwargs)
        self.bar = tqdm(total=total, desc=desc, **default_kwargs)

    def update(self, n: int = 1):
        """Update the progress bar by n steps."""
        self.bar.update(n)

    def set_postfix(self, **kwargs):
        """Set the postfix of the progress bar."""
        self.bar.set_postfix(**kwargs)

    def close(self):
        """Close the progress bar."""
        self.bar.close()

    def reset(self, total: int = None, desc: str = None):
        """
        Reset the progress bar for reuse.
        
        Args:
            total (int, optional): Total iterations for the new progress bar.
            desc (str, optional): New description.
        """
        self.bar.reset(total=total)
        if desc is not None:
            self.bar.set_description(desc)
