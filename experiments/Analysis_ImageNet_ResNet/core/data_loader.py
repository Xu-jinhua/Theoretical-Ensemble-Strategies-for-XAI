"""
Core Data Loading Module
MODIFIED: To read Zarr directories directly, removing ZipStore and temp copy.
"""
import io
import zarr
import torch
from pathlib import Path

class ZarrDataLoader:
    """
    Handles loading data from a single Zarr directory.
    For CUDA operations, it automatically pins tensor memory.
    """

    def __init__(self, task_params: dict, config: dict):
        """
        Initializes the DataLoader.

        Args:
            task_params (dict): Parameters for the current task.
            config (dict): The global configuration dictionary.
        """
        self.task_params = task_params
        self.config = config
        self.zarr_path = self._construct_path()
        self.root = None # Changed from self.store
        self.should_pin_memory = torch.cuda.is_available()

    def _construct_path(self) -> Path:
        """Constructs the full path to the Zarr archive."""
        data_source_config = self.config['data_source']
        naming_pattern = data_source_config['naming_pattern']
        filename = naming_pattern.format(**self.task_params)
        return Path(data_source_config['path']) / filename

    def load(self):
        """
        Opens the Zarr directory directly.

        Raises:
            FileNotFoundError: If the Zarr directory does not exist.
        """
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr directory not found at: {self.zarr_path}")
        
        if not self.zarr_path.is_dir():
             raise NotADirectoryError(f"Path is not a Zarr directory: {self.zarr_path}")

        print(f"Loading data from {self.zarr_path}...")
        self.root = zarr.open(self.zarr_path, mode='r')
        print("Data loaded successfully.")
        
    def _to_tensor(self, numpy_array):
        """Converts numpy array to a torch tensor, pinning memory if possible."""
        tensor = torch.from_numpy(numpy_array)
        if self.should_pin_memory:
            return tensor.pin_memory()
        return tensor

    def get_attributions(self, method_name: str) -> torch.Tensor:
        """Retrieves an attribution tensor, ensuring it's a contiguous CPU tensor."""
        return self._to_tensor(self.root[f'attributions/{method_name}'][:]).contiguous()

    def get_all_attribution_methods(self) -> list[str]:
        """Returns a list of all available attribution methods."""
        if 'attributions' in self.root:
            return list(self.root['attributions'].keys())
        return []

    def get_images(self) -> torch.Tensor:
        """Returns the image tensor, ensuring it's a contiguous CPU tensor."""
        return self._to_tensor(self.root['images'][:]).contiguous()

    def get_labels(self) -> torch.Tensor:
        """Returns the labels tensor, ensuring it's a contiguous CPU tensor."""
        return self._to_tensor(self.root['labels'][:]).contiguous()

    def get_predictions(self) -> torch.Tensor:
        """Returns the model prediction tensor, ensuring it's a contiguous CPU tensor."""
        predictions_tensor = self._to_tensor(self.root['model_predictions'][:])
        return torch.argmax(predictions_tensor, dim=1).contiguous()

    def get_global_mean(self) -> torch.Tensor:
        """Returns the global mean tensor, ensuring it's a contiguous CPU tensor."""
        return self._to_tensor(self.root['means/overall'][:]).contiguous()

    def get_class_mean(self, class_id: int) -> torch.Tensor:
        """Returns a class mean tensor, ensuring it's a contiguous CPU tensor."""
        return self._to_tensor(self.root[f'means/class_{class_id}'][:]).contiguous()
    
    def get_all_class_means(self) -> dict[int, torch.Tensor]:
        """Returns a dictionary of all class mean tensors."""
        class_means = {}
        if 'means' in self.root:
            for key in self.root['means'].keys():
                if key.startswith('class_'):
                    class_id = int(key.split('_')[1])
                    class_means[class_id] = self.get_class_mean(class_id)
        return class_means

    def get_n_class(self) -> int:
        """Returns the number of classes."""
        # For ImageNet, means might not be per-class
        if 'means' in self.root and len(self.get_all_class_means()) > 1:
            return len(self.get_all_class_means())
        # Fallback for ImageNet (1000 classes)
        if self.task_params.get('dataset') == 'imagenet':
            return 1000
        return 1 # Default fallback

    def get_n_channels(self) -> int:
        """Returns the number of image channels."""
        return self.get_images().shape[1]

    def close(self):
        """Closes the Zarr store."""
        # zarr.open() on a directory doesn't need explicit closing, 
        # but we'll clear the root for good practice.
        self.root = None
        
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
