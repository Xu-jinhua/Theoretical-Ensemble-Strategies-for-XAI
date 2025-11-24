"""
Core Data Loading Module
"""
import io
import zipfile
import shutil
import uuid

import zarr
from zarr.storage import ZipStore
import torch
from pathlib import Path

class ZarrDataLoader:
    """
    Handles loading data from a single Zarr ZipStore.
    
    To accelerate read operations from potentially slow storage, this class first
    copies the .zarr.zip file to a local temporary directory before opening it.
    The temporary directory is automatically cleaned up upon exit.
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
        self.store = None
        self.root = None
        self.temp_dir_path = None
        self.should_pin_memory = torch.cuda.is_available()

    def _construct_path(self) -> Path:
        """Constructs the full path to the Zarr archive."""
        data_source_config = self.config['data_source']
        naming_pattern = data_source_config['naming_pattern']
        filename = naming_pattern.format(**self.task_params)
        return Path(data_source_config['path']) / filename

    def load(self):
        """
        Copies the Zarr archive to a temporary directory and loads it.

        Raises:
            FileNotFoundError: If the Zarr archive does not exist.
        """
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr archive not found at: {self.zarr_path}")

        base_temp_dir = Path(self.config['temp_extraction_dir'])
        task_id = self.task_params.get('task_id', uuid.uuid4())
        self.temp_dir_path = base_temp_dir / f"zarr_task_{task_id}"
        self.temp_dir_path.mkdir(parents=True, exist_ok=True)
        
        copied_zarr_path = self.temp_dir_path / self.zarr_path.name
        print(f"Copying {self.zarr_path} to temporary location {copied_zarr_path}...")
        shutil.copy2(self.zarr_path, copied_zarr_path)

        print(f"Loading data from {copied_zarr_path}...")
        self.store = ZipStore(copied_zarr_path, mode='r')
        self.root = zarr.open(self.store, mode='r')
        print("Data loaded successfully from temporary copy.")
        
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
        return len(self.get_all_class_means())

    def get_n_channels(self) -> int:
        """Returns the number of image channels."""
        return self.get_images().shape[1]

    def close(self):
        """Closes the Zarr store and cleans up the temporary directory."""
        if self.store:
            self.store.close()
        self.store = None
        self.root = None
        
        if self.temp_dir_path and self.temp_dir_path.exists():
            print(f"Cleaning up temporary directory: {self.temp_dir_path}")
            shutil.rmtree(self.temp_dir_path)
            self.temp_dir_path = None

    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

