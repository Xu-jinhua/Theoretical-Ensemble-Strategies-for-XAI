"""
Core Data Loading Module
MODIFIED: To read Zarr directories directly, removing ZipStore and temp copy.
MODIFIED: Added fallback to original (non-decorrelated) data for means.
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
        self._original_root = None  # Store reference to original (non-decorrelated) data
        
    def _load_means_from_original(self):
        """Load means from original (non-decorrelated) Zarr file if not present in current data."""
        if self._original_root is not None:
            return self._original_root
            
        # Construct path to original data
        original_path = None
        if self.task_params['model'] == 'resnet18':
            # ResNet18 data is in final_results
            original_path = Path("/mnt/shared_storage/xujinhua/XAI/final_results") / f"dataset-{self.task_params['dataset']}_model-{self.task_params['model']}_noise-{self.task_params['noise']}_split-{self.task_params['split']}.zarr.zip"
        else:
            # DenseNet121 data is in final_results_medmnist_densenet
            original_path = Path("/mnt/shared_storage/xujinhua/XAI/final_results_medmnist_densenet") / f"dataset-{self.task_params['dataset']}_model-{self.task_params['model']}_noise-{self.task_params['noise']}_split-{self.task_params['split']}.zarr"
            
            # Try .zarr.zip if .zarr doesn't exist
            if not original_path.exists():
                original_path = Path("/mnt/shared_storage/xujinhua/XAI/final_results_medmnist_densenet") / f"dataset-{self.task_params['dataset']}_model-{self.task_params['model']}_noise-{self.task_params['noise']}_split-{self.task_params['split']}.zarr.zip"
        
        if original_path and original_path.exists():
            print(f"Loading means from original data: {original_path}")
            if original_path.suffix == '.zip':
                from zarr.storage import ZipStore
                store = ZipStore(str(original_path), mode='r')
                self._original_root = zarr.open(store, mode='r')
            else:
                self._original_root = zarr.open(str(original_path), mode='r')
            return self._original_root
        else:
            raise FileNotFoundError(f"Original data not found at: {original_path}")

    def _construct_path(self) -> Path:
        """Constructs the full path to the Zarr archive."""
        data_source_config = self.config['data_source']
        naming_pattern = data_source_config['naming_pattern']
        
        # First, replace the template variables in the pattern
        try:
            # Make a copy of task_params with only the keys we need for formatting
            # This avoids issues with extra keys like task_id, config_path, etc.
            format_params = {
                'dataset': self.task_params.get('dataset'),
                'model': self.task_params.get('model'),
                'noise': self.task_params.get('noise'),
                'split': self.task_params.get('split')
            }
            
            formatted_pattern = naming_pattern.format(**format_params)
        except KeyError as e:
            raise KeyError(f"Missing task parameter for pattern replacement: {e}. Available params: {list(self.task_params.keys())}")
        except Exception as e:
            raise Exception(f"Error formatting pattern: {e}. naming_pattern={naming_pattern}, task_params={self.task_params}")
        
        # Check if pattern contains wildcard for format detection
        if '*' in formatted_pattern:
            # Try both .zarr and .zarr.zip formats
            base_pattern = formatted_pattern.replace('*', '')
            zarr_path = Path(data_source_config['path']) / f"{base_pattern}.zarr"
            zarr_zip_path = Path(data_source_config['path']) / f"{base_pattern}.zarr.zip"
            
            if zarr_path.exists():
                return zarr_path
            elif zarr_zip_path.exists():
                return zarr_zip_path
            else:
                # Return one of them to trigger the error message
                return zarr_path
        else:
            # Use the formatted pattern as-is
            return Path(data_source_config['path']) / formatted_pattern

    def load(self):
        """
        Opens the Zarr store (supports both directory and zip formats).

        Raises:
            FileNotFoundError: If the Zarr store does not exist.
        """
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found at: {self.zarr_path}")
        
        print(f"Loading data from {self.zarr_path}...")
        
        # Check if it's a zip file or directory
        if self.zarr_path.suffix == '.zip':
            # It's a zarr.zip file, use ZipStore
            from zarr.storage import ZipStore
            store = ZipStore(str(self.zarr_path), mode='r')
            self.root = zarr.open(store, mode='r')
            self._store = store  # Keep reference to close later
        else:
            # It's a zarr directory
            if not self.zarr_path.is_dir():
                raise NotADirectoryError(f"Path is not a Zarr directory: {self.zarr_path}")
            self.root = zarr.open(self.zarr_path, mode='r')
            self._store = None
        
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
        try:
            return self._to_tensor(self.root['means/overall'][:]).contiguous()
        except KeyError:
            # Fallback to original data
            print("Warning: 'means/overall' not found in decorrelated data. Loading from original data...")
            original_root = self._load_means_from_original()
            return self._to_tensor(original_root['means/overall'][:]).contiguous()

    def get_class_mean(self, class_id: int) -> torch.Tensor:
        """Returns a class mean tensor, ensuring it's a contiguous CPU tensor."""
        try:
            return self._to_tensor(self.root[f'means/class_{class_id}'][:]).contiguous()
        except KeyError:
            # Fallback to original data
            print(f"Warning: 'means/class_{class_id}' not found in decorrelated data. Loading from original data...")
            original_root = self._load_means_from_original()
            return self._to_tensor(original_root[f'means/class_{class_id}'][:]).contiguous()
    
    def get_all_class_means(self) -> dict[int, torch.Tensor]:
        """Returns a dictionary of all class mean tensors."""
        class_means = {}
        if 'means' in self.root:
            for key in self.root['means'].keys():
                if key.startswith('class_'):
                    class_id = int(key.split('_')[1])
                    class_means[class_id] = self.get_class_mean(class_id)
        else:
            # Fallback to original data
            print("Warning: 'means' group not found in decorrelated data. Loading from original data...")
            original_root = self._load_means_from_original()
            if 'means' in original_root:
                for key in original_root['means'].keys():
                    if key.startswith('class_'):
                        class_id = int(key.split('_')[1])
                        class_means[class_id] = self._to_tensor(original_root[f'means/class_{class_id}'][:]).contiguous()
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
        # Close ZipStore if it was used
        if hasattr(self, '_store') and self._store is not None:
            self._store.close()
        # Close original data store if it was opened
        if hasattr(self, '_original_store') and self._original_store is not None:
            self._original_store.close()
        # Clear the root references
        self.root = None
        self._original_root = None
        
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
