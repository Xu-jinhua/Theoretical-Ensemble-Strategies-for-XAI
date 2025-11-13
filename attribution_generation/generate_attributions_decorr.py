#!/usr/bin/env python3
"""
generate_attributions_decorr.py - Decorrelated Attribution Generation Script

This script generates decorrelated attributions. 

Usage:
    python generate_attributions_decorr.py --config config.yml --data-flag bloodmnist --model-flag densenet121 --noise-name clean --split val --gpu-id 0
"""

import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import gc
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import zarr
from zarr import codecs
from zarr.storage import ZipStore
# from numcodecs import codecs
import blosc
import shutil

# Import model loader
# Import model loader
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "../Attributions_DenseNet_MedMNIST"))
try:
    from model_loader import load_medmnist_model
except ImportError as e:
    logger.error(f"Failed to import model_loader: {e}")
    logger.error(f"sys.path: {sys.path}")
    raise

sys.path.append(str(current_dir / "../Attributions_DenseNet_ImageNet"))
try:
    from model_loader_densenet import load_densenet_model
except ImportError as e:
    logger.error(f"Failed to import model_loader_densenet: {e}")
    raise


from captum.attr import (
    Saliency, InputXGradient, GuidedBackprop, Deconvolution,
    IntegratedGradients, DeepLift, GradientShap, DeepLiftShap
)

# --- Logging Setup ---
def setup_logger(name: str, log_file: Optional[Path] = None):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# --- Model Loading Functions ---
def load_resnet_model(model_flag: str, data_flag: str, weights_root: str,
                      device: torch.device, n_classes: int) -> nn.Module:
    """
    Load ResNet model and apply DeepLIFT patch
    """
    from torchvision import models as torchvision_models
    
    if model_flag == 'resnet18':
        model = torchvision_models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif model_flag == 'resnet50':
        model = torchvision_models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    else:
        raise NotImplementedError(f"Model '{model_flag}' is not implemented.")
    
    # Apply DeepLIFT patch
    for module in model.modules():
        if isinstance(module, torchvision_models.resnet.BasicBlock):
            module.relu.inplace = False
            if not hasattr(module, 'relu2'):
                module.relu2 = nn.ReLU(inplace=False)
            
            def new_basic_forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = out + identity
                out = self.relu2(out)
                return out
            
            module.forward = new_basic_forward.__get__(module, module.__class__)
        
        elif isinstance(module, torchvision_models.resnet.Bottleneck):
            module.relu.inplace = False
            if not hasattr(module, 'relu2'):
                module.relu2 = nn.ReLU(inplace=False)
            if not hasattr(module, 'relu3'):
                module.relu3 = nn.ReLU(inplace=False)
            
            def new_bottleneck_forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu2(out)
                out = self.conv3(out)
                out = self.bn3(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = out + identity
                out = self.relu3(out)
                return out
            
            module.forward = new_bottleneck_forward.__get__(module, module.__class__)
    
    model.to(device)
    model.eval()
    return model

# --- Attribution Generator ---
class AttributionGenerator:
    """Generate decorrelated attributions"""
    
    def __init__(self, config: Dict[str, Any], data_flag: str, model_flag: str,
                 noise_name: str, split: str, gpu_id: int):
        self.config = config
        self.data_flag = data_flag
        self.model_flag = model_flag
        self.noise_name = noise_name
        self.split = split
        self.gpu_id = gpu_id
        
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.paths = config['paths']
        self.model_method_mapping = config['model_method_mapping']
        
        # Set up logging (must be done before loading preflight results)
        self.logger = setup_logger(
            f"AttributionGenerator-{data_flag}-{model_flag}",
            Path(self.paths['scheduler_log_dir']) / f"attribution_{data_flag}_{model_flag}_{noise_name}_{split}.log"
        )
        
        # Load preflight results (to determine batch size)
        self.preflight_results = self._load_preflight_results()
        
        # Data storage
        self.images_in_ram = None
        self.labels_in_ram = None
        self.preds_in_ram = None
        self.attributions_in_ram = {}
        
        self.logger.info(f"Initialized attribution generator: {data_flag}/{model_flag}/{noise_name}/{split} on GPU {gpu_id}")
    
    def _load_preflight_results(self):
        """Load preflight results, get batch size for each method"""
        # Use relative path or path from config
        preflight_path = Path(self.paths.get('preflight_results_path', './preflight_results.json'))
        try:
            with open(preflight_path, 'r') as f:
                results = json.load(f)
            self.logger.info(f"Successfully loaded preflight results: {preflight_path}")
            return results
        except FileNotFoundError:
            self.logger.warning(f"Preflight results file not found: {preflight_path}, will use default batch size")
            return {}
    
    def run(self):
        """Main execution flow"""
        task_name = f"{self.data_flag}_{self.model_flag}_{self.noise_name}_{self.split}"
        self.logger.info(f"--- Starting attribution generation task: {task_name} ---")
        
        try:
            # 1. Load base data (reuse existing data)
            self._load_base_data()
            
            # 2. Generate attributions for each model-method pair
            self._generate_attributions_for_all_models()
            
            # 3. Save results
            self._save_results()
            
            self.logger.info(f"--- Task {task_name} completed successfully ---")
            return True
            
        except Exception as e:
            self.logger.critical(f"Task {task_name} failed: {e}", exc_info=True)
            return False
    
    def _load_base_data(self):
        """Load base data (reuse results from Attributions_Densenet_MedMNIST)"""
        self.logger.info("Step 1: Loading base data...")
        
        # Determine base data path
        if self.model_flag == 'densenet121':
            base_dir = Path(self.paths['base_data_dir'])
        elif self.model_flag == 'resnet18':
            base_dir = Path(self.paths['base_data_dir_resnet18'])
        else:
            raise ValueError(f"Unsupported model: {self.model_flag}")
        
        # Construct base data filename - choose correct format based on model type
        if self.model_flag == 'resnet18':
            # ResNet uses zarr.zip format
            base_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}.zarr.zip"
        else:
            # DenseNet uses zarr directory format
            base_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}.zarr"
        
        base_path = base_dir / base_filename
        
        if not base_path.exists():
            # Try alternative format
            if self.model_flag == 'resnet18':
                alt_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}.zarr"
            else:
                alt_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}.zarr.zip"
            
            alt_path = base_dir / alt_filename
            if alt_path.exists():
                base_path = alt_path
            else:
                raise FileNotFoundError(f"Base data not found: {base_path} or {alt_path}")
        
        self.logger.info(f"Loading base data from {base_path}")
        
        # Load zarr data
        if base_path.suffix == '.zip':
            # ResNet uses zarr.zip format, use ZipStore to load
            self.logger.info(f"Loading zarr.zip file using ZipStore: {base_path}")
            store = ZipStore(str(base_path), mode='r')
            root = zarr.open(store, mode='r')
        else:
            # DenseNet uses zarr directory format
            root = zarr.open(str(base_path), mode='r')
        
        self.images_in_ram = torch.from_numpy(root['images'][:])
        self.labels_in_ram = torch.from_numpy(root['labels'][:]).squeeze()
        self.preds_in_ram = torch.from_numpy(root['model_predictions'][:])
        
        self.logger.info(f"Base data loaded: images={self.images_in_ram.shape}, labels={self.labels_in_ram.shape}")
    
    def _generate_attributions_for_all_models(self):
        """Generate attributions for each model-method pair"""
        self.logger.info("Step 2: Generating decorrelated attributions...")
        
        num_samples = self.images_in_ram.shape[0]
        
        # Iterate over each model-method mapping
        for mapping in self.model_method_mapping:
            model_split_id = mapping['model_split_id']
            method_name = mapping['method_name']
            
            self.logger.info(f"Processing model split_{model_split_id} with method {method_name}")
            
            # Load corresponding model
            model = self._load_model(model_split_id)
            
            # Generate attributions for this model
            attributions = self._generate_attribution_for_model(model, method_name, num_samples)
            
            # Store attribution results
            # output_key = f"model_split_{model_split_id}_{method_name}"
            output_key = method_name
            self.attributions_in_ram[output_key] = attributions
            
            self.logger.info(f"Completed {output_key}")
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        self.logger.info(f"Generated {len(self.attributions_in_ram)} sets of attributions")
    
    def _load_model(self, split_id: int) -> torch.nn.Module:
        """Load model for specified split"""
        # Construct model path
        weights_root = Path(self.paths['decorr_weights_root'])
        model_dir = weights_root / f"{self.data_flag}_{self.model_flag}"
        
        # Determine corresponding seed value based on split_id
        seed_map = {
            0: 42,
            1: 123,
            2: 456,
            3: 789,
            4: 101112,
            5: 131415,
            6: 161718,
            7: 192021
        }
        seed = seed_map.get(split_id, 42)  # Default to 42
        
        model_path = model_dir / f"split_{split_id}_seed_{seed}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model architecture and weights
        if self.model_flag == 'densenet121':
            model = self._load_densenet_model(model_path)
        elif self.model_flag == 'resnet18':
            model = self._load_resnet_model(model_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_flag}")
        
        self.logger.info(f"Successfully loaded model: {model_path}")
        return model
    
    def _load_densenet_model(self, model_path: Path) -> torch.nn.Module:
        """Load DenseNet model"""
        # Load pre-trained DenseNet architecture
        imagenet_weights_root = Path(self.config['paths']['imagenet_weights_root'])
        model = load_densenet_model(
            model_flag='densenet121',
            weights_root=imagenet_weights_root,
            device=self.device
        )
        
        # Replace classifier head
        n_classes = len(self.preds_in_ram[0])
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)
        
        # Load decorrelated training weights
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['net'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_resnet_model(self, model_path: Path) -> torch.nn.Module:
        """Load ResNet model"""
        # Create ResNet architecture
        from torchvision import models
        model = models.resnet18(weights=None)
        
        # Replace classifier head
        n_classes = len(self.preds_in_ram[0])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        
        # Adjust input layer (MedMNIST requires 3 channels)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Load decorrelated training weights
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['net'])
        
        # Apply DeepLIFT patch
        self._apply_deeplift_patches(model)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _apply_deeplift_patches(self, model: torch.nn.Module):
        """Apply DeepLIFT compatibility patch"""
        from torchvision import models as torchvision_models
        
        for module in model.modules():
            if isinstance(module, torchvision_models.resnet.BasicBlock):
                module.relu.inplace = False
                if not hasattr(module, 'relu2'):
                    module.relu2 = nn.ReLU(inplace=False)
                
                def new_basic_forward(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)
                    out = self.conv2(out)
                    out = self.bn2(out)
                    if self.downsample is not None:
                        identity = self.downsample(x)
                    out = out + identity
                    out = self.relu2(out)
                    return out
                
                module.forward = new_basic_forward.__get__(module, module.__class__)
            
            elif isinstance(module, torchvision_models.resnet.Bottleneck):
                module.relu.inplace = False
                if not hasattr(module, 'relu2'):
                    module.relu2 = nn.ReLU(inplace=False)
                if not hasattr(module, 'relu3'):
                    module.relu3 = nn.ReLU(inplace=False)
                
                def new_bottleneck_forward(self, x):
                    identity = x
                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)
                    out = self.conv2(out)
                    out = self.bn2(out)
                    out = self.relu2(out)
                    out = self.conv3(out)
                    out = self.bn3(out)
                    if self.downsample is not None:
                        identity = self.downsample(x)
                    out = out + identity
                    out = self.relu3(out)
                    return out
                
                module.forward = new_bottleneck_forward.__get__(module, module.__class__)
    
    def _generate_attribution_for_model(self, model: torch.nn.Module,
                                       method_name: str, num_samples: int) -> torch.Tensor:
        """Generate attributions for a single model using specified method"""
        # Create explainer
        explainer = self._create_explainer(model, method_name)
        
        # Get batch size for this method (refer to Attributions_Densenet_MedMNIST approach)
        batch_size = self._get_batch_size_for_method(method_name)
        self.logger.info(f"Method {method_name} using batch size: {batch_size}")
        
        # Prepare storage for attribution results - use float16 to save memory
        attribution_results = torch.zeros_like(self.images_in_ram, dtype=torch.float16)
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size),
                         desc=f"Generating attributions: {method_name}"):
                start_idx = i
                end_idx = min(i + batch_size, num_samples)
                
                # Prepare batch data - move to GPU
                batch_images = self.images_in_ram[start_idx:end_idx].to(self.device, non_blocking=True)
                batch_targets = torch.argmax(self.preds_in_ram[start_idx:end_idx], dim=1).to(self.device, non_blocking=True)
                
                # Generate attributions
                attributions = self._get_attribution(explainer, method_name,
                                                   batch_images, batch_targets)
                
                # Store results - move back to CPU and use float16
                attribution_results[start_idx:end_idx] = attributions.detach().cpu().to(torch.float16)
                
                # Clean up GPU memory
                del batch_images, batch_targets, attributions
                torch.cuda.empty_cache()
        
        return attribution_results
    
    def _get_batch_size_for_method(self, method_name: str) -> int:
        """Get batch size for method name from preflight results"""
        # Map method name to preflight result key name
        method_key_map = {
            "Saliency": "Saliency",
            "InputXGradient": "InputXGradient",
            "GuidedBackprop": "GuidedBackprop",
            "Deconvolution": "Deconvolution",
            "IntegratedGradients_n_steps_50": "IntegratedGradients_baselines_zero_n_steps_50",
            "DeepLIFT": "DeepLIFT_baselines_global_mean",
            "GradientShap_n_samples_20": "GradientShap_baselines_combined_zng_n_samples_20",
            "DeepLIFT_SHAP": "DeepLIFT_SHAP_baselines_combined_zng"
        }
        
        preflight_key = method_key_map.get(method_name)
        if not preflight_key:
            self.logger.warning(f"Method {method_name} not found in preflight mapping, using default batch size 32")
            return 32
        
        # Get batch size from preflight results
        model_results = self.preflight_results.get(self.model_flag, {})
        batch_size = model_results.get(preflight_key, 32)
        
        if not batch_size or batch_size == 0:
            self.logger.warning(f"Batch size for method {method_name} is 0, using default value 32")
            return 32
        
        return batch_size
    
    def _create_explainer(self, model: torch.nn.Module, method_name: str):
        """Create explainer instance"""
        explainer_classes = {
            "Saliency": Saliency,
            "InputXGradient": InputXGradient,
            "GuidedBackprop": GuidedBackprop,
            "Deconvolution": Deconvolution,
            "IntegratedGradients": IntegratedGradients,
            "DeepLIFT": DeepLift,
            "GradientShap": GradientShap,
            "DeepLIFT_SHAP": DeepLiftShap
        }
        
        # Handle method names with parameters
        base_method_name = method_name.split('_')[0]
        
        if base_method_name not in explainer_classes:
            raise ValueError(f"Unsupported explanation method: {method_name}")
        
        return explainer_classes[base_method_name](model)
    
    def _get_attribution(self, explainer, method_name: str, images: torch.Tensor, 
                        targets: torch.Tensor) -> torch.Tensor:
        """Generate attributions"""
        kwargs = {'target': targets}
        
        # Handle special parameters
        if 'IntegratedGradients' in method_name:
            if 'n_steps_50' in method_name:
                kwargs['n_steps'] = 50
            elif 'n_steps_200' in method_name:
                kwargs['n_steps'] = 200
        
        elif 'GradientShap' in method_name:
            if 'n_samples_20' in method_name:
                kwargs['n_samples'] = 20
            elif 'n_samples_40' in method_name:
                kwargs['n_samples'] = 40
            # GradientShap requires baselines parameter - refer to Attributions_Densenet_MedMNIST approach
            # combined_zng = zero + noise + global_mean
            baselines_list = []
            baselines_list.append(torch.zeros_like(images))  # zero
            baselines_list.append(torch.randn_like(images))  # noise
            # Note: Here we need global_mean_baseline, but in current simplified version we use zero and noise
            baselines = torch.cat(baselines_list, dim=0)
            kwargs['baselines'] = baselines
        
        # Generate attributions
        with torch.cuda.amp.autocast():
            attributions = explainer.attribute(images, **kwargs)
        
        return attributions
    
    def _save_results(self):
        """Save attribution results to zarr (without compression)"""
        self.logger.info("Step 3: Saving attribution results (without compression)...")
        
        # Construct output path - choose correct format based on model type
        if self.model_flag == 'resnet18':
            # ResNet uses zarr.zip format
            output_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}_decorr.zarr.zip"
            output_path = Path(self.paths['final_output_dir']) / output_filename
            
            # Delete existing output
            if output_path.exists():
                output_path.unlink()
                self.logger.warning(f"Deleted existing output: {output_path}")
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load base data path
            base_path = self._get_base_data_path()
            self.logger.info(f"Loading base data structure from {base_path}")
            
            # ResNet: Use ZipStore to create zarr.zip
            store = ZipStore(str(output_path), mode='w')
            
            # Create root group
            root = zarr.group(store=store)
            
            # Copy base data
            if base_path.suffix == '.zip':
                base_store = ZipStore(str(base_path), mode='r')
                base_root = zarr.open(base_store, mode='r')
            else:
                base_root = zarr.open(str(base_path), mode='r')
            
            for key in ['images', 'labels', 'model_predictions', 'means']:
                if key in base_root:
                    if key == 'means':
                        # Copy means group and all its subkeys
                        means_group = root.create_group('means')
                        for mean_key in base_root['means'].keys():
                            data = base_root['means'][mean_key][:]
                            z_arr = means_group.create_array(
                                name=mean_key,
                                shape=data.shape,
                                chunks=base_root['means'][mean_key].chunks,
                                dtype=base_root['means'][mean_key].dtype
                            )
                            z_arr[:] = data
                        print(f"  Copied means group, containing {list(base_root['means'].keys())}")
                    else:
                        data = base_root[key][:]
                        # zarr v3's create_array does not support data parameter, need to create first then assign
                        # No compression
                        z_arr = root.create_array(
                            name=key,
                            shape=data.shape,
                            chunks=base_root[key].chunks,
                            dtype=base_root[key].dtype
                        )
                        z_arr[:] = data
            
            if base_path.suffix == '.zip':
                base_store.close()
            
        else:
            # DenseNet uses zarr directory format
            output_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}_decorr.zarr"
            output_path = Path(self.paths['final_output_dir']) / output_filename
            
            # Delete existing output
            if output_path.exists():
                shutil.rmtree(output_path)
                self.logger.warning(f"Deleted existing output: {output_path}")
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load base data path
            base_path = self._get_base_data_path()
            self.logger.info(f"Loading base data structure from {base_path}")
            
            # DenseNet: Copy base data directory
            if base_path.suffix == '.zip':
                # If base data is zip format, unzip first
                import zipfile
                with zipfile.ZipFile(base_path, 'r') as zip_ref:
                    zip_ref.extractall(output_path.parent)
                    extracted_dir = output_path.parent / base_path.stem
                    if extracted_dir.exists():
                        extracted_dir.rename(output_path)
            else:
                # Directly copy directory
                shutil.copytree(base_path, output_path)
                
                # Ensure means group is copied (if exists)
                if 'means' in base_root:
                    print(f"  Copied means group, containing {list(base_root['means'].keys())}")
            
            # Open zarr file
            store = zarr.storage.LocalStore(str(output_path))
            
            # Create root group
            root = zarr.group(store=store)
        
        # Delete existing attributions group (if exists)
        if 'attributions' in root:
            self.logger.info("Deleting existing attributions group in base data")
            del root['attributions']
        
        # Create new attributions group
        attr_group = root.create_group('attributions')
        
        # Save attribution data (without compression)
        for key, tensor in self.attributions_in_ram.items():
            self.logger.info(f"Saving attribution: {key}")
            
            tensor_np = tensor.numpy()
            z_attr = attr_group.create_array(
                name=key,
                shape=tensor_np.shape,
                chunks=(128, 3, 224, 224),  # Use 128 as chunk size
                dtype=tensor_np.dtype
            )
            z_attr[:] = tensor_np
            
            # Clean up memory
            del tensor_np
            gc.collect()
        
        # Add metadata
        root.attrs.update({
            'description': "Decorrelated attribution generation results (no compression)",
            'data_flag': self.data_flag,
            'model_flag': self.model_flag,
            'noise_name': self.noise_name,
            'split': self.split,
            'zarr_version': zarr.__version__,
            'compression': 'none'
        })
        
        if 'means' in root:
            root['means'].attrs['description'] = "Mean images calculated from the noisy dataset of this split."
        
        attr_group.attrs['description'] = "Attribution maps for various XAI methods (no compression)."
        
        # Close store
        store.close()
        
        self.logger.info(f"Successfully saved attribution results to: {output_path}")
    
    def _get_base_data_path(self) -> Path:
        """Get base data path"""
        if self.model_flag == 'densenet121':
            base_dir = Path(self.paths['base_data_dir'])
        elif self.model_flag == 'resnet18':
            base_dir = Path(self.paths['base_data_dir_resnet18'])
        else:
            raise ValueError(f"Unsupported model: {self.model_flag}")
        
        base_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}.zarr"
        base_path = base_dir / base_filename
        
        if not base_path.exists():
            zip_path = base_dir / f"{base_filename}.zip"
            if zip_path.exists():
                return zip_path
        
        return base_path

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Generate decorrelated attributions')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--data-flag', type=str, required=True, help='Dataset name')
    parser.add_argument('--model-flag', type=str, required=True, help='Model name')
    parser.add_argument('--noise-name', type=str, required=True, help='Noise condition')
    parser.add_argument('--split', type=str, required=True, help='Data split')
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator and run
    generator = AttributionGenerator(
        config=config,
        data_flag=args.data_flag,
        model_flag=args.model_flag,
        noise_name=args.noise_name,
        split=args.split,
        gpu_id=args.gpu_id
    )
    
    success = generator.run()
    
    if success:
        print("Attribution generation completed successfully")
        sys.exit(0)
    else:
        print("Attribution generation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()