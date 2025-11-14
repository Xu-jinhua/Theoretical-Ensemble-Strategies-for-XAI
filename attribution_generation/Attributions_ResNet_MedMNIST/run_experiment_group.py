import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from itertools import product
import gc
import time
from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import medmnist
from medmnist import INFO
from tqdm import tqdm
import numpy as np
import zarr
from zarr import codecs
from zarr.storage import ZipStore
import blosc
import zipfile
from datetime import datetime

from model_loader import load_medmnist_model
from captum.attr import (
    IntegratedGradients, GradientShap, Occlusion, Saliency, InputXGradient,
    GuidedBackprop, Deconvolution, DeepLift, DeepLiftShap, FeatureAblation
)


def setup_logger(name):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger("ExperimentGroupWorker")


class NoiseInjector:
    def __init__(self, device):
        self.device = device

    def add_noise(self, images: torch.Tensor, noise_type: str, params: dict) -> torch.Tensor:
        if noise_type == "gaussian":
            std = params.get('std', 0.15)
            return torch.clamp(images + torch.randn_like(images) * std, -1, 1)
        elif noise_type == "salt_pepper":
            amount = params.get('amount', 0.05)
            noisy_images = images.clone()
            num_pixels = images[0, 0].numel()
            num_salt = int(amount * num_pixels / 2)
            num_pepper = int(amount * num_pixels / 2)
            for i in range(images.size(0)):
                coords_salt = [torch.randint(0, dim, (num_salt,), device=self.device) for dim in images.shape[2:]]
                noisy_images[i, :, coords_salt[0], coords_salt[1]] = 1.0
                coords_pepper = [torch.randint(0, dim, (num_pepper,), device=self.device) for dim in images.shape[2:]]
                noisy_images[i, :, coords_pepper[0], coords_pepper[1]] = -1.0
            return noisy_images
        elif noise_type == "speckle":
            std = params.get('std', 0.15)
            return torch.clamp(images + images * (torch.randn_like(images) * std), -1, 1)
        elif noise_type == "none" or noise_type == "clean":
            return images
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

class ExperimentGroupWorker:
    
    def __init__(self, data_flag: str, model_flag: str, noise_name: str, split: str, gpu_id: int, 
                 config_path: Optional[str] = None, config: Optional[Dict] = None, subset_size: Optional[int] = None):
        self.data_flag = data_flag
        self.model_flag = model_flag
        self.noise_name = noise_name
        self.split = split
        self.gpu_id = gpu_id
        self.subset_size = subset_size

        if config:
            self.config = config
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("Must provide either a 'config' dictionary or a 'config_path'.")
            
        self.paths = self.config['paths']
        self.search_space = self.config['search_space']
        self.preflight_cfg = self.config['preflight_check']
        
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.noise_injector = NoiseInjector(self.device)
        self.model = None
        self.global_mean_baseline = None
        self.preflight_results = self._load_preflight_results()

        self.images_in_ram = None
        self.labels_in_ram = None
        self.preds_in_ram = None
        self.indices_in_ram = None
        self.mean_images_in_ram = {}
        self.attributions_in_ram = {}

    def _load_preflight_results(self):
        try:
            with open(self.preflight_cfg['results_path'], 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.critical(f"Pre-flight results not found at {self.preflight_cfg['results_path']}. Cannot proceed.")
            sys.exit(1)

    def run(self):
        
        task_name = f"{self.data_flag}_{self.model_flag}_{self.noise_name}_{self.split}"
        if self.subset_size:
            task_name += f"_subset{self.subset_size}"
        logger.info(f"--- Starting task group: {task_name} on GPU:{self.gpu_id} ---")
        
        try:
            self._prepare_data_in_memory()
            self._calculate_means_in_memory()
            self._generate_all_attributions()
            self._save_results_to_zarr()
            logger.info(f"--- Task group {task_name} completed successfully. ---")
        except Exception:
            logger.critical(f"Task group {task_name} failed!", exc_info=True)
            raise

    def _prepare_data_in_memory(self):
        logger.info("Step 1: Preparing data in memory...")
        
        self.model = load_medmnist_model(
            self.model_flag, self.data_flag, Path(self.paths['medmnist_weights_root']), self.device
        )

        
        info = INFO[self.data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        
        full_dataset = DataClass(
            split=self.split, transform=transform, download=True,
            size=224, as_rgb=True, root=Path(self.paths['medmnist_data_root'])
        )

        if self.subset_size and self.subset_size < len(full_dataset):
            logger.info(f"Using a subset of size {self.subset_size}")
            dataset = Subset(full_dataset, range(self.subset_size))
        else:
            dataset = full_dataset

        loader_batch_size = 512 if not self.subset_size else self.subset_size
        loader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=False, num_workers=4)

        
        all_images, all_labels, all_preds, all_indices = [], [], [], []
        noise_cfg = next(n for n in self.search_space['noise_conditions'] if n['name'] == self.noise_name)

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader, desc="Loading data & predicting")):
                images_gpu = images.to(self.device)
                
                noisy_images_gpu = self.noise_injector.add_noise(
                    images_gpu, noise_cfg['type'], noise_cfg['params']
                )
                
                preds_gpu = self.model(noisy_images_gpu)

                all_images.append(noisy_images_gpu.cpu())
                all_labels.append(labels.cpu())
                all_preds.append(preds_gpu.cpu())
                
                start_idx = i * loader_batch_size
                all_indices.append(torch.arange(start_idx, start_idx + len(images)))
        
        
        self.images_in_ram = torch.cat(all_images)
        self.labels_in_ram = torch.cat(all_labels).squeeze()
        self.preds_in_ram = torch.cat(all_preds)
        self.indices_in_ram = torch.cat(all_indices)
        logger.info(f"Data prepared in memory. Shape: {self.images_in_ram.shape}")

    def _calculate_means_in_memory(self):
        logger.info("Step 2: Calculating mean images...")
        
        self.mean_images_in_ram['overall'] = torch.mean(self.images_in_ram, dim=0)
        
        unique_labels = torch.unique(self.labels_in_ram)
        for label in tqdm(unique_labels, desc="Calculating per-class means"):
            label_int = label.item()
            mask = self.labels_in_ram == label_int
            class_images = self.images_in_ram[mask]
            if len(class_images) > 0:
                self.mean_images_in_ram[f'class_{label_int}'] = torch.mean(class_images, dim=0)
        logger.info(f"Calculated {len(self.mean_images_in_ram)} mean images.")

    def _generate_all_attributions(self):
        logger.info("Step 3: Generating all attributions...")
        self._prepare_baselines()
        
        num_samples = self.images_in_ram.shape[0]

        method_configs = []
        for method_template in self.search_space['attribution_methods']:
            params = method_template.get('params', {})
            param_keys = list(params.keys())
            param_values = [v if isinstance(v, list) else [v] for v in params.values()]
            for param_combo in product(*param_values):
                run_params = dict(zip(param_keys, param_combo))
                method_configs.append({
                    'name': method_template['name'],
                    'class': method_template['class'],
                    'params': run_params
                })

        for method_cfg in tqdm(method_configs, desc="Generating all attributions"):
            method_name = method_cfg['name']
            method_class_name = method_cfg['class']
            method_params = method_cfg['params']
            
            param_str = "_".join(sorted([f"{k}_{v}" for k, v in method_params.items()]))
            method_unique_key = f"{method_name}_{param_str}" if param_str else method_name
            
            batch_size = self.subset_size if self.subset_size else self.preflight_results.get(self.model_flag, {}).get(method_unique_key, 1)
            if not batch_size or batch_size == 0:
                logger.warning(f"Skipping {method_unique_key} due to batch size of 0.")
                continue

            explainer = globals()[method_class_name](self.model)
            
            attribution_results = torch.zeros_like(self.images_in_ram, dtype=torch.float16)

            for i in tqdm(range(0, num_samples, batch_size), desc=f"Method: {method_unique_key}", leave=False):
                start_idx, end_idx = i, min(i + batch_size, num_samples)
                
                
                batch_images = self.images_in_ram[start_idx:end_idx].to(self.device)
                batch_targets = torch.argmax(self.preds_in_ram[start_idx:end_idx], dim=1).to(self.device)
                
                attributions = self._get_attribution(explainer, method_params, batch_images, batch_targets)
                
                attribution_results[start_idx:end_idx] = attributions.detach().cpu().to(torch.float16)
            
            self.attributions_in_ram[method_unique_key] = attribution_results
        
        logger.info(f"Generated {len(self.attributions_in_ram)} sets of attributions.")
        gc.collect()
        torch.cuda.empty_cache()

    def _save_results_to_zarr(self):
        output_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}.zarr.zip"
        output_path = Path(self.paths['final_output_dir']) / output_filename
        
        logger.info(f"Step 4: Saving all results to Zarr store: {output_path}")

        
        num_cores = os.cpu_count()
        if num_cores:
            logger.info(f"Initializing Blosc compression with {num_cores} threads.")
            blosc.set_nthreads(num_cores)
        else:
            logger.warning("Could not determine CPU cores count. Blosc will use default settings.")
        
        if output_path.exists():
            
            os.remove(output_path)
            logger.warning(f"Removed existing Zarr zip file: {output_path}")

        
        blosc_compressor = codecs.Blosc(cname='zstd', clevel=6, shuffle=2)
        
        
        
        with ZipStore(str(output_path), mode='w') as store:
            root = zarr.group(store=store)

            
            BATCH_CHUNK_SIZE = 512

            chunk_shape_img = (BATCH_CHUNK_SIZE, 3, 224, 224)
            
            root.create_array('images', data=self.images_in_ram.numpy(), chunks=chunk_shape_img, compressors=[blosc_compressor])
            root.create_array('labels', data=self.labels_in_ram.numpy(), chunks=(BATCH_CHUNK_SIZE,), compressors=[blosc_compressor])
            root.create_array('model_predictions', data=self.preds_in_ram.numpy(), chunks=(BATCH_CHUNK_SIZE, self.preds_in_ram.shape[1]), compressors=[blosc_compressor])
            root.create_array('original_indices', data=self.indices_in_ram.numpy(), chunks=(BATCH_CHUNK_SIZE,), compressors=[blosc_compressor])

            means_group = root.create_group('means')
            for name, tensor in self.mean_images_in_ram.items():
                means_group.create_array(name, data=tensor.numpy(), compressors=[blosc_compressor])

            attr_group = root.create_group('attributions')
            for name, tensor in self.attributions_in_ram.items():
                attr_group.create_array(name, data=tensor.numpy(), chunks=chunk_shape_img, compressors=[blosc_compressor])

            root.attrs.update({
                'description': "Consolidated results for a single experiment group.",
                'data_flag': self.data_flag, 'model_flag': self.model_flag,
                'noise_name': self.noise_name, 'split': self.split
            })
            means_group.attrs['description'] = "Mean images calculated from the noisy dataset in this split."
            attr_group.attrs['description'] = "Attribution maps for various XAI methods."

        logger.info("Successfully saved results to Zarr zip store.")





    def _prepare_baselines(self):
        mean_path = Path(self.paths['means_dir']) / f"{self.data_flag}_mean.pt"
        if mean_path.exists():
            self.global_mean_baseline = torch.load(mean_path, map_location=self.device)
        else:
            logger.warning(f"Mean image for '{self.data_flag}' not found. 'global_mean' baseline will use a zero tensor.")
            self.global_mean_baseline = torch.zeros(3, 224, 224, device=self.device)

    def _get_attribution(self, explainer, params, images, targets):
        
        explainer_class = explainer.__class__
        kwargs = {'target': targets}
        baselines_val = params.get('baselines')
        
        if baselines_val:
            if baselines_val == 'zero':
                baselines = torch.zeros_like(images)
            elif baselines_val == 'global_mean':
                baselines = self.global_mean_baseline.expand_as(images)
            elif baselines_val == 'combined_zng':
                 baselines = torch.cat([
                    torch.zeros_like(images),
                    torch.randn_like(images),
                    self.global_mean_baseline.expand_as(images)
                ], dim=0)
            kwargs['baselines'] = baselines
        
        if explainer_class is Occlusion:
            patch_size = params.get('patch_size', 16)
            sliding_window_shapes = (3, patch_size * 2, patch_size * 2)
            kwargs['strides'] = (3, patch_size, patch_size)
            clean_params = {k: v for k, v in params.items() if k not in ['baselines', 'patch_size']}
            kwargs.update(clean_params)
            return explainer.attribute(images, sliding_window_shapes=sliding_window_shapes, **kwargs)

        if explainer_class is FeatureAblation:
            patch_size = params.get('patch_size', 16)
            num_patches = 224 // patch_size
            feature_mask = torch.arange(num_patches**2, device=self.device).view(num_patches, num_patches)
            feature_mask = feature_mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
            kwargs['feature_mask'] = feature_mask.expand(images.size(0), 3, -1, -1)
            clean_params = {k: v for k, v in params.items() if k not in ['baselines', 'patch_size']}
            kwargs.update(clean_params)
            return explainer.attribute(images, **kwargs)
        
        clean_params = {k: v for k, v in params.items() if k not in ['baselines']}
        kwargs.update(clean_params)
        return explainer.attribute(images, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Experiment Group Worker.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main config file.')
    parser.add_argument('--data-flag', type=str, required=True)
    parser.add_argument('--model-flag', type=str, required=True)
    parser.add_argument('--noise-name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--gpu-id', type=int, required=True)
    args = parser.parse_args()

    try:
        
        worker = ExperimentGroupWorker(
            config_path=args.config,
            data_flag=args.data_flag,
            model_flag=args.model_flag,
            noise_name=args.noise_name,
            split=args.split,
            gpu_id=args.gpu_id
        )
        worker.run()
    except Exception:
        sys.exit(2)