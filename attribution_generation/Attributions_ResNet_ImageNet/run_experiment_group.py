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

from torch import autocast

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import zarr
import blosc
import shutil

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
                 config_path: str):
        self.data_flag = data_flag
        self.model_flag = model_flag
        self.noise_name = noise_name
        self.split = split
        self.gpu_id = gpu_id
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
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

    def _load_preflight_results(self):
        preflight_results_path = self.preflight_cfg.get('results_path', './preflight_results.json')
        logger.info(f"Successfully loaded preflight results from {preflight_results_path}")
        try:
            with open(preflight_results_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.critical(f"Preflight results file not found: {preflight_results_path}. Please run preflight_checker.py first.")
            sys.exit(1)

    def run(self):
        task_name = f"{self.data_flag}_{self.model_flag}_{self.noise_name}_{self.split}"
        logger.info(f"--- Starting task group: {task_name} on GPU:{self.gpu_id} ---")
        
        try:
            self._prepare_data_in_memory()
            zarr_root = self._initialize_zarr_store_and_save_base_data()
            self._generate_and_save_attributions_incrementally(zarr_root)
            logger.info(f"--- Task group {task_name} completed successfully. ---")
        except Exception:
            logger.critical(f"Task group {task_name} failed!", exc_info=True)
            raise

    def _prepare_data_in_memory(self):
        if self.images_in_ram is not None:
            logger.info("Data already in memory, skipping preparation step.")
            return

        logger.info("Step 1: Preparing data in memory...")
        self.model = load_medmnist_model(
            self.model_flag, self.data_flag, Path(self.paths['medmnist_weights_root']), self.device
        )
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset_path = Path(self.paths['medmnist_data_root']) / self.split
        full_dataset = ImageFolder(root=dataset_path, transform=transform)
        
        loader = DataLoader(full_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
        all_images, all_labels, all_preds, all_indices = [], [], [], []
        noise_cfg = next(n for n in self.search_space['noise_conditions'] if n['name'] == self.noise_name)

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader, desc="Loading data and predicting")):
                images_gpu = images.to(self.device, non_blocking=True)
                noisy_images_gpu = self.noise_injector.add_noise(images_gpu, noise_cfg['type'], noise_cfg['params'])
                
                with autocast("cuda", dtype=torch.float16):
                    preds_gpu = self.model(noisy_images_gpu)

                all_images.append(noisy_images_gpu.cpu())
                all_labels.append(labels.cpu())
                all_preds.append(preds_gpu.cpu())
                start_idx = i * 256
                all_indices.append(torch.arange(start_idx, start_idx + len(images)))
        
        self.images_in_ram = torch.cat(all_images)
        self.labels_in_ram = torch.cat(all_labels).squeeze()
        self.preds_in_ram = torch.cat(all_preds)
        self.indices_in_ram = torch.cat(all_indices)
        logger.info(f"Data prepared in memory. Shape: {self.images_in_ram.shape}")

    def _initialize_zarr_store_and_save_base_data(self):
        output_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}.zarr"
        output_path = Path(self.paths['final_output_dir']) / output_filename
        
        logger.info(f"Initializing Zarr directory: {output_path}")

        if not output_path.exists():
            root = zarr.open(str(output_path), mode='w')
            chunk_size = 128
            
            logger.info("Saving base data (images, labels, predictions, means)...")
            
            images_np = self.images_in_ram.numpy()
            z_images = root.create_array('images', shape=images_np.shape, chunks=(chunk_size, 3, 224, 224), dtype=images_np.dtype)
            z_images[:] = images_np

            labels_np = self.labels_in_ram.numpy()
            z_labels = root.create_array('labels', shape=labels_np.shape, chunks=(chunk_size * 16,), dtype=labels_np.dtype)
            z_labels[:] = labels_np

            preds_np = self.preds_in_ram.numpy()
            z_preds = root.create_array('model_predictions', shape=preds_np.shape, chunks=(chunk_size, preds_np.shape[1]), dtype=preds_np.dtype)
            z_preds[:] = preds_np

            indices_np = self.indices_in_ram.numpy()
            z_indices = root.create_array('original_indices', shape=indices_np.shape, chunks=(chunk_size * 16,), dtype=indices_np.dtype)
            z_indices[:] = indices_np

            mean_images_in_ram = {}
            mean_images_in_ram['overall'] = torch.mean(self.images_in_ram, dim=0)
            unique_labels = torch.unique(self.labels_in_ram)
            for label in tqdm(unique_labels, desc="Calculating and saving per-class means"):
                mask = self.labels_in_ram == label.item()
                mean_images_in_ram[f'class_{label.item()}'] = torch.mean(self.images_in_ram[mask], dim=0)
            
            means_group = root.create_group('means')
            for name, tensor in mean_images_in_ram.items():
                tensor_np = tensor.numpy()
                z_mean = means_group.create_array(name, shape=tensor_np.shape, dtype=tensor_np.dtype)
                z_mean[:] = tensor_np
            
            root.create_group('attributions')

            root.attrs.update({
                'description': "Consolidated results of an experiment group.",
                'data_flag': self.data_flag, 'model_flag': self.model_flag,
                'noise_name': self.noise_name, 'split': self.split,
                'zarr_version': zarr.__version__
            })
            logger.info("Base data saved successfully.")
            return root
        else:
            logger.info("Found existing Zarr directory, opening in append mode.")
            return zarr.open(str(output_path), mode='a')

    def _generate_and_save_attributions_incrementally(self, zarr_root: zarr.Group):
        logger.info("Step 2: Generating and saving attributions incrementally...")
        self._prepare_baselines()
        
        attr_group = zarr_root.require_group('attributions')
        completed_attributions = set(attr_group.keys())
        logger.info(f"Completed attributions ({len(completed_attributions)}): {completed_attributions}")

        method_configs = self._get_method_configs()

        for method_cfg in tqdm(method_configs, desc="Processing all attribution methods"):
            method_unique_key = self._get_method_unique_key(method_cfg)
            
            if method_unique_key in completed_attributions:
                logger.info(f"Skipping existing attribution: {method_unique_key}")
                continue

            logger.info(f"Starting to compute attribution: {method_unique_key}")
            batch_size = self._get_batch_size(method_unique_key)
            if not batch_size:
                logger.warning(f"Skipping {method_unique_key} due to batch size of 0.")
                continue

            explainer = globals()[method_cfg['class']](self.model)
            num_samples = self.images_in_ram.shape[0]
            attribution_results = torch.zeros_like(self.images_in_ram, dtype=torch.float16)

            for i in tqdm(range(0, num_samples, batch_size), desc=f"Computing batches", leave=False):
                start_idx, end_idx = i, min(i + batch_size, num_samples)
                batch_images = self.images_in_ram[start_idx:end_idx].to(self.device)
                batch_targets = torch.argmax(self.preds_in_ram[start_idx:end_idx], dim=1).to(self.device)
                
                attributions = self._get_attribution(explainer, method_cfg['params'], batch_images, batch_targets)
                attribution_results[start_idx:end_idx] = attributions.detach().cpu().to(torch.float16)
            
            logger.info(f"Saving attribution: {method_unique_key}...")
            
            attr_np = attribution_results.numpy()
            z_attr = attr_group.create_array(
                method_unique_key, 
                shape=attr_np.shape, 
                chunks=(128, 3, 224, 224), 
                dtype=attr_np.dtype,
                overwrite=True
            )
            z_attr[:] = attr_np
            logger.info(f"Successfully saved {method_unique_key}.")
        
        logger.info("All attributions generated and saved.")
        gc.collect()
        torch.cuda.empty_cache()

    def _get_method_configs(self):
        method_configs = []
        for method_template in self.search_space['attribution_methods']:
            params = method_template.get('params', {})
            param_keys = list(params.keys())
            param_values = [v if isinstance(v, list) else [v] for v in params.values()]
            for param_combo in product(*param_values):
                run_params = dict(zip(param_keys, param_combo))
                method_configs.append({'name': method_template['name'], 'class': method_template['class'], 'params': run_params})
        return method_configs

    def _get_method_unique_key(self, method_cfg):
        param_str = "_".join(sorted([f"{k}_{v}" for k, v in method_cfg['params'].items()]))
        return f"{method_cfg['name']}_{param_str}" if param_str else method_cfg['name']

    def _get_batch_size(self, method_unique_key):
        batch_size = self.preflight_results.get(self.model_flag, {}).get(method_unique_key, 64)
        if 'IntegratedGradients' in method_unique_key or 'Shap' in method_unique_key:
            batch_size = self.preflight_results.get(self.model_flag, {}).get(method_unique_key, 8)
        return batch_size
        
    def _prepare_baselines(self):
        mean_path = Path(self.paths['means_dir']) / f"{self.data_flag}_mean.pt"
        if not mean_path.exists():
             logger.critical(f"Mean image '{mean_path}' not found! Please run calculate_means.py first.")
             sys.exit(1)
        self.global_mean_baseline = torch.load(mean_path, map_location=self.device)
        logger.info(f"Successfully loaded global mean baseline: {mean_path}")

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
                 baselines_list = []
                 if 'z' in baselines_val: baselines_list.append(torch.zeros_like(images))
                 if 'n' in baselines_val: baselines_list.append(torch.randn_like(images))
                 if 'g' in baselines_val: baselines_list.append(self.global_mean_baseline.expand_as(images))
                 baselines = torch.cat(baselines_list, dim=0)
            kwargs['baselines'] = baselines
        
        clean_params = {k: v for k, v in params.items() if k != 'baselines'}

        if explainer_class is Occlusion:
            patch_size = clean_params.pop('patch_size', 16)
            kwargs['sliding_window_shapes'] = (3, patch_size, patch_size)
        elif explainer_class is FeatureAblation:
            patch_size = clean_params.pop('patch_size', 16)
            num_patches = 224 // patch_size
            feature_mask = torch.arange(num_patches**2, device=self.device).view(num_patches, num_patches)
            feature_mask = feature_mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
            kwargs['feature_mask'] = feature_mask.expand(images.size(0), 3, -1, -1)
        
        kwargs.update(clean_params)

        with autocast("cuda", dtype=torch.float16):
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