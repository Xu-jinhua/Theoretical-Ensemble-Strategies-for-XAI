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
from typing import Dict, Any

import torch
from tqdm import tqdm
import numpy as np
import zarr

# Import new DenseNet loader
from model_loader_densenet import load_densenet_model

# Import from Captum
from captum.attr import (
    IntegratedGradients, GradientShap, Occlusion, Saliency, InputXGradient,
    GuidedBackprop, Deconvolution, DeepLift, DeepLiftShap, FeatureAblation
)

# --- Logging Setup ---
def setup_logger(name):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] (DenseNetGen) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger("DenseNetExplanationGenerator")


class NoiseInjector:
    """A simple class to inject different types of noise into image tensors."""
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

class AttributionGenerator:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.global_mean_baseline = None # Will be calculated at runtime

    def prepare_baselines(self, overall_mean_tensor):
        """
        Prepare baselines needed by Captum.
        Modification: No longer load from file, but use the mean calculated from current noise level.
        """
        self.global_mean_baseline = overall_mean_tensor.to(self.device)
        logger.info("Baseline set from 'overall_mean' of current noisy data.")

    def _get_attribution(self, explainer, params, images, targets):
        """Core logic for calling Captum methods."""
        explainer_class = explainer.__class__
        kwargs = {'target': targets}
        baselines_val = params.get('baselines')
        
        if baselines_val:
            if baselines_val == 'zero':
                baselines = torch.zeros_like(images)
            elif baselines_val == 'global_mean':
                if self.global_mean_baseline is None:
                    raise RuntimeError("global_mean baseline not set!")
                baselines = self.global_mean_baseline.expand_as(images)
            elif baselines_val == 'combined_zng':
                 if self.global_mean_baseline is None:
                    raise RuntimeError("global_mean baseline not set!")
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

    def get_batch_size(self, method_key: str) -> int:
        """Get batch size from configuration"""
        batch_sizes_cfg = self.config['settings']['batch_sizes']
        return batch_sizes_cfg.get(method_key, batch_sizes_cfg['default'])


def main(config_path: str):
    

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    paths = config['paths']
    settings = config['settings']
    model_flag = settings['model_flag']
    
    device = torch.device(f"cuda:{settings['gpu_id']}")
    noise_injector = NoiseInjector(device)


    logger.info(f"Loading model: {model_flag}")
    model = load_densenet_model(
        model_flag, 
        Path(paths['imagenet_weights_root']), 
        device
    )
    model = torch.compile(model) # Compile to accelerate
    

    logger.info(f" Zarr Load: {paths['input_zarr_path']}")
    zarr_root = None
    try:
        zarr_root = zarr.open(str(paths['input_zarr_path']), mode='r')
        images_in_ram = torch.from_numpy(zarr_root['images'][:])
        labels_in_ram = torch.from_numpy(zarr_root['labels'][:]).squeeze()
        num_samples = len(images_in_ram)
        logger.info(f"Successfully loaded {num_samples} images and labels into memory.")
    except Exception as e:
        logger.critical(f"Failed to load Zarr file: {e}", exc_info=True)
        return
    finally:
        if zarr_root:
            zarr_root.store.close()

    attr_gen = AttributionGenerator(model, config, device)

    for noise_cfg in tqdm(config['noise_conditions'], desc="Processing noise conditions"):
        noise_name = noise_cfg['name']
        logger.info(f"--- Starting to process noise: {noise_name} ---")
        

        output_dir = Path(paths['output_root_dir']) / f"model-{model_flag}_noise-{noise_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Applying noise...")
        noisy_images_cpu = noise_injector.add_noise(
            images_in_ram, noise_cfg['type'], noise_cfg['params']
        )
        
        logger.info("Generating new predictions for DenseNet...")
        preds_list = []

        inference_batch_size = settings['batch_sizes']['default']
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, inference_batch_size), desc="Inference"):
                batch_images = noisy_images_cpu[i:i+inference_batch_size].to(device)
                with torch.amp.autocast('cuda'):
                    preds_gpu = model(batch_images)
                preds_list.append(preds_gpu.cpu())
        
        all_preds_cpu = torch.cat(preds_list)


        logger.info("Calculating mean images...")
        mean_images_in_ram = {}
        mean_images_in_ram['overall'] = torch.mean(noisy_images_cpu, dim=0)
        
        unique_labels = torch.unique(labels_in_ram)
        for label in tqdm(unique_labels, desc="Computing class means", leave=False):
            label_int = label.item()
            mask = labels_in_ram == label_int
            class_images = noisy_images_cpu[mask]
            if len(class_images) > 0:
                mean_images_in_ram[f'class_{label_int}'] = torch.mean(class_images, dim=0)

   
        logger.info("Saving base .pt files (images, labels, predictions)...")
        torch.save(noisy_images_cpu, output_dir / "images.pt")
        torch.save(labels_in_ram, output_dir / "labels.pt")
        torch.save(all_preds_cpu, output_dir / "model_predictions.pt")
        
        means_dir = output_dir / "means"
        means_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor in mean_images_in_ram.items():
            torch.save(tensor, means_dir / f"{name}.pt")
        logger.info("Mean files saved successfully.")


        attr_gen.prepare_baselines(mean_images_in_ram['overall'])


        attr_output_dir = output_dir / "attributions"
        attr_output_dir.mkdir(parents=True, exist_ok=True)

        method_configs = []
        for method_template in config['attribution_methods']:
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
        
        logger.info(f"Starting to generate {len(method_configs)} attributions for {noise_name}...")
        for method_cfg in tqdm(method_configs, desc="Generating attributions"):
            method_name = method_cfg['name']
            method_class_name = method_cfg['class']
            method_params = method_cfg['params']
            
            param_str = "_".join(sorted([f"{k}_{v}" for k, v in method_params.items()]))
            method_unique_key = f"{method_name}_{param_str}" if param_str else method_name
            
            output_file = attr_output_dir / f"{method_unique_key}.pt"
            if output_file.exists():
                logger.debug(f"Skipping existing: {output_file.name}")
                continue

            batch_size = attr_gen.get_batch_size(method_unique_key)
            explainer = globals()[method_class_name](model)
            
            attribution_results = []
            
            try:
                for i in tqdm(range(0, num_samples, batch_size), desc=f"Method: {method_unique_key}", leave=False):
                    start_idx, end_idx = i, min(i + batch_size, num_samples)
                    
                    batch_images = noisy_images_cpu[start_idx:end_idx].to(device)
                    batch_targets = torch.argmax(all_preds_cpu[start_idx:end_idx], dim=1).to(device)
                    
                    attributions = attr_gen._get_attribution(explainer, method_params, batch_images, batch_targets)
                    attribution_results.append(attributions.detach().cpu().to(torch.float16))
                
                # Concatenate and save
                final_attr_tensor = torch.cat(attribution_results)
                torch.save(final_attr_tensor, output_file)
                
            except Exception as e:
                logger.error(f"Failed to generate {method_unique_key}: {e}", exc_info=True)
                if output_file.exists():
                    os.remove(output_file) # Remove incomplete file
            
            gc.collect()
            torch.cuda.empty_cache()
            
        logger.info(f"--- Finished processing noise: {noise_name} ---")

    logger.info("--- All tasks completed ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseNet XAI Explanation Generator.")
    parser.add_argument('--config', type=str, default="config_densenet.yml", 
                        help='Path to the DenseNet config file.')
    args = parser.parse_args()
    
    main(config_path=args.config)