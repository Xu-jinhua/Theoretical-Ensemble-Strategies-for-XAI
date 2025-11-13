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
from torch.utils.data import DataLoader, Subset, Dataset # [新] 导入 Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np 
import zarr
import blosc
import shutil

# Local imports
from model_loader import load_medmnist_model 
from captum.attr import (
    IntegratedGradients, GradientShap, Occlusion, Saliency, InputXGradient,
    GuidedBackprop, Deconvolution, DeepLift, DeepLiftShap, FeatureAblation
)

# --- Logging Setup (保持不变) ---
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

# --- NoiseInjector (保持不变) ---
#
class NoiseInjector:
    def __init__(self, device):
        self.device = device
    def add_noise(self, images: torch.Tensor, noise_type: str, params: dict) -> torch.Tensor:
        # ... (所有噪声逻辑保持不变) ...
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



class MedMNISTDatasetNPZ(Dataset):
    def __init__(self, npz_path: Path, split: str):
        try:
            data = np.load(npz_path)
        except FileNotFoundError:
            logger.error(f"数据文件未找到: {npz_path}")
            raise
        
        try:
            self.images = data[f'{split}_images']
            self.labels = data[f'{split}_labels']
        except KeyError:
            logger.critical(f"在 {npz_path} 中未找到 '{split}_images' 或 '{split}_labels'。")
            raise

        self.images = self.images.astype(np.float32) / 255.0
        
        if self.images.ndim == 4:
            self.images = np.transpose(self.images, (0, 3, 1, 2))
        elif self.images.ndim == 3:
            self.images = np.expand_dims(self.images, axis=1)
            self.images = self.images.repeat(3, axis=1)
        
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"成功加载 {split} 数据: {self.images.shape[0]} 个样本 (形状: {self.images.shape})")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        label = torch.tensor(self.labels[index], dtype=torch.long).squeeze()
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
# --- [新] 数据集类结束 ---


class ExperimentGroupWorker:
    # ... (init, _load_preflight_results 保持不变) ...
    #
    def __init__(self, data_flag: str, model_flag: str, noise_name: str, split: str, gpu_id: int, 
                 config_path: Optional[str] = None, config: Optional[Dict] = None):
        self.data_flag = data_flag
        self.model_flag = model_flag
        self.noise_name = noise_name
        self.split = split
        self.gpu_id = gpu_id
        
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

        # Data holders
        self.images_in_ram = None
        self.labels_in_ram = None
        self.preds_in_ram = None
        self.indices_in_ram = None
        self.mean_images_in_ram = {}
        self.attributions_in_ram = {}

    def _load_preflight_results(self):
        preflight_results_path = self.preflight_cfg.get('results_path', './preflight_results_from_MedMNIST.json')
        logger.info(f"成功从 {preflight_results_path} 加载预检结果")
        try:
            with open(preflight_results_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.critical(f"预检结果文件未找到: {preflight_results_path}。")
            logger.critical(f"请确保该文件存在，并且已包含 'densenet121' 的条目。")
            sys.exit(1)

    def run(self):
        """Main execution workflow for the task."""
        task_name = f"{self.data_flag}_{self.model_flag}_{self.noise_name}_{self.split}"
        logger.info(f"--- 开始任务组: {task_name} on GPU:{self.gpu_id} ---")
        
        try:
            self._prepare_data_in_memory()
            self._calculate_means_in_memory()
            self._generate_all_attributions()
            self._save_results_to_zarr()
            logger.info(f"--- 任务组 {task_name} 已成功完成。 ---")
        except Exception:
            logger.critical(f"任务组 {task_name} 失败！", exc_info=True)
            raise

    def _prepare_data_in_memory(self):
        """
        [重大修改] 
        此函数现在从 .npz 文件加载数据。
        """
        logger.info("步骤 1: 在内存中准备数据...")
        

        self.model = load_medmnist_model(
            self.model_flag, self.data_flag, Path(self.paths['medmnist_weights_root']), self.device
        )

        npz_path = Path(self.paths['medmnist_data_root']) / f"{self.data_flag}_224.npz"
        logger.info(f"从 {npz_path} 加载 MedMNIST .npz 数据")
        
        try:
            full_dataset = MedMNISTDatasetNPZ(npz_path, self.split)
        except FileNotFoundError:
            logger.critical(f"数据文件未找到: {npz_path}")
            sys.exit(1)
        except KeyError:
            logger.critical(f"在 {npz_path} 中未找到 '{self.split}' 分割。")
            sys.exit(1)
        
        logger.info(f"将完整的 {self.data_flag} {self.split} 集 ({len(full_dataset)} 张图像) 加载到 RAM...")
        
        loader_batch_size = 1024 
        loader = DataLoader(full_dataset, batch_size=loader_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        all_images, all_labels, all_preds, all_indices = [], [], [], []
        noise_cfg = next(n for n in self.search_space['noise_conditions'] if n['name'] == self.noise_name)

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader, desc="加载数据并预测")):
                images_gpu = images.to(self.device, non_blocking=True)
                
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
        logger.info(f"数据已在内存中准备就绪。形状: {self.images_in_ram.shape}")

    
    def _calculate_means_in_memory(self):
        logger.info("步骤 2: 计算平均图像...")
        self.mean_images_in_ram['overall'] = torch.mean(self.images_in_ram, dim=0)
        unique_labels = torch.unique(self.labels_in_ram)
        for label in tqdm(unique_labels, desc="计算每类平均值"):
            label_int = label.item()
            mask = self.labels_in_ram == label_int
            class_images = self.images_in_ram[mask]
            if len(class_images) > 0:
                self.mean_images_in_ram[f'class_{label_int}'] = torch.mean(class_images, dim=0)
        logger.info(f"已计算 {len(self.mean_images_in_ram)} 张平均图像。")

    def _generate_all_attributions(self):
        logger.info("步骤 3: 生成所有归因...")
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
        for method_cfg in tqdm(method_configs, desc="生成所有归因"):
            method_name = method_cfg['name']
            method_class_name = method_cfg['class']
            method_params = method_cfg['params']
            param_str = "_".join(sorted([f"{k}_{v}" for k, v in method_params.items()]))
            method_unique_key = f"{method_name}_{param_str}" if param_str else method_name
            
            batch_size = self.preflight_results.get(self.model_flag, {}).get(method_unique_key, 64)
            if 'IntegratedGradients' in method_unique_key or 'Shap' in method_unique_key:
                batch_size = self.preflight_results.get(self.model_flag, {}).get(method_unique_key, 8)
            if not batch_size or batch_size == 0:
                logger.warning(f"由于批次大小为 0，跳过 {method_unique_key}。")
                continue

            explainer = globals()[method_class_name](self.model)
            attribution_results = torch.zeros_like(self.images_in_ram, dtype=torch.float16) 
            for i in tqdm(range(0, num_samples, batch_size), desc=f"方法: {method_unique_key}", leave=False):
                start_idx, end_idx = i, min(i + batch_size, num_samples)
                batch_images = self.images_in_ram[start_idx:end_idx].to(self.device)
                batch_targets = torch.argmax(self.preds_in_ram[start_idx:end_idx], dim=1).to(self.device)
                attributions = self._get_attribution(explainer, method_params, batch_images, batch_targets)
                attribution_results[start_idx:end_idx] = attributions.detach().cpu().to(torch.float16)
            self.attributions_in_ram[method_unique_key] = attribution_results
        logger.info(f"已生成 {len(self.attributions_in_ram)} 组归因。")
        gc.collect()
        torch.cuda.empty_cache()

    def _save_results_to_zarr(self):
        output_filename = f"dataset-{self.data_flag}_model-{self.model_flag}_noise-{self.noise_name}_split-{self.split}.zarr"
        output_path = Path(self.paths['final_output_dir']) / output_filename
        
        logger.info(f"步骤 4: 将所有结果(无压缩)保存到 Zarr 目录: {output_path}")

        if output_path.exists():
            shutil.rmtree(output_path)
            logger.warning(f"已删除已存在的 Zarr 目录: {output_path}")

        
        root = zarr.open(str(output_path), mode='w')
        chunk_size = 128
        
        try:
            logger.debug("正在保存 images...")
            images_np = self.images_in_ram.numpy()
            z_images = root.create_array(
                'images', 
                shape=images_np.shape, 
                chunks=(chunk_size, 3, 224, 224), 
                dtype=images_np.dtype
            )
            z_images[:] = images_np

            # 保存 labels
            logger.debug("正在保存 labels...")
            labels_np = self.labels_in_ram.numpy()
            z_labels = root.create_array(
                'labels', 
                shape=labels_np.shape, 
                chunks=(chunk_size * 16,), 
                dtype=labels_np.dtype
            )
            z_labels[:] = labels_np

            # 保存 model_predictions
            logger.debug("正在保存 model_predictions...")
            preds_np = self.preds_in_ram.numpy()
            z_preds = root.create_array(
                'model_predictions', 
                shape=preds_np.shape, 
                chunks=(chunk_size, self.preds_in_ram.shape[1]), 
                dtype=preds_np.dtype
            )
            z_preds[:] = preds_np

            # 保存 original_indices
            logger.debug("正在保存 original_indices...")
            indices_np = self.indices_in_ram.numpy()
            z_indices = root.create_array(
                'original_indices', 
                shape=indices_np.shape, 
                chunks=(chunk_size * 16,), 
                dtype=indices_np.dtype
            )
            z_indices[:] = indices_np

            # 保存 means
            logger.debug("正在保存 means...")
            means_group = root.create_group('means')
            for name, tensor in self.mean_images_in_ram.items():
                tensor_np = tensor.numpy()
                z_mean = means_group.create_array(
                    name, 
                    shape=tensor_np.shape, 
                    dtype=tensor_np.dtype
                )
                z_mean[:] = tensor_np

            # 保存 attributions
            logger.debug("正在保存 attributions...")
            attr_group = root.create_group('attributions')
            for name, tensor in self.attributions_in_ram.items():
                tensor_np = tensor.numpy()
                z_attr = attr_group.create_array(
                    name, 
                    shape=tensor_np.shape, 
                    chunks=(chunk_size, 3, 224, 224), 
                    dtype=tensor_np.dtype
                )
                z_attr[:] = tensor_np
            
            root.attrs.update({
                'description': "一个实验组的整合结果。",
                'data_flag': self.data_flag, 'model_flag': self.model_flag,
                'noise_name': self.noise_name, 'split': self.split,
                'zarr_version': zarr.__version__
            })
            means_group.attrs['description'] = "从此 split 中的带噪数据集计算出的平均图像。"
            attr_group.attrs['description'] = "各种 XAI 方法的归因图。"

            logger.info("成功将结果(无压缩)保存到 Zarr 目录。")
        
        except Exception as e:
            logger.critical(f"保存 Zarr 期间发生致命错误: {e}", exc_info=True)
            if output_path.exists():
                shutil.rmtree(output_path)
                logger.warning(f"已删除不完整的 Zarr 目录: {output_path}")
            raise e

    def _prepare_baselines(self):
        mean_path = Path(self.paths['means_dir']) / f"{self.data_flag}_mean.pt"
        if not mean_path.exists():
             logger.critical(f"平均图像 '{mean_path}' 未找到！请先运行 calculate_means.py。")
             sys.exit(1)
        self.global_mean_baseline = torch.load(mean_path, map_location=self.device)
        logger.info(f"成功加载全局平均基线: {mean_path}")

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
                 if 'z' in baselines_val:
                     baselines_list.append(torch.zeros_like(images))
                 if 'n' in baselines_val:
                     baselines_list.append(torch.randn_like(images))
                 if 'g' in baselines_val:
                     baselines_list.append(self.global_mean_baseline.expand_as(images))
                 baselines = torch.cat(baselines_list, dim=0)
            kwargs['baselines'] = baselines
        if explainer_class is Occlusion:
            patch_size = params.get('patch_size', 16)
            sliding_window_shapes = (3, patch_size, patch_size)
            kwargs['sliding_window_shapes'] = sliding_window_shapes
            clean_params = {k: v for k, v in params.items() if k not in ['baselines', 'patch_size']}
            kwargs.update(clean_params)
            return explainer.attribute(images, **kwargs)
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