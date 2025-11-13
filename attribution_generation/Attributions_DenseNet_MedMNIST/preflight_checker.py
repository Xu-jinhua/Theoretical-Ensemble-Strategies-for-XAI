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

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder # <--- 导入 ImageFolder

from model_loader import load_medmnist_model
from captum.attr import (
    IntegratedGradients, GradientShap, Occlusion, Saliency, InputXGradient,
    GuidedBackprop, Deconvolution, DeepLift, DeepLiftShap, FeatureAblation
)

def setup_logging(log_dir: Path, filename: str, logger_name: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / filename
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] (PreflightChecker) %(message)s"))
    logger.addHandler(console_handler)
    
    return logger

class BatchSizeFinder:
    def __init__(self, config_path: str, gpu_id: int, logger: logging.Logger):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.logger = logger
        
        self.paths = self.config['paths']
        self.preflight_cfg = self.config['preflight_check']
        self.search_space = self.config['search_space']

    def _setup_dataset_and_model(self, model_flag: str, data_flag: str):
        self.logger.info(f"为预检设置数据集 '{data_flag}' 和模型 '{model_flag}'...")
        
        model = load_medmnist_model(
            model_flag, data_flag, Path(self.paths['medmnist_weights_root']), self.device
        )
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset_path = Path(self.paths['medmnist_data_root']) / 'val'
        if not dataset_path.exists():
             self.logger.critical(f"预检所需的数据集路径未找到: {dataset_path}")
             sys.exit(1)

        full_dataset = ImageFolder(root=dataset_path, transform=transform)
        
        subset_size = self.preflight_cfg['subset_size']
        if len(full_dataset) < subset_size:
            self.logger.warning(f"完整数据集大小 ({len(full_dataset)}) 小于 subset_size ({subset_size})。将使用完整数据集进行预检。")
            subset_indices = range(len(full_dataset))
        else:
            subset_indices = range(subset_size)
            
        dataset = Subset(full_dataset, subset_indices)
        
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        images, labels = next(iter(loader))
        images = images.to(self.device)
        
        return model, images

    def _get_attribution(self, explainer, params, images, targets):
        explainer_class = explainer.__class__
        kwargs = {'target': targets}
        baselines_val = params.get('baselines')
        
        if baselines_val:
            # !!! 核心修正 !!!
            # SHAP 类方法需要一个基线分布 (样本数 > 1)。
            if explainer_class in (DeepLiftShap, GradientShap):
                # 创建一个包含 3 个随机样本的小型基线分布。
                # 这能确保即使在测试批次大小为 1 时，断言也能通过。
                num_baseline_samples = 3 
                baselines = torch.randn(num_baseline_samples, *images.shape[1:], device=images.device)
            else:
                # 其他方法 (如 DeepLIFT) 只需要一个基线样本即可。
                # Captum 会自动将其扩展到整个批次。
                baselines = torch.randn_like(images[0:1]) 
            
            kwargs['baselines'] = baselines
        
        if explainer_class is Occlusion:
            patch_size = params.get('patch_size', 16)
            sliding_window_shapes = (3, patch_size * 2, patch_size * 2)
            kwargs['strides'] = (3, patch_size, patch_size)
            return explainer.attribute(images, sliding_window_shapes=sliding_window_shapes, **kwargs)

        if explainer_class is FeatureAblation:
            patch_size = params.get('patch_size', 16)
            num_patches = 224 // patch_size
            feature_mask = torch.arange(num_patches**2, device=self.device).view(num_patches, num_patches)
            feature_mask = feature_mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
            kwargs['feature_mask'] = feature_mask.expand(images.size(0), 3, -1, -1)
            return explainer.attribute(images, **kwargs)
        
        clean_params = {k: v for k, v in params.items() if k not in ['baselines', 'patch_size']}
        kwargs.update(clean_params)
        return explainer.attribute(images, **kwargs)

    def _test_batch_size(self, explainer, params, all_images, batch_size):
        try:
            targets = torch.randint(0, 1000, (batch_size,), device=self.device)
            _ = self._get_attribution(explainer, params, all_images[:batch_size], targets)
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            else:
                self.logger.error(f"在批次大小 {batch_size} 时发生非 OOM 的 RuntimeError: {e}", exc_info=True)
                return False
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def run(self):
        self.logger.info("--- 开始预检: 批次大小查找器 ---")
        final_results = {}
        
        data_flag = self.search_space['data_flags'][0]

        for model_flag in self.search_space['model_flags']:
            final_results[model_flag] = {}
            model, images = self._setup_dataset_and_model(model_flag, data_flag)
            
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
            
            pbar = tqdm(method_configs, desc=f"测试模型 {model_flag}")
            for method_cfg in pbar:
                method_name = method_cfg['name']
                method_class_name = method_cfg['class']
                method_params = method_cfg['params']

                param_str = "_".join(sorted([f"{k}_{v}" for k, v in method_params.items()]))
                method_unique_key = f"{method_name}_{param_str}" if param_str else method_name
                
                pbar.set_postfix_str(method_unique_key)
                
                if method_name in self.preflight_cfg.get('exempt_methods', {}):
                    exempt_bs = self.preflight_cfg['exempt_methods'][method_name]
                    self.logger.info(f"'{method_unique_key}' 是豁免项。设置批次大小为 {exempt_bs}。")
                    final_results[model_flag][method_unique_key] = exempt_bs
                    continue

                explainer = globals()[method_class_name](model)
                
                low = 1
                high = self.preflight_cfg['start_batch_size']
                safe_bs = 0

                while self._test_batch_size(explainer, method_params, images, high):
                    low = high
                    high *= 2
                    if high > len(images):
                        high = len(images)
                        break
                
                for _ in range(self.preflight_cfg['binary_search_depth']):
                    if high <= low: break
                    mid = (low + high) // 2
                    if mid == low: break
                    if self._test_batch_size(explainer, method_params, images, mid):
                        low = mid
                    else:
                        high = mid - 1
                
                safe_bs = low if self._test_batch_size(explainer, method_params, images, low) else 0

                self.logger.info(f"'{method_unique_key}' 的最佳批次大小: {safe_bs}")
                final_results[model_flag][method_unique_key] = safe_bs
            
            del model, images
            gc.collect()
            torch.cuda.empty_cache()

        results_path = Path(self.preflight_cfg['results_path'])
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        
        self.logger.info(f"--- 预检完成。结果已保存至 {results_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI 实验预检脚本。")
    parser.add_argument('--config', type=str, default="config.yml", help='主实验配置文件路径。')
    parser.add_argument('--gpu-id', type=int, required=True, help='用于检查的 GPU ID。')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    log_dir = Path(config['paths']['scheduler_log_dir'])
    
    logger = setup_logging(log_dir, "preflight_batch_size_check.log", "BatchSizeChecker")
    checker = BatchSizeFinder(config_path=args.config, gpu_id=args.gpu_id, logger=logger)
    checker.run()

