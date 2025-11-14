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
import random
import copy

from tqdm import tqdm
import torch
import medmnist
from medmnist import INFO

from model_loader import load_medmnist_model
from captum.attr import (
    IntegratedGradients, GradientShap, Occlusion, Saliency, InputXGradient,
    GuidedBackprop, Deconvolution, DeepLift, DeepLiftShap, FeatureAblation
)
# We now import the worker to reuse its logic directly
from run_experiment_group import ExperimentGroupWorker


def setup_logging(log_dir: Path, filename: str, logger_name: str):
    """配置一个指定的日志记录器。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / filename
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler for detailed logs
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    # Console handler for high-level status
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] (Preflight) %(message)s"))
    logger.addHandler(console_handler)
    
    return logger


class BatchSizeFinder:
    """
    Finds the maximum safe batch size for each (model, attribution method) combo.
    (This is the original logic of preflight_checker.py)
    """
    def __init__(self, config_path: str, gpu_id: int, logger: logging.Logger):
        self.config_path = config_path
        self.logger = logger
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.preflight_cfg = self.config['preflight_check']
        self.search_space = self.config['search_space']
        self.paths = self.config['paths']
        
        self.device = torch.device(f"cuda:{gpu_id}")
        self.proxy_data_flag = self._get_proxy_dataset()
        self.model = None
        self.dummy_data = None
        self.global_mean_baseline = None

    def _get_proxy_dataset(self) -> str:
        data_flags = self.search_space.get('data_flags', [])
        if data_flags: return data_flags[0]
        weights_root = Path(self.paths['medmnist_weights_root'])
        for dir_name in sorted(os.listdir(weights_root)):
            if 'weights_' in dir_name and '3d' not in dir_name:
                flag = dir_name.replace('weights_', '')
                if flag in INFO:
                    return flag
        self.logger.critical("Could not find any available 2D datasets.")
        sys.exit(1)

    def _prepare_proxy_data(self, model_flag: str):
        self.logger.info(f"Preparing proxy data for model '{model_flag}'...")
        info = INFO[self.proxy_data_flag]
        self.model = load_medmnist_model(
            model_flag=model_flag, data_flag=self.proxy_data_flag,
            weights_root=Path(self.paths['medmnist_weights_root']), device=self.device
        )
        subset_size = self.preflight_cfg['subset_size']
        self.dummy_data = {
            'images': torch.randn(subset_size, 3, 224, 224, device=self.device),
            'targets': torch.randint(0, len(info['label']), (subset_size,), device=self.device)
        }
        mean_path = Path(self.paths['means_dir']) / f"{self.proxy_data_flag}_mean.pt"
        self.global_mean_baseline = torch.load(mean_path, map_location=self.device) if mean_path.exists() else torch.zeros(3, 224, 224, device=self.device)

    # Note: _get_attribution and other helper methods are reused from the original file
    def _get_attribution(self, explainer, params, images, targets):
        explainer_class = explainer.__class__
        kwargs = {'target': targets}
        if params.get('baselines'):
            if params['baselines'] == 'zero': kwargs['baselines'] = torch.zeros_like(images)
            elif params['baselines'] == 'global_mean': kwargs['baselines'] = self.global_mean_baseline.expand_as(images)
            elif params['baselines'] == 'combined_zng':
                 kwargs['baselines'] = torch.cat([torch.zeros_like(images), torch.randn_like(images), self.global_mean_baseline.expand_as(images)], dim=0)
        
        if explainer_class is Occlusion:
            kwargs['sliding_window_shapes'] = (3, params.get('patch_size', 16), params.get('patch_size', 16))
            # 修复：添加 strides 参数以减少计算量
            kwargs['strides'] = kwargs['sliding_window_shapes']
        
        if explainer_class is FeatureAblation:
            patch_size = params.get('patch_size', 16)
            num_patches = 224 // patch_size
            feature_mask = torch.arange(num_patches**2, device=self.device).view(num_patches, num_patches)
            feature_mask = feature_mask.repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
            kwargs['feature_mask'] = feature_mask.expand(images.size(0), 3, -1, -1)

        # For Occlusion and FeatureAblation, remove 'patch_size' as it's not a valid captum argument
        clean_params = {k: v for k, v in params.items() if k not in ['baselines', 'patch_size']}
        
        # 对于其他方法，清理掉不支持的参数
        clean_params = {k: v for k, v in params.items() if k not in ['baselines']}
        kwargs.update(clean_params)
        return explainer.attribute(images, **kwargs)

    def _test_batch_size(self, explainer, params, bs: int):
        _ = self._get_attribution(explainer, params, self.dummy_data['images'][:bs], self.dummy_data['targets'][:bs])
        gc.collect()
        torch.cuda.empty_cache()

    def _binary_search(self, explainer, params, low: int, high: int) -> int:
        max_safe_bs = low
        while low <= high:
            mid = (low + high) // 2
            if mid == 0: break
            try:
                self._test_batch_size(explainer, params, mid)
                max_safe_bs = mid
                low = mid + 1
            except Exception:
                high = mid - 1
        return max_safe_bs

    def find_max_batch_size(self, method_cfg: dict) -> int:
        params = method_cfg.get('params', {})
        explainer = globals()[method_cfg['class']](self.model)
        try:
            self._test_batch_size(explainer, params, 1)
        except Exception as e:
            self.logger.critical(f"Method {method_cfg['name']} failed at bs=1.", exc_info=True)
            return 0
        return self._binary_search(explainer, params, 1, self.preflight_cfg['start_batch_size'])

    def run(self):
        self.logger.info("--- Starting Mode: [batch_size] ---")
        results = {}
        for model_flag in self.search_space['model_flags']:
            results[model_flag] = {}
            self._prepare_proxy_data(model_flag)
            
            method_configs_to_test = []
            for method_template in self.search_space['attribution_methods']:
                param_grid = method_template.get('params', {})
                for param_combo in (dict(zip(param_grid, v)) for v in product(*param_grid.values())):
                    method_configs_to_test.append({'name': method_template['name'], 'class': method_template['class'], 'params': param_combo})
            
            for method_cfg in tqdm(method_configs_to_test, desc=f"Testing {model_flag}"):
                params = method_cfg.get('params', {})
                param_str = "_".join(sorted([f"{k}_{v}" for k, v in params.items()]))
                unique_key = f"{method_cfg['name']}_{param_str}" if param_str else method_cfg['name']
                
                max_bs = self.find_max_batch_size(method_cfg)
                results[model_flag][unique_key] = max_bs
            
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

        results_path = Path(self.preflight_cfg['results_path'])
        with open(results_path, 'w') as f: json.dump(results, f, indent=4)
        self.logger.info(f"--- Batch size check complete! Results: {results_path} ---")


class FullWorkflowTester:
    """
    Runs a mini, end-to-end version of the experiment to test the full pipeline.
    """
    def __init__(self, config_path: str, gpu_id: int, logger: logging.Logger):
        self.config_path = config_path
        self.gpu_id = gpu_id
        self.logger = logger
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.test_output_dir = Path("./preflight_test_outputs")
        self.test_output_dir.mkdir(exist_ok=True)

    def _get_random_dataset(self) -> str:
        flags = self.config['search_space'].get('data_flags', [])
        if flags: return random.choice(flags)
        
        scanned_flags = []
        weights_root = Path(self.config['paths']['medmnist_weights_root'])
        for dir_name in sorted(os.listdir(weights_root)):
            if 'weights_' in dir_name and '3d' not in dir_name:
                flag = dir_name.replace('weights_', '')
                if flag in INFO: scanned_flags.append(flag)
        
        if not scanned_flags:
            self.logger.critical("No 2D datasets found to run the test.")
            sys.exit(1)
        return random.choice(scanned_flags)

    def run(self):
        self.logger.info("--- Starting Mode: [full_test] ---")
        self.logger.info(f"Test outputs will be saved to: {self.test_output_dir.resolve()}")
        
        data_flag = self._get_random_dataset()
        self.logger.info(f"Using random dataset '{data_flag}' for the test.")

        test_grid = list(product(
            self.config['search_space']['model_flags'],
            self.config['search_space']['noise_conditions'],
            self.config['search_space']['splits']
        ))
        
        total_tests = len(test_grid)
        failures = 0

        for i, (model_flag, noise_cfg, split) in enumerate(test_grid):
            test_name = f"Test {i+1}/{total_tests} ({model_flag}/{noise_cfg['name']}/{split})"
            self.logger.info(f"--- {test_name} STARTING ---")

            try:
                # Create a temporary config in memory for this specific test run
                test_config = copy.deepcopy(self.config)
                test_config['paths']['final_output_dir'] = str(self.test_output_dir)

                # Instantiate the worker directly, passing the config dict and subset_size
                worker = ExperimentGroupWorker(
                    config=test_config,
                    data_flag=data_flag,
                    model_flag=model_flag,
                    noise_name=noise_cfg['name'],
                    split=split,
                    gpu_id=self.gpu_id,
                    subset_size=2 # KEY: Run on a tiny subset
                )
                worker.run()
                self.logger.info(f"--- {test_name} SUCCEEDED ---")

            except Exception as e:
                self.logger.error(f"--- {test_name} FAILED ---", exc_info=True)
                failures += 1
        
        self.logger.info("--- Full Workflow Test Finished ---")
        if failures > 0:
            self.logger.error(f"Completed with {failures}/{total_tests} failures. Check log for details.")
            sys.exit(1)
        else:
            self.logger.info(f"All {total_tests}/{total_tests} tests passed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Experiment Pre-flight Checker.")
    parser.add_argument('--config', type=str, default="config.yml", help='Path to the main experiment configuration file.')
    parser.add_argument('--gpu-id', type=int, required=True, help='The GPU ID to use for the check.')
    parser.add_argument('--mode', type=str, default='batch_size', choices=['batch_size', 'full_test'], 
                        help="'batch_size' to find optimal batch sizes, 'full_test' to run an end-to-end pipeline test.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    log_dir = Path(config['paths']['scheduler_log_dir'])
    
    if args.mode == 'batch_size':
        logger = setup_logging(log_dir, "preflight_batch_size_check.log", "BatchSizeChecker")
        checker = BatchSizeFinder(config_path=args.config, gpu_id=args.gpu_id, logger=logger)
        checker.run()
    elif args.mode == 'full_test':
        logger = setup_logging(log_dir, "preflight_full_workflow_test.log", "FullWorkflowTester")
        tester = FullWorkflowTester(config_path=args.config, gpu_id=args.gpu_id, logger=logger)
        tester.run()
