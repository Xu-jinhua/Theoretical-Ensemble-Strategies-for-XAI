#!/usr/bin/env python3
import argparse
import json
import yaml
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import logging
import sys

script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from core.data_loader import ZarrDataLoader
from core.ensembles import EnsembleFactory
from core.evaluations import apply_mask_batch, evaluate_batch
from core.utils import (attribution_to_rank_batch, NumpyEncoder, setup_logging)
from core.model_loader_densenet import load_densenet_model as load_model

class ExperimentAnalyzer:
    def __init__(self, task_params: dict, config: dict, device: torch.device, logger: logging.Logger):
        self.task_params = task_params
        self.config = config
        self.analysis_params = config['analysis_params']
        self.device = device
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.dynamic_methods = set(self.analysis_params.get('dynamic_ensemble_methods', []))

    def run(self):
        output_file = self.output_dir / self._generate_output_filename()
        if output_file.exists():
            self.logger.info(f"Output file already exists. Skipping task: {output_file}")
            return

        self.logger.info(f"Computing all results from scratch for {output_file.name}.")

        with ZarrDataLoader(self.task_params, self.config) as loader:
            self.logger.info(f"Loading base data from Zarr: {loader.zarr_path}")
            images = loader.get_images()
            labels = loader.get_labels().squeeze()
            original_preds = loader.get_predictions()
            num_samples = len(images)
            self.logger.info(f"Base data loaded ({num_samples} samples).")

            all_attributions = {}
            attr_methods = loader.get_all_attribution_methods()
            if not attr_methods:
                self.logger.critical(f"No attributions found in {loader.zarr_path}.")
                exit(1)

            self.logger.info(f"Loading {len(attr_methods)} attribution methods...")
            for method_name in tqdm(attr_methods, desc="Loading attributions"):
                all_attributions[method_name] = loader.get_attributions(method_name)

            self.logger.info("Loading mean tensors from Zarr...")
            global_mean = loader.get_global_mean()
            class_means = loader.get_all_class_means()
            fill_tensors = self._prepare_fill_tensors_from_means(global_mean, class_means, original_preds)

        model = load_model(
            model_flag=self.task_params['model'],
            data_flag=self.task_params['dataset'], 
            weights_root=Path(self.config['weights_root']),
            device=self.device
        )
        model = torch.compile(model)
        self.logger.info("Model loaded and compiled.")

        final_results = self._perform_analysis(model, images, labels, original_preds, all_attributions, fill_tensors)

        with open(output_file, 'w') as f:
            json.dump(final_results, f, cls=NumpyEncoder, indent=2)
        self.logger.info(f"Analysis complete. New results saved to {output_file}")

    def _perform_analysis(self, model, images, labels, original_preds, all_attributions, fill_tensors):
        results = defaultdict(dict)
        batch_size = self.analysis_params.get('batch_size', 256)
        top_k_values = self.analysis_params['top_k_values']

        for patch_size in self.analysis_params['patch_size']:
            current_attributions = {name: attr for name, attr in all_attributions.items() 
                                    if f"patch_size_{patch_size}" in name or 'patch_size' not in name}
            if not current_attributions:
                continue

            single_method_ranks = {name: attribution_to_rank_batch(attr, patch_size) for name, attr in current_attributions.items()}
            single_method_values = {name: attr for name, attr in current_attributions.items()}

            for eval_config in self.analysis_params['evaluation_methods']:
                eval_params = eval_config['params']
                eval_key = f"patch_{patch_size}_fill_{eval_params['fill_type']}_replace_{eval_params['replace_top']}"
                
                current_results = defaultdict(lambda: defaultdict(list))
                
                for name, rank_tensor in tqdm(single_method_ranks.items(), desc=f"Evaluating single methods for {eval_key}"):
                    acc_list, cons_list = self._evaluate_method_for_all_k(model, images, rank_tensor, top_k_values, original_preds, labels, fill_tensors, batch_size, patch_size, eval_params['replace_top'])
                    current_results[name]['accuracy'].extend(acc_list)
                    current_results[name]['consistency'].extend(cons_list)

                missing_single_methods = [name for name in single_method_ranks.keys() if name not in current_results]
                if missing_single_methods:
                    self.logger.error(f"Cannot perform ensembling. Missing single method results for {eval_key}: {missing_single_methods}")
                    continue

                is_reversed = not eval_params['replace_top']
                sorted_methods = sorted(single_method_ranks.keys(), key=lambda m: np.mean(current_results[m]['accuracy']), reverse=is_reversed)

                self.logger.info("Pre-stacking all sorted ranks and values for STATIC path...")
                all_sorted_ranks = torch.stack([single_method_ranks[m] for m in sorted_methods], dim=1)
                all_sorted_values = torch.stack([single_method_values[m] for m in sorted_methods], dim=1)

                self.logger.info(f"Running STATIC ensembling for {eval_key}...")
                for n in self.analysis_params['ensemble_n_range']:
                    if len(sorted_methods) < n: continue
                    
                    top_n_ranks_stack = all_sorted_ranks[:, :n, ...]
                    top_n_values_stack = all_sorted_values[:, :n, ...]
                    
                    for ens_config in self.analysis_params['ensemble_methods']:
                        if ens_config['name'] in self.dynamic_methods:
                            continue
                            
                        method_name = self._get_ensemble_method_name(ens_config, n, is_dynamic=False)
                        
                        self.logger.info(f"Computing STATIC ensemble: {eval_key} / {method_name}")
                        ensemble_func, operates_on_ranks = EnsembleFactory.get_ensemble_method(ens_config)
                        input_stack = top_n_ranks_stack if operates_on_ranks else top_n_values_stack
                        
                        ensembled_result = self._get_ensembled_result_optimized(input_stack, ensemble_func, operates_on_ranks, ens_config, batch_size, patch_size)

                        acc_list, cons_list = self._evaluate_method_for_all_k(model, images, ensembled_result, top_k_values, original_preds, labels, fill_tensors, batch_size, patch_size, eval_params['replace_top'])
                        current_results[method_name]['accuracy'].extend(acc_list)
                        current_results[method_name]['consistency'].extend(cons_list)

                self.logger.info(f"Running DYNAMIC ensembling for {eval_key}...")
                for n in self.analysis_params['ensemble_n_range']:
                    for ens_config in self.analysis_params['ensemble_methods']:
                        if ens_config['name'] not in self.dynamic_methods:
                            continue
                        
                        method_name = self._get_ensemble_method_name(ens_config, n, is_dynamic=True)
                        
                        self.logger.info(f"Computing DYNAMIC ensemble: {eval_key} / {method_name}")
                        ensemble_func, operates_on_ranks = EnsembleFactory.get_ensemble_method(ens_config)
                        
                        acc_list, cons_list = self._evaluate_dynamic_ensemble(
                            model, images, labels, original_preds, fill_tensors,
                            single_method_ranks, single_method_values,
                            current_results, top_k_values, batch_size, patch_size,
                            eval_params['replace_top'], n, ensemble_func,
                            operates_on_ranks, ens_config
                        )
                        current_results[method_name]['accuracy'].extend(acc_list)
                        current_results[method_name]['consistency'].extend(cons_list)
                
                results[eval_key] = dict(current_results)
                
        return dict(results)

    def _get_ensembled_result_optimized(self, input_stack, ensemble_func, operates_on_ranks, ens_config, batch_size, patch_size):
        ensembled_results_list = []
        input_stack_cpu = input_stack.cpu()
        
        for i in range(0, input_stack_cpu.shape[0], batch_size):
            batch_input_stack = input_stack_cpu[i:i+batch_size].to(self.device)
            with torch.amp.autocast('cuda'):
                if ens_config.get('params', {}).get('k') is not None:
                    k_val = ens_config['params']['k']
                    batch_ensembled_result = ensemble_func(batch_input_stack, k_val)
                else:
                    batch_ensembled_result = ensemble_func(batch_input_stack)
            ensembled_results_list.append(batch_ensembled_result.cpu())
        
        ensembled_result = torch.cat(ensembled_results_list, dim=0)
        
        if operates_on_ranks:
            return ensembled_result
        else:
            return attribution_to_rank_batch(ensembled_result, patch_size)

    def _evaluate_dynamic_ensemble(self, model, images, labels, original_preds, fill_tensors,
                                   single_method_ranks, single_method_values,
                                   current_results, top_k_values, batch_size, patch_size,
                                   replace_top, n, ensemble_func, operates_on_ranks, ens_config):
        total_samples = len(images)
        acc_sums = {k: torch.tensor(0, dtype=torch.int64, device=self.device) for k in top_k_values}
        cons_sums = {k: torch.tensor(0, dtype=torch.int64, device=self.device) for k in top_k_values}
        
        is_reversed = not replace_top
        stream = torch.cuda.Stream()
        
        all_ranks_cpu = torch.stack(list(single_method_ranks.values()), dim=1)
        all_values_cpu = torch.stack(list(single_method_values.values()), dim=1)
        all_method_names = list(single_method_ranks.keys())
        
        with torch.no_grad(), torch.cuda.stream(stream):
            for i in tqdm(range(0, total_samples, batch_size), desc="Processing dynamic batches", leave=False):
                sl = slice(i, i + batch_size)
                
                batch_images = images[sl].to(self.device, non_blocking=True)
                batch_fills = fill_tensors[sl].to(self.device, non_blocking=True)
                batch_orig_preds = original_preds[sl].to(self.device, non_blocking=True)
                batch_labels = labels[sl].to(self.device, non_blocking=True)
                
                batch_all_ranks_cpu = all_ranks_cpu[sl]
                batch_all_values_cpu = all_values_cpu[sl]
                
                super_batch_image_list = []
                
                for k_idx, k_val in enumerate(top_k_values):
                    k_sorted_method_names = sorted(
                        all_method_names,
                        key=lambda m: np.mean(current_results[m]['accuracy'][k_idx]),
                        reverse=is_reversed
                    )
                    top_n_names = k_sorted_method_names[:n]
                    top_n_indices = [all_method_names.index(name) for name in top_n_names]
                    
                    if operates_on_ranks:
                        input_stack_cpu = batch_all_ranks_cpu[:, top_n_indices, ...]
                    else:
                        input_stack_cpu = batch_all_values_cpu[:, top_n_indices, ...]
                        
                    batch_input_stack_gpu = input_stack_cpu.to(self.device, non_blocking=True)
                    with torch.amp.autocast('cuda'):
                        if ens_config.get('params', {}).get('k') is not None:
                            k_param = ens_config['params']['k']
                            ensembled_result_gpu = ensemble_func(batch_input_stack_gpu, k_param)
                        else:
                            ensembled_result_gpu = ensemble_func(batch_input_stack_gpu)
                    
                    if not operates_on_ranks:
                        ensembled_rank_gpu = attribution_to_rank_batch(ensembled_result_gpu, patch_size)
                    else:
                        ensembled_rank_gpu = ensembled_result_gpu
                        
                    masked_images_for_k = apply_mask_batch(
                        batch_images,
                        ensembled_rank_gpu,
                        k_val,
                        patch_size,
                        replace_top,
                        batch_fills
                    )
                    super_batch_image_list.append(masked_images_for_k)

                final_super_batch = torch.cat(super_batch_image_list, dim=0)

                with torch.amp.autocast('cuda'):
                    masked_preds = model(final_super_batch).argmax(dim=-1)

                num_k = len(top_k_values)
                masked_preds_reshaped = masked_preds.view(num_k, batch_images.shape[0]).transpose(0, 1)

                for idx, k in enumerate(top_k_values):
                    acc, cons = evaluate_batch(masked_preds_reshaped[:, idx], batch_orig_preds, batch_labels)
                    acc_sums[k] += acc
                    cons_sums[k] += cons

        torch.cuda.synchronize()
        acc_list = [acc_sums[k].item() / total_samples for k in top_k_values]
        cons_list = [cons_sums[k].item() / total_samples for k in top_k_values]
        return acc_list, cons_list


    def _evaluate_method_for_all_k(self, model, images, rank_tensor, top_k_values, original_preds, labels, fill_tensors, batch_size, patch_size, replace_top):
        total_samples = len(images)
        acc_sums = {k: torch.tensor(0, dtype=torch.int64, device=self.device) for k in top_k_values}
        cons_sums = {k: torch.tensor(0, dtype=torch.int64, device=self.device) for k in top_k_values}

        stream = torch.cuda.Stream()
        
        rank_tensor_cpu = rank_tensor.cpu()

        with torch.no_grad(), torch.cuda.stream(stream):
            for i in tqdm(range(0, total_samples, batch_size), desc="Processing static batches", leave=False):
                sl = slice(i, i + batch_size)
                
                batch_images = images[sl].to(self.device, non_blocking=True)
                batch_ranks = rank_tensor_cpu[sl].to(self.device, non_blocking=True)
                batch_fills = fill_tensors[sl].to(self.device, non_blocking=True)
                batch_orig_preds = original_preds[sl].to(self.device, non_blocking=True)
                batch_labels = labels[sl].to(self.device, non_blocking=True)

                num_k = len(top_k_values)
                
                with torch.amp.autocast('cuda'):
                    super_batch_images = apply_mask_batch(
                        batch_images.repeat_interleave(num_k, dim=0),
                        batch_ranks.repeat_interleave(num_k, dim=0),
                        torch.tensor(top_k_values, device=self.device).repeat(batch_images.shape[0]),
                        patch_size,
                        replace_top,
                        batch_fills.repeat_interleave(num_k, dim=0)
                    )
                    masked_preds = model(super_batch_images).argmax(dim=-1)

                masked_preds = masked_preds.view(batch_images.shape[0], num_k)

                for idx, k in enumerate(top_k_values):
                    acc, cons = evaluate_batch(masked_preds[:, idx], batch_orig_preds, batch_labels)
                    acc_sums[k] += acc
                    cons_sums[k] += cons

        torch.cuda.synchronize()

        acc_list = [acc_sums[k].item() / total_samples for k in top_k_values]
        cons_list = [cons_sums[k].item() / total_samples for k in top_k_values]
        return acc_list, cons_list
    
    def _prepare_fill_tensors_from_means(self, global_mean: torch.Tensor, class_means: dict, original_preds: torch.Tensor):
        self.logger.info(f"Preparing fill tensors from loaded means...")

        original_preds_cpu = original_preds.cpu()
        fill_list = [class_means.get(pred.item(), global_mean) for pred in tqdm(original_preds_cpu, desc="Mapping class means")]

        return torch.stack(fill_list)

    def _generate_output_filename(self) -> str:
        return f"results_{self.task_params['dataset']}_{self.task_params['model']}_{self.task_params['noise']}_{self.task_params['split']}.json"

    def _get_ensemble_method_name(self, config, n, is_dynamic=False):
        name = config['name']
        
        if name == 'norm_ensemble':
            p = config['params']
            method_name = f"{name}_{p['normalization']}_{p['aggregation']}_n{n}"
        else:
            method_name = f"{name}_n{n}"

        if is_dynamic:
            return f"dynamic_{method_name}"
        else:
            return method_name


def main():
    parser = argparse.ArgumentParser(description="Run a single XAI analysis worker.")
    parser.add_argument('--task_json', type=str, required=True, help='JSON string of the task parameters')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use for this task')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    task_params = json.loads(args.task_json)
    
    config_path = task_params.get('config_path', 'config_analysis_densenet.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    task_id = task_params.get('task_id', 'unknown')
    p = task_params
    log_filename = f"task_{task_id}_{p.get('dataset', 'na')}_{p.get('model', 'na')}_{p.get('noise', 'na')}_{p.get('split', 'na')}.log"
    log_dir = Path(config['output_dir']) / "scheduler_logs"
    logger = setup_logging(log_dir / log_filename, name=f"Worker-T{task_id}")
    
    logger.info(f"Worker starting for task {task_id} on GPU {args.gpu_id}")
    logger.info(f"Task parameters: {task_params}")
    logger.info(f"Using config file: {config_path}")
    logger.info(f"Dynamic methods set: {config['analysis_params'].get('dynamic_ensemble_methods', [])}")
    logger.info(f"Output results dir: {config['output_dir']}")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        analyzer = ExperimentAnalyzer(task_params, config, device, logger)
        analyzer.run()
        logger.info(f"Task {task_id} completed successfully.")
    except Exception as e:
        logger.critical(f"Task {task_id} failed: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()