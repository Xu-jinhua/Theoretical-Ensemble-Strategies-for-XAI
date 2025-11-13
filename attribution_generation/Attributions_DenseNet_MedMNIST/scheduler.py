import os
# 在文件开头添加以下代码，用来解决MKL相关的线程冲突问题
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import sys
import yaml
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from itertools import product
from collections import deque
from typing import Dict, List, Deque, Optional, Any

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (Scheduler) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("Scheduler")


class Task:
    """Represents a single, consolidated experiment group task."""
    def __init__(self, task_id: int, params: Dict[str, Any]):
        self.id = task_id
        self.params = params
        self.status = "pending" # pending, running, completed, failed, timeout
        self.retries = 0
        self.process: Optional[subprocess.Popen] = None
        self.gpu_id: Optional[int] = None
        self.log_file: Optional[Path] = None
        self.start_time: Optional[float] = None

    def get_log_filename(self) -> str:
        p = self.params
        return f"task_{self.id}__{p['data_flag']}_{p['model_flag']}_{p['noise_name']}_{p['split']}_retry_{self.retries}.log"


class TaskScheduler:
    """A simplified scheduler for dispatching consolidated experiment group tasks."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paths = self.config['paths']
        self.scheduler_cfg = self.config['scheduler']
        self.preflight_cfg = self.config['preflight_check']
        
        self.log_dir = Path(self.paths['scheduler_log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.available_gpus: Deque[int] = deque(self.scheduler_cfg['available_gpus'])
        
        self.task_queue: Deque[Task] = deque()
        self.running_tasks: Dict[int, Task] = {}
        self.completed_tasks_count = 0
        self.total_tasks = 0

    def run(self):
        logger.info("--- Task Scheduler Started (Zarr Workflow) ---")

        if self.preflight_cfg.get('enabled', False):
            self._run_preflight_check()

        self._generate_tasks()
        
        if not self.task_queue:
            logger.info("No tasks to execute. Scheduler exiting.")
            return

        logger.info(f"Successfully generated {len(self.task_queue)} tasks.")
        self.total_tasks = len(self.task_queue)

        while self.task_queue or self.running_tasks:
            self._check_running_tasks()
            
            while self.available_gpus and self.task_queue:
                task = self.task_queue.popleft()
                self._start_task(task)
                
            time.sleep(self.scheduler_cfg['polling_interval'])

        logger.info(f"--- All {self.total_tasks} tasks have finished. Scheduler shutting down. ---")

    def _generate_tasks(self):
        logger.info("Generating experiment group tasks...")
        space = self.config['search_space']
        task_id_counter = 0

        data_flags = self._get_data_flags()
        if not data_flags:
            logger.warning("No data_flags found to process. Exiting.")
            return

        for data_flag, model_flag, noise_cfg, split in product(
            data_flags, space['model_flags'], space['noise_conditions'], space['splits']
        ):
            output_filename = f"dataset-{data_flag}_model-{model_flag}_noise-{noise_cfg['name']}_split-{split}.zarr"
            output_path = Path(self.paths['final_output_dir']) / output_filename
            if output_path.exists():
                logger.info(f"Skipping task, output already exists: {output_path.name}")
                continue

            task_params = {
                'data_flag': data_flag,
                'model_flag': model_flag,
                'noise_name': noise_cfg['name'],
                'split': split,
            }
            
            task = Task(task_id_counter, task_params)
            self.task_queue.append(task)
            task_id_counter += 1

    def _start_task(self, task: Task):
        gpu_id = self.available_gpus.popleft()
        task.gpu_id = gpu_id
        task.status = "running"
        task.start_time = time.time()
        task.log_file = self.log_dir / task.get_log_filename()
        
        p = task.params
        
        # --- [MODIFIED] Reverted to original command-line argument passing ---
        cmd = [
            sys.executable, "run_experiment_group.py",
            "--config", self.config_path,
            "--data-flag", p['data_flag'],
            "--model-flag", p['model_flag'],
            "--noise-name", p['noise_name'],
            "--split", p['split'],
            "--gpu-id", str(gpu_id)
        ]
        # --- End modification ---
        
        logger.info(f"Starting task {task.id} ({p['data_flag']}/{p['model_flag']}/{p['noise_name']}/{p['split']}) on GPU {gpu_id}")
        
        try:
            log_handle = open(task.log_file, 'w')
            process = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)
            task.process = process
            self.running_tasks[task.id] = task
        except Exception as e:
            logger.error(f"Failed to start process for task {task.id}: {e}", exc_info=True)
            log_handle.close()
            self.task_queue.appendleft(task)
            self.available_gpus.append(gpu_id)


    def _check_running_tasks(self):
        finished_task_ids = []
        for task_id, task in list(self.running_tasks.items()):
            return_code = task.process.poll()
            
            if return_code is not None or (time.time() - task.start_time > self.scheduler_cfg['task_timeout']):
                self.available_gpus.append(task.gpu_id)
                finished_task_ids.append(task_id)

                if return_code == 0:
                    logger.info(f"Task {task.id} finished successfully.")
                    task.status = "completed"
                    self.completed_tasks_count += 1
                elif return_code is not None:
                    logger.error(f"Task {task.id} failed with exit code {return_code}. Log: {task.log_file.name}")
                    self._handle_failed_task(task)
                else: # Timeout case
                    logger.warning(f"Task {task.id} timed out. Terminating...")
                    task.process.terminate()
                    time.sleep(2)
                    task.process.kill()
                    task.status = "timeout"
                    self._handle_failed_task(task)

        for task_id in finished_task_ids:
            del self.running_tasks[task_id]

    def _handle_failed_task(self, task: Task):
        task.retries += 1
        if task.retries <= self.scheduler_cfg['max_retries']:
            logger.info(f"Re-queueing task {task.id} (Retry {task.retries}/{self.scheduler_cfg['max_retries']})")
            task.status = "pending"
            self.task_queue.appendleft(task)
        else:
            logger.error(f"Task {task.id} reached max retries. Marking as permanently failed.")
            task.status = "failed"
            self.completed_tasks_count += 1
    
    def _run_preflight_check(self):
        results_path = Path(self.preflight_cfg['results_path'])
        if results_path.exists():
            logger.info(f"Found existing pre-flight results: {results_path}, skipping check.")
            return
        logger.info("No pre-flight results found, starting check...")
        if not self.available_gpus:
            logger.critical("No available GPUs to run pre-flight check.")
            sys.exit(1)
        gpu_id = self.available_gpus[0]
        cmd = [sys.executable, "preflight_checker.py", "--config", self.config_path, "--gpu-id", str(gpu_id)]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
            logger.info("Pre-flight check completed successfully.")
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.critical(f"Pre-flight check failed with exit code {e.returncode}.")
            logger.critical("STDOUT:\n" + e.stdout)
            logger.critical("STDERR:\n" + e.stderr)
            sys.exit(1)
        except Exception as e:
            logger.critical(f"An unexpected error occurred during pre-flight check: {e}", exc_info=True)
            sys.exit(1)

    def _get_data_flags(self) -> List[str]:
        # This logic is for MedMNIST auto-discovery and is not used for the ImageNet setup.
        # It's kept for potential future reusability.
        flags = self.config['search_space'].get('data_flags', [])
        if 'imagenet' in flags: # For our current setup, this will always be true
            return flags
            
        # Fallback logic for MedMNIST
        if not flags:
            weights_root = Path(self.paths['medmnist_weights_root'])
            if not weights_root.exists():
                logger.critical(f"Weights directory does not exist: {weights_root}")
                sys.exit(1)
            
            from medmnist import INFO
            scanned_flags = []
            for dir_name in sorted(os.listdir(weights_root)):
                if os.path.isdir(weights_root / dir_name) and dir_name.startswith('weights_'):
                    data_flag = dir_name.replace('weights_', '')
                    if data_flag in INFO and not INFO[data_flag].get('is_3d', False):
                        scanned_flags.append(data_flag)
            logger.info(f"Scanned and found {len(scanned_flags)} valid 2D datasets.")
            return scanned_flags
        return flags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Experiment Scheduler (Zarr Workflow)")
    parser.add_argument('--config', type=str, default="config.yml", help='Path to the configuration file.')
    args = parser.parse_args()
    
    scheduler = TaskScheduler(config_path=args.config)
    scheduler.run()

