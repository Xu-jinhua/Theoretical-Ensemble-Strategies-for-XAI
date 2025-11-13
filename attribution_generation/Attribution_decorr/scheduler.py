#!/usr/bin/env python3
"""
scheduler.py - Decorrelated Attribution Generation Task Scheduler

This scheduler manages multiple attribution generation tasks, distributing them to available GPUs.
Supports parallel generation of attributions for different datasets, models, and noise conditions.
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import sys
import yaml
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
    """Represents an attribution generation task"""
    
    def __init__(self, task_id: int, data_flag: str, model_flag: str, 
                 noise_name: str, split: str, config_path: str):
        self.id = task_id
        self.data_flag = data_flag
        self.model_flag = model_flag
        self.noise_name = noise_name
        self.split = split
        self.config_path = config_path
        
        self.status = "pending"  # pending, running, completed, failed, timeout
        self.retries = 0
        self.process: Optional[subprocess.Popen] = None
        self.gpu_id: Optional[int] = None
        self.log_file: Optional[Path] = None
        self.start_time: Optional[float] = None
    
    def get_log_filename(self) -> str:
        return f"task_{self.id}_{self.data_flag}_{self.model_flag}_{self.noise_name}_{self.split}_retry_{self.retries}.log"


class TaskScheduler:
    """Task scheduler, responsible for distributing attribution generation tasks to GPUs"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scheduler_cfg = self.config['scheduler']
        self.search_space = self.config['search_space']
        
        # Create log directory (from paths config)
        # self.log_dir = Path(self.paths['scheduler_log_dir'])
        self.log_dir = Path(self.config['paths']['scheduler_log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure log file
        file_handler = logging.FileHandler(self.log_dir / "scheduler_main.log")
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] (Scheduler) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
        
        # GPU resource management
        self.available_gpus: Deque[int] = deque(self.scheduler_cfg['available_gpus'])
        self.max_tasks_per_gpu = self.scheduler_cfg.get('max_tasks_per_gpu', 1)
        self.gpu_task_count: Dict[int, int] = {gpu_id: 0 for gpu_id in self.scheduler_cfg['available_gpus']}
        
        # Task queue
        self.task_queue: Deque[Task] = deque()
        self.running_tasks: Dict[int, Task] = {}
        self.completed_tasks_count = 0
        self.total_tasks = 0
        
        logger.info("--- Decorrelated attribution scheduler initialization completed ---")
        logger.info(f"Available GPUs: {self.scheduler_cfg['available_gpus']}")
        logger.info(f"Max tasks per GPU: {self.max_tasks_per_gpu}")
    
    def run(self):
        """Main run loop"""
        logger.info("--- Scheduler started running ---")
        
        # Generate all tasks
        self._generate_tasks()
        
        if not self.task_queue:
            logger.info("No tasks to execute, scheduler exiting.")
            return
        
        self.total_tasks = len(self.task_queue)
        logger.info(f"Successfully generated {self.total_tasks} attribution generation tasks")
        
        # Display GPU resource information
        total_gpu_slots = len(self.scheduler_cfg['available_gpus']) * self.max_tasks_per_gpu
        logger.info(f"Total available GPU slots: {total_gpu_slots}")
        
        # Main loop
        while self.task_queue or self.running_tasks:
            # Check running tasks
            self._check_running_tasks()
            
            # Start new tasks
            while self.task_queue and self._has_available_gpu_slot():
                task = self.task_queue.popleft()
                self._start_task(task)
            
            # Wait before checking again
            time.sleep(self.scheduler_cfg['polling_interval'])
        
        logger.info(f"--- All {self.total_tasks} tasks completed, scheduler shutting down ---")
        logger.info(f"Success: {self.completed_tasks_count}, Failed: {self.total_tasks - self.completed_tasks_count}")
    
    def _has_available_gpu_slot(self) -> bool:
        """Check if there are available GPU slots"""
        for gpu_id in self.scheduler_cfg['available_gpus']:
            if self.gpu_task_count[gpu_id] < self.max_tasks_per_gpu:
                return True
        return False
    
    def _get_available_gpu(self) -> Optional[int]:
        """Get a GPU with available slots"""
        for gpu_id in self.scheduler_cfg['available_gpus']:
            if self.gpu_task_count[gpu_id] < self.max_tasks_per_gpu:
                return gpu_id
        return None
    
    def _generate_tasks(self):
        """Generate all attribution generation tasks"""
        logger.info("Generating attribution generation tasks...")
        
        task_id_counter = 0
        
        # Iterate through all combinations of datasets, models, noise conditions, and splits
        for data_flag, model_flag, noise_cfg, split in product(
            self.search_space['data_flags'],
            self.search_space['model_flags'],
            self.search_space['noise_conditions'],
            self.search_space['splits']
        ):
            # Check if output file already exists
            output_dir = Path(self.config['paths']['final_output_dir'])
            output_filename = f"dataset-{data_flag}_model-{model_flag}_noise-{noise_cfg['name']}_split-{split}_decorr.zarr"
            output_path = output_dir / output_filename
            
            if output_path.exists():
                logger.info(f"Skipping task, output already exists: {output_path}")
                continue
            
            # Create task
            task = Task(
                task_id_counter,
                data_flag,
                model_flag,
                noise_cfg['name'],
                split,
                self.config_path
            )
            self.task_queue.append(task)
            task_id_counter += 1
        
        logger.info(f"Successfully generated {len(self.task_queue)} pending tasks")
        
        # Display number of skipped tasks
        total_possible_tasks = (
            len(self.search_space['data_flags']) *
            len(self.search_space['model_flags']) *
            len(self.search_space['noise_conditions']) *
            len(self.search_space['splits'])
        )
        skipped_tasks = total_possible_tasks - len(self.task_queue)
        if skipped_tasks > 0:
            logger.info(f"Skipped {skipped_tasks} already completed tasks")
    
    def _start_task(self, task: Task):
        """Start a task"""
        gpu_id = self._get_available_gpu()
        if gpu_id is None:
            logger.error("No available GPU slots")
            self.task_queue.appendleft(task)
            return
        
        task.gpu_id = gpu_id
        task.status = "running"
        task.start_time = time.time()
        task.log_file = self.log_dir / task.get_log_filename()
        
        # Update GPU task count
        self.gpu_task_count[gpu_id] += 1
        
        # Build command
        cmd = [
            sys.executable, "generate_attributions_decorr.py",
            "--config", task.config_path,
            "--data-flag", task.data_flag,
            "--model-flag", task.model_flag,
            "--noise-name", task.noise_name,
            "--split", task.split,
            "--gpu-id", str(gpu_id)
        ]
        
        logger.info(f"Starting task {task.id}: {task.data_flag}/{task.model_flag}/{task.noise_name}/{task.split} on GPU {gpu_id} (current GPU tasks: {self.gpu_task_count[gpu_id]}/{self.max_tasks_per_gpu})")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            log_handle = open(task.log_file, 'w')
            process = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)
            task.process = process
            self.running_tasks[task.id] = task
        except Exception as e:
            logger.error(f"Failed to start task {task.id}: {e}", exc_info=True)
            log_handle.close()
            # Restore GPU task count
            self.gpu_task_count[gpu_id] -= 1
            self.task_queue.appendleft(task)
    
    def _check_running_tasks(self):
        """Check status of running tasks"""
        finished_task_ids = []
        
        for task_id, task in list(self.running_tasks.items()):
            return_code = task.process.poll()
            
            # Check if task is completed or timed out
            if return_code is not None or (time.time() - task.start_time > self.scheduler_cfg['task_timeout']):
                # Update GPU task count
                self.gpu_task_count[task.gpu_id] -= 1
                finished_task_ids.append(task_id)
                
                if return_code == 0:
                    logger.info(f"Task {task.id} completed successfully (GPU {task.gpu_id} remaining tasks: {self.gpu_task_count[task.gpu_id]})")
                task.status = "completed"
                self.completed_tasks_count += 1
            elif return_code is not None:
                logger.error(f"Task {task.id} failed with exit code: {return_code}. Log: {task.log_file}")
                self._handle_failed_task(task)
            else:  # Task timeout
                logger.warning(f"Task {task.id} timed out, terminating...")
                task.process.terminate()
                time.sleep(2)
                task.process.kill()
                task.status = "timeout"
                self._handle_failed_task(task)
        
        # Clean up completed tasks
        for task_id in finished_task_ids:
            del self.running_tasks[task_id]
    
    def _handle_failed_task(self, task: Task):
        """Handle failed task"""
        task.retries += 1
        
        if task.retries <= self.scheduler_cfg['max_retries']:
            logger.info(f"Re-queuing task {task.id} (retry {task.retries}/{self.scheduler_cfg['max_retries']})")
            task.status = "pending"
            self.task_queue.appendleft(task)
        else:
            logger.error(f"Task {task.id} reached maximum retry count, marking as permanently failed")
            task.status = "failed"


def main():
    parser = argparse.ArgumentParser(description='Decorrelated attribution generation task scheduler')
    parser.add_argument('--config', type=str, default="config.yml", help='Configuration file path')
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        logger.error(f"Configuration file does not exist: {args.config}")
        sys.exit(1)
    
    scheduler = TaskScheduler(config_path=args.config)
    scheduler.run()


if __name__ == "__main__":
    main()