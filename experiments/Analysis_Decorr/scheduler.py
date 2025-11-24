#!/usr/bin/env python3
import sys
import os
import subprocess
import yaml
import json
import time
import argparse
import logging
from pathlib import Path
from itertools import product
from collections import deque
from typing import Dict, List, Deque, Optional, Any

sys.path.append(str(Path(__file__).parent))
from core.utils import setup_logging

class Task:
    def __init__(self, task_id: int, params: Dict[str, Any]):
        self.id = task_id
        self.params = params
        self.status = "pending"
        self.retries = 0
        self.process: Optional[subprocess.Popen] = None
        self.gpu_id: Optional[int] = None
        self.log_file: Optional[Path] = None
        self.start_time: Optional[float] = None

    def get_log_filename(self) -> str:
        p = self.params
        return f"task_{self.id}_{p['dataset']}_{p['model']}_{p['noise']}_{p['split']}.log"

class TaskScheduler:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scheduler_config = self.config['scheduler']
        self.output_dir = Path(self.config['output_dir'])
        self.log_dir = self.output_dir / "scheduler_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.log_dir / "scheduler_main.log", name="Scheduler")
        
        self.available_gpus: Deque[int] = deque(self.scheduler_config['available_gpus'])
        self.max_concurrent_tasks = len(self.available_gpus)
        
        self.task_queue: Deque[Task] = deque()
        self.running_tasks: Dict[int, Task] = {}
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []

    def run(self):
        self.logger.info("--- Task Scheduler started ---")
        self._generate_tasks()
        self.logger.info(f"Checking for completed tasks in {self.output_dir} before dispatching.")
        self._filter_completed_tasks()

        if not self.task_queue:
            self.logger.info("Task queue is empty. Exiting.")
            return

        while self.task_queue or self.running_tasks:
            self._check_running_tasks()
            
            while len(self.running_tasks) < self.max_concurrent_tasks and self.available_gpus and self.task_queue:
                task = self.task_queue.popleft()
                self._start_task(task)
            
            time.sleep(self.scheduler_config.get('polling_interval', 10))

        self.logger.info("--- All tasks executed, scheduler shutting down ---")
        self.logger.info(f"Completed: {len(self.completed_tasks)}, Failed: {len(self.failed_tasks)}")

    def _generate_tasks(self):
        experiment_matrix = self.config['experiments']
        task_id_counter = 0
        
        for experiment_group in experiment_matrix:
            keys = experiment_group.keys()
            values = [v if isinstance(v, list) else [v] for v in experiment_group.values()]
            
            for combo in product(*values):
                task_params = dict(zip(keys, combo))
                self.task_queue.append(Task(task_id=task_id_counter, params=task_params))
                task_id_counter += 1
                
        self.logger.info(f"Generated {len(self.task_queue)} tasks.")

    def _filter_completed_tasks(self):
        check_dir = Path(self.config['output_dir'])
        
        remaining_tasks = deque()
        self.logger.info(f"Checking for already completed tasks in {check_dir}...")
        for task in self.task_queue:
            p = task.params
            output_filename = f"results_{p['dataset']}_{p['model']}_{p['noise']}_{p['split']}.json"
            output_file = check_dir / output_filename
            
            if output_file.exists():
                self.logger.info(f"Skipping completed Task {task.id}: {output_file.name}")
                self.completed_tasks.append(task)
            else:
                remaining_tasks.append(task)
                
        self.logger.info(f"{len(remaining_tasks)} tasks remain to be executed.")
        self.task_queue = remaining_tasks

    def _start_task(self, task: Task):
        gpu_id = self.available_gpus.popleft()
        task.gpu_id = gpu_id
        task.status = "running"
        task.start_time = time.time()
        task.log_file = self.log_dir / task.get_log_filename()
        
        task_payload = task.params.copy()
        task_payload['task_id'] = task.id
        task_payload['config_path'] = "config_analysis_densenet_decoor.yml"
        
        task_json = json.dumps(task_payload)
        cmd = [
            sys.executable, 
            "worker.py", 
            "--task_json", task_json, 
            "--gpu_id", str(gpu_id)
        ]
        
        self.logger.info(f"Starting Task {task.id} on GPU {gpu_id}...")
        log_handle = open(task.log_file, 'w')
        process = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle)
        task.process = process
        self.running_tasks[process.pid] = task

    def _check_running_tasks(self):
        pids_to_remove = []
        for pid, task in list(self.running_tasks.items()):
            return_code = task.process.poll()
            if return_code is not None:
                self.available_gpus.append(task.gpu_id)
                pids_to_remove.append(pid)
                if return_code == 0:
                    self.logger.info(f"Task {task.id} (PID {pid}) on GPU {task.gpu_id} completed successfully.")
                    task.status = "completed"
                    self.completed_tasks.append(task)
                else:
                    self.logger.error(f"Task {task.id} (PID {pid}) failed with exit code {return_code}. Log: {task.log_file}")
                    self._handle_failed_task(task)
            elif time.time() - task.start_time > self.scheduler_config.get('task_timeout', 36000):
                self.logger.warning(f"Task {task.id} (PID {pid}) timed out. Terminating...")
                task.process.terminate()
                try:
                    task.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    task.process.kill()
                self.available_gpus.append(task.gpu_id)
                pids_to_remove.append(pid)
                task.status = "timeout"
                self._handle_failed_task(task)
                
        for pid in pids_to_remove:
            del self.running_tasks[pid]

    def _handle_failed_task(self, task: Task):
        task.retries += 1
        if task.retries <= self.scheduler_config.get('max_retries', 1):
            self.logger.info(f"Re-queueing Task {task.id} (Retry {task.retries})")
            task.status = "pending"
            self.task_queue.append(task)
        else:
            self.logger.error(f"Task {task.id} reached max retries. Marking as permanently failed.")
            self.failed_tasks.append(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Analysis Framework Scheduler")
    parser.add_argument('--config', type=str, default="config_analysis_densenet_decoor.yml", help='Path to the configuration file.')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if str(script_dir) not in sys.path:
        sys.path.append(str(script_dir))
    
    try:
        scheduler = TaskScheduler(config_path=args.config)
        scheduler.run()
    except Exception as e:
        logging.basicConfig()
        logging.critical(f"Scheduler failed to start: {e}", exc_info=True)
        try:
            scheduler.logger.critical(f"Scheduler failed to start: {e}", exc_info=True)
        except:
            pass