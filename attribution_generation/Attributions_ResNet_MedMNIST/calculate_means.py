import os
import sys
import yaml
import logging
from pathlib import Path

import torch
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("MeanCalculator")


class MeanImageCalculator:
    
    def __init__(self, config_path: str):
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration file: {config_path}")
        except FileNotFoundError:
            logger.critical(f"Error: Configuration file '{config_path}' not found.")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Error loading or parsing configuration file: {e}")
            sys.exit(1)
        
        self.paths = self.config['paths']
        self.search_space = self.config['search_space']
        self.means_dir = Path(self.paths['means_dir'])
        self.medmnist_data_root = Path(self.paths['medmnist_data_root'])
        
        self.means_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"All mean images will be saved to: {self.means_dir}")

    def _get_target_datasets(self) -> list[str]:
        
        target_flags = self.search_space.get('data_flags', [])
        if not target_flags:
            logger.info("'data_flags' is empty, automatically scanning 2D datasets from weights directory...")
            weights_root = Path(self.paths['medmnist_weights_root'])
            if not weights_root.exists():
                logger.critical(f"Weights directory does not exist: {weights_root}")
                sys.exit(1)
            
            scanned_flags = []
            for dir_name in sorted(os.listdir(weights_root)):
                if os.path.isdir(weights_root / dir_name) and 'weights_' in dir_name and not '3d' in dir_name:
                    data_flag = dir_name.replace('weights_', '')
                    if data_flag in INFO:
                        scanned_flags.append(data_flag)
            
            logger.info(f"Scanned {len(scanned_flags)} 2D datasets: {scanned_flags}")
            return scanned_flags
        else:
            logger.info(f"Computing mean images for {len(target_flags)} datasets specified in configuration: {target_flags}")
            return target_flags

    def calculate_and_save_all(self):
        
        target_datasets = self._get_target_datasets()
        if not target_datasets:
            logger.warning("No target datasets found. Script will exit.")
            return

        for data_flag in tqdm(target_datasets, desc="Processing all datasets"):
            save_path = self.means_dir / f"{data_flag}_mean.pt"
            if save_path.exists():
                logger.info(f"Skipping: Mean image for '{data_flag}' already exists at {save_path}")
                continue

            logger.info(f"Starting to compute mean image for '{data_flag}'...")
            
            try:
                info = INFO[data_flag]
                DataClass = getattr(medmnist, info['python_class'])
                
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])

                train_dataset = DataClass(
                    split='train',
                    transform=transform,
                    download=True,
                    size=224,
                    as_rgb=True,
                    root=self.medmnist_data_root
                )
                
                loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, num_workers=4)
                
                sum_images = torch.zeros(3, 224, 224, dtype=torch.float64)
                num_samples = 0

                for images, _ in tqdm(loader, desc=f"Computing {data_flag}", leave=False):
                    sum_images += torch.sum(images.to(torch.float64), dim=0)
                    num_samples += images.shape[0]

                if num_samples == 0:
                    logger.warning(f"Training set for '{data_flag}' is empty, cannot compute mean image.")
                    continue

                mean_image = sum_images / num_samples
                
                mean_image_float32 = mean_image.to(torch.float32)
                
                torch.save(mean_image_float32, save_path)
                logger.info(f"Successfully saved mean image for '{data_flag}' to: {save_path}")

            except Exception as e:
                logger.error(f"Error computing mean image for '{data_flag}': {e}", exc_info=True)
        
        logger.info("Mean image computation for all datasets completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute and save mean images for MedMNIST datasets for use as baselines in attribution methods.")
    parser.add_argument('--config', type=str, default="config.yml", help='Path to the main experiment configuration file.')
    args = parser.parse_args()
    
    calculator = MeanImageCalculator(config_path=args.config)
    calculator.calculate_and_save_all()
