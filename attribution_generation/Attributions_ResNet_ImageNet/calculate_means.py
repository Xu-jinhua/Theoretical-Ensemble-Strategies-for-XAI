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
from torchvision.datasets import ImageFolder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("MeanCalculator")


class MeanImageCalculator:
    """
    A utility class for calculating and saving dataset mean images.
    """
    def __init__(self, config_path: str):
        """
        Initialize the calculator.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Successfully loaded config file: {config_path}")
        except FileNotFoundError:
            logger.critical(f"Error: Config file '{config_path}' not found.")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Error loading or parsing config file: {e}")
            sys.exit(1)
        
        self.paths = self.config['paths']
        self.search_space = self.config['search_space']
        self.means_dir = Path(self.paths['means_dir'])
        self.medmnist_data_root = Path(self.paths['medmnist_data_root'])
        
        self.means_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"All mean images will be saved to: {self.means_dir}")

    def _get_target_datasets(self) -> list[str]:
        """
        Determine the list of datasets that need mean image calculation.
        """
        target_flags = self.search_space.get('data_flags', [])
        if not target_flags:
            logger.warning("The 'data_flags' in the config file is empty, no means will be calculated.")
        return target_flags

    def calculate_and_save_all(self):
        """
        Execute mean image calculation and saving for all target datasets.
        """
        target_datasets = self._get_target_datasets()
        if not target_datasets:
            return

        for data_flag in tqdm(target_datasets, desc="Processing all datasets"):
            save_path = self.means_dir / f"{data_flag}_mean.pt"
            if save_path.exists():
                logger.info(f"Skipping: mean image for '{data_flag}' already exists at {save_path}")
                continue

            logger.info(f"Starting to calculate mean image for '{data_flag}'...")
            
            try:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

                if data_flag == 'imagenet':
                    dataset_path = self.medmnist_data_root / 'val' 
                    
                    if not dataset_path.exists():
                        logger.error(f"ImageNet 'val' directory not found: {dataset_path}")
                        continue
                    
                    logger.info(f"Using ImageFolder to load: {dataset_path} (for mean calculation)")
                    data_dataset = ImageFolder(
                        root=dataset_path,
                        transform=transform
                    )
                else:
                    info = INFO[data_flag]
                    DataClass = getattr(medmnist, info['python_class'])
                    
                    data_dataset = DataClass(
                        split='train',
                        transform=transform,
                        download=True, 
                        size=224,
                        as_rgb=True, 
                        root=self.medmnist_data_root
                    )
                
                loader = DataLoader(dataset=data_dataset, batch_size=512, shuffle=False, num_workers=8)
                
                sum_images = torch.zeros(3, 224, 224, dtype=torch.float64)
                num_samples = 0

                for images, _ in tqdm(loader, desc=f"Calculating {data_flag}", leave=False):
                    sum_images += torch.sum(images.to(torch.float64), dim=0)
                    num_samples += images.shape[0]

                if num_samples == 0:
                    logger.warning(f"The dataset for '{data_flag}' is empty, cannot calculate mean image.")
                    continue

                mean_image = sum_images / num_samples
                mean_image_float32 = mean_image.to(torch.float32)
                
                torch.save(mean_image_float32, save_path)
                logger.info(f"Successfully saved mean image for '{data_flag}' to: {save_path}")

            except Exception as e:
                logger.error(f"Error calculating mean image for '{data_flag}': {e}", exc_info=True)
        
        logger.info("Mean image calculation for all datasets completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate and save mean images for datasets to use as baselines for attribution methods.")
    parser.add_argument('--config', type=str, default="config.yml", help='Path to the experiment configuration file.')
    args = parser.parse_args()
    
    calculator = MeanImageCalculator(config_path=args.config)
    calculator.calculate_and_save_all()
