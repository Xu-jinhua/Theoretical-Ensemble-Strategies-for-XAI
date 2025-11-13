import os
import sys
import yaml
import logging
from pathlib import Path

import torch
import numpy as np
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("MeanCalculator")


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


class MeanImageCalculator:
    def __init__(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
        except FileNotFoundError:
            logger.critical(f"错误: 找不到配置文件 '{config_path}'。")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"加载或解析配置文件时出错: {e}")
            sys.exit(1)
        
        self.paths = self.config['paths']
        self.search_space = self.config['search_space']
        self.means_dir = Path(self.paths['means_dir'])
        self.medmnist_data_root = Path(self.paths['medmnist_data_root'])
        
        self.means_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"所有平均图将保存到: {self.means_dir}")

    def _get_target_datasets(self) -> list[str]:
        target_flags = self.search_space.get('data_flags', [])
        if not target_flags:
            logger.warning("配置文件中的 'data_flags' 为空，将不计算任何均值。")
        return target_flags

    def calculate_and_save_all(self):
        target_datasets = self._get_target_datasets()
        if not target_datasets:
            return

        for data_flag in tqdm(target_datasets, desc="处理所有数据集"):
            save_path = self.means_dir / f"{data_flag}_mean.pt"
            if save_path.exists():
                logger.info(f"跳过: '{data_flag}' 的平均图已存在于 {save_path}")
                continue

            logger.info(f"开始计算 '{data_flag}' 的平均图...")
            
            try:
                if data_flag == 'imagenet':
                    logger.warning("calculate_means.py 正在跳过 'imagenet'。请使用 ImageNet 专用均值。")
                    continue
                else:
                    npz_path = self.medmnist_data_root / f"{data_flag}_224.npz"
                    if not npz_path.exists():
                        logger.error(f"MedMNIST .npz 文件未找到: {npz_path}")
                        continue
                        
                    data_dataset = MedMNISTDatasetNPZ(npz_path, 'train')
                
                loader = DataLoader(dataset=data_dataset, batch_size=512, shuffle=False, num_workers=8)
                sum_images = torch.zeros(3, 224, 224, dtype=torch.float64)
                num_samples = 0

                for images, _ in tqdm(loader, desc=f"计算 {data_flag}", leave=False):
                    sum_images += torch.sum(images.to(torch.float64), dim=0)
                    num_samples += images.shape[0]

                if num_samples == 0:
                    logger.warning(f"'{data_flag}' 的数据集为空，无法计算平均图。")
                    continue

                mean_image = sum_images / num_samples
                mean_image_float32 = mean_image.to(torch.float32)
                
                torch.save(mean_image_float32, save_path)
                logger.info(f"成功保存 '{data_flag}' 的平均图到: {save_path}")

            except Exception as e:
                logger.error(f"为 '{data_flag}' 计算平均图时发生错误: {e}", exc_info=True)
        
        logger.info("所有数据集的平均图计算完成。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="为数据集计算并保存平均图，用作归因方法的基线。")
    parser.add_argument('--config', type=str, default="config.yml", help='指向实验总配置文件的路径。')
    args = parser.parse_args()
    
    calculator = MeanImageCalculator(config_path=args.config)
    calculator.calculate_and_save_all()