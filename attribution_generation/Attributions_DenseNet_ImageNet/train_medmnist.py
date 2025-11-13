import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import sys
import logging
from medmnist import INFO
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from model_loader_densenet import load_densenet_model

logger = logging.getLogger("MedMNIST-Train")
logger.setLevel(logging.INFO)
logger.propagate = False

DATASETS_TO_TRAIN = ["breastmnist"]

MODEL_FLAG = "densenet121"

# Set the path to your MedMNIST data directory
DATA_ROOT = Path("path/to/your/medmnist/data")

# Set the path to your ImageNet pretrained weights directory
IMAGENET_WEIGHTS_ROOT = Path("path/to/your/imagenet/weights")

# Set the path where you want to save trained MedMNIST weights
OUTPUT_WEIGHTS_ROOT = Path("path/to/save/medmnist/weights")

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MedMNISTDatasetNPZ(Dataset):
    """
    A PyTorch dataset class for directly reading MedMNIST .npz files.
    Automatically handles both 3D grayscale and 4D RGB images,
    and converts all images to 3-channel tensors.
    """
    
    def __init__(self, npz_path: Path, split: str):
        try:
            data = np.load(npz_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {npz_path}")
            raise

        self.images = data[f'{split}_images']
        self.labels = data[f'{split}_labels']

        self.images = self.images.astype(np.float32) / 255.0
        
        if self.images.ndim == 4:
            self.images = np.transpose(self.images, (0, 3, 1, 2))
            logger.info(f"Detected 4D RGB images (N, H, W, C), converted to (N, C, H, W).")
            
        elif self.images.ndim == 3:
            logger.info(f"Detected 3D grayscale images (N, H, W), converting to 3 channels...")
            self.images = np.expand_dims(self.images, axis=1)
            self.images = self.images.repeat(3, axis=1)
            
        else:
            raise ValueError(f"Unrecognized image array dimensions: {self.images.shape}")

        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"Successfully loaded {split} data: {self.images.shape[0]} samples (shape: {self.images.shape})")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        label = torch.tensor(self.labels[index], dtype=torch.long).squeeze()
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return acc

def main():
    logger.info(f"--- Starting MedMNIST model training on {DEVICE} ---")
    
    for data_flag in DATASETS_TO_TRAIN:
        
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] (Train-{data_flag}) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        log_file_path = f"training_{data_flag}.log"
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        logger.info(f"--- Processing dataset: {data_flag} ---")
        logger.info(f"Logs will be saved to: {log_file_path}")
        
        try:
            npz_path = DATA_ROOT / f"{data_flag}_224.npz"
            train_dataset = MedMNISTDatasetNPZ(npz_path, 'train')
            val_dataset = MedMNISTDatasetNPZ(npz_path, 'val')
            test_dataset = MedMNISTDatasetNPZ(npz_path, 'test')
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
        except FileNotFoundError:
            logger.warning(f"Skipping {data_flag}, because {npz_path} was not found.")
            continue
            
        info = INFO[data_flag]
        n_classes = len(info['label'])
        logger.info(f"Dataset {data_flag} has {n_classes} classes.")

        logger.info(f"Loading ImageNet pretrained {MODEL_FLAG}...")
        model = load_densenet_model(
            model_flag=MODEL_FLAG,
            weights_root=IMAGENET_WEIGHTS_ROOT,
            device=DEVICE
        )

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)
        model.to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_acc = evaluate(model, val_loader, DEVICE)
            
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Training Loss: {train_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                logger.info(f"*** New best model (validation accuracy: {best_val_acc:.4f}) ***")

        if best_model_state is None:
            logger.warning(f"Training failed for {data_flag}, model not saved.")
            continue
            
        output_filename = f"{MODEL_FLAG}_224_1.pth" 
        output_dir = OUTPUT_WEIGHTS_ROOT / f"weights_{data_flag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        torch.save({'net': best_model_state}, output_path)
        logger.info(f"Best model saved to: {output_path}")

        model.load_state_dict(best_model_state)
        test_acc = evaluate(model, test_loader, DEVICE)
        logger.info(f"--- {data_flag} final test accuracy: {test_acc:.4f} ---")

if __name__ == "__main__":
    main()
