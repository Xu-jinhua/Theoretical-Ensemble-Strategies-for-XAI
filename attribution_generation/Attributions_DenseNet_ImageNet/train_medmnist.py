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

# Import our loader for loading ImageNet weights
from model_loader_densenet import load_densenet_model

# --- Logging setup ---
# [Modified] Remove basicConfig, we'll configure dynamically in main function
logger = logging.getLogger("MedMNIST-Train")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent logs from propagating to possibly configured root logger

# --- 1. Core Configuration ---

# Datasets to train
DATASETS_TO_TRAIN = ["breastmnist"]

# Model to use
MODEL_FLAG = "densenet121"

# [Reference] Input data path (from your ls)
DATA_ROOT = Path("/mnt/shared_storage/xujinhua/XAI/data")

# [Reference] Path to ImageNet pretrained weights
IMAGENET_WEIGHTS_ROOT = Path("/mnt/shared_storage/xujinhua/XAI/weights")

# [Reference] Output path for new MedMNIST weights (!!must match 'medmnist_weights_root' in your old config.yml)
OUTPUT_WEIGHTS_ROOT = Path("/mnt/shared_storage/xujinhua/XAI/weights")

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 2. Custom Dataset Class ---

class MedMNISTDatasetNPZ(Dataset):
    """
    [Fixed]
    A PyTorch dataset class for directly reading MedMNIST .npz files.
    Now handles both 3D grayscale and 4D RGB images,
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

        # 1. Convert to float32 and normalize to [0, 1]
        self.images = self.images.astype(np.float32) / 255.0
        
        # --- [Critical fix] ---
        if self.images.ndim == 4:
            # 4D array (N, H, W, C), assumed to be RGB
            # Transpose axes (N, H, W, C) -> (N, C, H, W)
            self.images = np.transpose(self.images, (0, 3, 1, 2))
            logger.info(f"Detected 4D RGB images (N, H, W, C), converted to (N, C, H, W).")
            
        elif self.images.ndim == 3:
            # 3D array (N, H, W), assumed to be grayscale
            logger.info(f"Detected 3D grayscale images (N, H, W), converting to 3 channels...")
            # 1. Add channel dimension: (N, H, W) -> (N, 1, H, W)
            self.images = np.expand_dims(self.images, axis=1)
            # 2. Repeat this channel 3 times: (N, 1, H, W) -> (N, 3, H, W)
            self.images = self.images.repeat(3, axis=1)
            
        else:
            raise ValueError(f"Unrecognized image array dimensions: {self.images.shape}")
        # --- [Fix end] ---

        # [Reference] Define normalization consistent with your previous framework (-1 to 1)
        # (transform here is (x - 0.5) * 2)
        # All images are now 3-channel, so this normalization is safe
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"Successfully loaded {split} data: {self.images.shape[0]} samples (shape: {self.images.shape})")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        # Convert to tensor
        image = torch.from_numpy(self.images[index])
        
        # [Reference] Labels in MedMNIST .npz are (N, 1), CrossEntropyLoss needs (N,)
        label = torch.tensor(self.labels[index], dtype=torch.long).squeeze()
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# --- 3. Training and Evaluation Functions ---

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

# --- 4. Main Execution Loop ---
def main():
    # This log will only print to console once
    logger.info(f"--- Starting MedMNIST model training on {DEVICE} ---")
    
    for data_flag in DATASETS_TO_TRAIN:
        
        # --- [New] Set up dedicated logging for this task ---
        
        # 1. Remove all old handlers from this logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # 2. Define formatter
        formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] (Train-{data_flag}) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 3. Create file handler (e.g., training_bloodmnist.log)
        # Logs will be saved in the directory where you run the script
        log_file_path = f"training_{data_flag}.log"
        fh = logging.FileHandler(log_file_path, mode='w')  # 'w' mode ensures fresh log each run
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # 4. Create stream handler (for console output)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        # --- Logging setup complete ---

        logger.info(f"--- Processing dataset: {data_flag} ---")
        logger.info(f"Logs will be saved to: {log_file_path}")
        
        # 1. Load data
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
            
        # 2. Get model metadata
        info = INFO[data_flag]
        n_classes = len(info['label'])
        logger.info(f"Dataset {data_flag} has {n_classes} classes.")

        # 3. Load pretrained model
        #
        logger.info(f"Loading ImageNet pretrained {MODEL_FLAG}...")
        model = load_densenet_model(
            model_flag=MODEL_FLAG,
            weights_root=IMAGENET_WEIGHTS_ROOT,
            device=DEVICE
        )

        # 4. Replace classifier head for transfer learning
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, n_classes)
        model.to(DEVICE)
        
        # 5. Set up training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_acc = 0.0
        best_model_state = None

        # 6. Training loop
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_acc = evaluate(model, val_loader, DEVICE)
            
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Training Loss: {train_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                logger.info(f"*** New best model (validation accuracy: {best_val_acc:.4f}) ***")

        # 7. Save best model
        if best_model_state is None:
            logger.warning(f"Training failed for {data_flag}, model not saved.")
            continue
            
        # [Reference] Target path format from old model_loader.py
        output_filename = f"{MODEL_FLAG}_224_1.pth" 
        # [Reference] Target directory structure from old config.yml and model_loader.py
        output_dir = OUTPUT_WEIGHTS_ROOT / f"weights_{data_flag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        # [Reference] Save format {'net': ...} from old model_loader.py
        torch.save({'net': best_model_state}, output_path)
        logger.info(f"Best model saved to: {output_path}")

        # 8. (Optional) Evaluate on test set
        model.load_state_dict(best_model_state)
        test_acc = evaluate(model, test_loader, DEVICE)
        logger.info(f"--- {data_flag} final test accuracy: {test_acc:.4f} ---")

if __name__ == "__main__":
    main()
