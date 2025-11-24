"""
General utility functions for the analysis framework.
"""
import torch
import json
import numpy as np
import logging
from pathlib import Path
import torch.nn as nn
from torchvision import models as torchvision_models
from medmnist import INFO

# ===================================================================
# Logging
# ===================================================================

def setup_logging(log_file_path, level=logging.INFO, name="Experiment"):
    """Sets up a logger that outputs to both a file and the console."""
    logger = logging.getLogger(name)
    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
            
    logger.setLevel(level)
    formatter = logging.Formatter(
        f"[%(asctime)s] [%(levelname)s] [{name}/%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Ensure log directory exists
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # File handler
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler (console)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

# ===================================================================
# Model Loading (MedMNIST Official Method)
# ===================================================================

def load_model(model_flag: str, data_flag: str, weights_root: str, device: torch.device, run_index: int = 1) -> nn.Module:
    """
    Loads a MedMNIST official pretrained model and applies all necessary patches
    to ensure compatibility with Captum's DeepLIFT.
    """
    print(f"Loading model '{model_flag}' for dataset '{data_flag}'...")

    info = INFO[data_flag]
    n_classes = len(info['label'])
    n_channels = 3  # All MedMNIST 2D datasets are 3-channel

    if model_flag == 'resnet18':
        model = torchvision_models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif model_flag == 'resnet50':
        model = torchvision_models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    else:
        raise NotImplementedError(f"Model '{model_flag}' is not implemented in the loader.")

    weight_filename = f"{model_flag}_224_{run_index}.pth"
    weight_path = Path(weights_root) / f"weights_{data_flag}" / weight_filename

    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found at: {weight_path}")

    try:
        pretrain = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(pretrain['net'])
        print(f"Successfully loaded weights from {weight_path}")
    except Exception as e:
        print(f"Failed to load weights from {weight_path}. Error: {e}")
        raise e

    # --- Apply patches to ResNet blocks for full DeepLIFT compatibility ---
    print("Applying patches to ResNet blocks for full DeepLIFT compatibility...")
    for module in model.modules():
        if isinstance(module, torchvision_models.resnet.BasicBlock):
            module.relu.inplace = False
            module.relu2 = nn.ReLU(inplace=False)

            def new_basic_forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = out + identity
                out = self.relu2(out)
                return out
            module.forward = new_basic_forward.__get__(module, module.__class__)

        if isinstance(module, torchvision_models.resnet.Bottleneck):
            module.relu.inplace = False
            module.relu2 = nn.ReLU(inplace=False)
            module.relu3 = nn.ReLU(inplace=False)

            def new_bottleneck_forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu2(out)
                out = self.conv3(out)
                out = self.bn3(out)
                if self.downsample is not None:
                    identity = self.downsample(x)
                out = out + identity
                out = self.relu3(out)
                return out
            module.forward = new_bottleneck_forward.__get__(module, module.__class__)

    print("ResNet blocks patched successfully.")

    model.to(device)
    model.eval()

    return model

# ===================================================================
# Core Data Transformation
# ===================================================================

def attribution_to_rank_batch(attr_images, patch_size):
    """
    Converts a batch of attribution images to a rank matrix.
    Input shape: [B, C, H, W]
    """
    # Take the mean across channels and then the absolute value
    abs_images = torch.abs(attr_images).mean(dim=1, keepdim=True)
    
    # Create non-overlapping patches
    patches = abs_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # Calculate the mean score for each patch
    patch_scores = patches.mean(dim=(-1, -2)).squeeze(1)
    
    # Get ranks from scores (lower rank is more important)
    flat_scores = patch_scores.flatten(start_dim=1)
    ranks = torch.argsort(torch.argsort(flat_scores, dim=1, descending=True), dim=1, descending=False)
    
    return ranks.reshape(patch_scores.shape)

# ===================================================================
# JSON Serialization Helper
# ===================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy and PyTorch types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)