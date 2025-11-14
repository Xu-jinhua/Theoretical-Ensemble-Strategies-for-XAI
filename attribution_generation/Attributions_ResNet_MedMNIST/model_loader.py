import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from pathlib import Path
import sys
import logging

from medmnist import INFO

logger = logging.getLogger("ModelLoader")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] (ModelLoader) %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _get_model(model_flag: str, in_channels: int, num_classes: int) -> nn.Module:
    if model_flag == 'resnet18':
        model = torchvision_models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_flag == 'resnet50':
        model = torchvision_models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise NotImplementedError(f"Model '{model_flag}' is not implemented in the loader.")
    return model


def load_medmnist_model(model_flag: str, data_flag: str, weights_root: Path, device: torch.device, run_index: int = 1) -> nn.Module:
    logger.info(f"Loading model '{model_flag}' for dataset '{data_flag}'...")
    
    info = INFO[data_flag]
    n_classes = len(info['label'])
    n_channels = 3

    model = _get_model(model_flag, n_channels, n_classes)

    weight_filename = f"{model_flag}_224_{run_index}.pth"
    weight_path = weights_root / f"weights_{data_flag}" / weight_filename

    if not weight_path.exists():
        logger.critical(f"Weight file not found at: {weight_path}")
        sys.exit(1)

    try:
        pretrain = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(pretrain['net'])
        logger.info(f"Successfully loaded weights from {weight_path}")
    except Exception as e:
        logger.critical(f"Failed to load weights from {weight_path}. Error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Applying patches to ResNet blocks for full DeepLIFT compatibility...")
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
            
    logger.info("ResNet blocks patched successfully.")
    
    model.to(device)
    model.eval()

    return model

