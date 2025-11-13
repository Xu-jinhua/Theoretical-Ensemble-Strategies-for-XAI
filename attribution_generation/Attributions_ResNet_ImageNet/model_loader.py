import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from pathlib import Path
import sys
import logging

# Set up a simple logger
logger = logging.getLogger("ModelLoader")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] (ModelLoader) %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_medmnist_model(model_flag: str, data_flag: str, weights_root: Path, device: torch.device, run_index: int = 1) -> nn.Module:
    """
    Load a torchvision pretrained ImageNet model (from local file), and apply DeepLIFT patches.
    """
    logger.info(f"Loading ImageNet-pretrained model '{model_flag}' from LOCAL weights...")

    # Define the local weights directory path
    local_weights_dir = Path("/Path/to/the/model/weights") 
    weight_paths = {
        'resnet18': local_weights_dir / 'resnet18-f37072fd.pth',
        'resnet50': local_weights_dir / 'resnet50-11ad3fa6.pth'
    }

    weight_path = weight_paths.get(model_flag)
    if not weight_path or not weight_path.exists():
        logger.critical(f"Local weight file not found: {weight_path}")
        logger.critical("Please modify the 'local_weights_dir' variable in model_loader.py.")
        sys.exit(1)

    # Initialize the model without pretrained weights
    if model_flag == 'resnet18':
        model = torchvision_models.resnet18(weights=None)
    elif model_flag == 'resnet50':
        model = torchvision_models.resnet50(weights=None)
    else:
        raise NotImplementedError(f"Model '{model_flag}' is not implemented.")

    # Load weights from local file
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded local weights from {weight_path}")
    except Exception as e:
        logger.critical(f"Failed to load local weights from {weight_path}. Error: {e}", exc_info=True)
        sys.exit(1)

    # Apply DeepLIFT patches
    logger.info("Applying patches to ResNet blocks for full DeepLIFT compatibility...")
    for module in model.modules():
        # Patch BasicBlock (used in ResNet-18)
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

        # Patch Bottleneck (used in ResNet-50)
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
