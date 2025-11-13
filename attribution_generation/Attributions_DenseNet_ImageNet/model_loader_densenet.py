import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from pathlib import Path
import sys
import logging
from collections import OrderedDict 

# 设置一个简单的 logger
logger = logging.getLogger("DenseNetModelLoader")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] (ModelLoader) %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_densenet_model(model_flag: str, weights_root: Path, device: torch.device) -> nn.Module:
    """
    Load a torchvision pretrained DenseNet model (from local file), and apply DeepLIFT patches.
    Added state_dict key conversion logic.
    """
    logger.info(f"Loading ImageNet-pretrained model '{model_flag}' from LOCAL weights...")

    local_weights_dir = weights_root
    weight_paths = {
        'densenet121': local_weights_dir / 'densenet121-a639ec97.pth'
    }

    weight_path = weight_paths.get(model_flag)
    if not weight_path or not weight_path.exists():
        logger.critical(f"Local weight file not found: {weight_path}")
        logger.critical(f"Please ensure 'densenet121-a639ec97.pth' exists in {local_weights_dir}")
        sys.exit(1)

    if model_flag == 'densenet121':
        model = torchvision_models.densenet121(weights=None)
    else:
        raise NotImplementedError(f"Model '{model_flag}' is not implemented.")

    try:
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        
        new_state_dict = OrderedDict()
        
        logger.info("Converting old DenseNet state_dict key names...")
        for k, v in state_dict.items():
            k = k.replace("norm.1", "norm1")
            k = k.replace("conv.1", "conv1")
            k = k.replace("norm.2", "norm2")
            k = k.replace("conv.2", "conv2")
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        
        logger.info(f"Successfully loaded and converted weights from {weight_path}")
        
    except Exception as e:
        logger.critical(f"Failed to load local weights from {weight_path}. Error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Applying patches to DenseNet blocks for DeepLIFT compatibility...")
    for module in model.modules():
        if isinstance(module, nn.ReLU) and hasattr(module, 'inplace'):
            module.inplace = False
            
    logger.info("DenseNet blocks patched successfully.")
    
    model.to(device)
    model.eval()

    return model