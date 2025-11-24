import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from pathlib import Path
import sys
import logging
from collections import OrderedDict # <--- [新] 导入 OrderedDict

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
    [已修改] 加载一个 torchvision 预训练的 DenseNet 模型 (从本地文件)，并应用 DeepLIFT 补丁。
    添加了 state_dict 键名转换逻辑。
    """
    logger.info(f"Loading ImageNet-pretrained model '{model_flag}' from LOCAL weights...")

    # --- 1. 定义你的本地权重路径 ---
    local_weights_dir = weights_root
    weight_paths = {
        'densenet121': local_weights_dir / 'densenet121-a639ec97.pth'
    }

    weight_path = weight_paths.get(model_flag)
    if not weight_path or not weight_path.exists():
        logger.critical(f"本地权重文件未找到: {weight_path}")
        logger.critical(f"请确保 'densenet121-a639ec97.pth' 存在于 {local_weights_dir}")
        sys.exit(1)

    # --- 2. 初始化模型 (不带预训练权重) ---
    if model_flag == 'densenet121':
        model = torchvision_models.densenet121(weights=None)
    else:
        raise NotImplementedError(f"Model '{model_flag}' is not implemented.")

    # --- 3. 从本地文件加载权重 [已修改] ---
    try:
        # [修改] 添加 weights_only=True 来解决 FutureWarning
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        
        # [关键修复] 创建一个新的 state_dict 来存放重命名后的键
        new_state_dict = OrderedDict()
        
        logger.info("正在转换旧的 DenseNet state_dict 键名...")
        for k, v in state_dict.items():
            # 这 4 行是修复的关键
            # 将 '...norm.1...' 替换为 '...norm1...'
            # 将 '...conv.1...' 替换为 '...conv1...'
            # 将 '...norm.2...' 替换为 '...norm2...'
            # 将 '...conv.2...' 替换为 '...conv2...'
            k = k.replace("norm.1", "norm1")
            k = k.replace("conv.1", "conv1")
            k = k.replace("norm.2", "norm2")
            k = k.replace("conv.2", "conv2")
            new_state_dict[k] = v
        
        # [修改] 加载我们重命名后的 state_dict
        model.load_state_dict(new_state_dict)
        
        logger.info(f"Successfully loaded and converted weights from {weight_path}")
        
    except Exception as e:
        logger.critical(f"Failed to load local weights from {weight_path}. Error: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. (关键) 应用 DeepLIFT 补丁 ---
    logger.info("Applying patches to DenseNet blocks for DeepLIFT compatibility...")
    for module in model.modules():
        if isinstance(module, nn.ReLU) and hasattr(module, 'inplace'):
            module.inplace = False
            
    logger.info("DenseNet blocks patched successfully.")
    
    model.to(device)
    model.eval()

    return model