import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from pathlib import Path
import sys
import logging
from collections import OrderedDict # <--- [新] 导入 OrderedDict
from medmnist import INFO # [新] 需要这个来获取类别数

# 设置一个简单的 logger
logger = logging.getLogger("DenseNetModelLoader")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] (ModelLoader) %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_densenet_model(model_flag: str, data_flag: str, weights_root: Path, device: torch.device) -> nn.Module:
    """
    [已修改] 加载 DenseNet 模型。
    - 如果 data_flag == 'imagenet', 加载 ImageNet 预训练权重。
    - 如果 data_flag 是 MedMNIST 数据集, 加载对应的 MedMNIST 训练权重。
    """
    logger.info(f"Loading model '{model_flag}' for dataset '{data_flag}'...")

    # --- 1. MedMNIST 权重加载逻辑 ---
    if data_flag in ["bloodmnist", "breastmnist", "dermamnist"]:
        logger.info(f"Loading MedMNIST-trained model from LOCAL weights...")

        # 1a. 获取 MedMNIST 特定信息
        info = INFO[data_flag]
        n_classes = len(info['label'])
        n_channels = 3 # 假设所有都是 3 通道

        # 1b. 构建权重路径
        # e.g., /.../weights/weights_bloodmnist/densenet121_224_1.pth
        weight_path = weights_root / f"weights_{data_flag}" / f"{model_flag}_224_1.pth"

        if not weight_path.exists():
            logger.critical(f"MedMNIST 权重文件未找到: {weight_path}")
            sys.exit(1)

        # 1c. 初始化模型 (不带预训练权重，但有正确的输出类别)
        if model_flag == 'densenet121':
            model = torchvision_models.densenet121(weights=None)
            # 替换全连接层以匹配 MedMNIST 的类别数
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_classes)
        else:
            raise NotImplementedError(f"Model '{model_flag}' is not implemented for MedMNIST.")

        # 1d. 加载 MedMNIST 权重 (通常是 checkpoint 字典)
        try:
            # [修改] MedMNIST 权重是 checkpoint，不是 state_dict
            pretrain = torch.load(weight_path, map_location='cpu', weights_only=False) # 设为 False

            # [关键] MedMNIST 权重通常保存在 'net' 键中
            # 并且它们可能没有经过转换
            state_dict = pretrain['net']

            new_state_dict = OrderedDict()
            logger.info("正在转换 MedMNIST DenseNet state_dict 键名...")
            for k, v in state_dict.items():
                # 模块可能被包裹在 'module.' 中 (如果是 DataParallel 训练的)
                if k.startswith('module.'):
                    k = k[7:] # 移除 'module.'

                k = k.replace("norm.1", "norm1")
                k = k.replace("conv.1", "conv1")
                k = k.replace("norm.2", "norm2")
                k = k.replace("conv.2", "conv2")
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            logger.info(f"Successfully loaded and converted MedMNIST weights from {weight_path}")

        except Exception as e:
            logger.critical(f"Failed to load local MedMNIST weights from {weight_path}. Error: {e}", exc_info=True)
            sys.exit(1)

    # --- 2. ImageNet 权重加载逻辑 (保留) ---
    elif data_flag == 'imagenet':
        logger.info(f"Loading ImageNet-pretrained model '{model_flag}' from LOCAL weights...")

        local_weights_dir = weights_root
        weight_paths = {
            'densenet121': local_weights_dir / 'densenet121-a639ec97.pth'
        }
        weight_path = weight_paths.get(model_flag)

        if not weight_path or not weight_path.exists():
            logger.critical(f"本地 ImageNet 权重文件未找到: {weight_path}")
            sys.exit(1)

        if model_flag == 'densenet121':
            model = torchvision_models.densenet121(weights=None) # 默认 1000 类
        else:
            raise NotImplementedError(f"Model '{model_flag}' is not implemented.")

        try:
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
            new_state_dict = OrderedDict()
            logger.info("正在转换旧的 ImageNet DenseNet state_dict 键名...")
            for k, v in state_dict.items():
                k = k.replace("norm.1", "norm1")
                k = k.replace("conv.1", "conv1")
                k = k.replace("norm.2", "norm2")
                k = k.replace("conv.2", "conv2")
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            logger.info(f"Successfully loaded and converted ImageNet weights from {weight_path}")

        except Exception as e:
            logger.critical(f"Failed to load local ImageNet weights from {weight_path}. Error: {e}", exc_info=True)
            sys.exit(1)

    else:
        raise ValueError(f"未知的数据集: {data_flag}")

    # --- 3. (关键) 应用 DeepLIFT 补丁 (通用) ---
    logger.info("Applying patches to DenseNet blocks for DeepLIFT compatibility...")
    for module in model.modules():
        if isinstance(module, nn.ReLU) and hasattr(module, 'inplace'):
            module.inplace = False

    logger.info("DenseNet blocks patched successfully.")

    model.to(device)
    model.eval()

    return model