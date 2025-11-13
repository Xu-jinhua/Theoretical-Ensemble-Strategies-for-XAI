# XAI Attribution Generation Experiments

This repository contains code for generating attribution explanations for various deep learning models on MedMNIST and ImageNet datasets.

## Project Structure

```
attribution_generation/
├── Attribution_decorr/           # Decorrelated attribution generation
├── Attributions_DenseNet_ImageNet/  # DenseNet on ImageNet
├── Attributions_DenseNet_MedMNIST/  # DenseNet on MedMNIST
├── Attributions_ResNet_ImageNet/    # ResNet on ImageNet
└── Attributions_ResNet_MedMNIST/    # ResNet on MedMNIST
```

## Experiments Overview

### 1. Decorrelated Attribution Generation (Attribution_decorr/)
Generates decorrelated attributions using independent models, each paired with a specific explanation method.

**Key Features:**
- Decorrelated models for dataset
- Different explanation methods
- Each model-method pair generates unique attributions

**Usage:**
```bash
cd Attribution_decorr/
python scheduler.py --config config.yml
```

### 2. DenseNet ImageNet (Attributions_DenseNet_ImageNet/)
Generates attributions for DenseNet models on ImageNet dataset

**Key Features:**
- DenseNet121 architecture
- ImageNet-1k dataset
- Multiple noise conditions and explanation methods

**Usage:**
```bash
cd Attributions_DenseNet_ImageNet/
python scheduler.py --config config_densenet.yml
```

### 3. DenseNet MedMNIST (Attributions_DenseNet_MedMNIST/)
Generates attributions for DenseNet models on MedMNIST datasets.

**Key Features:**
- DenseNet121 architecture
- Multiple MedMNIST datasets (bloodmnist, dermamnist, breastmnist)
- Comprehensive explanation method coverage

**Usage:**
```bash
cd Attributions_DenseNet_MedMNIST/
python scheduler.py --config config.yml
```

### 4. ResNet ImageNet (Attributions_ResNet_ImageNet/)
Generates attributions for ResNet models on ImageNet dataset.

**Key Features:**
- ResNet18 and ResNet50 architectures
- ImageNet dataset
- Multiple noise conditions

**Usage:**
```bash
cd Attributions_ResNet_ImageNet/
python scheduler.py --config config.yml
```

### 5. ResNet MedMNIST (Attributions_ResNet_MedMNIST/)
Generates attributions for ResNet models on MedMNIST datasets.

**Key Features:**
- ResNet18 and ResNet50 architectures
- Multiple MedMNIST datasets
- Comprehensive explanation methods

**Usage:**
```bash
cd Attributions_ResNet_MedMNIST/
python scheduler.py --config config.yml
```

## Common Workflow

All experiments follow a similar workflow:

1. **Configuration**: Edit the config.yml file to set paths and parameters
2. **Pre-computation**: Calculate mean images (one-time setup)
3. **Pre-flight Check** (Optional but recommended): Generate batch size configurations
4. **Attribution Generation**: Run the scheduler to generate attributions
5. **Output**: Results saved as Zarr files or .pt files

### Pre-flight Check

Some experiments require a pre-flight check to determine optimal batch sizes for each explanation method. This prevents out-of-memory errors and improves efficiency.

**To run pre-flight check:**
```bash
cd <experiment_directory>/
python preflight_checker.py --config config.yml
```

This will generate a `preflight_results.json` file that the scheduler will use to set appropriate batch sizes for each explanation method.

**Note**: The pre-flight check is enabled by default in most config files (`preflight_check.enabled: true`). If you want to skip it or use default batch sizes, you can set `preflight_check.enabled: false` in the config file.

## Configuration

Before running any experiment, you need to configure the paths in the respective config.yml files:

### Required Path Configurations

```yaml
paths:
  # Model weights directory
  medmnist_weights_root: "<PATH_TO_MEDMNIST_WEIGHTS>"
  
  # Dataset directory
  medmnist_data_root: "<PATH_TO_MEDMNIST_DATA>"
  
  # Pre-computed means directory
  means_dir: "<PATH_TO_PRECOMPUTED_MEANS>"
  
  # Output directory
  final_output_dir: "<PATH_TO_FINAL_OUTPUT>"
```

### Explanation Methods

All experiments support the following explanation methods:
- **Saliency**
- **InputXGradient**
- **GuidedBackprop**
- **Deconvolution**
- **IntegratedGradients** (n_steps: 50, 200)
- **DeepLIFT**
- **GradientShap** (n_samples: 20, 40)
- **DeepLIFT_SHAP**
- **FeatureAblation** (patch_size: 8, 14, 16)
- **Occlusion** (patch_size: 8, 14, 16)

### Noise Conditions

- **clean**: No noise
- **gaussian_std_0.15**: Gaussian noise with std=0.15
- **salt_pepper_amount_0.05**: Salt and pepper noise with amount=0.05
- **speckle_std_0.15**: Speckle noise with std=0.15

## Data Format

### Input
- MedMNIST: .npz files
- ImageNet: Zarr files

### Output
- **Zarr format**: For most experiments, results saved as .zarr or .zarr.zip files
- **PT format**: For DenseNet ImageNet experiment, results saved as .pt files

Each output file contains:
- Images
- Labels
- Model predictions
- Attribution maps for all explanation methods
- Mean images (global and per-class)

## GPU Requirements

- Minimum 1 GPU with 8GB VRAM
- Recommended: Multiple GPUs for parallel processing
- Experiments automatically distribute tasks across available GPUs

## Troubleshooting

1. **Out of Memory**: Reduce batch sizes in config.yml
2. **Missing Files**: Ensure all path configurations are correct
3. **CUDA Errors**: Check GPU availability and PyTorch installation



