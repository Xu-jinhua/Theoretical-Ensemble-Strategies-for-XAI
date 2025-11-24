# XAI Ensemble Analysis & Experiments

This repository contains the core analysis code for evaluating ensemble strategies. It corresponds to the experiments (**EXP1** and **EXP2**) presented in the paper, calculating **Fidelity**, **Consistency**, and **Robustness** metrics for various explanation methods and ensemble techniques.

## Project Structure

    experiments/
    ├── Analysis_Decorr/              # Analysis for decorrelated models (Independence validation)
    ├── Analysis_ImageNet_Densenet/   # DenseNet121 analysis on ImageNet
    ├── Analysis_ImageNet_ResNet/     # ResNet18/50 analysis on ImageNet
    ├── Analysis_MedMNIST_Densenet/   # DenseNet121 analysis on MedMNIST
    └── Analysis_MedMNIST_ResNet/     # ResNet18/50 analysis on MedMNIST

## Experiments Overview

The code in these directories reproduces the results for the two main experiments discussed in the paper:

* **EXP1 (Independence Hypothesis):** Validates the impact of statistical independence by comparing standard ensembles vs. models trained on disjoint data splits.
* **EXP2 (Noise Model Alignment):** Benchmarks Naive vs. Theory-Guided ensembles (Borda, Schulze, RRF, Kemeny-Young) across different noise conditions.

---

## 1. Analysis Modules

Each subdirectory is a self-contained analysis module for a specific Dataset-Model pair.

### A. MedMNIST Experiments
**Location:** `Analysis_MedMNIST_ResNet/` and `Analysis_MedMNIST_Densenet/`
* **Target:** Reproduces tables for BloodMNIST, DermaMNIST, and BreastMNIST.
* **Input:** Requires attributions generated for MedMNIST (saved in `.zarr` or `.npz` format).
* **Usage:**

    cd Analysis_MedMNIST_ResNet/
    python scheduler.py --config config_analysis.yml

### B. ImageNet Experiments
**Location:** `Analysis_ImageNet_ResNet/` and `Analysis_ImageNet_Densenet/`
* **Target:** Reproduces tables for ImageNet-1k (5k sample subset).
* **Input:** Requires attributions generated for ImageNet.
* **Usage:**

    cd Analysis_ImageNet_Densenet/
    python scheduler.py --config config_analysis_densenet.yml

### C. Decorrelated Analysis (Independence Check)
**Location:** `Analysis_Decorr/`
* **Target:** Specifically analyzes the decorrelated models to validate the Independence Hypothesis (EXP1).
* **Usage:**

    cd Analysis_Decorr/
    python scheduler.py --config config_analysis_decoor.yml

---

## Common Workflow

All analysis modules follow a unified workflow powered by a `scheduler.py` and `worker.py` architecture for efficient processing.

1.  **Attribution Generation:** Ensure you have generated the attribution maps using the scripts in `../attribution_generation/`.
2.  **Configuration:** Edit the `config_analysis.yml` in the target folder to point to your generated attribution files.
3.  **Execution:** Run the `scheduler.py`.
4.  **Results:** The scripts will compute metrics (Fidelity, Consistency, Robustness) and save them.

---

## Configuration

Before running, you must configure the paths in the `config_analysis.yml` file within each directory.


## Ensemble Strategies

The analysis code evaluates the following aggregation methods:
* Simple Averaging (Baseline)
* Borda Count
* Schulze Method
* Reciprocal Rank Fusion (RRF)
* Kemeny-Young

## Evaluation Metrics

The scripts calculate the three key metrics defined in the paper:

* **Fidelity ($\mathcal{F}$):** Measures the drop in classification accuracy when the top-k important patches are masked.
* **Consistency ($\mathcal{C}$):** Measures the percentage of predictions that remain stable (or flip) after masking.
* **Robustness ($\mathcal{R}$):** Measures the stability of explanations when input images are subjected to noise (Gaussian, Salt & Pepper, Speckle).

---

## Troubleshooting

* **FileNotFoundError:** Double-check that `attribution_input_dir` in the config file correctly points to the output of the attribution generation step.
* **Worker Timeout/Errors:** If `worker.py` fails, check the logs in the output directory. It usually indicates an issue with loading a specific attribution file.
