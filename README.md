# Theoretical-Ensemble-Strategies-for-XAI
Official implementation for "Theoretical Insights into Ensemble Strategies for Image Post-Hoc Explanation"

**Abstract.**  *Post-hoc explanation methods of image recognition models often exhibit high variance or disagreement across explanations when the input data are perturbed, the underlying models are modified, or different explainability techniques are employed. To mitigate this issue, several approaches have been proposed, among which ensemble strategies that aggregate multiple explanations have attracted particular attention. Although some of these methods demonstrate good empirical performance, most existing works remain largely empirical, with limited theoretical justification or understanding of why ensemble strategies work and when they fail. In this study, we investigate different datasets, convolutional neural network architectures, post-hoc explanation techniques, and ensembling strategies to identify the most influential image patches. In particular, we compare various ensembling strategies based on distinct voting principles - namely, Borda Count, Kemeny–Young, Reciprocal Rank Fusion, and the Schulze method - and show that the performance of such approaches depends on the degree of satisfaction of their underlying theoretical assumptions.*


## Paper & Supplementary Materials

The full results and detailed tables corresponding to the experiments in our paper can be found below:

| Document | Description | Link |
| :--- | :--- | :--- |
| **Supplementary A** | **Detailed Results for Experiment 1 (EXP1):** Validation of the Independence Hypothesis and comparison of ensembling strategies on disjoint data subsets. | [Download PDF](./results/Exp1Table.pdf) |
| **Supplementary B** | **Detailed Results for Experiment 2 (EXP2):** Comprehensive performance comparison of Naive vs. Theory-Guided ensembles across ImageNet and MedMNIST datasets. | [Download PDF](./results/Exp2Table.pdf) |

## Data Preparation and Attribution Generation (`attribution_generation/`)

This section handles all data preparation and the generation of the attribution maps used in our study.

### 1. Datasets

Our experiments rely on two primary datasets. Please download them and set the correct paths in the config files inside `attribution_generation/`.

* **MedMNIST v2:** A collection of 12 2D and 3D biomedical image datasets. The `.npz` files can be found from the official repository:
    * **Download:** [MedMNIST GitHub Repository](https://github.com/MedMNIST/MedMNIST)
    * *Set the `medmnist_data_root` path in the config files.*

* **ImageNet:**
    * The ImageNet dataset is available at [https://www.image-net.org/](https://www.image-net.org/).

### 2. Attribution Generation Process

The `attribution_generation/` folder contains all scripts required to generate the **different explanation attribution maps** (Saliency, IntegratedGradients, DeepLIFT, Occlusion, etc.) for multiple models (ResNet, DenseNet) under **different noise conditions** (clean, gaussian, salt-pepper, speckle).

The generation process is highly configurable and managed by `scheduler.py` scripts. For a complete, step-by-step guide on configuring paths, running pre-flight checks, and generating the final Zarr/PT files, please refer to the detailed README within that folder:

* [Full Instructions: `attribution_generation/README.md`](./attribution_generation/README.md)
