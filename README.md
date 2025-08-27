# DETR3D: A PyTorch Re-implementation

This repository contains an unofficial PyTorch re-implementation of the paper: **"DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries"**.

DETR3D presents a streamlined, end-to-end approach for 3D object detection from multiple camera views, eliminating the need for non-maximum suppression (NMS) and other complex post-processing steps.

**Link to the original paper:** https://arxiv.org/abs/2110.06922

## 📋 Features

- **Training Script:** A simple and clean script to train the DETR3D model on the nuScenes dataset.
- **Configurable Parameters:** Easily manage all hyperparameters, dataset paths, and model configurations through a single YAML file.
- **Built-in Visualization:** Automatically generates and saves visualizations of detection results in the `vis/` directory after each validation epoch.

## 🚀 Getting Started

Follow these steps to set up the environment, prepare the dataset, and start training the model.

### 1. Installation

First, clone the repository and install the required dependencies. It's recommended to use a virtual environment.

### 2. Dataset Preparation

This implementation uses the nuScenes dataset. You need to preprocess it before training.

1. Download the official nuScenes dataset from their website.

2. Run the provided script to generate the preprocessed data files required for training:

```bash
python3 datasets_all/save_nusc_3d.py
```

This script will create the necessary data splits and annotations.

### 3. Configuration

All settings for training are managed in the `configs/red.yaml` file. Before running, make sure to update it with the correct paths and desired hyperparameters, ex - `batch_size`, `learning_rate`, etc. Adjust these as needed for your setup.

### 4. Training

To start training the model, run the `train.py` script:

```bash
python3 train.py
```

**Note on GPUs:** To specify the number of GPUs for training, please modify the relevant parameter directly within the `train.py` script before running it.

### 5. Visualization

During training, the model's performance on the validation set will be visualized automatically. After each validation step, output images showing the 3D bounding box predictions will be saved to the `vis/` directory.

## 📜 Citation

**If you find this work useful in your research, please consider citing the original DETR3D paper:**

```bibtex
@inproceedings{wang2022detr3d,
  title={{DETR3D}: {3D} Object Detection from Multi-view Images via {3D-to-2D} Queries},
  author={Wang, Yue and Gu, Vignesh and Ogale, Aniket and Dai, Davis and Gremmes, Eric and Pfrommer, Thomas},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2022}
}
```
