AdaptFNO: Adaptive Fourier Neural Operator for Climate Modeling & Inpainting
Original Authors: Hiep Vo Dang (Yeshiva University), Bach D. G. Nguyen (Michigan State University), Phong C. H. Nguyen (Phenikaa University), Truong-Son Hy (UAB)

Implementation Extension: Wind Field Inpainting on CERRA Data

📖 Overview
Fourier Neural Operators (FNOs) are powerful for modeling spatio-temporal dynamics but often overlook fine-scale details. AdaptFNO introduces an adaptive variant that combines Global Operators (low-res context) and Local Operators (high-res details) via a cross-attention mechanism.

🌪️ Extension: Wind Field Inpainting
While the original paper focused on forecasting, this repository has been adapted for Spatial Inpainting (Reconstruction) of wind fields from sparse observations.

This implementation:

Reconstructs dense wind fields from sparse station-like data (masked inputs).

Uses CERRA reanalysis data (NetCDF format).

Operates in a multiscale manner: A Global Operator sees the coarse sparse grid, while the Local Operator fills fine details using cross-attention context.

📐 Architecture
The model uses a dual-branch architecture:

Global Branch: Processes downsampled sparse inputs to capture large-scale flow.

Local Branch: Processes high-resolution sparse patches.

Cross-Attention: Bridges the two, allowing the local reconstruction to be consistent with global atmospheric patterns.

📂 Repository Structure
The code is organized for modular training and grid search on HPC clusters:

Plaintext

AdaptFNO/
│
├── data/
│   ├── cerra_dataset.py       # Xarray/NetCDF loader for CERRA (Sparse + Mask)
│   └── __init__.py
│
├── models/
│   ├── modules.py             # Core AFNO layers (PatchEmbed, MLP, etc.)
│   ├── operators.py           # Global & Local Operator definitions
│   └── adaptfno_inpainting.py # Wrapper combining Global+Local for inpainting
│
├── utils/
│   ├── loss.py                # Inpainting Loss (MSE + L1 + Gradient + Consistency)
│   └── __init__.py
│
├── configs/                   # Configuration files
│   ├── config.yaml            # Main training configuration
│   └── search/                # Generated configs for grid search
│
├── train.py                   # Main training script (loads config.yaml)
├── grid_search.py             # Script to generate configs for hyperparam search
├── submit_grid.sh             # Slurm script for running job arrays
└── README.md
⚙️ Installation
Clone the repository and install dependencies. Note the addition of xarray and netCDF4 for handling CERRA data.

Bash

git clone https://github.com/YourUsername/AdaptFNO.git
cd AdaptFNO
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xarray netCDF4 pyyaml tqdm
Requirements:

Python 3.9+

PyTorch 2.0+

xarray, netCDF4 (Data Loading)

pyyaml (Configuration)

📊 Dataset: CERRA
This implementation is designed for CERRA (Copernicus European Regional Reanalysis) data in NetCDF (.nc) format.

Input Format: The model expects pairs of NetCDF files for training and validation:

Ground Truth (.nc): Dense wind speed data.

Mask (.nc): Binary mask (1 = valid observation, 0 = missing).

Preprocessing: The CERRADataset loader automatically:

Reads .nc files using xarray.

Applies the mask to create sparse inputs (Sparse = GT * Mask).

Auto-detects variable names (e.g., ws, u10, mask).

🚀 Usage
1. Configuration (config.yaml)
All training parameters are controlled via config.yaml. Edit this file to point to your specific NetCDF paths:

YAML

dataset:
  train_gt: "path/to/train_wind.nc"
  train_mask: "path/to/train_mask.nc"
  val_gt: "path/to/val_wind.nc"
  val_mask: "path/to/val_mask.nc"

architecture:
  img_size: [128, 128]      # Will be auto-detected if matching data
  patch_size: [16, 16]      # Must divide img_size!
  embedding_dim: 256
  
training:
  batch_size: 4
  learning_rate: 1.0e-4
  n_epochs: 100
  save_dir: "./checkpoints"
2. Training
Run the training script with your configuration:

Bash

python train.py --config config.yaml
The script will auto-detect your image resolution and warn you if the patch_size is incompatible.

🔬 Hyperparameter Search (HPC / Slurm)
For research experiments, use the included grid search tools to run parameter sweeps on a Slurm cluster.

Step 1: Generate Configs Edit grid_search.py to define your parameter grid (e.g., learning rates, embedding dims). Then run:

Bash

python grid_search.py
# Output: Generated 12 configuration files in configs/search/
Step 2: Submit Job Array Update submit_grid.sh to match the number of files (e.g., #SBATCH --array=0-11) and submit:

Bash

sbatch submit_grid.sh
This runs all experiments in parallel, saving separate checkpoints for each run.

📑 Citation
If you use the AdaptFNO architecture, please cite the original workshop paper:

Snippet di codice

@inproceedings{dang2025adaptfno,
  title={AdaptFNO: Adaptive Fourier Neural Operator with Dynamic Spectral Modes and Multiscale Learning for Climate Modeling},
  author={Dang, Hiep Vo and Nguyen, Bach D.G. and Nguyen, Phong C.H. and Hy, Truong-Son},
  booktitle={NeurIPS 2025 Workshop on Machine Learning and the Physical Sciences},
  year={2025}
}
🤝 Acknowledgements
Original AdaptFNO implementation by HySonLab.

CERRA dataset (Copernicus Climate Change Service).

Fourier Neural Operator (Li et al., ICLR 2021).
