# AdaptFNO: Adaptive Fourier Neural Operator for Climate Modeling & Inpainting

[![NeurIPS 2025 Workshop](https://img.shields.io/badge/NeurIPS%202025-Workshop-blue)](https://neurips.cc/Conferences/2025/Schedule)

**Original Authors:** Hiep Vo Dang (Yeshiva University), Bach D. G. Nguyen (Michigan State University), Phong C. H. Nguyen (Phenikaa University), Truong-Son Hy (University of Alabama at Birmingham) - Correspondence to thy@uab.edu

**Implementation Extension:** Wind Field Inpainting on CERRA Data

---

## 📖 Overview

Fourier Neural Operators (FNOs) are powerful for modeling spatio-temporal dynamics but often emphasize low-frequency patterns, overlooking fine-scale details critical in climate forecasting. **AdaptFNO** introduces an adaptive variant that:

- **Dynamically adjusts spectral modes** based on input frequency content.
- **Combines global and local operators** for multiscale learning.
- Uses a **cross-attention mechanism** to align global and local features.

### 🌪️ Extension: Wind Field Inpainting
While the original paper focused on forecasting future states, this repository has been adapted for **Spatial Inpainting** (Reconstruction) of wind fields from sparse observations. 

This implementation:
- **Reconstructs dense wind fields** from sparse station-like data (masked inputs).
- Uses **CERRA reanalysis data** (NetCDF format).
- Operates in a **multiscale manner**: A Global Operator sees the coarse sparse grid, while the Local Operator fills fine details using cross-attention context.

---

## 📐 Architecture

The model uses a dual-branch architecture:
1.  **Global Branch:** Processes downsampled sparse inputs to capture large-scale flow.
2.  **Local Branch:** Processes high-resolution sparse patches.
3.  **Cross-Attention:** Bridges the two, allowing the local reconstruction to be consistent with global atmospheric patterns.

![AdaptFNO Architecture](AdaptFNO.png)

---

## 📂 Repository Structure

The code is organized for modular training and grid search on HPC clusters:

```text
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
