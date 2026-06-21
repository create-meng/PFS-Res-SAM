# PFS-Res-SAM: Progressive Full-image Scoring Res-SAM for GPR Anomaly Detection

A progressive GPR anomaly detection framework built around a small normal feature bank,
full-image patch anomaly scoring, dense score-map construction, connected-component
region extraction, and hierarchical false-positive suppression.

## Overview

PFS-Res-SAM detects subsurface anomalies in GPR B-scans using only a small set of normal
reference samples (default: 20 intact images). The pipeline:

1. **Feature bank construction** — extract 2D-ESN dynamic features from normal patches
2. **Patch-level anomaly scoring** — compute patch-wise L2 distance to the normal bank
3. **Dense score map** — project patch scores onto the full image plane
4. **Connected-region extraction** — threshold and segment the score map
5. **Hierarchical filtering** — suppress false positives in resized and original image coordinates
6. **Image-level scoring** — aggregate patch scores for ROC/AUC evaluation

---

## Quick Start

### Prerequisites

- Python 3.9
- CUDA 11.8 (GPU) or CPU-only

### 1. Clone

```bash
git clone https://github.com/create-meng/PFS-Res-SAM.git
cd PFS-Res-SAM
```

### 2. Environment

**GPU (recommended):**
```bash
conda create -n pfs-res-sam python=3.9 -y
conda activate pfs-res-sam
pip install -r requirements-gpu.txt
```

**CPU:**
```bash
conda create -n pfs-res-sam python=3.9 -y
conda activate pfs-res-sam
pip install -r requirements-cpu.txt
```

### 3. Download SAM weight

Download the SAM ViT-L checkpoint (required by the pipeline):

```bash
# Linux / WSL
aria2c -x 16 -s 16 -k 1M \
  -o sam_vit_l_0b3195.pth \
  -d sam \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
```

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -OutFile sam\sam_vit_l_0b3195.pth
```

The file should be placed at `sam/sam_vit_l_0b3195.pth` (~1.2 GB).

### 4. Prepare data

Download the GPR dataset from Mendeley Data:

> Mojahid et al. (2024) — [Intelligent recognition of subsurface utilities and voids:
> A Ground Penetrating Radar dataset for Deep Learning applications](https://doi.org/10.17632/ww7fd9t325.1)

Expected layout under a `data/` directory (create if missing):

```
data/GPR_data/
├── augmented_intact/        # Normal samples (900 images) — for feature bank + AUC normal pool
├── augmented_cavities/      # Cavity anomalies (553 images, with VOC/YOLO annotations)
└── augmented_utilities/     # Utility-type anomalies (786 images, with VOC/YOLO annotations)
```

### 5. Run

```bash
# Step 1: Build feature bank
python experiments/_feature_bank.py

# Step 2: Run inference
python experiments/_inference_auto.py

# Step 3: Evaluate results
python experiments/_evaluate.py

# Or run the full ablation suite
python experiments/run_ablation_main.py
```

---

## Project Structure

```
PFS-Res-SAM/
├── PatchRes/                      # Core modules
│   ├── ESN_2D_nobatch.py         # 2D-ESN implementation
│   ├── PatchRes.py               # PatchRes main class
│   ├── ResSAM.py                 # Res-SAM core module
│   ├── common.py                 # Nearest-neighbor search (faiss/sklearn)
│   ├── functions.py              # Data loading, mask generation
│   ├── logger.py                 # Logging utilities
│   └── sam_integration.py        # SAM integration
├── experiments/                   # Pipeline scripts
│   ├── _feature_bank.py          # Feature bank construction
│   ├── _inference_auto.py        # Fully automatic inference
│   ├── _evaluate.py              # Evaluation metrics
│   ├── run_ablation_main.py      # Ablation study runner
│   ├── dataset_layout.py         # Dataset path management
│   ├── paper_constants.py        # Paper-aligned constants
│   └── resize_policy.py          # Image resize policy
├── sam/
│   └── sam.py                    # SAM interface (weight downloaded separately)
├── scripts/                       # Setup and batch scripts
├── requirements-cpu.txt          # CPU dependencies
├── requirements-gpu.txt          # GPU dependencies (CUDA 11.8)
└── README.md
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 50×50 | 2D-ESN patch size |
| `stride` | 5 | Sliding window stride |
| `hidden_size` | 30 | Reservoir size |
| `beta_threshold` (β) | 0.1 | Anomaly threshold (Eq.9) |
| `init_normal_samples` | 20 | Feature bank sample count |
| `eval IoU threshold` | 0.5 | Box matching criterion |

---

## Dataset

The dataset used in this work is from the **Mendeley Data** repository by Mojahid et al.
(2024), containing 2,239 GPR B-scan images of buried utilities, voids, and intact zones.
The current pipeline uses an augmented subset: 900 intact (normal), 553 cavity, and
786 utility-type images.

- **Paper**: Mojahid, A., El Ouai, D., El Amraoui, K., El Hami, K. & Aitbenamer, H. (2024)
- **DOI**: [10.17632/ww7fd9t325.1](https://doi.org/10.17632/ww7fd9t325.1)

## License

This project is provided for research and academic use.
