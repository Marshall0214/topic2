# CURVATURE-GUIDED TASK SYNERGY FOR SKELETON BASED TEMPORAL ACTION SEGMENTATION (ICLR 2026)

> **Status:** Code is available, pretrained models **coming soon**.  
> **Paper:** *Curvature-Guided Task Synergy for Skeleton-based Temporal Action Segmentation* (ICLR 2026)  
> **ArXiv/OpenReview:** coming soon

<!-- Optional badges (coming soon) -->
<!-- ![PyTorch](https://img.shields.io/badge/PyTorch-1.10.1+-red) -->
<!-- ![License](https://img.shields.io/badge/License-MIT-green) -->
<!-- ![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b) -->

---

## TL;DR
CurvSeg restores cross-task synergy between **action classification** and **boundary localization** for skeleton-based temporal action segmentation by:
1) using **representation curvature** as a geometric prior for boundary detection, and  
2) using localization to **refine classification feature geometry** via a bidirectional consistency objective,  
plus an **Expert-Driven Decoupling (EDD)** module (Mixture-of-Experts) to provide task-adaptive features.

![framework](./imgs/framework.jpg) 

---

## Results
### Quantitative Results
- Main tables / comparisons: coming soon

### Qualitative Results
- Visualization examples: coming soon

---

## Installation
### Environment
- **PyTorch 1.10.1** (required)

### Setup
1. Install PyTorch 1.10.1 (match your CUDA version).
2. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

We evaluate on:

- MCFS-22 / MCFS-130
- PKU-MMD
- LARa

Links: [Google Driver](https://drive.google.com/drive/folders/1xBzaXV7-DAegHCopTp7xawAgOgto3x-3?usp=drive_link)

## Usage

We provide one unified script for training + testing.
For different datasets, we provide separate .sh launchers.

### Quick Start

```bash
# Example (placeholder)
bash pku_sub.sh
```

## Checkpoints

Pretrained models: coming soon

## Citation
```
If you find this work helpful, please cite:

@inproceedings{curvseg2026,
  title     = {Curvature-Guided Task Synergy for Skeleton-based Temporal Action Segmentation},
  author    = {Coming soon},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```


## License

This repository is released under the coming soon license.
(Recommendation: choose a standard OSI-approved license before release.)

## Acknowledgements

This codebase builds upon / refers to:

DeST (coming soon)

LaSA (coming soon)

Other dependencies: coming soon
