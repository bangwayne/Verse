# Verse_1.0

# VerSe: Integrating Multiple Queries as Prompts for Versatile Cardiac MRI Segmentation

## Abstract
Despite the advances in learning-based image segmentation
approach, the accurate segmentation of cardiac structures from magnetic
resonance imaging (MRI) remains a critical challenge. While existing
automatic segmentation methods have shown promise, they still require
extensive manual corrections of the segmentation results by human experts, 
particularly in complex regions such as the basal and apical parts
of the heart. Recent efforts have been made on developing interactive
image segmentation methods that enable human-in-the-loop learning.
However, they are semi-automatic and inefficient, due to their reliance on
click-based prompts, especially for 3D cardiac MRI volumes. To address
these limitations, we propose VerSe, a Versatile Segmentation frame-
work to unify automatic and interactive segmentation through mutiple
queries. Our key innovation lies in the joint learning of object and click
queries as prompts for a shared segmentation backbone. VerSe supports
both fully automatic segmentation, through object queries, and interactive mask 
refinement, by providing click queries when needed. With the
proposed integrated prompting scheme, VerSe demonstrates significant
improvement in performance and efficiency over existing methods, on
both cardiac MRI and out-of-distribution medical imaging datasets.

## Features
- Unified framework for automatic and interactive segmentation.
- Integration of object and click queries for enhanced versatility.
- Evaluated on both cardiac MRI and out-of-distribution medical imaging datasets.

## Example
Below is an example visualization of our VerSe framework:

![VerSe Framework](figure1.png)

## Updates
* 03/03/2024 Upload the code and the weights of Verse.
* 02/13/2025 Accepted by IPMI2025!


## Quick Start

### Prepare the environment.

Tested on Ubuntu only.
- Python 3.10+
- PyTorch 1.13+ and corresponding torchvision

Install required python packages: `pip install -r requirements.txt`

### Prepare datasets

Please download the original datasets from their official website. We preprocess the nii.gz files using gray normalization and resizing.


```bash
Verse
└── datasets
     ├── ACDC
     │   ├── train
     │   │   ├── image
     │   │       ├── patient001.nii.gz
     │   │       └── ...
     │   │   ├── annotations
     │   │       ├── patient001_gt.nii.gz
     │   │       └── ...
     │   ├── test
     │       ├── image
     │       ├── annotations
     ├── MM
     │   ├── train
     │   └── test
     ├── ...
     └── data_mapper

```

### Training
The training configurations are under `Verse_1_0_Cardiac/configs/Cardiac_config.Training_config.py`. The model configurations are under `Verse_1_0_Cardiac/configs/Cardiac_config.Cardiac_config.py`.

To train Verse: `sh run_train.sh`

The Pre-trained Verse weight can be downloaded in https://drive.google.com/file/d/1rdRwfJ9JpJK9elbm02hGTLwgDSjJkjcN/view?usp=drive_link.

### Evaluation
The testing configurations are under `Verse_1_0_Cardiac/configs/Cardiac_config.Testing_config.py`.

To evaluate Verse: `sh run_eval.sh`


## Citation

```bibtex
@article{guo2024verse,
  title={VerSe: Integrating Multiple Queries as Prompts for Versatile Cardiac MRI Segmentation},
  author={Guo, Bangwei and Ye, Meng and Gao, Yunhe and Xin, Bingyu and Axel, Leon and Metaxas, Dimitris},
  journal={arXiv preprint arXiv:2412.16381},
  year={2024}
}
```
