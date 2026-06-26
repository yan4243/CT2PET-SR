# PET-CDM

Minimal code snapshot for CT-conditioned PET reconstruction with a
cross-attention diffusion model and projection-domain physical data
consistency.

This staging snapshot contains the manuscript's main training and inference
paths. The private dataset loader is left as an explicit placeholder.

## Included

- CT encoder and cross-attention PET UNet
- Gaussian diffusion training and DDIM sampling
- Gaussian PSF degradation
- ASTRA parallel-beam projection and FBP
- angular/radial sinogram rebinning
- dose-scaled projection-domain Poisson sampling
- physical data-consistency updates
- 6 mm, 8 mm, and 12 mm inference settings

## Layout

```text
train.py                 Cross-attention diffusion training
infer.py                 Physical degradation and DC inference
models/                  Diffusion and cross-attention model
utils/                   Physics, DC, registration, metrics, checkpoints
datasets/                Dataset-interface placeholders
configs/train.yaml       Model and training configuration
configs/infer_*.yaml     Paper physical settings
scripts/train.sh         Single-GPU training
scripts/train_slurm.sh   Generic Slurm/torchrun training
scripts/infer_all.sh     Run the three paper inference settings
```

## Data interface

Before running the code, implement:

- `datasets/PET_CT_Datasets.py`
- `datasets/PET_CT_Datasets_case.py`

The slice-level loader must return:

```python
pet, pet_affine, ct, ct_affine, key
```

Images should have shape `[1, H, W]`; affine matrices should have shape
`[4, 4]`.

## Installation

```bash
conda create -n pet-cdm python=3.10
conda activate pet-cdm
pip install -r requirements.txt
```

ASTRA GPU support depends on the local CUDA environment.

## Training

Edit the dataset paths in `configs/train.yaml`, then run:

```bash
bash scripts/train.sh
```

For four GPUs under Slurm:

```bash
sbatch scripts/train_slurm.sh
```

## Inference

```bash
bash scripts/infer_all.sh /path/to/checkpoint.pth outputs 0
```

| Setting | PSF FWHM | Angular rebin | Radial rebin | Dose |
|---|---:|---:|---:|---:|
| 6 mm | 4.5 mm | 2 | 1 | 0.10 |
| 8 mm | 6.0 mm | 2 | 2 | 0.10 |
| 12 mm | 8.0 mm | 3 | 2 | 0.05 |

## Status

This is an early staging release. Dataset integration, tests, packaging, and
further removal of compatibility code can be completed incrementally.
