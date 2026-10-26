# SFM-DETR

**Wavelet-Guided Fusion for Wind Turbine Blade Defect Detection**

This repository provides the SFM-DETR implementation, training configurations, dataset validation scripts, and experimental figures accompanying the manuscript. The code builds on the PyTorch implementation of RT-DETRv2 and retains the upstream Apache License 2.0 notices.

## Key Contributions

1. **Coupled semantic, frequency, and morphological modeling.** The four-subband preference `D_l` produced by SCWA guides both detail selection and coarse-scale context sampling, connecting the two feature paths through a shared selection signal.
2. **Semantic-Conditioned Wavelet Attention (SCWA).** Semantic-conditioned attention operates jointly over four same-resolution Haar subbands and their 3 × 3 neighborhoods, preserving signed high-frequency responses and producing an interpretable four-channel preference.
3. **Morphology-aware Frequency-guided Attention Neck (MFA-Neck).** The shared preference `D_l` conditions sampling and attention biases. Bounded translation, rotation, and anisotropic scaling adapt the sampling geometry for cross-scale feature fusion.

## Repository Structure

```text
src/                     SFM-DETR and RT-DETRv2 implementation
configs/sfm_detr/        Full-model and ablation templates for WTBD and WTBs2025
tools/train.py           Training and evaluation entry point
tools/prepare_config.py  Generate runnable configurations from templates
tools/validate_coco.py   Validate COCO-format datasets
assets/                  Architecture, training curves, and qualitative comparisons
results/                 Three-run results reported in manuscript Table 2
docs/DATASETS.md          Dataset download links and directory structure
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Training requires PyTorch 2.x, torchvision, CUDA, and COCO evaluation dependencies. An NVIDIA GPU is recommended. The optional `requirements-deploy.txt` file contains dependencies for ONNX export and deployment.

## Datasets

Download the datasets from their Figshare release pages using the links below:

| Dataset | Download | DOI |
|---|---|---|
| WTBD | [Download WTBD](https://doi.org/10.6084/m9.figshare.30210175.v1) | `10.6084/m9.figshare.30210175.v1` |
| WTBs2025 | [Download WTBs2025](https://doi.org/10.6084/m9.figshare.28876391.v1) | `10.6084/m9.figshare.28876391.v1` |

After downloading, arrange the datasets in the COCO directory structure described in [the dataset preparation guide](docs/DATASETS.md). Dataset files are not included in this repository. Validate the prepared dataset before training:

```bash
python tools/validate_coco.py /path/to/WTBD/coco
```

## Training

Generate a local configuration from a template. The script replaces `__DATA_ROOT_WTBD__` or `__DATA_ROOT_WTBS2025__` with the dataset directory you provide:

```bash
python tools/prepare_config.py \
  --template configs/sfm_detr/wtbd_full_r18_100e.yml.template \
  --data-root /path/to/WTBD/coco \
  --output configs/local_wtbd_full.yml \
  --output-dir ./runs/wtbd_full
python tools/train.py -c configs/local_wtbd_full.yml \
  -t /path/to/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --use-amp
```

To train on WTBs2025, use `wtbs2025_full_r18_100e.yml.template` and update the dataset, configuration, and output paths accordingly. The `baseline`, `scwa_only`, and `mfa_only` templates support component ablations. To resume an interrupted run, use the same configuration with `-r runs/.../last.pth`.

COCO-pretrained weights are available from the [official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR). Model weights are not included in this repository.

Evaluate an existing checkpoint:

```bash
python tools/train.py -c configs/local_wtbd_full.yml -r runs/wtbd_full/last.pth --test-only
```

Training logs are saved to the configured `output_dir`. EMA weights can be used for validation during training; the controlled ablation endpoints reported in the manuscript use raw weights from epoch 100.

## Benchmark Results

Values are reported as the mean ± sample standard deviation over three independent runs. AP50 is COCO AP averaged over classes at IoU = 0.50, commonly called mAP50. AP additionally averages over IoU thresholds from 0.50 to 0.95 in steps of 0.05. For DETR models, precision (P) and recall (R) are evaluated at the maximum-F1 operating point of the interpolated IoU-0.50 precision–recall curve; YOLO models retain their native P/R definitions.

| Dataset | Model | P | R | AP50 (mAP50) | AP |
|---|---|---:|---:|---:|---:|
| WTBD | YOLOv8-s | 0.7441 ± 0.0138 | 0.7063 ± 0.0119 | 0.7342 ± 0.0302 | 0.4174 ± 0.0126 |
| WTBD | YOLOv9-s | 0.5604 ± 0.0368 | 0.5284 ± 0.0285 | 0.5434 ± 0.0368 | 0.3059 ± 0.0258 |
| WTBD | YOLOv10-s | 0.7003 ± 0.0874 | 0.6309 ± 0.0466 | 0.6903 ± 0.0599 | 0.3804 ± 0.0296 |
| WTBD | RT-DETR-R18 | 0.7973 ± 0.0416 | 0.6400 ± 0.0361 | 0.7414 ± 0.0147 | 0.4635 ± 0.0122 |
| WTBD | RT-DETRv2-R18 | 0.8005 ± 0.0469 | 0.6900 ± 0.0520 | 0.7832 ± 0.0046 | 0.4953 ± 0.0061 |
| WTBD | Position-DETR R18 | 0.8917 ± 0.0542 | 0.7489 ± 0.0051 | 0.7850 ± 0.0120 | 0.4293 ± 0.0097 |
| WTBD | Co-Deformable-DETR R18 | 0.8156 ± 0.0366 | 0.7117 ± 0.0100 | 0.7175 ± 0.0129 | 0.3995 ± 0.0191 |
| WTBD | SFM-DETR | 0.9444 ± 0.0183 | 0.6300 ± 0.0047 | 0.8035 ± 0.0109 | 0.4856 ± 0.0093 |
| WTBs2025 | YOLOv8-s | 0.4881 ± 0.0105 | 0.4357 ± 0.0071 | 0.4409 ± 0.0028 | 0.2926 ± 0.0042 |
| WTBs2025 | YOLOv9-s | 0.4273 ± 0.0092 | 0.4545 ± 0.0058 | 0.4377 ± 0.0033 | 0.2615 ± 0.0039 |
| WTBs2025 | YOLOv10-s | 0.5131 ± 0.0076 | 0.4458 ± 0.0061 | 0.4874 ± 0.0042 | 0.3439 ± 0.0021 |
| WTBs2025 | RT-DETR-R18 | 0.6872 ± 0.0087 | 0.5713 ± 0.0105 | 0.6361 ± 0.0057 | 0.4233 ± 0.0064 |
| WTBs2025 | RT-DETRv2-R18 | 0.7264 ± 0.0103 | 0.6611 ± 0.0064 | 0.6634 ± 0.0035 | 0.4306 ± 0.0046 |
| WTBs2025 | Position-DETR R18 | 0.8012 ± 0.0113 | 0.6893 ± 0.0092 | 0.7214 ± 0.0029 | 0.4473 ± 0.0051 |
| WTBs2025 | Co-Deformable-DETR R18 | 0.7478 ± 0.0054 | 0.6701 ± 0.0083 | 0.6585 ± 0.0038 | 0.4241 ± 0.0054 |
| WTBs2025 | SFM-DETR | 0.8324 ± 0.0093 | 0.7497 ± 0.0074 | 0.7813 ± 0.0046 | 0.4613 ± 0.0069 |

## Qualitative Results

The following figure shows representative comparisons from the manuscript. The 5 × 8 grid presents, from left to right: original image, ground-truth annotations, SFM-DETR, RT-DETRv2-R18, Co-Deformable-DETR R18, Position-DETR R18, RT-DETR-R18, and YOLOv8-s. These examples were selected to illustrate favorable SFM-DETR detections across defect categories; they are qualitative illustrations rather than a random-sample estimate of performance.

![Qualitative detection comparison across multiple models](assets/qualitative.png)

Challenging examples and error analysis:

![Detection results on challenging examples](assets/hard_examples.png)

Training curves and component attribution maps:

![Precision and AP50 training curves](assets/training_curves.png)

![Grad-CAM heatmaps for the component ablation study](assets/ablation_heatmaps.png)

## License and Citation

The upstream RT-DETRv2 code is distributed under the Apache License 2.0 included in this repository. Upstream copyright and license notices are preserved in [LICENSE](LICENSE). SFM-DETR additions and experimental configurations are provided with this repository. Use of the datasets is subject to the respective dataset providers' terms.

If you use this code in your research, please cite the SFM-DETR manuscript, the RT-DETR/RT-DETRv2 papers, and the papers associated with the datasets you use.
