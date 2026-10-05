# Crop Segmentation from Satellite Image Time Series with Video Transformers

> Master's project, M2 Vision & Machine Intelligence, Université Paris Cité (2026). Team of 2.

Can a video foundation model pre-trained on natural videos be transferred to **dense
semantic segmentation of Sentinel-2 time series**? This project adapts two video
transformers, **TimeSformer** (supervised, Kinetics-400) and **V-JEPA 2.1** (self-supervised),
to the [PASTIS](https://github.com/VSainteuf/pastis-benchmark) benchmark: 18 crop classes
plus background, 10 spectral bands, 33 to 61 irregular acquisitions per patch, evaluated
with 5-fold geographic cross-validation.

![RGB | ground truth | prediction](utils/docs/qualitative.png)

## Methods

| | Method 1: TimeSformer | Method 2: V-JEPA 2.1 |
|---|---|---|
| Pre-training | Supervised, Kinetics-400 | Self-supervised (JEPA), ViT-L |
| Spectral adaptation | Weight inflation 3 → 10 bands (variance-preserving) | RGB warm start + Kaiming init for NIR/SWIR |
| Temporal aggregation | Learned-query **attention pooling** with padding mask | Masked temporal mean pooling over 3D tokens |
| Decoder | **HIRES**: progressive upsampling + high-res image skip, on a multi-scale feature pyramid (layers 4/8/12) | Segmentation head |
| Transfer | LoRA (r=8), full fine-tuning and linear probing compared | LoRA (r=16) on `qkv`/`proj`, multi-GPU DDP |
| Report | [README_TImeSFormer.md](README_TImeSFormer.md) (FR) | [README_vjepa2.md](README_vjepa2.md) |

## Key results

| Model | mIoU | Macro-F1 |
|---|:-:|:-:|
| TimeSformer + HIRES + LoRA (5-fold) | 0.285 ± 0.009 | 0.409 |
| V-JEPA 2.1 + LoRA (10 bands) | 0.296 | 0.42 |
| *U-TAE (specialized reference)* | *0.631* | – |

- **The decoder was the bottleneck, not the backbone.** Replacing a single ×16 upsampling
  with the HIRES decoder **tripled the mIoU** (0.095 → 0.291). The transfer strategy
  (LoRA, full fine-tuning or linear probe) only moves it within 0.27–0.29.
- **Robust to clouds.** With 50 % of the dates masked, TimeSformer keeps 79 % of its mIoU,
  and masked dates receive exactly zero attention.
- **Spectral bands matter.** V-JEPA drops from 0.296 to 0.273 mIoU when restricted to RGB,
  and collapses without LoRA because the domain shift is too large for a frozen encoder.
- **Honest negative result.** Generic video transformers stay far behind a convolutional
  model built for this task. The 8×8 token grid limits the spatial resolution of the
  segmentation.

## Repository layout

```
TimeSformer/   Method 1: model, training, 5-fold cross-validation, TTA, YAML configs, results
V_JEPA/        Method 2: 10-band adaptation, DDP training, experiment orchestration
utils/         PASTIS loader (variable-length series + masks), U-TAE-compatible metrics, figures
```

## Authors

- **Auguste Calmanovic-Plescoff**: Method 1 (TimeSformer)
- **Mayeul**: Method 2 (V-JEPA 2.1)

Originally developed in a private team repository; published here with the full commit
history.
