# Tiny ImageNet (Bonus) - Task3 Framework

This directory reuses Task3's custom CUDA ops and autograd to train on Tiny ImageNet (200 classes, 64x64).

## 1) Build Task3 CUDA Extension

```bash
cd Task3
python setup.py build_ext --inplace
```

## 2) Dataset Layout

Point `--data-dir` to the Tiny ImageNet root (e.g. `tiny-imagenet-200`). The scripts do not auto-download by default (default data dir: `Tiny-imagenet/data`):
```shell
./Tiny-imagenet/hfd.sh zh-plus/tiny-imagenet --dataset --include "*.parquet" --local-dir Tiny-imagenet/data -x 10
```

```
tiny-imagenet-200/
  train/
    n01443537/
      images/xxx.JPEG
      n01443537_boxes.txt
  val/
    images/xxx.JPEG
    val_annotations.txt
  wnids.txt
```

The loader uses `wnids.txt` (if present) to keep class order consistent.

Parquet alternative: if `--data-dir` contains parquet files (e.g. Hugging Face datasets), the loader will use them instead of image folders. Filenames must include `train` and `val` (or `validation`) to identify splits, and you need `datasets` installed.

## 3) Train

Download:
```bash
python Tiny-imagenet/train.py --data-dir Tiny-imagenet/data --download
```

Train:
```bash
python Tiny-imagenet/train.py --data-dir Tiny-imagenet/data
```

## 4) Evaluate

```bash
python Tiny-imagenet/eval.py --data-dir /path/to/tiny-imagenet-200 --ckpt Tiny-imagenet/checkpoint/ckpt.pth
```
