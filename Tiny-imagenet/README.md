# Tiny ImageNet (Bonus) - Task3 Framework

This directory reuses Task3's custom CUDA ops and autograd to train on Tiny ImageNet (200 classes, 64x64).

## 1) Build Task3 CUDA Extension

```bash
cd Task3
python setup.py build_ext --inplace
```

## 2) Dataset Layout

Point `--data-dir` to the Tiny ImageNet root (e.g. `tiny-imagenet-200`). If missing, the scripts will auto-download to the `--data-dir` folder (default: `Tiny-imagenet/data`):

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

## 3) Train

```bash
python Tiny-imagenet/train.py
```

Common knobs:

- `--arch` VGG variant (VGG11/13/16/19)
- `--image-size` (default 64)
- `--autoaugment`, `--random-erasing`
- `--ema`
- `--micro-batch-size` to split large batches if you hit CUDA OOM

## 4) Evaluate

```bash
python Tiny-imagenet/eval.py --data-dir /path/to/tiny-imagenet-200 --ckpt Tiny-imagenet/checkpoint/ckpt.pth
```
