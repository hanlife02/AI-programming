# Tiny-ImageNet（Task3 CUDA 训练）

本目录提供 Tiny-ImageNet 训练脚本，使用 Task3 的自定义 CUDA 算子与最小框架：

- **卷积/池化/BN/线性/损失** 等算子均来自 Task3 的 CUDA 扩展（`Task3/csrc/ops.cu`）。
- Python 侧只负责：计算图、优化器与训练循环（`Task3/minifw/*` + `Tiny-imagenet/train.py`）。
- 数据加载使用 `torchvision.datasets.ImageFolder`，默认 **64×64** 输入。
- 训练脚本支持 **自动下载 Tiny-ImageNet**，并将 `val` 预处理成 ImageFolder 结构。

## 1) 构建扩展

在有 CUDA 的环境中（需要 `nvcc` 与可用 GPU）：

```bash
cd Task3
python setup.py build_ext --inplace
```

## 2) 训练

默认数据路径（在repo根目录的 `input/` 下）：

- 训练集：`input/tiny-imagenet/tiny-imagenet-200/train`
- 验证集：`input/tiny-imagenet/tiny-imagenet-200/val`

从仓库根目录运行：

```bash
python Tiny-imagenet/train.py
```

常用参数：

- `--train-dir`: 训练集 ImageFolder 路径
- `--val-dir`: 验证集 ImageFolder 路径（必填/必须存在）
- `--download/--no-download`: 自动下载 Tiny-ImageNet（默认开启）
- `--force-download`: 强制重新下载（忽略本地已有数据）
- `--check-integrity/--no-check-integrity`: 完整性校验（默认开启）
- `--data-root`: 自动下载根目录（默认 `input/tiny-imagenet`）
- `--dataset-url`: Tiny-ImageNet zip 下载地址（默认 CS231n 官方）
- `--num-classes`: 类别数（默认 200）
- `--model`: VGG 配置（默认 `VGG16`）
- `--epochs` / `--batch-size` / `--num-workers` / `--lr`
- `--scheduler`: `cosine` / `none`

## 3) 说明

- 训练使用 `ImageFolder`，目录结构需为 `class_name/xxx.jpg` 的标准形式。
- 若 `train/val` 路径不存在且开启 `--download`，脚本会自动下载并整理 `val/images + val_annotations.txt` 为 ImageFolder 结构。
- 完整性校验会检查类别数与图像数量（200 类、训练 100000、验证 10000）；不完整会触发重新下载并报错提示。
- 模型内部使用 `GlobalAvgPool2d` 适配 64×64 输入，因此无需手动改输入尺寸。
- `--num-classes` 与数据集类别不一致时会提示警告。

## 4) 测试集评估

```bash
python Tiny-imagenet/eval.py --ckpt Tiny-imagenet/checkpoint/ckpt.pth
```