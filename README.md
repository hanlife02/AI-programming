# AI-Programming

2025-2026秋季学期《人工智能中的编程》大作业内容，仅供学习使用。

本repo包含3个task与1个bonus：

- Task1：PyTorch CIFAR-10 单卡训练/评估
- Task2：PyTorch DDP 数据并行训练/评估
- Task3：自研 CUDA 算子 + 最小框架训练/评估
- Bonus：基于 Task3 算子的 Tiny-ImageNet 训练与评估

## 环境配置

使用根目录的 `environment.yml`：

```bash
conda env create -f environment.yml
conda activate <env_name>
```

注：本实验是在RTX5090显卡上运行得到的结果。

## Task1：CIFAR-10 单卡

```bash
python Task1/train.py
python Task1/eval.py --ckpt Task1/checkpoints/cifar_net.pth
```

更多参数见 `Task1/README.md`,实验报告见`Task1/report.pdf`

## Task2：CIFAR-10 DDP

```bash
CUDA_VISIBLE_DEVICES=0 python Task2/train.py --run-name single
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 Task2/train.py --run-name ddp2
```

更多参数与对齐方案见 `Task2/README.md`,实验报告见`Task2/report.pdf`

## Task3：自研 CUDA 框架

```bash
cd Task3
python setup.py build_ext --inplace
cd ..
python Task3/train.py --epochs 50 --batch-size 256 --num-workers 8 --lr 0.1
python Task3/eval.py --ckpt Task3/checkpoint/ckpt.pth
```

更多参数见 `Task3/README.md`,实验报告见`Task3/report.pdf`

## Tiny-ImageNet（基于 Task3）

训练（默认会自动下载到 `input/tiny-imagenet`）：

```bash
python Tiny-imagenet/train.py
```

```bash
python Tiny-imagenet/eval.py  --ckpt Tiny-imagenet/checkpoint/ckpt.pth
```

更多参数与说明见 `Tiny-imagenet/README.md`,实验报告见`Tiny-imagenet/report.pdf`