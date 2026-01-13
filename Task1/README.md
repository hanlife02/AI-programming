## Task 1: PyTorch CIFAR-10

### 目录结构

```
Task1/
  README.md
  requirements.txt
  train.py
  eval.py
  plot_loss.py
  models/
    __init__.py
    cnn.py

  # 训练时自动生成（默认路径）
  data/
  checkpoints/
    cifar_net.pth
  outputs/
    loss.csv
    loss_curve.png
```

### 环境

- Python 3.10+
- 依赖：见 `Task1/requirements.txt`

### 训练

从仓库根目录运行（推荐）：

- `python Task1/train.py`

可选参数：

- `--model`（默认 `cnn_bn`，可选 `cnn`）
- `--epochs`（默认 50）
- `--batch-size`（默认 128）
- `--lr`（默认 0.1）
- `--optimizer`（默认 `sgd`，可选 `adamw`）
- `--weight-decay`（默认 5e-4）
- `--scheduler`（默认 `cosine`，可选 `none`/`step`）
- `--augment/--no-augment`（默认开启数据增强）
- `--val-split`（默认 0.1，用训练集切出验证集，避免边训边盯 test）
- `--num-workers`（默认 8）
- `--data-dir`（默认 `Task1/data`）
- `--ckpt`（默认 `Task1/checkpoints/cifar_net.pth`）
- `--best-ckpt`（默认 `Task1/checkpoints/cifar_net_best.pth`）
- `--loss-csv`（默认 `Task1/outputs/loss.csv`）
- `--loss-plot`（默认 `Task1/outputs/loss_curve.png`）
- `--device`（默认自动选择 cuda/cpu）

### 默认配置（等价命令）

`python Task1/train.py` 默认就等价于：

- `python Task1/train.py --model cnn_bn --epochs 50 --batch-size 128 --optimizer sgd --lr 0.1 --momentum 0.9 --weight-decay 5e-4 --scheduler cosine --augment --val-split 0.1`

如果想快速跑通流程（更少 epoch），可以：

- `python Task1/train.py --epochs 2`

### 绘制 loss curve

训练脚本默认会输出：

- `Task1/outputs/loss.csv`
- `Task1/outputs/loss_curve.png`

如果已经有 `loss.csv`，也可以单独绘图：

- `python Task1/plot_loss.py --csv Task1/outputs/loss.csv --out Task1/outputs/loss_curve.png`

### 评估

- `python Task1/eval.py --ckpt Task1/checkpoints/cifar_net.pth`
- 如果想用训练过程中保存的最佳验证集模型：`python Task1/eval.py --ckpt Task1/checkpoints/cifar_net_best.pth`
