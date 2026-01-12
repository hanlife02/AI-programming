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

- `--epochs`（默认 2）
- `--batch-size`（默认 4）
- `--lr`（默认 1e-3）
- `--num-workers`（默认 0，macOS 建议保持 0）
- `--data-dir`（默认 `Task1/data`）
- `--ckpt`（默认 `Task1/checkpoints/cifar_net.pth`）
- `--loss-csv`（默认 `Task1/outputs/loss.csv`）
- `--loss-plot`（默认 `Task1/outputs/loss_curve.png`）
- `--device`（默认自动选择 cuda/cpu）

### 绘制 loss curve

训练脚本默认会输出：

- `Task1/outputs/loss.csv`
- `Task1/outputs/loss_curve.png`

如果你已经有 `loss.csv`，也可以单独绘图：

- `python Task1/plot_loss.py --csv Task1/outputs/loss.csv --out Task1/outputs/loss_curve.png`

### 评估

- `python Task1/eval.py --ckpt Task1/checkpoints/cifar_net.pth`
