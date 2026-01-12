## Task 1: PyTorch CIFAR-10

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
- `--device`（默认自动选择 cuda/cpu）

### 评估

- `python Task1/eval.py --ckpt Task1/checkpoints/cifar_net.pth`
