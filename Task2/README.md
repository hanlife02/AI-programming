## Task 2: PyTorch Data Parallel (DDP) on CIFAR-10

本 Task 只做**数据并行**（DistributedDataParallel, DDP），不做模型并行。

### 目录结构

```
Task2/
  README.md
  requirements.txt
  train.py
  eval.py
  utils.py
  models/
    __init__.py
    cnn.py

  # 运行后自动生成
  data/
  outputs/
    <run_name>/
      metrics.csv
      ckpt.pth
      ckpt_best.pth
```

---

## 1. 并行准备：检查 GPU 数量

训练脚本启动时会打印：

- `torch.cuda.is_available`
- `torch.cuda.device_count`
- 当前进程的 `rank/world_size/local_rank`（DDP 时）

你也可以手动检查：

- `nvidia-smi`

---

## 2. 单卡训练（baseline）

建议显式指定只用 1 张卡：

```bash
CUDA_VISIBLE_DEVICES=0 python Task2/train.py --epochs 2 --batch-size 128 --num-workers 4 --run-name single
```

输出：

- `Task2/outputs/single/metrics.csv`：每个 epoch 的耗时、吞吐（images/sec）、loss、accuracy
- `Task2/outputs/single/ckpt.pth`：最终模型权重
- `Task2/outputs/single/ckpt_best.pth`：验证集最优模型（若启用 `--val-split` 且未 `--no-eval`）

---

## 3. 双卡数据并行训练（DDP）

同机 2 GPU（推荐 `torchrun`）：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 Task2/train.py --epochs 2 --batch-size 128 --num-workers 4 --run-name ddp2
```

说明：

- `--batch-size` 是 **per-GPU batch size**；双卡时全局 batch 会变成 `batch_size * world_size`
- 若你想保持全局 batch 不变：双卡时把 `--batch-size` 减半（例如单卡 128，双卡用 64）

输出：

- `Task2/outputs/ddp2/metrics.csv`
- `Task2/outputs/ddp2/ckpt.pth`（只由 rank0 保存）
- `Task2/outputs/ddp2/ckpt_best.pth`（只由 rank0 保存）

---

## 3.1 与 Task1 对齐的训练设置（推荐）

为了让单卡/双卡的**准确率**更接近 Task1 的训练口径，`train.py` 额外支持：

- `--model`：`cnn` 或 `cnn_bn`（通常 `cnn_bn` 更高准确率）
- `--augment/--no-augment`：训练集随机裁剪+翻转
- `--val-split`：从训练集切验证集（默认 0.1），`metrics.csv` 里记录 `val_acc`
- `--optimizer`/`--weight-decay`/`--scheduler`：更常见的 CIFAR-10 配置

示例（保持全局 batch 不变）：

- 单卡：`CUDA_VISIBLE_DEVICES=0 python Task2/train.py --model cnn_bn --epochs 2 --batch-size 128 --optimizer sgd --lr 0.1 --momentum 0.9 --weight-decay 5e-4 --scheduler cosine --augment --val-split 0.1 --run-name single_bn`
- 双卡：`CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 Task2/train.py --model cnn_bn --epochs 2 --batch-size 64 --optimizer sgd --lr 0.1 --momentum 0.9 --weight-decay 5e-4 --scheduler cosine --augment --val-split 0.1 --run-name ddp2_bn`

---

## 4. 性能对比（速度与准确率）

对比方式（建议用同样 epoch 数、同样数据处理、同样 eval 设置）：

1) 读取 `Task2/outputs/single/metrics.csv` 和 `Task2/outputs/ddp2/metrics.csv`
2) 关注：
   - `epoch_time_sec`：越小越快
   - `images_per_sec`：越大越快
   - `val_acc`：验证集准确率是否与单卡接近（允许有轻微差异）

---

## 评估（可选）

对已训练的 checkpoint 做一次单卡评估：

```bash
CUDA_VISIBLE_DEVICES=0 python Task2/eval.py --ckpt Task2/outputs/ddp2/ckpt.pth
```
