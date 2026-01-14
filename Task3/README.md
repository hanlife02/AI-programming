# Task 3: Custom Implementation (CIFAR-10)

本目录实现了一个最小但可训练的“自研”CNN框架：

- 卷积/反传、ReLU、MaxPool、GlobalAvgPool、Linear、CrossEntropy 都由 `CUDA + pybind11` 扩展提供
- Python 侧实现自动微分（反向图）与 SGD 优化器
- 数据处理允许使用 `torchvision`（符合 `hw.md` 的 Tips）

## 1) 构建扩展

在有 CUDA 的环境中（需要 `nvcc` 与可用 GPU）：

```bash
cd Task3
python setup.py build_ext --inplace
```

成功后会生成 `task3_ops.*.so`。

如果你遇到类似 `nvcc fatal: Unsupported gpu architecture 'compute_120'` 的报错，通常是 **PyTorch 编译时的 CUDA 版本** 与 **本机 nvcc 版本** 不一致导致自动选择了 nvcc 不支持的架构。本项目的 `Task3/setup.py` 会自动过滤不支持的架构；你也可以手动指定：

```bash
# 仅为示例：按你的 GPU 计算能力调整（如 8.6 / 8.9 / 9.0）
TASK3_CUDA_ARCH_LIST="8.6" python setup.py build_ext --inplace
```

## 2) 训练

```bash
python Task3/train.py --epochs 50 --batch-size 256 --num-workers 8 --lr 0.1
```

多卡（DDP / torchrun），`--batch-size` 为 **单卡** batch：

```bash
torchrun --standalone --nproc_per_node=2 Task3/train.py --epochs 50 --batch-size 256 --num-workers 8 --lr 0.1
```

指定卡数量/指定哪些卡：

- **卡数量**：用 `--nproc_per_node=<N>` 指定启动的进程数（通常等于使用的 GPU 数）。
  - 例如用 2 卡：`torchrun --standalone --nproc_per_node=2 Task3/train.py ...`
- **指定具体 GPU**：用 `CUDA_VISIBLE_DEVICES` 限制可见 GPU，再把 `--nproc_per_node` 设为可见卡数。
  - 例如只用 0,1 号卡（2 卡）：`CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 Task3/train.py ...`

常用参数：

- `--data-dir`: CIFAR-10 下载/缓存目录（默认 `Task3/data`）
- `--ckpt`: 最优验证集 checkpoint 路径（默认 `Task3/checkpoints/task3_ckpt.pt`）
- `--no-augment`: 关闭数据增强（更快但通常准确率更低）

## 3) 说明

- 本实现不使用 PyTorch 的 autograd / nn.Module 来训练；仅用其 Tensor 作为显存容器以及 torchvision 的数据集/加载器。
- 若你的评测机器有更强 GPU，可优先提高 `--batch-size` 与 `--num-workers` 以获得更高吞吐。

## 4) 测试集评估

```bash
python Task3/eval.py --ckpt Task3/checkpoints/task3_ckpt.pt
```

多卡评估：

```bash
torchrun --standalone --nproc_per_node=4 Task3/eval.py --ckpt Task3/checkpoints/task3_ckpt.pt
```
