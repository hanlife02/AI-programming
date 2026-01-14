# Task 3: Custom Implementation (CIFAR-10)

本目录实现了一个最小但可训练的“自研”CNN 框架，满足作业要求：

- **卷积网络（forward/backward）全部由自定义 CUDA 算子实现**：`Task3/csrc/ops.cu`
- 用 **pybind11** 导出为 Python 可调用扩展模块 `task3_ops`：`Task3/csrc/bindings.cpp`
  - 其中卷积提供了 `task3_ops.Conv2dOp`（class），并在 Python 侧缓存复用：`Task3/minifw/ops.py`
- Python 侧只负责：自动微分（计算图反传）、优化器（SGD）、训练循环与数据加载：`Task3/minifw/tensor.py` / `Task3/minifw/optim.py` / `Task3/train.py`
- 数据处理不做限制：这里使用 `torchvision.datasets.CIFAR10 + DataLoader`

## 1) 构建扩展

在有 CUDA 的环境中（需要 `nvcc` 与可用 GPU）：

```bash
cd Task3
python setup.py build_ext --inplace
```

成功后会生成 `task3_ops.*.so`（Linux/macOS）或 `task3_ops.*.pyd`（Windows）。

如果你遇到类似 `nvcc fatal: Unsupported gpu architecture 'compute_120'` 的报错，通常是 **PyTorch 编译时的 CUDA 版本** 与 **本机 nvcc 版本** 不一致导致自动选择了 nvcc 不支持的架构。本项目的 `Task3/setup.py` 会自动过滤不支持的架构；你也可以手动指定：

```bash
# 仅为示例：按你的 GPU 计算能力调整（如 8.6 / 8.9 / 9.0）
TASK3_CUDA_ARCH_LIST="8.6" python setup.py build_ext --inplace
```

如果编译能过但运行时报 `no kernel image is available for execution on the device`，说明扩展没有为你的 GPU 生成可执行的内核镜像。此时请确保 **最高档架构带 `+PTX`**（便于驱动在新 GPU 上 JIT）：

```bash
# 仅为示例：用你环境里 nvcc 支持的最高架构（例如 9.0）并加 +PTX
TASK3_CUDA_ARCH_LIST="9.0+PTX" python setup.py build_ext --inplace
```

## 2) 训练

```bash
python Task3/train.py --epochs 50 --batch-size 256 --num-workers 8 --lr 0.1
```

说明：

- 训练过程使用自研 `Tensor` 的计算图反传（不依赖 `torch.autograd`），并调用 CUDA 扩展完成各层的 forward/backward 与 `sgd_update_` 参数更新。
- 学习率可选余弦退火：`--scheduler cosine --t-max <int>`；或关闭：`--scheduler none`。

常用参数：

- `--model`: `mynet`（默认且唯一）
- `--data-dir`: CIFAR-10 下载/缓存目录（默认 `Task3/data`）
- `--ckpt`: checkpoint 路径（默认 `Task3/checkpoint/ckpt.pth`）
- `--save-loss-plot/--no-save-loss-plot`: 训练结束自动保存 loss curve（默认开启，保存到 `Task3/outputs/loss_curve.png`）
- `--loss-plot`: 自定义 loss curve 图片输出路径
- `--loss-csv`: 可选保存 loss 到 CSV（与 `--loss-log-every` 配合）
- `--loss-log-every`: 每 N step 记录一次 loss（默认 10；N 越大越省 CPU/内存，但曲线更稀疏）
- `--log-every`: 每 N step 刷新一次进度条（默认 1；可明显减少终端输出带来的开销）
- `--no-augment`: 关闭数据增强（更快但通常准确率更低）
- `--wd-mode weights`: 仅对权重（ndim>=2）做 weight decay（更常见、更容易提升验证集 acc）；`--wd-mode all` 为旧行为（所有参数都 decay）
- `--warmup-epochs`: 学习率线性 warmup（默认 5），通常能让训练更稳定
- `--autoaugment/--random-erasing`: 更强的数据增强（通常能提高验证集 acc，但训练更慢）
- `--ema`: 启用参数 EMA（通常能提高验证集 acc；`eval.py` 会默认优先用 EMA 权重）
- `-r/--resume`: 从 `--ckpt` 恢复训练（会校验模型结构 `arch` 是否一致）

## 3) 说明

- 本实现不使用 PyTorch 的 autograd / nn.Module 来训练；仅用其 Tensor 作为显存容器以及 torchvision 的数据集/加载器。
- 若你的评测机器有更强 GPU，可优先提高 `--batch-size` 与 `--num-workers` 以获得更高吞吐。

## 4) 测试集评估

```bash
python Task3/eval.py --ckpt Task3/checkpoint/ckpt.pth
```

如 checkpoint 中包含 EMA 权重，`eval.py` 默认会使用 EMA（可用 `--no-use-ema` 关闭）。
