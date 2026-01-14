from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

try:
    from Task3.minifw.nn import SimpleCifarNet
    from Task3.minifw.tensor import Tensor
except ModuleNotFoundError:  # supports: cd Task3 && python eval.py
    from minifw.nn import SimpleCifarNet
    from minifw.tensor import Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task3: Evaluate checkpoint on CIFAR-10 test set (single GPU or DDP).")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--ckpt", type=str, default="Task3/checkpoints/task3_ckpt.pt")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--backend", type=str, default="", help="DDP backend override (default: nccl).")
    return p.parse_args()


def build_test_transform() -> transforms.Compose:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def correct_count(logits: torch.Tensor, targets: torch.Tensor) -> int:
    pred = logits.argmax(dim=1)
    return int((pred == targets).sum().item())


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    return int(val)


def get_dist_info() -> DistInfo:
    world_size = _env_int("WORLD_SIZE", 1)
    enabled = world_size > 1
    if not enabled:
        return DistInfo(enabled=False, rank=0, local_rank=0, world_size=1)
    return DistInfo(
        enabled=True,
        rank=_env_int("RANK", 0),
        local_rank=_env_int("LOCAL_RANK", 0),
        world_size=world_size,
    )


def setup_dist(args: argparse.Namespace, info: DistInfo) -> None:
    if not info.enabled:
        return
    if dist.is_initialized():
        return
    backend = args.backend.strip() or "nccl"
    torch.cuda.set_device(info.local_rank)
    dist.init_process_group(backend=backend, init_method="env://", device_id=torch.device(f"cuda:{info.local_rank}"))


def cleanup_dist(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(info: DistInfo, msg: str) -> None:
    if info.rank == 0:
        print(msg, flush=True)


def main() -> None:
    args = parse_args()
    info = get_dist_info()
    setup_dist(args, info)

    if not torch.cuda.is_available():
        raise RuntimeError("Task3 evaluation requires CUDA (custom CUDA kernels).")
    torch.cuda.set_device(info.local_rank)
    device = torch.device(f"cuda:{info.local_rank}")

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser()
    ckpt = torch.load(ckpt_path, map_location=device)

    model = SimpleCifarNet(device=device)
    state = ckpt.get("model", {})
    for name, p in model.named_parameters().items():
        if name not in state:
            raise KeyError(f"Missing parameter in checkpoint: {name}")
        p.data.copy_(state[name].to(device))

    buffers = ckpt.get("buffers", {})
    for name, b in model.named_buffers().items():
        if name in buffers:
            b.copy_(buffers[name].to(device))

    model.eval()

    if info.enabled and info.rank != 0:
        dist.barrier()
        ds = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=False, transform=build_test_transform())
    else:
        ds = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=build_test_transform())
        if info.enabled:
            dist.barrier()

    sampler = (
        torch.utils.data.distributed.DistributedSampler(ds, num_replicas=info.world_size, rank=info.rank, shuffle=False)
        if info.enabled
        else None
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    torch.set_grad_enabled(False)
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous()
        labels = labels.to(device, non_blocking=True)
        logits = model(Tensor(images, requires_grad=False)).data
        correct += correct_count(logits, labels)
        total += int(images.size(0))

    if info.enabled:
        t = torch.tensor([correct, total], device=device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        correct_g, total_g = int(t[0].item()), int(t[1].item())
    else:
        correct_g, total_g = correct, total

    rank0_print(info, f"ckpt: {ckpt_path} | test acc: {(correct_g / max(1, total_g)):.4f}")
    cleanup_dist(info)


if __name__ == "__main__":
    main()
