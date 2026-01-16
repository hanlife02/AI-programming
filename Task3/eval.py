from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

try:
    from Task3.minifw.nn import MyNet
    from Task3.minifw.tensor import Tensor
except ModuleNotFoundError:
    from minifw.nn import MyNet
    from minifw.tensor import Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task3: Evaluate checkpoint on CIFAR-10 test set (single GPU or DDP).")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True, help="use EMA weights if present in checkpoint")
    p.add_argument("--backend", type=str, default="", help="DDP backend override (default: nccl).")
    return p.parse_args()


def build_test_transform():
    try:
        import torchvision.transforms as transforms
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Task3 evaluation requires torchvision. Install it with: pip install torchvision") from e

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
    try:
        import torchvision
    except ModuleNotFoundError as e: 
        raise ModuleNotFoundError("Task3 evaluation requires torchvision. Install it with: pip install torchvision") from e
    info = get_dist_info()
    setup_dist(args, info)

    if not torch.cuda.is_available():
        raise RuntimeError("Task3 evaluation requires CUDA (custom CUDA kernels).")
    torch.cuda.set_device(info.local_rank)
    device = torch.device(f"cuda:{info.local_rank}")

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (task_dir / "checkpoint" / "ckpt.pth")
    ckpt = torch.load(ckpt_path, map_location=device)

    arch = str(ckpt.get("arch", "mynet"))
    if arch in {"mynet", "vgg16"}:
        model = MyNet(vgg_name="VGG16", device=device)
    else:
        raise ValueError(f"Unknown checkpoint arch: {arch} (supported: 'mynet', legacy: 'vgg16')")
    state = ckpt.get("net", {})
    if bool(args.use_ema):
        ema_state = ckpt.get("ema", None)
        if isinstance(ema_state, dict) and len(ema_state) > 0:
            state = ema_state
    for name, p in model.named_parameters().items():
        if name not in state:
            raise KeyError(f"Missing parameter in checkpoint: {name}")
        p.data.copy_(state[name].to(device))

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
    class_names = list(getattr(ds, "classes", []))
    if len(class_names) == 0:
        num_classes = 10
        class_names = [str(i) for i in range(num_classes)]
    else:
        num_classes = len(class_names)
    correct_per_class = torch.zeros(num_classes, device=device, dtype=torch.int64)
    total_per_class = torch.zeros(num_classes, device=device, dtype=torch.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True).contiguous()
        labels = labels.to(device, non_blocking=True)
        logits = model(Tensor(images, requires_grad=False)).data
        preds = logits.argmax(dim=1)
        correct_mask = preds == labels
        correct += int(correct_mask.sum().item())
        total += int(images.size(0))
        total_per_class += torch.bincount(labels, minlength=num_classes)
        if correct_mask.any():
            correct_per_class += torch.bincount(labels[correct_mask], minlength=num_classes)

    if info.enabled:
        t = torch.tensor([correct, total], device=device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        correct_g, total_g = int(t[0].item()), int(t[1].item())
        dist.all_reduce(correct_per_class, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_per_class, op=dist.ReduceOp.SUM)
    else:
        correct_g, total_g = correct, total

    rank0_print(info, f"ckpt: {ckpt_path} | test acc: {(correct_g / max(1, total_g)):.4f}")
    if info.rank == 0:
        correct_list = correct_per_class.cpu().tolist()
        total_list = total_per_class.cpu().tolist()
        rank0_print(info, "per-class acc:")
        for idx, name in enumerate(class_names):
            acc = (correct_list[idx] / max(1, total_list[idx])) if idx < len(total_list) else 0.0
            rank0_print(info, f"  {idx:02d}-{name}: {acc:.4f}")
    cleanup_dist(info)


if __name__ == "__main__":
    main()
