from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision

try:
    from Task2.models.cnn import create_model
    from Task2.utils import build_transforms, seed_everything
except ModuleNotFoundError:
    from models.cnn import create_model
    from utils import build_transforms, seed_everything


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 with single GPU or DDP (Task 2).")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine"])
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "cnn_bn"])
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--backend", type=str, default="", help="DDP backend override (default: nccl if cuda else gloo).")
    parser.add_argument("--no-eval", action="store_true", help="Skip validation evaluation at the end of each epoch.")
    return parser.parse_args()


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
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def setup_dist(args: argparse.Namespace, info: DistInfo) -> None:
    if not info.enabled:
        return
    if dist.is_initialized():
        return
    backend = args.backend.strip()
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    device_id: torch.device | None = None
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
        device_id = torch.device(f"cuda:{info.local_rank}")
    dist.init_process_group(backend=backend, init_method="env://", device_id=device_id)


def cleanup_dist(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def get_device(info: DistInfo) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
        return torch.device(f"cuda:{info.local_rank}")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    seed_everything(seed)


def rank0_print(info: DistInfo, msg: str) -> None:
    if info.rank == 0:
        print(msg, flush=True)


def build_run_dir(run_name: str, info: DistInfo) -> Path:
    task_dir = Path(__file__).resolve().parent
    outputs_dir = task_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        return outputs_dir / run_name
    ts = time.strftime("%Y%m%d_%H%M%S")
    return outputs_dir / f"run_{ts}_ws{info.world_size}"


def write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "world_size",
        "per_gpu_batch",
        "global_batch",
        "epoch_time_sec",
        "images_per_sec",
        "train_loss",
        "val_acc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


@torch.no_grad()
def evaluate(model: nn.Module, device: torch.device, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 0.0 if total == 0 else (100.0 * correct / total)


def main() -> None:
    args = parse_args()
    info = get_dist_info()
    setup_dist(args, info)

    device = get_device(info)
    set_seed(args.seed + info.rank)

    rank0_print(
        info,
        f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()} "
        f"rank={info.rank} local_rank={info.local_rank} world_size={info.world_size} device={device}",
    )

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    run_dir = build_run_dir(args.run_name, info)
    if info.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if info.enabled:
        dist.barrier()

    train_transform, test_transform = build_transforms(args.augment)

    if info.enabled and info.rank != 0:
        dist.barrier()
        train_dataset_aug = torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=False,
            transform=train_transform,
        )
    else:
        train_dataset_aug = torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=True,
            transform=train_transform,
        )
        if info.enabled:
            dist.barrier()

    train_dataset_plain = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
        transform=test_transform,
    )

    generator = torch.Generator().manual_seed(args.seed)
    total_size = len(train_dataset_aug)
    val_size = int(total_size * max(0.0, min(0.9, float(args.val_split))))
    indices = torch.randperm(total_size, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    trainset: torch.utils.data.Dataset[object] = torch.utils.data.Subset(train_dataset_aug, train_indices)
    valset: torch.utils.data.Dataset[object] | None = (
        torch.utils.data.Subset(train_dataset_plain, val_indices) if val_size > 0 else None
    )

    sampler: torch.utils.data.Sampler[int] | None = None
    if info.enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(
            trainset,
            num_replicas=info.world_size,
            rank=info.rank,
            shuffle=True,
            drop_last=False,
        )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    valloader = (
        torch.utils.data.DataLoader(
            valset,
            batch_size=512,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if valset is not None
        else None
    )

    net: nn.Module = create_model(args.model).to(device)
    if info.enabled:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[info.local_rank] if device.type == "cuda" else None,
            output_device=info.local_rank if device.type == "cuda" else None,
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            net.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    if args.scheduler == "cosine":
        scheduler: object | None = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metrics_rows: list[dict[str, object]] = []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        if info.enabled and isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(epoch)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        net.train()
        loss_sum = 0.0
        sample_count = 0

        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bsz = int(inputs.size(0))
            loss_sum += float(loss.item()) * bsz
            sample_count += bsz

        loss_sum_t = torch.tensor(loss_sum, device=device)
        sample_count_t = torch.tensor(sample_count, device=device, dtype=torch.long)
        if info.enabled:
            dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count_t, op=dist.ReduceOp.SUM)

        if scheduler is not None:
            scheduler.step()

        val_acc = ""
        if (not args.no_eval) and info.rank == 0 and valloader is not None:
            model_for_eval = net.module if hasattr(net, "module") else net
            acc = evaluate(model_for_eval, device, valloader)
            val_acc = f"{acc:.2f}"
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(
                    {"model": args.model, "state_dict": model_for_eval.state_dict(), "epoch": epoch, "val_acc": acc},
                    run_dir / "ckpt_best.pth",
                )

        if info.enabled:
            dist.barrier()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - t0

        epoch_time_t = torch.tensor(epoch_time, device=device)
        if info.enabled:
            dist.all_reduce(epoch_time_t, op=dist.ReduceOp.MAX)

        total_samples = int(sample_count_t.item())
        train_loss = float(loss_sum_t.item()) / max(1, total_samples)
        epoch_time_sec = float(epoch_time_t.item())
        images_per_sec = 0.0 if epoch_time_sec <= 0 else (total_samples / epoch_time_sec)

        if info.rank == 0:
            row: dict[str, object] = {
                "epoch": epoch,
                "world_size": info.world_size,
                "per_gpu_batch": args.batch_size,
                "global_batch": args.batch_size * info.world_size,
                "epoch_time_sec": f"{epoch_time_sec:.4f}",
                "images_per_sec": f"{images_per_sec:.2f}",
                "train_loss": f"{train_loss:.6f}",
                "val_acc": val_acc,
            }
            metrics_rows.append(row)
            print(
                f"epoch={epoch} time={epoch_time_sec:.3f}s img/s={images_per_sec:.1f} "
                f"train_loss={train_loss:.4f} val_acc={val_acc}",
                flush=True,
            )

    if info.rank == 0:
        ckpt_path = run_dir / "ckpt.pth"
        model_to_save = net.module if hasattr(net, "module") else net
        torch.save({"model": args.model, "state_dict": model_to_save.state_dict(), "epoch": args.epochs}, ckpt_path)
        write_metrics_csv(metrics_rows, run_dir / "metrics.csv")
        print(f"Saved checkpoint to: {ckpt_path}", flush=True)
        print(f"Saved metrics to: {run_dir / 'metrics.csv'}", flush=True)

    cleanup_dist(info)


if __name__ == "__main__":
    main()
