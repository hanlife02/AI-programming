from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Tuple

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

try:
    from Task3.minifw.nn import SimpleCifarNet
    from Task3.minifw.optim import SGD
    from Task3.minifw.tensor import Tensor
except ModuleNotFoundError:  # supports: cd Task3 && python train.py
    from minifw.nn import SimpleCifarNet
    from minifw.optim import SGD
    from minifw.tensor import Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task3: Custom CUDA framework CIFAR-10 training (single GPU or DDP).")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "step", "cosine"])
    p.add_argument("--step-size", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--backend", type=str, default="", help="DDP backend override (default: nccl).")
    p.add_argument("--no-eval", action="store_true", help="Skip validation evaluation each epoch.")
    p.add_argument("--debug-step", action="store_true", help="Run exactly 1 train step and print debug stats.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean, std)
    train_t = []
    if augment:
        train_t.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    train_t.extend([transforms.ToTensor(), normalize])
    test_t = [transforms.ToTensor(), normalize]
    return transforms.Compose(train_t), transforms.Compose(test_t)


@dataclass
class CudaPrefetcher:
    loader: torch.utils.data.DataLoader
    device: torch.device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        stream = torch.cuda.Stream(device=self.device)
        it = iter(self.loader)
        next_x = None
        next_y = None
        try:
            x, y = next(it)
        except StopIteration:
            return
        with torch.cuda.stream(stream):
            next_x = x.to(self.device, non_blocking=True)
            next_y = y.to(self.device, non_blocking=True)
        for x, y in it:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            cur_x, cur_y = next_x, next_y
            with torch.cuda.stream(stream):
                next_x = x.to(self.device, non_blocking=True)
                next_y = y.to(self.device, non_blocking=True)
            yield cur_x, cur_y
        torch.cuda.current_stream(self.device).wait_stream(stream)
        yield next_x, next_y


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


def broadcast_params(info: DistInfo, model: SimpleCifarNet) -> None:
    if not info.enabled:
        return
    for _, p in model.named_parameters().items():
        dist.broadcast(p.data, src=0)


def build_grad_sync(params: list[Tensor], info: DistInfo) -> Callable[[], None]:
    if not info.enabled:
        def _noop() -> None:
            return
        return _noop

    numels: list[int] = [int(p.data.numel()) for p in params]
    offsets: list[int] = [0]
    for n in numels:
        offsets.append(offsets[-1] + n)
    total = offsets[-1]
    grad_buffer = torch.empty((total,), device=params[0].data.device, dtype=torch.float32)

    def _sync() -> None:
        off = 0
        for p, n in zip(params, numels, strict=True):
            if p.grad is None:
                raise RuntimeError("found None grad during distributed sync")
            grad_buffer[off : off + n].copy_(p.grad.reshape(-1))
            off += n

        dist.all_reduce(grad_buffer, op=dist.ReduceOp.SUM)
        grad_buffer.mul_(1.0 / float(info.world_size))

        off = 0
        for p, n in zip(params, numels, strict=True):
            p.grad = grad_buffer[off : off + n].view_as(p.data)
            off += n

    return _sync


def _tensor_stats(x: torch.Tensor) -> str:
    x_f = x.float()
    return (
        f"shape={tuple(x.shape)} dtype={str(x.dtype).replace('torch.', '')} "
        f"mean={x_f.mean().item():.4g} std={x_f.std(unbiased=False).item():.4g} "
        f"min={x_f.min().item():.4g} max={x_f.max().item():.4g}"
    )


def _lr_for_epoch(
    base_lr: float,
    scheduler: str,
    epoch: int,
    epochs: int,
    step_size: int,
    gamma: float,
    min_lr: float,
) -> float:
    if scheduler == "none":
        return float(base_lr)
    if scheduler == "step":
        step_size = max(1, int(step_size))
        k = max(0, (int(epoch) - 1) // step_size)
        return float(base_lr) * (float(gamma) ** k)
    if scheduler == "cosine":
        t = (float(epoch) - 1.0) / max(1.0, float(epochs))
        return float(min_lr) + 0.5 * (float(base_lr) - float(min_lr)) * (1.0 + math.cos(math.pi * t))
    raise ValueError(f"unknown scheduler: {scheduler}")


def _debug_one_step(model: SimpleCifarNet, images: torch.Tensor, labels: torch.Tensor, optim: SGD, sync_grads: Callable[[], None], info: DistInfo) -> None:
    try:
        from Task3.minifw import ops as fw_ops
    except ModuleNotFoundError:
        from minifw import ops as fw_ops

    if info.rank == 0:
        ext_mod = fw_ops.ext()
        ext_path = getattr(ext_mod, "__file__", "<unknown>")
        print(f"[debug] task3_ops loaded from: {ext_path}", flush=True)

    x = Tensor(images.contiguous(), requires_grad=False)
    t1 = model.conv1(x)
    t2 = t1.relu()
    t3 = model.conv2(t2)
    t4 = t3.relu()
    t5 = model.pool1(t4)
    t6 = model.conv3(t5)
    t7 = t6.relu()
    t8 = model.pool2(t7)
    t9 = model.gap(t8)
    logits = model.fc(t9)
    loss = logits.cross_entropy(labels)

    if info.rank == 0:
        print(f"[debug] conv1 out: {_tensor_stats(t1.data)}", flush=True)
        print(f"[debug] pool2 out: {_tensor_stats(t8.data)}", flush=True)
        print(f"[debug] gap out : {_tensor_stats(t9.data)}", flush=True)
        print(f"[debug] logits  : {_tensor_stats(logits.data)}", flush=True)
        print(f"[debug] loss    : {loss.data.item():.6f}", flush=True)

    loss.backward()
    sync_grads()

    if info.rank == 0:
        named = model.named_parameters()
        for k in ["conv1.w", "conv1.b", "conv2.w", "conv2.b", "conv3.w", "conv3.b", "fc.w", "fc.b"]:
            p = named.get(k)
            if p is None:
                continue
            g = p.grad
            if g is None:
                print(f"[debug] grad {k}: None", flush=True)
            else:
                g_f = g.float()
                print(
                    f"[debug] grad {k}: mean={g_f.mean().item():.4g} "
                    f"abs_mean={g_f.abs().mean().item():.4g} max={g_f.abs().max().item():.4g} "
                    f"norm={g_f.norm().item():.4g}",
                    flush=True,
                )

    # check parameters actually change after the step
    before = {}
    if info.rank == 0:
        for name, p in model.named_parameters().items():
            if name in {"conv1.w", "fc.w", "fc.b"}:
                before[name] = p.data.detach().clone()

    optim.step()
    optim.zero_grad()

    if info.rank == 0:
        for name, p in model.named_parameters().items():
            if name not in before:
                continue
            delta = (p.data - before[name]).float()
            print(
                f"[debug] update {name}: abs_mean={delta.abs().mean().item():.4g} "
                f"max={delta.abs().max().item():.4g} norm={delta.norm().item():.4g}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    info = get_dist_info()
    setup_dist(args, info)

    if not torch.cuda.is_available():
        raise RuntimeError("Task3 requires CUDA (custom CUDA kernels).")
    torch.cuda.set_device(info.local_rank)
    device = torch.device(f"cuda:{info.local_rank}")

    # same model init on all ranks; data shuffling handled by DistributedSampler.
    seed_everything(args.seed)
    torch.set_grad_enabled(False)

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (task_dir / "checkpoints" / "task3_ckpt.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    rank0_print(
        info,
        f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()} "
        f"rank={info.rank} local_rank={info.local_rank} world_size={info.world_size} device={device} "
        f"per_gpu_batch={args.batch_size} global_batch={args.batch_size * info.world_size}",
    )

    train_t, test_t = build_transforms(not args.no_augment)
    if info.enabled and info.rank != 0:
        dist.barrier()
        train_ds_aug = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=train_t)
    else:
        train_ds_aug = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_t)
        if info.enabled:
            dist.barrier()
    train_ds_plain = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=False, transform=test_t)

    # Important: train/val split should be identical across ranks for correctness/reproducibility.
    g = torch.Generator().manual_seed(args.seed)
    total = len(train_ds_aug)
    val_size = int(total * max(0.0, min(0.9, float(args.val_split))))
    idx = torch.randperm(total, generator=g).tolist()
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]
    train_ds = torch.utils.data.Subset(train_ds_aug, tr_idx)
    val_ds = torch.utils.data.Subset(train_ds_plain, val_idx) if val_size > 0 else None

    sampler: torch.utils.data.Sampler[int] | None = None
    if info.enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds,
            num_replicas=info.world_size,
            rank=info.rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        generator=g,
        worker_init_fn=seed_worker,
    )
    val_loader = (
        torch.utils.data.DataLoader(
            val_ds,
            batch_size=max(256, args.batch_size),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
            worker_init_fn=seed_worker,
        )
        if val_ds is not None
        else None
    )

    model = SimpleCifarNet(device=device)
    rank0_print(info, f"model params: {sum(int(p.data.numel()) for p in model.parameters()):,}")
    broadcast_params(info, model)

    params = list(model.parameters())
    optim = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sync_grads = build_grad_sync(params, info)
    base_lr = float(args.lr)

    best_val = 0.0
    try:
        for epoch in range(1, args.epochs + 1):
            optim.lr = _lr_for_epoch(
                base_lr=base_lr,
                scheduler=args.scheduler,
                epoch=epoch,
                epochs=args.epochs,
                step_size=args.step_size,
                gamma=args.gamma,
                min_lr=args.min_lr,
            )
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            model.train()
            optim.zero_grad()

            torch.cuda.synchronize()
            t0 = time.time()
            n_seen = 0
            loss_sum = 0.0
            correct_sum = 0

            prefetch = CudaPrefetcher(train_loader, device)
            torch.cuda.synchronize()
            step_t0 = time.time()
            for step, (images, labels) in enumerate(prefetch, start=1):
                images = images.contiguous()
                labels = labels.to(device, non_blocking=True)

                if args.debug_step:
                    model.train()
                    optim.zero_grad()
                    _debug_one_step(model, images, labels, optim, sync_grads, info)
                    return

                logits = model(Tensor(images, requires_grad=False))
                loss = logits.cross_entropy(labels)
                loss.backward()
                sync_grads()
                optim.step()
                optim.zero_grad()

                bs = int(images.size(0))
                n_seen += bs
                loss_sum += float(loss.data.item()) * bs
                correct_sum += correct_count(logits.data, labels)

                if args.log_every > 0 and step % args.log_every == 0:
                    torch.cuda.synchronize()
                    dt = time.time() - step_t0
                    ips = (args.log_every * args.batch_size) / max(dt, 1e-6)
                    rank0_print(
                        info,
                        f"epoch {epoch:03d} step {step:04d} | "
                        f"loss {loss_sum/n_seen:.4f} acc {correct_sum/n_seen:.4f} | {ips:.0f} img/s (per-rank)",
                    )
                    step_t0 = time.time()

            torch.cuda.synchronize()
            dt = time.time() - t0

            if info.enabled:
                loss_t = torch.tensor([loss_sum], device=device, dtype=torch.float32)
                cnt_t = torch.tensor([correct_sum, n_seen], device=device, dtype=torch.int64)
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
                loss_sum_g = float(loss_t.item())
                correct_sum_g = int(cnt_t[0].item())
                n_seen_g = int(cnt_t[1].item())
            else:
                loss_sum_g, correct_sum_g, n_seen_g = loss_sum, correct_sum, n_seen

            train_loss = loss_sum_g / max(1, n_seen_g)
            train_acc = correct_sum_g / max(1, n_seen_g)
            global_img_s = (n_seen_g / dt) if dt > 0 else 0.0
            rank0_print(
                info,
                f"epoch {epoch:03d} train | lr {optim.lr:.4g} loss {train_loss:.4f} acc {train_acc:.4f} | {global_img_s:.0f} img/s (global)",
            )

            if (not args.no_eval) and val_loader is not None:
                model.eval()
                v_correct = 0
                v_seen = 0
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True).contiguous()
                    labels = labels.to(device, non_blocking=True)
                    logits = model(Tensor(images, requires_grad=False)).data
                    v_correct += correct_count(logits, labels)
                    v_seen += int(images.size(0))

                if info.enabled:
                    v_t = torch.tensor([v_correct, v_seen], device=device, dtype=torch.int64)
                    dist.all_reduce(v_t, op=dist.ReduceOp.SUM)
                    v_correct_g, v_seen_g = int(v_t[0].item()), int(v_t[1].item())
                else:
                    v_correct_g, v_seen_g = v_correct, v_seen

                val_acc = 0.0 if v_seen_g == 0 else (v_correct_g / v_seen_g)
                rank0_print(info, f"epoch {epoch:03d} val   | acc {val_acc:.4f}")

                if info.rank == 0 and val_acc > best_val:
                    best_val = val_acc
                    torch.save(
                        {
                            "epoch": epoch,
                            "val_acc": val_acc,
                            "world_size": info.world_size,
                            "model": {k: v.data for k, v in model.named_parameters().items()},
                        },
                        ckpt_path,
                    )
                    rank0_print(info, f"saved best checkpoint to: {ckpt_path}")

        rank0_print(info, f"best val acc: {best_val:.4f}")
    finally:
        cleanup_dist(info)


if __name__ == "__main__":
    main()
