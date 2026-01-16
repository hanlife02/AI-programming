from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch

try:
    from Task3.minifw.optim import SGD
    from Task3.minifw.tensor import Tensor
    from Task3.utils import progress_bar
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Task3.minifw.optim import SGD
    from Task3.minifw.tensor import Tensor
    from Task3.utils import progress_bar

from model import TinyVGG
from tiny_imagenet import TinyImageNet, ensure_tiny_imagenet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny ImageNet Training (Task3 custom CUDA framework)")
    p.add_argument("--arch", type=str, default="VGG16", choices=["VGG11", "VGG13", "VGG16", "VGG19"])
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1, help="learning rate")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help="Split each batch into smaller micro-batches to reduce memory (0 disables).",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=1, help="update progress bar every N steps (reduces CPU overhead)")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument(
        "--wd-mode",
        type=str,
        default="weights",
        choices=["weights", "all"],
        help="weight decay policy: 'weights' applies decay only to params with ndim>=2 (recommended); 'all' matches old behavior",
    )
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--autoaugment", action=argparse.BooleanOptionalAction, default=False, help="use torchvision AutoAugment(ImageNet)")
    p.add_argument("--random-erasing", type=float, default=0.0, help="RandomErasing probability (e.g. 0.25)")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=False, help="download Tiny ImageNet if missing")
    p.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--t-max", type=int, default=200, help="T_max for cosine scheduler")
    p.add_argument("--warmup-epochs", type=int, default=5, help="linear warmup epochs (0 to disable)")
    p.add_argument("--min-lr", type=float, default=0.0, help="minimum lr for cosine schedule")
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=False, help="enable EMA of model parameters")
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay (if --ema)")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument(
        "--loss-csv",
        type=str,
        default="",
        help="Path to save per-step loss as CSV (default: disabled).",
    )
    p.add_argument(
        "--save-loss-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a loss curve PNG at the end of training (default: enabled).",
    )
    p.add_argument(
        "--loss-plot",
        type=str,
        default="",
        help="Path to save loss curve image (default: Tiny-imagenet/outputs/loss_curve.png when --save-loss-plot is enabled).",
    )
    p.add_argument(
        "--loss-log-every",
        type=int,
        default=10,
        help="record loss every N steps for CSV/plot (reduces overhead; default: 10)",
    )
    return p.parse_args()


def build_transforms(image_size: int, augment: bool, autoaugment: bool, random_erasing_p: float) -> tuple[object, object]:
    try:
        import torchvision.transforms as transforms
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Tiny ImageNet training requires torchvision. Install it with: pip install torchvision") from e

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = max(int(image_size), int(round(float(image_size) / 0.875)))

    train_ops: list[transforms.Transform] = []
    if augment:
        train_ops.append(transforms.RandomResizedCrop(image_size))
        train_ops.append(transforms.RandomHorizontalFlip())
        if autoaugment:
            if not hasattr(transforms, "AutoAugment"):
                raise RuntimeError("torchvision.transforms.AutoAugment is unavailable in this torchvision version")
            train_ops.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
    else:
        train_ops.extend([transforms.Resize(resize_size), transforms.CenterCrop(image_size)])
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if random_erasing_p > 0:
        train_ops.append(transforms.RandomErasing(p=float(random_erasing_p)))

    test_ops = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def cosine_lr(base_lr: float, epoch: int, t_max: int, min_lr: float = 0.0) -> float:
    if t_max <= 0:
        return float(base_lr)
    t = min(max(epoch, 0), t_max)
    cos = 0.5 * (1.0 + math.cos(math.pi * float(t) / float(t_max)))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cos


def scheduled_lr(args: argparse.Namespace, epoch: int) -> float:
    base_lr = float(args.lr)
    warmup_epochs = max(0, int(args.warmup_epochs))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    if args.scheduler == "cosine":
        t_max = int(args.t_max)
        if t_max <= 0:
            t_max = int(args.epochs)
        t_cur = epoch - warmup_epochs
        t_den = max(1, t_max - warmup_epochs)
        return cosine_lr(base_lr, t_cur, t_den, min_lr=float(args.min_lr))
    return base_lr


def write_loss_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["global_step", "epoch", "batch_idx", "loss"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def maybe_save_loss_plot(steps: list[int], losses: list[float], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping loss curve plot. Install via: pip install matplotlib")
        return

    if not steps or not losses:
        print("No loss rows to plot; skipping.")
        return

    ema_losses: list[float] = []
    alpha = 0.05
    ema = losses[0]
    for loss in losses:
        ema = alpha * loss + (1 - alpha) * ema
        ema_losses.append(ema)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, losses, linewidth=0.8, alpha=0.25, label="loss (raw)")
    plt.plot(steps, ema_losses, linewidth=1.8, label=f"loss (EMA, alpha={alpha})")
    plt.xlabel("global step")
    plt.ylabel("loss")
    plt.title("Tiny ImageNet Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@torch.no_grad()
def test(net, device: torch.device, loader: torch.utils.data.DataLoader) -> float:
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True).contiguous()
        targets = targets.to(device, non_blocking=True)
        outputs = net(Tensor(inputs, requires_grad=False)).data
        _, predicted = outputs.max(1)
        total += int(targets.size(0))
        correct += int(predicted.eq(targets).sum().item())
        progress_bar(batch_idx, len(loader), "Acc: %.3f%% (%d/%d)" % (100.0 * correct / max(1, total), correct, total))
    return 100.0 * correct / max(1, total)


def main() -> None:
    args = parse_args()
    try:
        import torchvision
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Tiny ImageNet training requires torchvision. Install it with: pip install torchvision") from e

    if not torch.cuda.is_available():
        raise RuntimeError("Tiny ImageNet training requires CUDA (custom CUDA kernels).")
    device = torch.device("cuda")

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (task_dir / "checkpoint" / "ckpt.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    loss_csv_path = Path(args.loss_csv).expanduser() if args.loss_csv else None
    loss_plot_path = (
        (Path(args.loss_plot).expanduser() if args.loss_plot else (task_dir / "outputs" / "loss_curve.png"))
        if bool(args.save_loss_plot)
        else None
    )

    print("==> Preparing data..", flush=True)
    transform_train, transform_test = build_transforms(
        int(args.image_size),
        bool(args.augment),
        bool(args.autoaugment),
        float(args.random_erasing),
    )
    data_root = ensure_tiny_imagenet(data_dir, download=bool(args.download))
    trainset = TinyImageNet(root=data_root, split="train", transform=transform_train)
    if len(trainset.classes) != int(args.num_classes):
        raise ValueError(f"num-classes={args.num_classes} but dataset has {len(trainset.classes)} classes")
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    valset = TinyImageNet(root=data_root, split="val", transform=transform_test, class_to_idx=trainset.class_to_idx)
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    print("==> Building model..", flush=True)
    net = TinyVGG(vgg_name=str(args.arch), num_classes=int(args.num_classes), device=device)
    best_acc = 0.0
    start_epoch = 0
    ema: dict[str, torch.Tensor] | None = None
    ema_backup: dict[str, torch.Tensor] | None = None

    if args.resume:
        print("==> Resuming from checkpoint..", flush=True)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        arch = str(ckpt.get("arch", args.arch))
        if arch != str(args.arch):
            raise ValueError(f"Checkpoint arch mismatch: ckpt={arch!r} args.arch={args.arch!r}")
        num_classes = int(ckpt.get("num_classes", args.num_classes))
        if num_classes != int(args.num_classes):
            raise ValueError(f"Checkpoint num_classes mismatch: ckpt={num_classes} args.num_classes={args.num_classes}")
        ckpt_class_to_idx = ckpt.get("class_to_idx", None)
        if isinstance(ckpt_class_to_idx, dict) and ckpt_class_to_idx != trainset.class_to_idx:
            raise ValueError("Checkpoint class_to_idx mismatch with dataset")
        state = ckpt.get("net", {})
        for name, p in net.named_parameters().items():
            if name not in state:
                raise KeyError(f"Missing parameter in checkpoint: {name}")
            p.data.copy_(state[name].to(device))
        best_acc = float(ckpt.get("acc", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if bool(args.ema):
            ema_state = ckpt.get("ema", None)
            if isinstance(ema_state, dict):
                ema = {k: v.to(device) for k, v in ema_state.items()}

    params = list(net.parameters())
    if str(args.wd_mode) == "weights":
        decay, no_decay = [], []
        for p in params:
            (decay if p.data.ndim >= 2 else no_decay).append(p)
        optimizer = SGD(
            [
                {"params": decay, "weight_decay": float(args.weight_decay)},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = SGD(params, lr=float(args.lr), momentum=float(args.momentum), weight_decay=float(args.weight_decay))

    if bool(args.ema) and ema is None:
        ema = {k: v.data.detach().clone() for k, v in net.named_parameters().items()}
    if ema is not None:
        ema_backup = {k: torch.empty_like(v.data) for k, v in net.named_parameters().items()}

    def ema_update() -> None:
        if ema is None:
            return
        d = float(args.ema_decay)
        for name, p in net.named_parameters().items():
            ema[name].mul_(d).add_(p.data, alpha=(1.0 - d))

    def ema_apply() -> None:
        if ema is None or ema_backup is None:
            return
        for name, p in net.named_parameters().items():
            ema_backup[name].copy_(p.data)
            p.data.copy_(ema[name])

    def ema_restore() -> None:
        if ema is None or ema_backup is None:
            return
        for name, p in net.named_parameters().items():
            p.data.copy_(ema_backup[name])

    record_loss = (loss_csv_path is not None) or (loss_plot_path is not None)
    log_every = max(1, int(args.log_every))
    loss_log_every = max(1, int(args.loss_log_every))
    global_step = int(start_epoch) * len(trainloader)

    loss_steps: list[int] = []
    loss_values: torch.Tensor | None = None
    loss_write_idx = 0
    if record_loss:
        total_steps = (int(args.epochs) - int(start_epoch)) * len(trainloader)
        max_records = (total_steps + loss_log_every - 1) // loss_log_every
        loss_values = torch.empty((max_records,), device=device, dtype=torch.float32)

    for epoch in range(start_epoch, int(args.epochs)):
        print("\nEpoch: %d" % epoch, flush=True)

        optimizer.lr = scheduled_lr(args, epoch)

        net.train()
        train_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        correct_sum = torch.zeros((), device=device, dtype=torch.int64)
        total = 0
        micro_batch_size = max(0, int(args.micro_batch_size))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = int(targets.size(0))
            micro_batch = micro_batch_size if micro_batch_size > 0 else batch_size
            micro_batch = min(micro_batch, batch_size)

            if micro_batch == batch_size:
                inputs = inputs.to(device, non_blocking=True).contiguous()
                targets = targets.to(device, non_blocking=True)

                logits = net(Tensor(inputs, requires_grad=False))
                loss = logits.cross_entropy(targets)
                loss.backward()
                optimizer.step()
                ema_update()
                optimizer.zero_grad()

                batch_loss = loss.data.detach()
                _, predicted = logits.data.max(1)
                total += int(targets.size(0))
                correct_sum.add_(predicted.eq(targets).sum())
            else:
                batch_loss = torch.zeros((), device=device, dtype=torch.float32)
                optimizer.zero_grad()
                for start in range(0, batch_size, micro_batch):
                    end = min(start + micro_batch, batch_size)
                    micro_inputs = inputs[start:end].to(device, non_blocking=True).contiguous()
                    micro_targets = targets[start:end].to(device, non_blocking=True)

                    logits = net(Tensor(micro_inputs, requires_grad=False))
                    loss = logits.cross_entropy(micro_targets)
                    scale = float(end - start) / float(batch_size)
                    if scale != 1.0:
                        scaled_loss = loss * scale
                        scaled_loss.backward()
                    else:
                        loss.backward()

                    batch_loss.add_(loss.data.detach() * scale)
                    _, predicted = logits.data.max(1)
                    total += int(micro_targets.size(0))
                    correct_sum.add_(predicted.eq(micro_targets).sum())

                optimizer.step()
                ema_update()
                optimizer.zero_grad()

            train_loss_sum.add_(batch_loss)

            if record_loss and (global_step % loss_log_every == 0) and loss_values is not None:
                if loss_write_idx < int(loss_values.numel()):
                    loss_values[loss_write_idx] = batch_loss
                    loss_steps.append(int(global_step))
                    loss_write_idx += 1
            global_step += 1

            if (batch_idx % log_every == 0) or (batch_idx == len(trainloader) - 1):
                avg_loss = float((train_loss_sum / float(batch_idx + 1)).item())
                acc = float((correct_sum.float() * (100.0 / float(max(1, total)))).item())
                progress_bar(
                    batch_idx,
                    len(trainloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.4g"
                    % (avg_loss, acc, int(correct_sum.item()), total, optimizer.lr),
                )

        print("\n==> Testing..", flush=True)
        if ema is not None:
            ema_apply()
        acc = test(net, device, valloader)
        if ema is not None:
            ema_restore()

        if acc > best_acc:
            print("Saving..", flush=True)
            state = {
                "arch": str(args.arch),
                "num_classes": int(args.num_classes),
                "image_size": int(args.image_size),
                "class_to_idx": trainset.class_to_idx,
                "net": {k: v.data for k, v in net.named_parameters().items()},
                "ema": ema,
                "acc": acc,
                "epoch": epoch,
            }
            torch.save(state, ckpt_path)
            best_acc = acc

    if record_loss:
        losses: list[float] = []
        if loss_values is not None and loss_write_idx > 0:
            losses = loss_values[:loss_write_idx].detach().cpu().tolist()
        if loss_csv_path is not None:
            rows = [
                {
                    "global_step": int(s),
                    "epoch": int(s // len(trainloader)),
                    "batch_idx": int(s % len(trainloader)),
                    "loss": float(l),
                }
                for s, l in zip(loss_steps, losses)
            ]
            write_loss_csv(rows, loss_csv_path)
            print(f"Saved loss CSV to: {loss_csv_path}", flush=True)
        if loss_plot_path is not None:
            maybe_save_loss_plot(loss_steps, losses, loss_plot_path)
            if loss_plot_path.exists():
                print(f"Saved loss curve to: {loss_plot_path}", flush=True)

    print("best acc: %.3f%%" % best_acc, flush=True)


if __name__ == "__main__":
    main()
