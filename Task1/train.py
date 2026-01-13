from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
    from Task1.models.cnn import create_model
    from Task1.utils import build_transforms, seed_everything
except ModuleNotFoundError:
    from models.cnn import create_model
    from utils import build_transforms, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 (Task 1).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "step", "cosine"])
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--best-ckpt", type=str, default="")
    parser.add_argument("--model", type=str, default="cnn_bn", choices=["cnn", "cnn_bn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument(
        "--loss-csv",
        type=str,
        default="",
        help="Path to save per-step loss as CSV (default: disabled).",
    )
    parser.add_argument(
        "--save-loss-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a loss curve PNG at the end of training (default: enabled).",
    )
    parser.add_argument(
        "--loss-plot",
        type=str,
        default="",
        help="Path to save loss curve image (default: Task1/outputs/loss_curve.png when --save-loss-plot is enabled).",
    )
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_loss_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["global_step", "epoch", "batch_idx", "loss"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def maybe_save_loss_plot(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping loss curve plot. Install via: pip install matplotlib")
        return

    if not rows:
        print("No loss rows to plot; skipping.")
        return

    steps = [int(r["global_step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]

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
    plt.title("Task1 Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (task_dir / "checkpoints" / "cifar_net.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = (
        Path(args.best_ckpt).expanduser()
        if args.best_ckpt
        else ckpt_path.with_name(f"{ckpt_path.stem}_best{ckpt_path.suffix}")
    )
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    loss_csv_path = Path(args.loss_csv).expanduser() if args.loss_csv else None
    loss_plot_path = (
        (Path(args.loss_plot).expanduser() if args.loss_plot else (task_dir / "outputs" / "loss_curve.png"))
        if args.save_loss_plot
        else None
    )

    device = get_device(args.device)
    print(f"Using device: {device}")

    train_transform, test_transform = build_transforms(args.augment)
    train_dataset_aug = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
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

    trainset: torch.utils.data.Dataset[Any] = torch.utils.data.Subset(train_dataset_aug, train_indices)
    valset: torch.utils.data.Dataset[Any] | None = (
        torch.utils.data.Subset(train_dataset_plain, val_indices) if val_size > 0 else None
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )
    valloader = (
        torch.utils.data.DataLoader(
            valset,
            batch_size=max(64, args.batch_size),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if valset is not None
        else None
    )

    net = create_model(args.model).to(device)
    param_count = sum(p.numel() for p in net.parameters())
    print(f"Model: {args.model} | params: {param_count:,}")
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
        scheduler: Any = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    record_loss = (loss_csv_path is not None) or (loss_plot_path is not None)
    loss_rows: list[dict[str, Any]] = [] if record_loss else []
    global_step = 0
    best_val_loss = float("inf")

    @torch.no_grad()
    def eval_loader(loader: torch.utils.data.DataLoader) -> float:
        net.eval()
        total = 0
        loss_sum = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss_sum += float(loss.item()) * int(labels.size(0))
            total += int(labels.size(0))
        avg_loss = 0.0 if total == 0 else (loss_sum / total)
        return avg_loss

    for epoch in range(args.epochs):
        running_loss = 0.0
        epoch_samples = 0
        epoch_loss_sum = 0.0
        net.train()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            bsz = int(labels.size(0))
            epoch_samples += bsz
            epoch_loss_sum += float(loss.item()) * bsz
            if record_loss:
                loss_rows.append(
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "batch_idx": i,
                        "loss": float(loss.item()),
                    }
                )
            global_step += 1

            if args.log_every > 0 and i % args.log_every == (args.log_every - 1):
                denom = float(args.log_every)
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / denom:.3f}")
                running_loss = 0.0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_time_sec = time.perf_counter() - t0
        train_img_per_sec = 0.0 if train_time_sec <= 0 else (epoch_samples / train_time_sec)

        if scheduler is not None:
            scheduler.step()

        train_loss = 0.0 if epoch_samples == 0 else (epoch_loss_sum / epoch_samples)
        msg = (
            f"Epoch {epoch + 1}/{args.epochs} | train loss: {train_loss:.4f}"
            f" | train time: {train_time_sec:.2f}s | train img/s: {train_img_per_sec:.1f}"
        )
        if valloader is not None:
            val_loss = eval_loader(valloader)
            msg += f" | val loss: {val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": args.model, "state_dict": net.state_dict(), "epoch": epoch + 1, "val_loss": val_loss},
                    best_ckpt_path,
                )
        print(msg)

    print("Finished Training")
    torch.save({"model": args.model, "state_dict": net.state_dict(), "epoch": args.epochs}, ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")
    if best_ckpt_path.exists():
        print(f"Saved best checkpoint to: {best_ckpt_path}")
    if record_loss:
        if loss_csv_path is not None:
            write_loss_csv(loss_rows, loss_csv_path)
            print(f"Saved loss CSV to: {loss_csv_path}")
        if loss_plot_path is not None:
            maybe_save_loss_plot(loss_rows, loss_plot_path)
            if loss_plot_path.exists():
                print(f"Saved loss curve to: {loss_plot_path}")


if __name__ == "__main__":
    main()
