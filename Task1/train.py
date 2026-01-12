from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
    from Task1.models.cnn import Cifar10CNN
except ModuleNotFoundError:
    from models.cnn import Cifar10CNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 (Task 1).")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument(
        "--loss-csv",
        type=str,
        default="",
        help="Path to save per-step loss as CSV (default: Task1/outputs/loss.csv).",
    )
    parser.add_argument(
        "--loss-plot",
        type=str,
        default="",
        help="Path to save loss curve image (default: Task1/outputs/loss_curve.png).",
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

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (task_dir / "checkpoints" / "cifar_net.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    outputs_dir = task_dir / "outputs"
    loss_csv_path = Path(args.loss_csv).expanduser() if args.loss_csv else (outputs_dir / "loss.csv")
    loss_plot_path = Path(args.loss_plot).expanduser() if args.loss_plot else (outputs_dir / "loss_curve.png")

    device = get_device(args.device)
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    net = Cifar10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    loss_rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_rows.append(
                {
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "batch_idx": i,
                    "loss": float(loss.item()),
                }
            )
            global_step += 1
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")
    torch.save(net.state_dict(), ckpt_path)
    print(f"Saved checkpoint to: {ckpt_path}")
    write_loss_csv(loss_rows, loss_csv_path)
    print(f"Saved loss CSV to: {loss_csv_path}")
    maybe_save_loss_plot(loss_rows, loss_plot_path)
    if loss_plot_path.exists():
        print(f"Saved loss curve to: {loss_plot_path}")


if __name__ == "__main__":
    main()
