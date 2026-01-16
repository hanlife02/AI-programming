from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Task1 loss curve from CSV.")
    parser.add_argument("--csv", type=str, default="", help="Loss CSV path (default: Task1/outputs/loss.csv).")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output image path (default: Task1/outputs/loss_curve.png).",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.05, help="EMA smoothing alpha.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_dir = Path(__file__).resolve().parent
    csv_path = Path(args.csv).expanduser() if args.csv else (task_dir / "outputs" / "loss.csv")
    out_path = Path(args.out).expanduser() if args.out else (task_dir / "outputs" / "loss_curve.png")

    import matplotlib.pyplot as plt 

    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")

    steps = [int(r["global_step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]

    ema_losses: list[float] = []
    alpha = float(args.ema_alpha)
    ema = losses[0]
    for loss in losses:
        ema = alpha * loss + (1 - alpha) * ema
        ema_losses.append(ema)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, losses, linewidth=0.8, alpha=0.25, label="loss (raw)")
    plt.plot(steps, ema_losses, linewidth=1.8, label=f"loss (EMA, alpha={alpha})")
    plt.xlabel("global step")
    plt.ylabel("loss")
    plt.title("Task1 Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to: {out_path}")


if __name__ == "__main__":
    main()

