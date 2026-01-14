"""Train CIFAR-10 with the custom CUDA framework (Task3).

This file follows the same training loop style as the provided PyTorch reference:
- progress bar per batch
- optional resume from checkpoint
- cosine LR scheduler (optional)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

try:
    from Task3.minifw.nn import DLALikeCifarNet, SimpleCifarNet, VGG
    from Task3.minifw.optim import SGD
    from Task3.minifw.tensor import Tensor
    from Task3.utils import progress_bar
except ModuleNotFoundError:  # supports: cd Task3 && python train.py
    from minifw.nn import DLALikeCifarNet, SimpleCifarNet, VGG
    from minifw.optim import SGD
    from minifw.tensor import Tensor
    from utils import progress_bar


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task3 CIFAR-10 Training (custom CUDA framework)")
    p.add_argument("--model", type=str, default="vgg16", choices=["vgg16", "dla", "simple"])
    p.add_argument("--lr", type=float, default=0.1, help="learning rate")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
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
    p.add_argument("--autoaugment", action=argparse.BooleanOptionalAction, default=False, help="use torchvision AutoAugment(CIFAR10)")
    p.add_argument("--random-erasing", type=float, default=0.0, help="RandomErasing probability (e.g. 0.25)")
    p.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--t-max", type=int, default=200, help="T_max for cosine scheduler")
    p.add_argument("--warmup-epochs", type=int, default=5, help="linear warmup epochs (0 to disable)")
    p.add_argument("--min-lr", type=float, default=0.0, help="minimum lr for cosine schedule")
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=False, help="enable EMA of model parameters")
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay (if --ema)")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    return p.parse_args()


def build_transforms(augment: bool, autoaugment: bool, random_erasing_p: float) -> tuple[object, object]:
    try:
        import torchvision.transforms as transforms
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Task3 training requires torchvision. Install it with: pip install torchvision") from e

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_ops: list[transforms.Transform] = []
    if augment:
        train_ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    if autoaugment:
        if not hasattr(transforms, "AutoAugment"):
            raise RuntimeError("torchvision.transforms.AutoAugment is unavailable in this torchvision version")
        train_ops.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if random_erasing_p > 0:
        train_ops.append(transforms.RandomErasing(p=float(random_erasing_p)))
    test_ops = [transforms.ToTensor(), transforms.Normalize(mean, std)]
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
        raise ModuleNotFoundError("Task3 training requires torchvision. Install it with: pip install torchvision") from e

    if not torch.cuda.is_available():
        raise RuntimeError("Task3 requires CUDA (custom CUDA kernels).")
    device = torch.device("cuda")

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (task_dir / "checkpoint" / "ckpt.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("==> Preparing data..", flush=True)
    transform_train, transform_test = build_transforms(bool(args.augment), bool(args.autoaugment), float(args.random_erasing))
    trainset = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    testset = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    print("==> Building model..", flush=True)
    if args.model == "dla":
        net = DLALikeCifarNet(device=device)
    elif args.model == "simple":
        net = SimpleCifarNet(device=device)
    else:
        net = VGG("VGG16", device=device)
    best_acc = 0.0
    start_epoch = 0
    ema: dict[str, torch.Tensor] | None = None
    ema_backup: dict[str, torch.Tensor] | None = None

    if args.resume:
        print("==> Resuming from checkpoint..", flush=True)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        arch = str(ckpt.get("arch", args.model))
        if arch != args.model:
            raise ValueError(f"Checkpoint arch mismatch: ckpt={arch!r} args.model={args.model!r}")
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

    for epoch in range(start_epoch, int(args.epochs)):
        print("\nEpoch: %d" % epoch, flush=True)

        optimizer.lr = scheduled_lr(args, epoch)

        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device, non_blocking=True).contiguous()
            targets = targets.to(device, non_blocking=True)

            logits = net(Tensor(inputs, requires_grad=False))
            loss = logits.cross_entropy(targets)
            loss.backward()
            optimizer.step()
            ema_update()
            optimizer.zero_grad()

            train_loss += float(loss.data.item())
            _, predicted = logits.data.max(1)
            total += int(targets.size(0))
            correct += int(predicted.eq(targets).sum().item())
            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.4g" % (train_loss / (batch_idx + 1), 100.0 * correct / max(1, total), correct, total, optimizer.lr),
            )

        print("\n==> Testing..", flush=True)
        if ema is not None:
            ema_apply()
        acc = test(net, device, testloader)
        if ema is not None:
            ema_restore()

        if acc > best_acc:
            print("Saving..", flush=True)
            state = {
                "arch": str(args.model),
                "net": {k: v.data for k, v in net.named_parameters().items()},
                "ema": ema,
                "acc": acc,
                "epoch": epoch,
            }
            torch.save(state, ckpt_path)
            best_acc = acc

    print("best acc: %.3f%%" % best_acc, flush=True)


if __name__ == "__main__":
    main()
