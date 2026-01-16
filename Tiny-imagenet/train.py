from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import torch

try:
    from Task3.minifw.nn import (
        Module,
        BatchNorm2d,
        Conv2d,
        GlobalAvgPool2d,
        Linear,
        MaxPool2d,
        ReLU,
        Sequential,
    )
    from Task3.minifw.optim import SGD
    from Task3.minifw.tensor import Tensor
    from Task3.utils import progress_bar
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from Task3.minifw.nn import (
        Module,
        BatchNorm2d,
        Conv2d,
        GlobalAvgPool2d,
        Linear,
        MaxPool2d,
        ReLU,
        Sequential,
    )
    from Task3.minifw.optim import SGD
    from Task3.minifw.tensor import Tensor
    from Task3.utils import progress_bar


_VGG_CFG = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class TinyImageNetNet(Module):

    def __init__(self, vgg_name: str = "VGG16", num_classes: int = 200, device: torch.device | None = None) -> None:
        device = device or torch.device("cuda")
        if vgg_name not in _VGG_CFG:
            raise ValueError(f"Unknown VGG config: {vgg_name}")
        self.features = self._make_layers(_VGG_CFG[vgg_name], device=device)
        self.pool = GlobalAvgPool2d()
        self.classifier = Linear(512, int(num_classes), device=device)

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.pool(out)
        n = int(out.data.size(0))
        out = out.reshape(n, -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg: list[object], device: torch.device) -> Sequential:
        layers: list[Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(MaxPool2d(kernel=2, stride=2))
            else:
                out_channels = int(v)
                layers.append(Conv2d(in_channels, out_channels, kernel=3, padding=1, device=device))
                layers.append(BatchNorm2d(out_channels, device=device))
                layers.append(ReLU())
                in_channels = out_channels
        return Sequential(*layers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny-ImageNet training with Task3 CUDA ops")
    p.add_argument("--train-dir", type=str, default="../input/tiny-imagenet/tiny-imagenet-200/train")
    p.add_argument("--val-dir", type=str, default="../input/tiny-imagenet/tiny-imagenet-200/val")
    p.add_argument("--dataset-id", type=str, default="akash2sharma/tiny-imagenet")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True, help="auto download if missing or incomplete")
    p.add_argument("--check-integrity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--model", type=str, default="VGG16", choices=sorted(_VGG_CFG.keys()))
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--t-max", type=int, default=200, help="T_max for cosine scheduler")
    p.add_argument("--warmup-epochs", type=int, default=5, help="linear warmup epochs (0 to disable)")
    p.add_argument("--min-lr", type=float, default=0.0, help="minimum lr for cosine schedule")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    return p.parse_args()


def build_transforms(augment: bool) -> tuple[object, object]:
    try:
        import torchvision.transforms as transforms
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Training requires torchvision. Install it with: pip install torchvision") from e

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_ops: list[transforms.Transform] = []
    if augment:
        train_ops.extend([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip()])
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
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
def evaluate(net: Module, device: torch.device, loader: torch.utils.data.DataLoader) -> float:
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


def _find_dataset_root(base: Path) -> Path:
    candidates = [base]
    direct = base / "tiny-imagenet-200"
    if direct.exists():
        candidates.append(direct)
    if base.exists():
        for item in base.iterdir():
            if item.is_dir():
                candidates.append(item)
    for cand in candidates:
        if (cand / "train").is_dir() and (cand / "val").is_dir():
            return cand
    for train_dir in base.rglob("train"):
        if train_dir.is_dir() and (train_dir.parent / "val").is_dir():
            return train_dir.parent
    raise FileNotFoundError(f"Cannot locate tiny-imagenet-200 under: {base}")


def _download_tiny_imagenet(dataset_id: str) -> Path:
    try:
        import kagglehub
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Auto download requires kagglehub. Install it with: pip install kagglehub") from e
    print(f"==> Downloading Tiny-ImageNet via kagglehub ({dataset_id}) ..", flush=True)
    dataset_path = Path(kagglehub.dataset_download(dataset_id))
    return _find_dataset_root(dataset_path)


def _val_is_imagefolder(val_dir: Path) -> bool:
    if not val_dir.exists():
        return False
    for item in val_dir.iterdir():
        if item.is_dir() and item.name != "images":
            if any(item.iterdir()):
                return True
    return False


def _prepare_val_imagefolder(val_dir: Path) -> None:
    if _val_is_imagefolder(val_dir):
        return
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"
    if not images_dir.exists() or not ann_path.exists():
        raise FileNotFoundError(f"val folder is not ImageFolder and missing {images_dir} or {ann_path}")

    mapping: dict[str, str] = {}
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]

    moved = 0
    for filename, cls in mapping.items():
        src = images_dir / filename
        if not src.exists():
            continue
        dst_dir = val_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_dir / filename))
        moved += 1

    if moved == 0:
        raise RuntimeError("val preprocessing failed; no images moved")

    remaining = list(images_dir.iterdir()) if images_dir.exists() else []
    if not remaining:
        images_dir.rmdir()


def _count_images(root: Path) -> int:
    exts = {".jpeg", ".jpg", ".png"}
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in exts:
                total += 1
    return total


def _count_class_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir())


def _dataset_is_complete(train_dir: Path, val_dir: Path, num_classes: int) -> bool:
    expected_classes = int(num_classes)
    expected_train = expected_classes * 500
    expected_val = expected_classes * 50
    train_classes = _count_class_dirs(train_dir)
    val_classes = _count_class_dirs(val_dir)
    if train_classes != expected_classes or val_classes != expected_classes:
        return False
    return _count_images(train_dir) == expected_train and _count_images(val_dir) == expected_val


def _ensure_dataset(
    train_dir: Path,
    val_dir: Path,
    dataset_id: str,
    download: bool,
    check_integrity: bool,
    num_classes: int,
) -> tuple[Path, Path]:
    if train_dir.exists() and val_dir.exists():
        _prepare_val_imagefolder(val_dir)
        if not check_integrity or _dataset_is_complete(train_dir, val_dir, num_classes):
            return train_dir, val_dir
        if not download:
            raise RuntimeError("dataset incomplete and auto-download disabled")
        print("==> Dataset incomplete, re-downloading..", flush=True)
    elif not download:
        raise FileNotFoundError(f"train/val dir not found: {train_dir} {val_dir}")

    dataset_root = _download_tiny_imagenet(dataset_id)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    _prepare_val_imagefolder(val_dir)
    if check_integrity and not _dataset_is_complete(train_dir, val_dir, num_classes):
        raise RuntimeError("downloaded dataset is incomplete; please retry download")
    return train_dir, val_dir


def main() -> None:
    args = parse_args()
    try:
        import torchvision
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Training requires torchvision. Install it with: pip install torchvision") from e

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA (Task3 custom CUDA kernels).")
    device = torch.device("cuda")

    script_dir = Path(__file__).resolve().parent
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (script_dir / "checkpoint" / "ckpt.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.train_dir).expanduser()
    val_dir = Path(args.val_dir).expanduser()

    expected_classes = int(args.num_classes) if int(args.num_classes) > 0 else 200
    train_dir, val_dir = _ensure_dataset(
        train_dir,
        val_dir,
        str(args.dataset_id),
        bool(args.download),
        bool(args.check_integrity),
        expected_classes,
    )

    print("==> Preparing data..", flush=True)
    transform_train, transform_val = build_transforms(bool(args.augment))
    dataset = torchvision.datasets.ImageFolder(str(train_dir), transform=transform_train)
    trainset = dataset
    valset = torchvision.datasets.ImageFolder(str(val_dir), transform=transform_val)
    if len(trainset) == 0:
        raise RuntimeError("train dataset is empty")
    if len(valset) == 0:
        raise RuntimeError("val dataset is empty")
    if len(valset.classes) != len(trainset.classes):
        print(
            f"warning: class count mismatch train={len(trainset.classes)} val={len(valset.classes)}",
            flush=True,
        )

    num_classes = int(args.num_classes)
    if num_classes <= 0:
        num_classes = len(trainset.classes)
    elif num_classes != len(trainset.classes):
        print(
            f"warning: --num-classes={num_classes} but train has {len(trainset.classes)} classes",
            flush=True,
        )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
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
    net = TinyImageNetNet(vgg_name=str(args.model), num_classes=num_classes, device=device)
    best_acc = 0.0
    start_epoch = 0

    if args.resume:
        print("==> Resuming from checkpoint..", flush=True)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("net", {})
        for name, p in net.named_parameters().items():
            if name not in state:
                raise KeyError(f"Missing parameter in checkpoint: {name}")
            p.data.copy_(state[name].to(device))
        best_acc = float(ckpt.get("acc", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    optimizer = SGD(list(net.parameters()), lr=float(args.lr), momentum=float(args.momentum), weight_decay=float(args.weight_decay))
    log_every = max(1, int(args.log_every))

    for epoch in range(start_epoch, int(args.epochs)):
        print("\nEpoch: %d" % epoch, flush=True)
        optimizer.lr = scheduled_lr(args, epoch)

        net.train()
        train_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        correct_sum = torch.zeros((), device=device, dtype=torch.int64)
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device, non_blocking=True).contiguous()
            targets = targets.to(device, non_blocking=True)

            logits = net(Tensor(inputs, requires_grad=False))
            loss = logits.cross_entropy(targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_sum.add_(loss.data.detach())
            _, predicted = logits.data.max(1)
            total += int(targets.size(0))
            correct_sum.add_(predicted.eq(targets).sum())

            if (batch_idx % log_every == 0) or (batch_idx == len(trainloader) - 1):
                avg_loss = float((train_loss_sum / float(batch_idx + 1)).item())
                acc = float((correct_sum.float() * (100.0 / float(max(1, total)))).item())
                progress_bar(
                    batch_idx,
                    len(trainloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.4g"
                    % (avg_loss, acc, int(correct_sum.item()), total, optimizer.lr),
                )

        print("\n==> Validating..", flush=True)
        acc = evaluate(net, device, valloader)

        if acc > best_acc:
            print("Saving..", flush=True)
            state = {
                "arch": str(args.model),
                "net": {k: v.data for k, v in net.named_parameters().items()},
                "acc": acc,
                "epoch": epoch,
            }
            torch.save(state, ckpt_path)
            best_acc = acc

    print("best acc: %.3f%%" % best_acc, flush=True)


if __name__ == "__main__":
    main()
