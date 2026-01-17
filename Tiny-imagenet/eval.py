from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser(description="Tiny-ImageNet evaluation with Task3 CUDA ops")
    p.add_argument("--val-dir", type=str, default="input/tiny-imagenet/tiny-imagenet-200/val")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--model", type=str, default="VGG16", choices=sorted(_VGG_CFG.keys()))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def build_val_transform() -> object:
    try:
        import torchvision.transforms as transforms
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Evaluation requires torchvision. Install it with: pip install torchvision") from e

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


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
        src.rename(dst_dir / filename)
        moved += 1

    if moved == 0:
        raise RuntimeError("val preprocessing failed; no images moved")

    remaining = list(images_dir.iterdir()) if images_dir.exists() else []
    if not remaining:
        images_dir.rmdir()


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


def main() -> None:
    args = parse_args()
    try:
        import torchvision
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError("Evaluation requires torchvision. Install it with: pip install torchvision") from e

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA (Task3 custom CUDA kernels).")
    device = torch.device("cuda")

    script_dir = Path(__file__).resolve().parent
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (script_dir / "checkpoint" / "ckpt.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    val_dir = Path(args.val_dir).expanduser()
    if not val_dir.exists():
        raise FileNotFoundError(f"val dir not found: {val_dir}")
    _prepare_val_imagefolder(val_dir)

    transform = build_val_transform()
    valset = torchvision.datasets.ImageFolder(str(val_dir), transform=transform)
    if len(valset) == 0:
        raise RuntimeError("val dataset is empty")

    num_classes = int(args.num_classes)
    if num_classes <= 0:
        num_classes = len(valset.classes)
    elif num_classes != len(valset.classes):
        print(f"warning: --num-classes={num_classes} but val has {len(valset.classes)} classes", flush=True)

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    net = TinyImageNetNet(vgg_name=str(args.model), num_classes=num_classes, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = str(ckpt.get("arch", args.model))
    if arch not in {str(args.model), "vgg16"}:
        raise ValueError(f"Checkpoint arch mismatch: ckpt={arch!r} args.model={args.model!r}")
    state = ckpt.get("net", {})
    for name, p in net.named_parameters().items():
        if name not in state:
            raise KeyError(f"Missing parameter in checkpoint: {name}")
        p.data.copy_(state[name].to(device))

    print("==> Evaluating..", flush=True)
    acc = evaluate(net, device, valloader)
    print(f"val acc: {acc:.3f}%", flush=True)


if __name__ == "__main__":
    main()
