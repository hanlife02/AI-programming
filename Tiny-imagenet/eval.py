from __future__ import annotations

import argparse
import csv
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
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--val-dir", type=str, default="input/tiny-imagenet/tiny-imagenet-200/val")
    p.add_argument("--test-dir", type=str, default="input/tiny-imagenet/tiny-imagenet-200/test")
    p.add_argument("--train-dir", type=str, default="input/tiny-imagenet/tiny-imagenet-200/train")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--num-classes", type=int, default=200)
    p.add_argument("--model", type=str, default="VGG16", choices=sorted(_VGG_CFG.keys()))
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-preds", type=str, default="", help="save test predictions to CSV when test has no labels")
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


def _list_image_files(root: Path) -> list[Path]:
    exts = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    files: list[Path] = []
    if root.is_dir():
        for name in sorted(root.iterdir()):
            if name.is_file() and name.suffix in exts:
                files.append(name)
    return files


class ImageListDataset(torch.utils.data.Dataset):

    def __init__(self, files: list[Path], transform) -> None:
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        try:
            from PIL import Image
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError("Evaluation requires Pillow. Install it with: pip install pillow") from e
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path.name


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


@torch.no_grad()
def predict(
    net: Module,
    device: torch.device,
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    out_path: Path,
) -> None:
    net.eval()
    rows: list[tuple[str, str]] = []
    for batch_idx, (inputs, names) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True).contiguous()
        outputs = net(Tensor(inputs, requires_grad=False)).data
        _, predicted = outputs.max(1)
        preds = predicted.detach().cpu().tolist()
        for filename, idx in zip(list(names), preds):
            rows.append((filename, class_names[int(idx)]))
        progress_bar(batch_idx, len(loader), "Pred: %d" % (len(rows)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "pred"])
        writer.writerows(rows)
    print(f"saved predictions to: {out_path}", flush=True)


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

    split = str(args.split)
    transform = build_val_transform()
    val_dir = Path(args.val_dir).expanduser()
    test_dir = Path(args.test_dir).expanduser()
    train_dir = Path(args.train_dir).expanduser()

    requested_num_classes = int(args.num_classes)
    dataset = None
    loader = None
    class_names: list[str] | None = None
    unlabeled = False

    if split == "val":
        if not val_dir.exists():
            raise FileNotFoundError(f"val dir not found: {val_dir}")
        _prepare_val_imagefolder(val_dir)
        dataset = torchvision.datasets.ImageFolder(str(val_dir), transform=transform)
        if len(dataset) == 0:
            raise RuntimeError("val dataset is empty")
        class_names = dataset.classes
    else:
        if not test_dir.exists():
            raise FileNotFoundError(f"test dir not found: {test_dir}")
        if _val_is_imagefolder(test_dir):
            dataset = torchvision.datasets.ImageFolder(str(test_dir), transform=transform)
            if len(dataset) == 0:
                raise RuntimeError("test dataset is empty")
            class_names = dataset.classes
        else:
            images_root = test_dir / "images" if (test_dir / "images").exists() else test_dir
            files = _list_image_files(images_root)
            if not files:
                raise RuntimeError(f"no test images found under: {images_root}")
            if not train_dir.exists():
                raise FileNotFoundError(f"train dir not found for class names: {train_dir}")
            class_names = torchvision.datasets.ImageFolder(str(train_dir)).classes
            dataset = ImageListDataset(files, transform)
            unlabeled = True

    if class_names is None or dataset is None:
        raise RuntimeError("failed to initialize dataset")

    num_classes = requested_num_classes if requested_num_classes > 0 else len(class_names)
    if len(class_names) != num_classes:
        raise ValueError(f"num_classes mismatch: dataset has {len(class_names)} classes, num_classes={num_classes}")

    loader = torch.utils.data.DataLoader(
        dataset,
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

    if split == "val":
        print("==> Evaluating (val)..", flush=True)
        acc = evaluate(net, device, loader)
        print(f"val acc: {acc:.3f}%", flush=True)
        return

    if not unlabeled:
        print("==> Evaluating (test)..", flush=True)
        acc = evaluate(net, device, loader)
        print(f"test acc: {acc:.3f}%", flush=True)
        return

    out_path = Path(args.save_preds).expanduser() if args.save_preds else (script_dir / "outputs" / "test_preds.csv")
    print("==> Predicting (test, unlabeled)..", flush=True)
    predict(net, device, loader, class_names, out_path)


if __name__ == "__main__":
    main()
