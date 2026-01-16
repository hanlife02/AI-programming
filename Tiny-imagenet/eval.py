from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from Task3.minifw.tensor import Tensor
    from Task3.utils import progress_bar
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Task3.minifw.tensor import Tensor
    from Task3.utils import progress_bar

from model import TinyVGG
from tiny_imagenet import TinyImageNet, ensure_tiny_imagenet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny ImageNet evaluation (single GPU).")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True, help="use EMA weights if present in checkpoint")
    p.add_argument("--image-size", type=int, default=0, help="override image size for eval (default: from ckpt or 64)")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=False, help="download Tiny ImageNet if missing")
    return p.parse_args()


def build_test_transform(image_size: int):
    try:
        import torchvision.transforms as transforms
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Tiny ImageNet evaluation requires torchvision. Install it with: pip install torchvision") from e

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = max(int(image_size), int(round(float(image_size) / 0.875)))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    try:
        import torchvision  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Tiny ImageNet evaluation requires torchvision. Install it with: pip install torchvision") from e

    if not torch.cuda.is_available():
        raise RuntimeError("Tiny ImageNet evaluation requires CUDA (custom CUDA kernels).")
    device = torch.device("cuda")

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (task_dir / "checkpoint" / "ckpt.pth")
    ckpt = torch.load(ckpt_path, map_location=device)

    arch = str(ckpt.get("arch", "VGG16"))
    num_classes = int(ckpt.get("num_classes", 200))
    class_to_idx = ckpt.get("class_to_idx", None)
    image_size = int(args.image_size) if int(args.image_size) > 0 else int(ckpt.get("image_size", 64))

    model = TinyVGG(vgg_name=arch, num_classes=num_classes, device=device)
    state = ckpt.get("net", {})
    if bool(args.use_ema):
        ema_state = ckpt.get("ema", None)
        if isinstance(ema_state, dict) and len(ema_state) > 0:
            state = ema_state
    for name, p in model.named_parameters().items():
        if name not in state:
            raise KeyError(f"Missing parameter in checkpoint: {name}")
        p.data.copy_(state[name].to(device))

    data_root = ensure_tiny_imagenet(data_dir, download=bool(args.download))
    ds = TinyImageNet(root=data_root, split="val", transform=build_test_transform(image_size), class_to_idx=class_to_idx)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    model.eval()
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True).contiguous()
        labels = labels.to(device, non_blocking=True)
        logits = model(Tensor(images, requires_grad=False)).data
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(images.size(0))
        progress_bar(batch_idx, len(loader), "Acc: %.3f%% (%d/%d)" % (100.0 * correct / max(1, total), correct, total))

    print(f"ckpt: {ckpt_path} | val acc: {(correct / max(1, total)):.4f}", flush=True)


if __name__ == "__main__":
    main()
