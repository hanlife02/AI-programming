from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

try:
    from Task1.models.cnn import Cifar10CNN
except ModuleNotFoundError:
    from models.cnn import Cifar10CNN


CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CNN on CIFAR-10 (Task 1).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    task_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else (task_dir / "data")
    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else (task_dir / "checkpoints" / "cifar_net.pth")

    device = get_device(args.device)
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    net = Cifar10CNN().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in CLASSES}
    total_pred = {classname: 0 for classname in CLASSES}

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                label_idx = int(label.item())
                pred_idx = int(prediction.item())
                total_pred[CLASSES[label_idx]] += 1
                if label_idx == pred_idx:
                    correct_pred[CLASSES[label_idx]] += 1

    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")
    for classname in CLASSES:
        denom = total_pred[classname]
        acc = 0.0 if denom == 0 else (100.0 * correct_pred[classname] / denom)
        print(f"Accuracy for class: {classname:5s} is {acc:.1f}%")


if __name__ == "__main__":
    main()
