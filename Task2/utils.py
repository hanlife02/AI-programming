from __future__ import annotations

import random

import torch
import torchvision.transforms as transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    train_transforms: list[transforms.Transform] = []
    if augment:
        train_transforms.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_transforms.extend([transforms.ToTensor(), normalize])

    test_transforms: list[transforms.Transform] = [transforms.ToTensor(), normalize]
    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)

