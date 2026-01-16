from __future__ import annotations

from typing import List

import torch

try:
    from Task3.minifw.nn import (
        BatchNorm2d,
        Conv2d,
        GlobalAvgPool2d,
        Linear,
        MaxPool2d,
        Module,
        ReLU,
        Sequential,
    )
    from Task3.minifw.tensor import Tensor
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Task3.minifw.nn import (
        BatchNorm2d,
        Conv2d,
        GlobalAvgPool2d,
        Linear,
        MaxPool2d,
        Module,
        ReLU,
        Sequential,
    )
    from Task3.minifw.tensor import Tensor


_VGG_CFG = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class TinyVGG(Module):
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

    def _make_layers(self, cfg: List[object], device: torch.device) -> Sequential:
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
