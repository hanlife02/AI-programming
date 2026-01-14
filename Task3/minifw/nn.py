from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch

from .tensor import Tensor


class Module:
    def parameters(self) -> Iterator[Tensor]:
        for _, v in self.named_parameters().items():
            yield v

    def named_parameters(self) -> Dict[str, Tensor]:
        params: Dict[str, Tensor] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                params[name] = value
            elif isinstance(value, Module):
                for child_name, child_param in value.named_parameters().items():
                    params[f"{name}.{child_name}"] = child_param
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        for child_name, child_param in item.named_parameters().items():
                            params[f"{name}.{i}.{child_name}"] = child_param
        return params

    def train(self) -> None:
        return

    def eval(self) -> None:
        return

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def kaiming_uniform_(w: torch.Tensor, fan_in: int) -> torch.Tensor:
    bound = (6.0 / fan_in) ** 0.5
    return w.uniform_(-bound, bound)


@dataclass
class Conv2d(Module):
    w: Tensor
    b: Optional[Tensor]
    stride: int
    padding: int

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0, bias: bool = True, device: torch.device | None = None):
        device = device or torch.device("cuda")
        w = torch.empty((out_ch, in_ch, kernel, kernel), device=device, dtype=torch.float32)
        kaiming_uniform_(w, fan_in=in_ch * kernel * kernel)
        self.w = Tensor(w, requires_grad=True)
        self.b = Tensor(torch.zeros((out_ch,), device=device, dtype=torch.float32), requires_grad=True) if bias else None
        self.stride = int(stride)
        self.padding = int(padding)

    def forward(self, x: Tensor) -> Tensor:
        return x.conv2d(self.w, self.b, stride=self.stride, padding=self.padding)


@dataclass
class Linear(Module):
    w: Tensor
    b: Optional[Tensor]

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: torch.device | None = None):
        device = device or torch.device("cuda")
        w = torch.empty((out_features, in_features), device=device, dtype=torch.float32)
        kaiming_uniform_(w, fan_in=in_features)
        self.w = Tensor(w, requires_grad=True)
        self.b = Tensor(torch.zeros((out_features,), device=device, dtype=torch.float32), requires_grad=True) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return x.linear(self.w, self.b)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


@dataclass
class MaxPool2d(Module):
    kernel: int = 2
    stride: int = 2

    def forward(self, x: Tensor) -> Tensor:
        return x.maxpool2d(kernel=self.kernel, stride=self.stride)


class GlobalAvgPool2d(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.global_avg_pool2d()


class SimpleCifarNet(Module):
    def __init__(self, device: torch.device | None = None) -> None:
        device = device or torch.device("cuda")
        # Match Task1/Task2 `cnn`:
        # Conv(5x5)->ReLU->Pool -> Conv(5x5)->ReLU->Pool -> Flatten -> FC -> FC -> FC
        self.conv1 = Conv2d(3, 6, kernel=5, device=device)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, kernel=5, device=device)
        self.relu = ReLU()
        self.fc1 = Linear(16 * 5 * 5, 120, device=device)
        self.fc2 = Linear(120, 84, device=device)
        self.fc3 = Linear(84, 10, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(int(x.data.size(0)), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
