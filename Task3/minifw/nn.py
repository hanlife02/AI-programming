from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

import torch

from .tensor import Tensor


class Module:
    training: bool = True

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
        self.training = True
        for child in _iter_modules(self):
            child.train()

    def eval(self) -> None:
        self.training = False
        for child in _iter_modules(self):
            child.eval()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def _iter_modules(m: Module) -> Iterator[Module]:
    for _, value in m.__dict__.items():
        if isinstance(value, Module):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Module):
                    yield item


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
class BatchNorm2d(Module):
    weight: Tensor
    bias: Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor
    momentum: float
    eps: float

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5, device: torch.device | None = None):
        device = device or torch.device("cuda")
        self.weight = Tensor(torch.ones((num_features,), device=device, dtype=torch.float32), requires_grad=True)
        self.bias = Tensor(torch.zeros((num_features,), device=device, dtype=torch.float32), requires_grad=True)
        self.running_mean = torch.zeros((num_features,), device=device, dtype=torch.float32)
        self.running_var = torch.ones((num_features,), device=device, dtype=torch.float32)
        self.momentum = float(momentum)
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        return x.batchnorm2d(
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            training=bool(self.training),
            momentum=float(self.momentum),
            eps=float(self.eps),
        )


@dataclass
class MaxPool2d(Module):
    kernel: int = 2
    stride: int = 2

    def forward(self, x: Tensor) -> Tensor:
        return x.maxpool2d(kernel=self.kernel, stride=self.stride)


class GlobalAvgPool2d(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.global_avg_pool2d()


@dataclass
class AvgPool2d(Module):
    kernel: int = 1
    stride: int = 1

    def forward(self, x: Tensor) -> Tensor:
        if int(self.kernel) == 1 and int(self.stride) == 1:
            return x
        raise NotImplementedError("AvgPool2d is only implemented for kernel=1,stride=1 (identity) in Task3.")


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


_VGG_CFG = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGGNet(Module):
    def __init__(self, cfg_name: str = "VGG16", num_classes: int = 10, device: torch.device | None = None) -> None:
        raise RuntimeError(
            "VGGNet has been replaced by MyNet (VGG-style Conv+BN+ReLU blocks + Flatten head). "
            "Use MyNet(vgg_name='VGG16') instead."
        )


class MyNet(Module):

    def __init__(self, vgg_name: str = "VGG16", num_classes: int = 10, device: torch.device | None = None) -> None:
        device = device or torch.device("cuda")
        if vgg_name not in _VGG_CFG:
            raise ValueError(f"Unknown VGG config: {vgg_name}")
        self.features = self._make_layers(_VGG_CFG[vgg_name], device=device)
        self.classifier = Linear(512, int(num_classes), device=device)

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
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
        layers.append(AvgPool2d(kernel=1, stride=1))
        return Sequential(*layers)
