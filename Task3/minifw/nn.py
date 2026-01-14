from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

import torch

from . import ops
from .tensor import Tensor, _Node


class Module:
    def train(self) -> None:
        self._training = True
        for _, child in self._named_children().items():
            child.train()

    def eval(self) -> None:
        self._training = False
        for _, child in self._named_children().items():
            child.eval()

    def _named_children(self) -> Dict[str, "Module"]:
        children: Dict[str, Module] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                children[name] = value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        children[f"{name}.{i}"] = item
        return children

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

    def named_buffers(self) -> Dict[str, torch.Tensor]:
        bufs: Dict[str, torch.Tensor] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor) and (not value.requires_grad):
                bufs[name] = value
            elif isinstance(value, Module):
                for child_name, child_buf in value.named_buffers().items():
                    bufs[f"{name}.{child_name}"] = child_buf
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        for child_name, child_buf in item.named_buffers().items():
                            bufs[f"{name}.{i}.{child_name}"] = child_buf
        return bufs

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


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, device: torch.device | None = None) -> None:
        device = device or torch.device("cuda")
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = Tensor(torch.ones((self.num_features,), device=device, dtype=torch.float32), requires_grad=True)
        self.bias = Tensor(torch.zeros((self.num_features,), device=device, dtype=torch.float32), requires_grad=True)
        self.running_mean = torch.zeros((self.num_features,), device=device, dtype=torch.float32)
        self.running_var = torch.ones((self.num_features,), device=device, dtype=torch.float32)
        self._training = True

    def forward(self, x: Tensor) -> Tensor:
        if x.data.dim() != 4:
            raise ValueError("BatchNorm2d expects NCHW input")
        if int(x.data.size(1)) != self.num_features:
            raise ValueError(f"BatchNorm2d expects C={self.num_features}, got {int(x.data.size(1))}")

        training = bool(getattr(self, "_training", True))
        if training:
            y, mean, invstd, var = ops.batchnorm2d_forward_train(x.data, self.weight.data, self.bias.data, eps=self.eps)
            ops.batchnorm2d_update_running_(self.running_mean, self.running_var, mean, var, momentum=self.momentum)
        else:
            y = ops.batchnorm2d_forward_eval(
                x.data, self.weight.data, self.bias.data, self.running_mean, self.running_var, eps=self.eps
            )
            mean = None
            invstd = None

        out_requires_grad = training and (x.requires_grad or self.weight.requires_grad or self.bias.requires_grad)
        out = Tensor(y, requires_grad=out_requires_grad)

        def _backward(g: torch.Tensor) -> None:
            if not training:
                return
            assert mean is not None
            assert invstd is not None

            need_dx = x.requires_grad
            need_dw = self.weight.requires_grad
            need_db = self.bias.requires_grad
            dx, dw, db = ops.batchnorm2d_backward(g, x.data, self.weight.data, mean, invstd, need_dx, need_dw, need_db)

            if need_dx:
                x.grad = dx if x.grad is None else (x.grad + dx)
            if need_dw:
                self.weight.grad = dw if self.weight.grad is None else (self.weight.grad + dw)
            if need_db:
                self.bias.grad = db if self.bias.grad is None else (self.bias.grad + db)

        if out.requires_grad:
            out._node = _Node((x,), _backward)
        return out


class SimpleCifarNet(Module):
    def __init__(self, device: torch.device | None = None) -> None:
        device = device or torch.device("cuda")
        # Match Task1/Task2 `cnn_bn` backbone:
        # (64,64)->pool -> (128,128)->pool -> (256,256)->pool -> GAP -> Linear(256->10)
        self.conv1 = Conv2d(3, 64, kernel=3, padding=1, bias=False, device=device)
        self.bn1 = BatchNorm2d(64, device=device)
        self.conv2 = Conv2d(64, 64, kernel=3, padding=1, bias=False, device=device)
        self.bn2 = BatchNorm2d(64, device=device)
        self.pool1 = MaxPool2d(2, 2)

        self.conv3 = Conv2d(64, 128, kernel=3, padding=1, bias=False, device=device)
        self.bn3 = BatchNorm2d(128, device=device)
        self.conv4 = Conv2d(128, 128, kernel=3, padding=1, bias=False, device=device)
        self.bn4 = BatchNorm2d(128, device=device)
        self.pool2 = MaxPool2d(2, 2)

        self.conv5 = Conv2d(128, 256, kernel=3, padding=1, bias=False, device=device)
        self.bn5 = BatchNorm2d(256, device=device)
        self.conv6 = Conv2d(256, 256, kernel=3, padding=1, bias=False, device=device)
        self.bn6 = BatchNorm2d(256, device=device)
        self.pool3 = MaxPool2d(2, 2)
        self.relu = ReLU()
        self.gap = GlobalAvgPool2d()
        self.fc = Linear(256, 10, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        x = self.gap(x)
        x = self.fc(x)
        return x
