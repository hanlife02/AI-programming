from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch

from . import ops


@dataclass
class _Node:
    parents: tuple["Tensor", ...]
    backward: Callable[[torch.Tensor], None]


class Tensor:
    def __init__(self, data: torch.Tensor, requires_grad: bool = False, node: Optional[_Node] = None):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Tensor(data=...) must wrap a torch.Tensor")
        if data.requires_grad:
            data = data.detach()
        self.data = data
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[torch.Tensor] = None
        self._node = node

    @staticmethod
    def zeros(shape: Iterable[int], device: torch.device, requires_grad: bool = False) -> "Tensor":
        return Tensor(torch.zeros(tuple(shape), device=device, dtype=torch.float32), requires_grad=requires_grad)

    @staticmethod
    def randn(shape: Iterable[int], device: torch.device, std: float = 1.0, requires_grad: bool = False) -> "Tensor":
        return Tensor(torch.randn(tuple(shape), device=device, dtype=torch.float32) * float(std), requires_grad=requires_grad)

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def view(self, *shape: int) -> "Tensor":
        y = self.data.view(*shape)
        out = Tensor(y, requires_grad=self.requires_grad)

        def _backward(g: torch.Tensor) -> None:
            if self.requires_grad:
                dx = g.view_as(self.data)
                self.grad = dx if self.grad is None else (self.grad + dx)

        if out.requires_grad:
            out._node = _Node((self,), _backward)
        return out

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Optional[torch.Tensor] = None) -> None:
        if not self.requires_grad:
            return
        if grad is None:
            if self.data.numel() != 1:
                raise ValueError("grad must be provided for non-scalar tensors")
            grad = torch.ones_like(self.data)
        else:
            if grad.shape != self.data.shape:
                raise ValueError("grad shape mismatch")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(v: Tensor) -> None:
            vid = id(v)
            if vid in visited:
                return
            visited.add(vid)
            if v._node is not None:
                for p in v._node.parents:
                    build(p)
            topo.append(v)

        build(self)

        self.grad = grad if self.grad is None else (self.grad + grad)

        for v in reversed(topo):
            if v._node is None or v.grad is None:
                continue
            v._node.backward(v.grad)

    # --- ops ---
    def __add__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(torch.tensor(other, device=self.data.device, dtype=self.data.dtype))
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward(g: torch.Tensor) -> None:
            if self.requires_grad:
                self.grad = g if self.grad is None else (self.grad + g)
            if other.requires_grad:
                other.grad = g if other.grad is None else (other.grad + g)

        if out.requires_grad:
            out._node = _Node((self, other), _backward)
        return out

    def __mul__(self, other: float) -> "Tensor":
        s = float(other)
        out = Tensor(self.data * s, requires_grad=self.requires_grad)

        def _backward(g: torch.Tensor) -> None:
            if self.requires_grad:
                self.grad = (g * s) if self.grad is None else (self.grad + g * s)

        if out.requires_grad:
            out._node = _Node((self,), _backward)
        return out

    def relu(self) -> "Tensor":
        y = ops.relu_forward(self.data)
        out = Tensor(y, requires_grad=self.requires_grad)

        def _backward(g: torch.Tensor) -> None:
            if self.requires_grad:
                dx = ops.relu_backward(g, self.data)
                self.grad = dx if self.grad is None else (self.grad + dx)

        if out.requires_grad:
            out._node = _Node((self,), _backward)
        return out

    def conv2d(self, w: "Tensor", b: Optional["Tensor"], stride: int = 1, padding: int = 0) -> "Tensor":
        y = ops.conv2d_forward(self.data, w.data, None if b is None else b.data, stride=stride, padding=padding)
        out = Tensor(y, requires_grad=self.requires_grad or w.requires_grad or (b.requires_grad if b is not None else False))

        def _backward(g: torch.Tensor) -> None:
            need_dx = self.requires_grad
            need_dw = w.requires_grad
            need_db = (b is not None) and b.requires_grad
            dx, dw, db = ops.conv2d_backward(
                g,
                self.data,
                w.data,
                need_grad_x=need_dx,
                need_grad_w=need_dw,
                need_grad_b=need_db,
                stride=stride,
                padding=padding,
            )
            if self.requires_grad:
                self.grad = dx if self.grad is None else (self.grad + dx)
            if w.requires_grad:
                w.grad = dw if w.grad is None else (w.grad + dw)
            if need_db and b is not None:
                b.grad = db if b.grad is None else (b.grad + db)

        if out.requires_grad:
            parents = (self, w) if b is None else (self, w, b)
            out._node = _Node(parents, _backward)
        return out

    def maxpool2d(self, kernel: int = 2, stride: int = 2) -> "Tensor":
        y, ctx = ops.maxpool2d_forward(self.data, kernel=kernel, stride=stride)
        out = Tensor(y, requires_grad=self.requires_grad)

        def _backward(g: torch.Tensor) -> None:
            if self.requires_grad:
                dx = ops.maxpool2d_backward(g, ctx)
                self.grad = dx if self.grad is None else (self.grad + dx)

        if out.requires_grad:
            out._node = _Node((self,), _backward)
        return out

    def global_avg_pool2d(self) -> "Tensor":
        h = int(self.data.size(2))
        w = int(self.data.size(3))
        y = ops.global_avg_pool2d_forward(self.data)
        out = Tensor(y, requires_grad=self.requires_grad)

        def _backward(g: torch.Tensor) -> None:
            if self.requires_grad:
                dx = ops.global_avg_pool2d_backward(g, h=h, w=w)
                self.grad = dx if self.grad is None else (self.grad + dx)

        if out.requires_grad:
            out._node = _Node((self,), _backward)
        return out

    def linear(self, w: "Tensor", b: Optional["Tensor"]) -> "Tensor":
        y = ops.linear_forward(self.data, w.data, None if b is None else b.data)
        out = Tensor(y, requires_grad=self.requires_grad or w.requires_grad or (b.requires_grad if b is not None else False))

        def _backward(g: torch.Tensor) -> None:
            need_dx = self.requires_grad
            need_dw = w.requires_grad
            need_db = (b is not None) and b.requires_grad
            dx, dw, db = ops.linear_backward(g, self.data, w.data, need_grad_x=need_dx, need_grad_w=need_dw, need_grad_b=need_db)
            if self.requires_grad:
                self.grad = dx if self.grad is None else (self.grad + dx)
            if w.requires_grad:
                w.grad = dw if w.grad is None else (w.grad + dw)
            if need_db and b is not None:
                b.grad = db if b.grad is None else (b.grad + db)

        if out.requires_grad:
            parents = (self, w) if b is None else (self, w, b)
            out._node = _Node(parents, _backward)
        return out

    def cross_entropy(self, targets: torch.Tensor) -> "Tensor":
        loss, ctx = ops.cross_entropy_forward(self.data, targets)
        out = Tensor(loss, requires_grad=self.requires_grad)

        def _backward(g: torch.Tensor) -> None:
            if self.requires_grad:
                dlogits = ops.cross_entropy_backward(g, ctx)
                self.grad = dlogits if self.grad is None else (self.grad + dlogits)

        if out.requires_grad:
            out._node = _Node((self,), _backward)
        return out
