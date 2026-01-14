from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch

from . import ops
from .tensor import Tensor


@dataclass
class SGD:
    params: list[Tensor]
    lr: float
    momentum: float = 0.9
    weight_decay: float = 5e-4

    def __post_init__(self) -> None:
        self._vel: Dict[int, torch.Tensor] = {}
        if self.momentum < 0:
            raise ValueError("momentum must be >= 0")

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            vel: Optional[torch.Tensor] = None
            if self.momentum != 0.0:
                vid = id(p)
                vel = self._vel.get(vid)
                if vel is None:
                    vel = torch.zeros_like(p.data)
                    self._vel[vid] = vel
            ops.sgd_update_(p.data, p.grad, vel, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

