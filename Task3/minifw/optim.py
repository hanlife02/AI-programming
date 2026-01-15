from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional, Sequence

import torch

from . import ops
from .tensor import Tensor


class SGD:

    def __init__(
        self,
        params: Sequence[Tensor] | Sequence[dict[str, Any]],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be >= 0")
        if momentum < 0:
            raise ValueError("momentum must be >= 0")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")

        self._vel: Dict[int, torch.Tensor] = {}
        self.param_groups: list[dict[str, Any]] = self._normalize_groups(params, lr, momentum, weight_decay)

    @staticmethod
    def _normalize_groups(
        params: Sequence[Tensor] | Sequence[dict[str, Any]],
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        if len(params) == 0:
            raise ValueError("params must be non-empty")
        first = params[0]
        if isinstance(first, Tensor):
            return [{"params": list(params), "lr": float(lr), "momentum": float(momentum), "weight_decay": float(weight_decay)}]
        if not isinstance(first, dict):
            raise TypeError("params must be a sequence of Tensor or a sequence of param group dicts")

        groups: list[dict[str, Any]] = []
        for g in params:  # type: ignore[assignment]
            if not isinstance(g, dict):
                raise TypeError("param group must be a dict")
            if "params" not in g:
                raise KeyError("param group missing required key: 'params'")
            group_params = list(g["params"])
            if len(group_params) == 0:
                continue
            groups.append(
                {
                    "params": group_params,
                    "lr": float(g.get("lr", lr)),
                    "momentum": float(g.get("momentum", momentum)),
                    "weight_decay": float(g.get("weight_decay", weight_decay)),
                }
            )
        if not groups:
            raise ValueError("no parameters found in param groups")
        return groups

    def parameters(self) -> Iterator[Tensor]:
        for g in self.param_groups:
            for p in g["params"]:
                yield p

    @property
    def lr(self) -> float:
        return float(self.param_groups[0]["lr"])

    @lr.setter
    def lr(self, value: float) -> None:
        v = float(value)
        if v < 0:
            raise ValueError("lr must be >= 0")
        for g in self.param_groups:
            g["lr"] = v

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def step(self) -> None:
        for g in self.param_groups:
            lr = float(g["lr"])
            momentum = float(g["momentum"])
            weight_decay = float(g["weight_decay"])
            for p in g["params"]:
                if p.grad is None:
                    continue
                vel: Optional[torch.Tensor] = None
                if momentum != 0.0:
                    vid = id(p)
                    vel = self._vel.get(vid)
                    if vel is None:
                        vel = torch.zeros_like(p.data)
                        self._vel[vid] = vel
                ops.sgd_update_(p.data, p.grad, vel, lr=lr, momentum=momentum, weight_decay=weight_decay)
