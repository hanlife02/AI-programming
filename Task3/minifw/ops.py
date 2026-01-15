from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _load_ext():
    try:
        import task3_ops
    except ModuleNotFoundError as e:
        import sys
        from pathlib import Path

        task3_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(task3_dir))
        try:
            import task3_ops
        except ModuleNotFoundError as e2:
            raise ModuleNotFoundError(
                "task3_ops extension not built. Build it with:\n"
                "  cd Task3 && python setup.py build_ext --inplace"
            ) from e2
    return task3_ops


_ext = None
_conv2d_op_cache = {}
_bn2d_op_cache = {}


def ext():
    global _ext
    if _ext is None:
        _ext = _load_ext()
    return _ext


def _conv2d_op(stride: int, padding: int, has_bias: bool):
    key = (int(stride), int(padding), bool(has_bias))
    op = _conv2d_op_cache.get(key)
    if op is None:
        op = ext().Conv2dOp(key[0], key[1], key[2])
        _conv2d_op_cache[key] = op
    return op


def _bn2d_op(momentum: float, eps: float):
    key = (float(momentum), float(eps))
    op = _bn2d_op_cache.get(key)
    if op is None:
        op = ext().BatchNorm2dOp(key[0], key[1])
        _bn2d_op_cache[key] = op
    return op


@dataclass(frozen=True)
class MaxPoolContext:
    indices: torch.Tensor
    in_h: int
    in_w: int
    kernel: int
    stride: int


@dataclass(frozen=True)
class CrossEntropyContext:
    targets: torch.Tensor
    probs: torch.Tensor


@dataclass(frozen=True)
class BatchNorm2dContext:
    saved_mean: torch.Tensor
    saved_invstd: torch.Tensor
    momentum: float
    eps: float


def conv2d_forward(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], stride: int, padding: int) -> torch.Tensor:
    if w.size(2) > 7 or w.size(3) > 7:
        raise ValueError("Task3 conv2d supports kernel sizes up to 7x7")
    return _conv2d_op(stride, padding, has_bias=b is not None).forward(x, w, b)


def conv2d_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    need_grad_x: bool,
    need_grad_w: bool,
    need_grad_b: bool,
    stride: int,
    padding: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _conv2d_op(stride, padding, has_bias=True).backward(
        grad_out,
        x,
        w,
        bool(need_grad_x),
        bool(need_grad_w),
        bool(need_grad_b),
    )


def relu_forward(x: torch.Tensor) -> torch.Tensor:
    return ext().relu_forward(x)


def relu_backward(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ext().relu_backward(grad_out, x)


def maxpool2d_forward(x: torch.Tensor, kernel: int, stride: int) -> tuple[torch.Tensor, MaxPoolContext]:
    y, idx = ext().maxpool2d_forward(x, int(kernel), int(stride))
    return y, MaxPoolContext(indices=idx, in_h=int(x.size(2)), in_w=int(x.size(3)), kernel=int(kernel), stride=int(stride))


def maxpool2d_backward(grad_out: torch.Tensor, ctx: MaxPoolContext) -> torch.Tensor:
    return ext().maxpool2d_backward(
        grad_out,
        ctx.indices,
        int(ctx.in_h),
        int(ctx.in_w),
        int(ctx.kernel),
        int(ctx.stride),
    )


def global_avg_pool2d_forward(x: torch.Tensor) -> torch.Tensor:
    return ext().global_avg_pool2d_forward(x)


def global_avg_pool2d_backward(grad_out: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return ext().global_avg_pool2d_backward(grad_out, int(h), int(w))


def linear_forward(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    return ext().linear_forward(x, w, b)


def linear_backward(
    grad_out: torch.Tensor, x: torch.Tensor, w: torch.Tensor, need_grad_x: bool, need_grad_w: bool, need_grad_b: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ext().linear_backward(grad_out, x, w, bool(need_grad_x), bool(need_grad_w), bool(need_grad_b))


def cross_entropy_forward(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, CrossEntropyContext]:
    loss, probs = ext().cross_entropy_forward(logits, targets)
    return loss, CrossEntropyContext(targets=targets, probs=probs)


def cross_entropy_backward(grad_out: torch.Tensor, ctx: CrossEntropyContext) -> torch.Tensor:
    return ext().cross_entropy_backward(ctx.probs, ctx.targets, grad_out)


def sgd_update_(param: torch.Tensor, grad: torch.Tensor, velocity: Optional[torch.Tensor], lr: float, momentum: float, weight_decay: float) -> None:
    ext().sgd_update_(param, grad, velocity, float(lr), float(momentum), float(weight_decay))


def batchnorm2d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[torch.Tensor, BatchNorm2dContext]:
    y, mean, invstd = _bn2d_op(momentum, eps).forward(x, weight, bias, running_mean, running_var, bool(training))
    return y, BatchNorm2dContext(saved_mean=mean, saved_invstd=invstd, momentum=float(momentum), eps=float(eps))


def batchnorm2d_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    saved_mean: torch.Tensor,
    saved_invstd: torch.Tensor,
    momentum: float,
    eps: float,
    need_grad_x: bool,
    need_grad_w: bool,
    need_grad_b: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _bn2d_op(momentum, eps).backward(
        grad_out,
        x,
        weight,
        saved_mean,
        saved_invstd,
        bool(need_grad_x),
        bool(need_grad_w),
        bool(need_grad_b),
    )
