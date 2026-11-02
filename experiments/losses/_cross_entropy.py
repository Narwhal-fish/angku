"""Cross-entropy that also runs under ``torch.use_deterministic_algorithms(True)`` on CUDA.

PyTorch's CUDA ``nll_loss2d`` (what ``F.cross_entropy`` uses for spatial targets) has no
deterministic kernel, so with ``training.deterministic`` (the default) every GPU run would
raise on its first batch. ``CrossEntropy`` is ``nn.CrossEntropyLoss`` with one change: on a
CUDA tensor while deterministic algorithms are on, it computes the same loss from
``log_softmax`` and a one-hot target (elementwise ops and reductions only, all deterministic).
Everywhere else it is exactly ``nn.CrossEntropyLoss``, so CPU runs (and every bundle made
before this) are bit-for-bit unchanged. The two formulas agree to rounding (~1e-7), not bit for
bit: bit-exactness is promised per device, never across CPU and GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def one_hot_cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """``F.cross_entropy`` semantics (class-index targets) without ``nll_loss``.

    Same weighting and normalisation as PyTorch: ``mean`` divides by the summed weight of the
    non-ignored targets; label smoothing adds ``eps / C * -sum_c w_c log p_c`` normalised the
    same way; ignored positions contribute 0 (``none`` returns 0 there)."""
    n_classes = logits.shape[1]
    logp = F.log_softmax(logits, dim=1)
    keep = target != ignore_index
    onehot = F.one_hot(torch.where(keep, target, 0), n_classes).movedim(-1, 1).to(logp.dtype)
    if weight is not None:
        w = weight.to(logp).view(1, n_classes, *([1] * (logits.dim() - 2)))
        logp_w = logp * w
        denom = ((onehot * w).sum(1) * keep).sum()
    else:
        logp_w = logp
        denom = keep.sum().to(logp.dtype)
    nll = -(onehot * logp_w).sum(1) * keep

    def reduce(per: Tensor) -> Tensor:
        if reduction == "mean":
            return per.sum() / denom
        if reduction == "sum":
            return per.sum()
        return per

    loss = reduce(nll)
    if label_smoothing > 0:
        smooth = -logp_w.sum(1) * keep
        loss = (1 - label_smoothing) * loss + (label_smoothing / n_classes) * reduce(smooth)
    return loss


class CrossEntropy(nn.CrossEntropyLoss):
    """``nn.CrossEntropyLoss``; on CUDA with deterministic algorithms on, ``one_hot_cross_entropy``."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.is_cuda and torch.are_deterministic_algorithms_enabled() and not target.is_floating_point():
            return one_hot_cross_entropy(
                input,
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
                reduction=self.reduction,
            )
        return super().forward(input, target)
