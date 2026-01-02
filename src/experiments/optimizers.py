from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional, Tuple

import torch
from torch import no_grad
from torch.optim import Optimizer


class StructuralHeadOptimizer(Optimizer):
    def __init__(
        self,
        params,
        base_optimizer_cls,
        lr: float = 1e-3,
        warmup_steps: int = 400,
        eps: float = 1e-8,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        ema_beta: float = 0.9,
        **base_kwargs,
    ):
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.ema_beta = float(ema_beta)

        self.step_count = 0
        self.base = base_optimizer_cls(params, lr=lr, **base_kwargs)
        self.param_groups = self.base.param_groups

        self._idx_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._group_ema: List[Optional[torch.Tensor]] = [None] * len(self.param_groups)

    def _rand_idx(self, n: int, k: int, device: torch.device) -> torch.Tensor:
        if k <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        return torch.randint(n, (k,), device=device, dtype=torch.long)

    @no_grad()
    def step(self, closure=None):
        self.step_count += 1
        alpha = min(1.0, self.step_count / max(1, self.warmup_steps))

        for gi, group in enumerate(self.param_groups):
            group_scale_sum = 0.0
            group_weight_sum = 0.0

            for p in group["params"]:
                g = p.grad
                if g is None or not g.is_floating_point():
                    continue

                x = g.view(-1)
                n = x.numel()
                if n < 8:
                    continue

                m = min(n, max(256, int(sqrt(n))))
                if m < n:
                    s = x[self._rand_idx(n, m, x.device)]
                else:
                    s = x

                abs_s = s.abs()
                mean_abs = abs_s.mean()

                k = max(4, m // 16)
                k = min(k, m)

                kth = abs_s.kthvalue(m - k + 1).values
                head_mask = abs_s >= kth
                head = s[head_mask]
                if head.numel() == 0:
                    continue

                tail = s[self._rand_idx(m, min(k, m), s.device)]
                if tail.numel() == 0:
                    continue

                head_dir = head.sign().sum()
                if head_dir == 0:
                    continue
                head_dir = head_dir.sign()

                consensus = (tail.sign() == head_dir).float().mean()

                head_abs_mean = head.abs().mean()
                tail_abs_mean = tail.abs().mean()
                strength = head_abs_mean / (mean_abs + self.eps)

                penalty = (head.mean() - tail.mean()).abs() + (head_abs_mean - tail_abs_mean).abs()

                raw_scale = (1.0 + 0.5 * strength * (2.0 * consensus - 1.0)) / (1.0 + penalty + self.eps)
                raw_scale = raw_scale.clamp(self.scale_min, self.scale_max)
                scale = (1.0 - alpha) + alpha * raw_scale

                w = float(n)
                group_scale_sum += float(scale) * w
                group_weight_sum += w

            if group_weight_sum > 0.0:
                group_scale = torch.tensor(group_scale_sum / group_weight_sum, device=group["params"][0].device)

                prev = self._group_ema[gi]
                if prev is None:
                    ema = group_scale
                else:
                    ema = prev.mul(self.ema_beta).add(group_scale, alpha=1.0 - self.ema_beta)
                self._group_ema[gi] = ema

                applied = ema.clamp(self.scale_min, self.scale_max).item()

                for p in group["params"]:
                    if p.grad is not None and p.grad.is_floating_point():
                        p.grad.mul_(applied)

        return self.base.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self.base.zero_grad(set_to_none=set_to_none)
