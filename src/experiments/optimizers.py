from torch.optim import Optimizer
from torch import no_grad, topk, randperm, tensor

class StructuralHeadOptimizer(Optimizer):
    def __init__(
        self,
        params,
        base_optimizer_cls,
        lr=1e-3,
        head_k=64,
        tail_k=64,
        sample_k=1024,
        warmup_steps=400,
        eps=1e-8,
        scale_min=0.5,
        scale_max=2.0,
        **base_kwargs,
    ):
        self.head_k = head_k
        self.tail_k = tail_k
        self.sample_k = sample_k
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.step_count = 0
        self.base = base_optimizer_cls(params, lr=lr, **base_kwargs)
        self.param_groups = self.base.param_groups

    @no_grad()
    def step(self, closure=None):
        self.step_count += 1
        alpha = min(1.0, self.step_count / self.warmup_steps)

        total_scale = 0.0
        total_weight = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None or not g.is_floating_point():
                    continue

                x = g.view(-1)
                n = x.numel()
                if n < 8:
                    continue

                m = min(n, self.sample_k)
                if m < n:
                    idx = randperm(n, device=x.device)[:m]
                    s = x[idx]
                else:
                    s = x

                abs_s = s.abs()
                hk = min(m, self.head_k)
                tk = min(m, self.tail_k)

                head = s[topk(abs_s, hk).indices]
                tail = s[randperm(m, device=s.device)[:tk]]

                head_dir = head.sign().sum().sign()
                if head_dir == 0:
                    head_dir = tensor(1.0, device=s.device)

                consensus = (tail.sign() == head_dir).float().mean()

                strength = head.abs().mean() / (abs_s.mean() + self.eps)
                penalty = (head.mean() - tail.mean()).abs() + (head.abs().mean() - tail.abs().mean()).abs()

                raw_scale = (1.0 + 0.5 * strength * (2.0 * consensus - 1.0)) / (1.0 + penalty + self.eps)
                raw_scale = raw_scale.clamp(self.scale_min, self.scale_max)

                scale = (1.0 - alpha) + alpha * raw_scale
                w = float(n)

                total_scale += scale * w
                total_weight += w

        if total_weight > 0.0:
            global_scale = total_scale / total_weight
            for group in self.param_groups:
                for p in group["params"]:
                    g = p.grad
                    if g is not None and g.is_floating_point():
                        g.mul_(global_scale)

        return self.base.step(closure)

    def zero_grad(self, set_to_none=False):
        self.base.zero_grad(set_to_none=set_to_none)