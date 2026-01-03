from torch import Tensor, tensor, cat
from torch.nn import Module, Linear, RMSNorm, Dropout, Parameter
from torch.nn.functional import softmax, sigmoid, silu

class ScoreSpaceEnsembleLayer(Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        head_dim_target: int = 32,
        max_heads: int = 8,
    ):
        super().__init__()

        num_heads = max(1, min(max_heads, dim // head_dim_target))
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = RMSNorm(dim)
        self.score_proj = Linear(dim, dim, bias=False)

        self.feature_proj = Linear(num_heads * 4, dim * 2, bias=False)
        self.output_proj = Linear(dim, dim, bias=False)
        self.dropout = Dropout(dropout)

        self.gate_proj = Linear(dim, dim)
        self.residual_scale = Parameter(tensor(0.0))

        self.feature_proj.weight.data.normal_(0.0, 0.02)
        self.output_proj.weight.data.zero_()
        self.gate_proj.bias.data.fill_(-4.0)

        self.register_buffer("eps", tensor(1e-6))

    def _weighted_stats(
        self,
        values: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean = (weights * values).sum(dim=-1)
        second_moment = (weights * values.square()).sum(dim=-1)
        variance = (second_moment - mean.square()).clamp_min(self.eps)
        concentration = (weights.square()).sum(dim=-1)
        return mean, variance, concentration

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch_size = input_tensor.size(0)
        residual = input_tensor

        normalized = self.norm(input_tensor)
        scores = self.score_proj(normalized)

        values = normalized.view(batch_size, self.num_heads, self.head_dim)
        score_heads = scores.view(batch_size, self.num_heads, self.head_dim)
        score_heads = score_heads / (
            score_heads.norm(dim=-1, keepdim=True) + self.eps
        )

        weight_pos = softmax(score_heads, dim=-1)
        weight_neg = softmax(-score_heads, dim=-1)

        mean_pos, var_pos, conc_pos = self._weighted_stats(values, weight_pos)
        mean_neg, var_neg, conc_neg = self._weighted_stats(values, weight_neg)

        entropy = -(weight_pos * (weight_pos + self.eps).log()).sum(dim=-1)
        mix = sigmoid(entropy).clamp(0.05, 0.95)

        mixed_mean = mix * mean_pos + (1.0 - mix) * mean_neg
        mixed_var = mix * var_pos + (1.0 - mix) * var_neg
        mixed_conc = mix * conc_pos + (1.0 - mix) * conc_neg

        features = cat(
            [mixed_mean, mixed_var, mixed_conc, entropy],
            dim=-1,
        )

        fused = self.feature_proj(features)
        fused_a, fused_b = fused.chunk(2, dim=-1)

        delta = self.output_proj(
            self.dropout(fused_a * silu(fused_b))
        )

        gate = sigmoid(self.gate_proj(delta))
        return residual + (self.residual_scale * gate) * delta
