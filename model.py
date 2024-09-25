import dataclasses
import math

import einops
import torch


def create_position(hidden_dimension: int, heads: int, sequence_length: int) -> torch.Tensor:
    """Create RoPE position."""

    theta = torch.logspace(
        start=math.log10(0.5 * math.pi),  
        end=math.log10(0.5 * math.pi * sequence_length),
        steps=(hidden_dimension // heads) // 2,
    ).repeat_interleave(2, dim=-1)

    position = torch.arange(sequence_length)/sequence_length
    position = torch.outer(position, theta)
    position = torch.stack([position.cos(), position.sin()], dim=0)

    return position


def apply_rope(x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Apply RoPE."""

    x_hat = torch.cat((-x[..., 1 :: 2], x[..., :: 2]), dim=-1)
    
    return x*position[0, : x.size(-2)] + x_hat*position[1, : x.size(-2)]


class Attention(torch.nn.Module):
    def __init__(self, hidden_dimension: int, heads: int) -> None:
        super().__init__()

        self.heads = heads
        self.linear_1 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_2 = torch.nn.Linear(hidden_dimension, hidden_dimension, bias=False)
        self.norm = torch.nn.LayerNorm(hidden_dimension // heads, bias=False)

        torch.nn.init.zeros_(self.linear_2.weight)
    
    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        q, k, v = einops.rearrange(x, 'b t (n h e) -> n b h t e', n=3, h=self.heads)
        q = apply_rope(self.norm(q), position)
        k = apply_rope(self.norm(k), position)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.linear_2(einops.rearrange(x, 'b h t e -> b t (h e)'))

        return x


class MLP(torch.nn.Module):
    def __init__(self, hidden_dimension: int) -> None:
        super().__init__()

        self.linear_1 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_2 = torch.nn.Linear(hidden_dimension * 3, hidden_dimension, bias=False)   

        torch.nn.init.zeros_(self.linear_2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = torch.nn.functional.silu(x)
        x = self.linear_2(x)

        return x


@dataclasses.dataclass
class TransformerConfig:
    hidden_dimension: int = 256
    heads: int = 4
    layers: int = 4
    vocabulary_size: int = 256
    sequence_length: int = 1024
    bos_id: int | None = None
    eos_id: int | None = None
    pad_id: int | None = None


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.attention = Attention(config.hidden_dimension, config.heads)
        self.mlp = MLP(config.hidden_dimension)
        self.norm_1 = torch.nn.LayerNorm(config.hidden_dimension)
        self.norm_2 = torch.nn.LayerNorm(config.hidden_dimension)

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_1(x), position)
        x = x + self.mlp(self.norm_2(x))

        return x
    

class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.config = config
        self.embed = torch.nn.Embedding(config.vocabulary_size, config.hidden_dimension, config.pad_id)
        self.blocks = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.register_buffer('position', create_position(config.hidden_dimension, config.heads, config.sequence_length))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)

        for block in self.blocks:
            x = block(x, self.position)
        
        return x @ self.embed.weight.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        logits = self.predict(x[..., : -1])
        
        return torch.nn.functional.nll_loss(logits.view(-1, logits.size(-1)), x[..., 1:].flatten())
