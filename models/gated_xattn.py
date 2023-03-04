"""
The flamingo is built from lucidrains and dhansmair's implementation.
"""
import torch
from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum, nn


def exists(val):
    return val is not None


def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )


# gated cross attention

class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        media,          # media tensor, represents information from other modality, encoded by perceiver resample - (batch, latents, dim)
        media_mask
    ):
        """This has the same inputs as the GatedCrossAttentionBlock
        Args:
            x (FloatTensor):
                language features (n_batch, n_token, d_token)
            media (FloatTensor, optional):
                hetero features   (n_batch, n_latents, d_media). Defaults to None.
            media_mask (LongTensor | BoolTensor, optional):
                mask for hetero features (n_batch, n_latents). Defaults to None.
        Returns:
            FloatTensor: Tensor (n_batch, n_token, d_token)
        """
        h = self.heads

        x = self.norm(x)

        # d_inner = d_head * n_head
        # (batch, n_token, d_token) -> (batch, n_token, d_inner)
        q = self.to_q(x)

        # (batch, n_latents, d_media) -> (batch, n_latents, d_inner) for k, v
        k, v = self.to_kv(media).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)  # (batch, n_token, n_latents)

        if media_mask is not None:
            sim = sim.masked_fill(~media_mask, float('-inf'))

        # What is this for? For numerical stability?
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)  # (batch, n_token, d_inner)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # (batch, n_token, d_token)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, latents, dim)
    ):
        x = self.attn(x, media) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh()  + x
        return x
