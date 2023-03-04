"""
The gated cross attention is built from lucidrains and dhansmair's implementation.
The 'media' in the code refers to the other modality, it can be knowledge graph, passage
embedding, image etc.
"""
import torch
from einops import rearrange
from einops_exts import rearrange_many
from torch import einsum, nn


def exists(val):
    return val is not None


def FeedForward(dim, mult = 4):
    """Feedforward layer with GELU activation."""
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
        dim_media,
        dim_head = 64,
        heads = 8,
    ):
        """
        Args:
            dim (int): dimension of the input language token embedding
            dim_media (int): dimension of the input media token embedding
            dim_head (int, optional): dimension of the q, k, v inside the attention head. Defaults to 64.
            heads (int, optional): number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_media, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x: torch.FloatTensor,
        media: torch.FloatTensor,
        media_mask: torch.LongTensor | torch.BoolTensor,
        previous_kv: tuple = None,
        output_kv: bool = False
    ):
        """This has the same inputs as the GatedCrossAttentionBlock
        Args:
            x (FloatTensor):
                language features (n_batch, n_token, d_token)
            media (FloatTensor, optional):
                media features, represents information from other modality, e.g. encoded by perceiver resample
                (n_batch, n_latents, d_media). Defaults to None.
            media_mask (LongTensor | BoolTensor, optional):
                mask for media features (n_batch, n_latents). Defaults to None.
            previous_kv (tuple, optional):
                tuple of previous keys and values. Passed when caching is used during text generation.
                Defaults to None.
            output_kv (bool, optional):
                whether to return the keys and values. Defaults to False.
        Returns:
            FloatTensor: Tensor (n_batch, n_token, d_token)
        """
        h = self.heads

        x = self.norm(x)

        # d_inner = d_head * n_head
        # (batch, n_token, d_token) -> (batch, n_token, d_inner)
        q = self.to_q(x)
        q = q * self.scale

        if previous_kv is None:
            # (batch, n_latents, d_media) -> (batch, n_latents, d_inner) for k, v
            k, v = self.to_kv(media).chunk(2, dim=-1)
            q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)
        else:
            # media can be ignored, k, v already computed
            k, v = previous_kv
            q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        # compute the attention scores from the queries and keys
        sim = einsum('... i d, ... j d -> ... i j', q, k)  # (batch, n_token, n_latents)

        if media_mask is not None:
            sim = sim.masked_fill(~media_mask, float('-inf'))

        # What is this for? For numerical stability?
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # compute the context vectors from the attention scores and values
        out = einsum('... i j, ... j d -> ... i d', attn, v)  # (batch, n_token, d_inner)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if output_kv:
            return self.to_out(out), (k, v)
        return self.to_out(out), None  # (batch, n_token, d_token)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_media,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        """
        Args:
            dim (int): dimension of the input language token embedding
            dim_media (int): dimension of the input media token embedding
            dim_head (int, optional): dimension of the q, k, v inside the attention head. Defaults to 64.
            heads (int, optional): number of attention heads. Defaults to 8.
            ff_mult (int, optional): multiplier for the hidden dimension of the feedforward layer. Defaults to 4.
        """
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_media=dim_media, dim_head=dim_head, heads=heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x: torch.FloatTensor,
        media: torch.FloatStorage,
        media_mask: torch.LongTensor | torch.BoolTensor,
    ):
        """
        Args:
            x (FloatTensor): language features (n_batch, n_token, d_token)
            media (FloatTensor, optional): media features, e.g. encoded by perceiver resample (n_batch, n_latents, d_media).
            media_mask (LongTensor | BoolTensor, optional): mask for media features (n_batch, n_latents).
        """
        x = self.attn(x, media, media_mask) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh()  + x
        return x
