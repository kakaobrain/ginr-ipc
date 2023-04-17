import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class AttentionBlockConfig:
    mask: bool = True
    embed_dim: int = MISSING
    n_head: int = MISSING
    mlp_bias: bool = True
    attn_bias: bool = True
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.1
    gelu: str = "v1"


@dataclass
class AttentionStackConfig:
    mask: bool = False
    n_layer: int = MISSING
    block: AttentionBlockConfig = AttentionBlockConfig()


class GELU(nn.Module):
    def __init__(self, version="v1"):
        super().__init__()
        assert version == "v1" or version == "v2"

        self.version = version

    def forward(self, x):
        if self.version == "v1":
            return F.gelu(x)
        else:
            return x * torch.sigmoid(1.702 * x)


class MultiSelfAttention(nn.Module):
    """
    Optimized by batched matmul operations
    """

    def __init__(self, config: AttentionBlockConfig, mask=True):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim, bias=config.attn_bias)
        self.query = nn.Linear(config.embed_dim, config.embed_dim, bias=config.attn_bias)
        self.value = nn.Linear(config.embed_dim, config.embed_dim, bias=config.attn_bias)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop, inplace=False)
        self.resid_drop = nn.Dropout(config.resid_pdrop, inplace=True)
        # output projection
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, config.attn_bias)

        self.n_head = config.n_head
        self.mask = mask

    def forward(self, x, contexts=None, caching=False, past_kv=None):
        (B, T, C) = x.shape
        if contexts is not None:
            assert past_kv is None
            B_ctx, T_ctx, C_ctx = contexts.shape

        if not caching:
            assert past_kv is None

        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(T, B * self.n_head, C // self.n_head).transpose(0, 1)  # (B*nh, T, hs)

        if contexts is None:
            k = self.key(x).view(T, B * self.n_head, C // self.n_head).transpose(0, 1)  # (B*nh, T, hs)
            v = self.value(x).view(T, B * self.n_head, C // self.n_head).transpose(0, 1)  # (B*nh, T, hs)
        else:
            contexts = contexts.transpose(0, 1).contiguous()  # (B, T_ctx, C) -> (T_ctx, B, C)
            x_with_ctx = torch.cat([contexts, x], dim=0)
            T_x_with_ctx = x_with_ctx.shape[0]
            k = (
                self.key(x_with_ctx).view(T_x_with_ctx, B * self.n_head, C // self.n_head).transpose(0, 1)
            )  # (B*nh, T_ctx, hs)
            v = (
                self.value(x_with_ctx).view(T_x_with_ctx, B * self.n_head, C // self.n_head).transpose(0, 1)
            )  # (B*nh, T_ctx, hs)

        if past_kv is not None:
            """
            we assume the past_kv is always given without contexts,
            becuase only one-pass of inference at the first time is needed for contexts.
            """
            assert contexts is None
            past_key, past_value = past_kv
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
            T_past = past_key.shape[1]
        else:
            T_past = 0

        if caching:
            # when context is not None, the context is caching.
            present = torch.stack([k, v])
        else:
            present = None

        # Tensor shape below: query: (B * nh, T, hs) X key: (B * nh, hs, T_past+T) -> (B * nh, T, T_past+T)
        att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
        if self.mask:
            if contexts is None:
                # assume it is not cross-attention or cross-attention with past_kv
                if past_kv is None:
                    # non-cross-attention
                    mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                    mask = mask.view(1, T, T)
                    att = att.masked_fill(~mask[:, :T, :T], float("-inf"))
                else:
                    # cross attention btw query & key
                    mask_qk = torch.ones(T, T_past, device=q.device, dtype=torch.bool)
                    mask_qq = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                    mask = torch.cat([mask_qk, mask_qq], dim=-1).unsqueeze(1)
                    att = att.masked_fill(~mask[:, :T, : T_past + T], float("-inf"))
                    # cross attention
            else:
                # assume it is cross-attention
                mask_qk = torch.ones(T, T_ctx, device=q.device, dtype=torch.bool)
                mask_qq = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                mask = torch.cat([mask_qk, mask_qq], dim=-1).unsqueeze(0)
                att = att.masked_fill(~mask[:, :T, : T_ctx + T], float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.bmm(att, v)  # (B*nh, T, T_past+T) X (B*nh, T_past+T, hs) -> (B*nh, T, hs)
        y = y.transpose(0, 1).contiguous().view(T, B, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        if caching:
            return y.transpose(0, 1).contiguous(), present  # (T, B, C) -> (B, T, C)
        else:
            return y.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)


class AttentionBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config: AttentionBlockConfig, mask: bool):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

        self.attn = MultiSelfAttention(config, mask=mask)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=config.mlp_bias),
            GELU(config.gelu),
            nn.Linear(4 * config.embed_dim, config.embed_dim, bias=config.mlp_bias),
            nn.Dropout(config.resid_pdrop, inplace=True),
        )
        self._cache = None

    def forward(self, x, contexts=None):
        attn = self.attn(self.ln1(x), contexts=contexts)

        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x

    def cached_forward(self, x_present, contexts=None):
        if self._cache["past_kv"] is not None:
            # contexts vectors (encoder's outputs) are computed only at the first time.
            # After that, the contexts vectors are already cached.
            contexts = None
        attn, present = self.attn(self.ln1(x_present), contexts=contexts, caching=True, past_kv=self._cache["past_kv"])
        self._cache["past_kv"] = present

        x_present = x_present + attn
        x_present = x_present + self.mlp(self.ln2(x_present))

        return x_present

    def init_cache(self):
        self._cache = {"past_kv": None}


class AttentionStack(nn.Module):
    blocks: Iterable[AttentionBlock]

    def __init__(self, config: AttentionStackConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([AttentionBlock(config.block, config.mask) for _ in range(config.n_layer)])

    def forward(self, x, contexts=None):
        """
        if contexts is not None.
        the attention stack conducts cross attention as
          query = x
          key, value = [contexts, x]
        """

        for block in self.blocks:
            x = block(x, contexts=contexts)
        return x

    def cached_forward(self, x_present, contexts=None):
        for block in self.blocks:
            x_present = block.cached_forward(x_present, contexts=contexts)
        return x_present

    def init_cache(self):
        for block in self.blocks:
            block.init_cache()
