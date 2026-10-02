"""A GPT (decoder-only transformer) in plain PyTorch: the model every
strategy in tandem trains. Pre-norm blocks with causal self-attention and a
GELU MLP. The LM head is not tied to the embedding, so that every parameter
belongs to exactly one block when the model is sharded or split."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 256
    seq: int = 128
    layers: int = 4
    heads: int = 4
    dim: int = 128

    @staticmethod
    def preset(name):
        return {
            "tiny": Config(),
            "small": Config(seq=256, layers=6, heads=6, dim=384),  # 11M
            "medium": Config(seq=512, layers=12, heads=12, dim=768),  # 85M
            "large": Config(seq=512, layers=24, heads=16, dim=1024),  # 302M
        }[name]


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.qkv = nn.Linear(c.dim, 3 * c.dim)
        self.proj = nn.Linear(c.dim, c.dim)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.dim)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.dim)
        self.fc = nn.Linear(c.dim, 4 * c.dim)
        self.out = nn.Linear(4 * c.dim, c.dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class Embed(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.tok = nn.Embedding(c.vocab, c.dim)
        self.pos = nn.Embedding(c.seq, c.dim)

    def forward(self, idx):
        return self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))


class Head(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c.dim)
        self.lm = nn.Linear(c.dim, c.vocab, bias=False)

    def forward(self, x):
        return self.lm(self.ln(x))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embed = Embed(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.head = Head(c)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2's scaled init for residual projections
            for lin in (b.attn.proj, b.out):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def units(self):
        """The model as a sequence of modules: what ZeRO-3 shards one at a
        time and pipeline parallelism splits across stages."""
        return [self.embed, *self.blocks, self.head]

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(x)
        if targets is None:
            return logits
        return loss_fn(logits, targets)


def loss_fn(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
