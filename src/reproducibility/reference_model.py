"""SYNTHETIC_FIXTURE reference model.

It is a small, honest Pythia-style causal decoder
implementing the same five PE injection points described in the project
plan, used only so `scripts/audit_parameters.py`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

PE_TYPES = ("learned", "sinusoidal", "rope", "alibi", "nope")


def _sinusoidal_table(max_len: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    def _pow2_slopes(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = _pow2_slopes(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        slopes = _pow2_slopes(closest)
        extra = _pow2_slopes(2 * closest)[0::2][: num_heads - closest]
        slopes = slopes + extra
    return torch.tensor(slopes, dtype=torch.float32)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, pe_type: str, rotary_pct: float, max_len: int):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.head_dim = hidden // heads
        self.pe_type = pe_type
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)

        if pe_type == "rope":
            rot_dim = max(2, int(self.head_dim * rotary_pct))
            rot_dim -= rot_dim % 2
            inv_freq = 1.0 / (10000 ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
            t = torch.arange(max_len, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("rope_cos", emb.cos(), persistent=False)
            self.register_buffer("rope_sin", emb.sin(), persistent=False)
            self.rot_dim = rot_dim
        else:
            self.rot_dim = 0

        if pe_type == "alibi":
            self.register_buffer("alibi_slopes", _alibi_slopes(heads), persistent=False)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Dh]

        if self.pe_type == "rope" and self.rot_dim > 0:
            cos = self.rope_cos[:L].to(q.dtype)
            sin = self.rope_sin[:L].to(q.dtype)
            rd = self.rot_dim
            q_rot, q_pass = q[..., :rd], q[..., rd:]
            k_rot, k_pass = k[..., :rd], k[..., rd:]
            q_rot = q_rot * cos + _rotate_half(q_rot) * sin
            k_rot = k_rot * cos + _rotate_half(k_rot) * sin
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.pe_type == "alibi":
            pos = torch.arange(L, device=x.device)
            rel = (pos[None, :] - pos[:, None]).clamp(max=0).float()  # j-i, <=0 for j<=i
            bias = self.alibi_slopes.to(x.device).view(1, -1, 1, 1) * rel.view(1, 1, L, L)
            scores = scores + bias

        scores = scores.masked_fill(~self.causal_mask[:L, :L], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn, hidden)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, hidden, heads, ffn, pe_type, rotary_pct, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden, eps=1e-5)
        self.attn = CausalSelfAttention(hidden, heads, pe_type, rotary_pct, max_len)
        self.ln2 = nn.LayerNorm(hidden, eps=1e-5)
        self.ffn = FFN(hidden, ffn)

    def forward(self, x):
        # Pythia-style parallel residual: both attn and ffn read the same
        # ln(x) and are added back to the same residual stream.
        h = self.ln1(x)
        x = x + self.attn(h) + self.ffn(self.ln2(h))
        return x


class ReferenceCausalLM(nn.Module):
    def __init__(self, pe_type: str, config: dict):
        super().__init__()
        if pe_type not in PE_TYPES:
            raise ValueError(f"unknown pe_type {pe_type!r}, expected one of {PE_TYPES}")
        m = config["model"]
        hidden = m["hidden_size"]
        layers = m["num_layers"]
        heads = m["num_attention_heads"]
        ffn = m["ffn_size"]
        vocab = m["vocab_size"]
        max_len = m["context_length"]
        rotary_pct = m.get("rotary_pct", 0.25)

        self.pe_type = pe_type
        self.tok_emb = nn.Embedding(vocab, hidden)

        if pe_type == "learned":
            self.pos_emb = nn.Embedding(max_len, hidden)
        elif pe_type == "sinusoidal":
            self.register_buffer("pos_emb_table", _sinusoidal_table(max_len, hidden), persistent=False)

        self.blocks = nn.ModuleList(
            [Block(hidden, heads, ffn, pe_type, rotary_pct, max_len) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(hidden, eps=1e-5)
        self.head = nn.Linear(hidden, vocab, bias=False)  # untied

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        if self.pe_type == "learned":
            pos = torch.arange(L, device=input_ids.device)
            x = x + self.pos_emb(pos)[None, :, :]
        elif self.pe_type == "sinusoidal":
            x = x + self.pos_emb_table[:L][None, :, :].to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def build_model(pe_type: str, config: dict) -> nn.Module:
    return ReferenceCausalLM(pe_type, config)
