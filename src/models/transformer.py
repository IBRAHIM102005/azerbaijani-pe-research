"""The shared base model.

Every positional-encoding arm uses *this* module, unmodified.  The scheme is
injected as a :class:`~src.models.positional.PositionalScheme` and touched only
through its three hooks, so no arm gets a different attention implementation,
a different residual layout, or a different initialisation rule.
"""

from __future__ import annotations

import math
import zlib
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .positional import PositionalScheme, build_positional_scheme

__all__ = ["CausalSelfAttention", "TransformerBlock", "PELanguageModel", "init_deterministic"]

NEG_INF = float("-inf")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with hooks for RoPE and ALiBi."""

    def __init__(self, config: ModelConfig, pe: PositionalScheme) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        # The scheme is *shared* with the top-level model and every other
        # layer.  Holding it inside a tuple keeps it out of this layer's
        # ``.parameters()``/``.modules()``, so a learned positional table is
        # owned -- and counted -- exactly once, by the model.
        self._pe_ref = (pe,)

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._causal_cache: Optional[torch.Tensor] = None

    @property
    def pe(self) -> PositionalScheme:
        return self._pe_ref[0]

    def _causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        cache = self._causal_cache
        if cache is None or cache.shape[-1] < seq_len or cache.device != device:
            full = torch.full((seq_len, seq_len), NEG_INF, device=device)
            self._causal_cache = torch.triu(full, diagonal=1)
            cache = self._causal_cache
        return cache[:seq_len, :seq_len].to(dtype).view(1, 1, seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (B, H, T, hd)

        # hook 2: RoPE rotates q/k in place of any additive signal
        q, k = self.pe.rotate_qk(q, k)

        # additive attention mask = causal mask (+ hook 3: ALiBi bias)
        attn_mask = self._causal_mask(T, x.device, q.dtype)
        bias = self.pe.attention_bias(T, x.device, q.dtype)
        if bias is not None:
            attn_mask = attn_mask + bias

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff, bias=True)
        self.proj = nn.Linear(config.d_ff, config.d_model, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block with a **parallel** residual:

    ``x + attn(ln1(x)) + mlp(ln2(x))``

    Both branches are computed from the block input, not chained.  This is
    fixed by the frozen spec and is identical in all five arms.
    """

    def __init__(self, config: ModelConfig, pe: PositionalScheme) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config, pe)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel residual: attention and MLP both read the *same* block
        # input, so the MLP does not see the attention output.  Each branch
        # keeps its own LayerNorm (this is the layout the frozen spec's
        # parameter budget corresponds to; a shared-LN variant would be
        # 3,072 parameters smaller).
        return x + self.attn(self.ln1(x)) + self.mlp(self.ln2(x))


class PELanguageModel(nn.Module):
    """Small decoder-only LM whose positional encoding is configurable."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.is_template:
            raise ValueError(
                "this config carries the template seed and is not runnable; "
                "resolve a production seed from configs/run_matrix.json first "
                "(src.models.run_config.resolve_run_config)"
            )
        self.config = config
        self.pe = build_positional_scheme(config)

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, self.pe) for _ in range(config.n_layer)
        )
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        init_deterministic(self, config)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(logits, loss)``; ``loss`` is ``None`` when no labels given.

        Labels are expected already shifted by the data pipeline is *not*
        assumed: this method shifts internally, so ``labels`` may simply be
        ``input_ids``.
        """
        B, T = input_ids.shape
        x = self.wte(input_ids)

        # hook 1: absolute schemes add to the embedding
        pos_emb = self.pe.additive_embedding(T, x.device, x.dtype)
        if pos_emb is not None:
            x = x + pos_emb

        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


# ---------------------------------------------------------------------------
# deterministic initialisation
# ---------------------------------------------------------------------------
def _param_seed(name: str, base_seed: int) -> int:
    """Stable per-parameter seed.

    Python's builtin ``hash`` of a string is salted per process, so CRC32 is
    used instead: the same parameter name yields the same seed on every
    machine and in every run.
    """
    return (base_seed * 1_000_003 + zlib.crc32(name.encode("utf-8"))) % (2**31 - 1)


@torch.no_grad()
def init_deterministic(model: nn.Module, config: ModelConfig) -> None:
    """Seed every parameter independently, by name, using the frozen scheme.

    Two things are happening here.

    **Which distribution.**  The frozen spec is GPT-NeoX/Pythia's pairing:

    * ``small_init`` -- ``std = sqrt(2 / (5 * d_model))`` -- for the token
      embedding, the output head, the learned positional table and the input
      projections (``qkv``, ``mlp.fc``).  From Nguyen & Salazar (2019).
    * ``wang_init`` -- ``std = 2 / (n_layer * sqrt(d_model))`` -- for the two
      projections that write into the residual stream (``attn.proj``,
      ``mlp.proj``).  From Ben Wang's GPT-J.  This is what keeps the residual
      variance from growing with depth; it replaces the ad-hoc
      ``1/sqrt(2*n_layer)`` rescaling of a fixed sigma.

    LayerNorm weights are ones, every bias is zero.

    **Which random numbers.**  Each parameter is seeded from its own name
    (CRC32 of the name mixed with ``init_seed``) rather than from one global
    RNG stream.  With a single stream, allocating the learned arm's positional
    table would consume random numbers and shift the initial values of every
    later layer, so the arms would differ in more than their positional
    encoding.  Per-name seeding makes every shared weight bit-identical across
    all five arms.
    """
    if config.init_scheme != "pythia":       # guarded in ModelConfig too
        raise ValueError(f"unknown init_scheme {config.init_scheme!r}")
    small_std = config.small_init_std
    wang_std = config.wang_init_std

    for name, param in model.named_parameters():
        gen = torch.Generator(device="cpu").manual_seed(_param_seed(name, config.init_seed))

        if name.endswith("bias"):
            param.zero_()
        elif ".ln" in name or name.startswith("ln_f"):
            param.fill_(1.0)
        elif name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
            # residual-writing projections -> wang_init
            param.copy_(torch.normal(0.0, wang_std, param.shape, generator=gen))
        else:
            # embeddings, output head, positional table, input projections
            # -> small_init
            param.copy_(torch.normal(0.0, small_std, param.shape, generator=gen))

    # LayerNorm weights are matched by the ".ln" branch above; make sure any
    # LayerNorm that slipped through is still unit-scaled.
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.weight.fill_(1.0)
            module.bias.zero_()


def build_model(config: ModelConfig) -> PELanguageModel:
    return PELanguageModel(config)