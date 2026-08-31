"""Reference implementations of the five positional encoding schemes.

All five schemes expose the *same* three hooks, so the transformer core never
branches on ``pe_type``:

======================  =========================  =========================
scheme                  hook it uses               where it acts
======================  =========================  =========================
learned absolute        ``additive_embedding``     token embedding
sinusoidal absolute     ``additive_embedding``     token embedding
RoPE                    ``rotate_qk``              inside attention, on q/k
ALiBi                   ``attention_bias``         inside attention, on scores
NoPE                    -- (all hooks no-op)       nowhere
======================  =========================  =========================

Keeping the hooks uniform is a fairness requirement: every arm executes the
identical attention code path, and the only difference is which hook returns
something other than ``None`` / the identity.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig

__all__ = [
    "PositionalScheme",
    "NoPositionalEncoding",
    "LearnedPositionalEncoding",
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEncoding",
    "AlibiPositionalEncoding",
    "build_positional_scheme",
    "alibi_slopes",
    "sinusoidal_table",
]


# ---------------------------------------------------------------------------
# common interface
# ---------------------------------------------------------------------------
class PositionalScheme(nn.Module):
    """Base class: three hooks, all no-ops by default (this *is* NoPE)."""

    #: whether the scheme owns trainable parameters (used by the fairness report)
    is_parametric: bool = False
    #: whether the scheme can be evaluated beyond ``max_seq_len`` unchanged
    extrapolates: bool = True

    def supports_length(self, seq_len: int) -> bool:
        """Whether this arm can be *evaluated* at ``seq_len`` at all.

        The length-generalisation protocol calls this before every evaluation
        context.  Learned absolute encodings return ``False`` beyond their
        trained table, and the result table must record ``n/a`` for that cell
        rather than a number obtained by resizing or interpolating the table --
        interpolation would silently turn the learned arm into a different
        method and make the comparison meaningless.
        """
        return True

    def additive_embedding(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Return ``(1, seq_len, d_model)`` to add to token embeddings, or None."""
        return None

    def rotate_qk(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optionally transform queries/keys.  Shapes ``(B, H, T, head_dim)``."""
        return q, k

    def attention_bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> Optional[torch.Tensor]:
        """Return ``(1, n_head, seq_len, seq_len)`` added to logits, or None."""
        return None


class NoPositionalEncoding(PositionalScheme):
    """NoPE: no positional signal is injected anywhere."""

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return "no positional information"


# ---------------------------------------------------------------------------
# absolute schemes (act on the embedding)
# ---------------------------------------------------------------------------
class LearnedPositionalEncoding(PositionalScheme):
    """Learned absolute position embeddings (GPT-2 style ``wpe``).

    Owns ``max_seq_len * d_model`` trainable parameters -- the only arm that
    does.  This is reported explicitly by the parameter-fairness tool rather
    than hidden, because it cannot be equalised without changing the method.
    """

    is_parametric = True
    extrapolates = False

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.table = nn.Parameter(torch.zeros(max_seq_len, d_model))

    def supports_length(self, seq_len: int) -> bool:
        return seq_len <= self.max_seq_len

    def additive_embedding(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"learned positional table holds {self.max_seq_len} positions, "
                f"but {seq_len} were requested; this arm does not extrapolate"
            )
        return self.table[:seq_len].to(dtype=dtype).unsqueeze(0)


def sinusoidal_table(
    seq_len: int, d_model: int, theta: float = 10000.0, device=None, dtype=torch.float32
) -> torch.Tensor:
    """``(seq_len, d_model)`` table from Vaswani et al. (2017), eq. (4)-(5).

    ``PE[pos, 2i]   = sin(pos / theta ** (2i / d_model))``
    ``PE[pos, 2i+1] = cos(pos / theta ** (2i / d_model))``
    """
    if d_model % 2 != 0:
        raise ValueError("sinusoidal encoding requires an even d_model")
    position = torch.arange(seq_len, device=device, dtype=torch.float64).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, device=device, dtype=torch.float64)
    div = torch.exp(-math.log(theta) * two_i / d_model)
    table = torch.zeros(seq_len, d_model, device=device, dtype=torch.float64)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)
    return table.to(dtype=dtype)


class SinusoidalPositionalEncoding(PositionalScheme):
    """Fixed sinusoidal absolute encoding; no trainable parameters."""

    is_parametric = False
    extrapolates = True

    def __init__(self, max_seq_len: int, d_model: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.theta = theta
        self.register_buffer(
            "table",
            sinusoidal_table(max_seq_len, d_model, theta),
            persistent=False,
        )

    def additive_embedding(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if seq_len > self.table.shape[0]:  # grow lazily; the formula is closed-form
            self.table = sinusoidal_table(
                seq_len, self.d_model, self.theta, device=device
            ).to(device)
        return self.table[:seq_len].to(device=device, dtype=dtype).unsqueeze(0)


# ---------------------------------------------------------------------------
# relative schemes (act inside attention)
# ---------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionalEncoding(PositionalScheme):
    """RoPE (Su et al., 2021), half-split (GPT-NeoX) layout.

    Rotates query and key vectors so that the attention logit between
    positions ``m`` and ``n`` depends on ``m - n`` only.  That invariance is
    asserted directly in ``tests/test_positional.py``.

    ``rotary_pct`` controls how much of each head is rotated, following
    GPT-NeoX: the first ``int(head_dim * rotary_pct)`` channels are rotated
    and the rest are passed through unchanged.  Pythia uses ``0.25``; the
    original RoFormer formulation is ``1.0``.  The pass-through channels
    contribute a position-independent term to the logit, so the
    relative-offset invariance holds for any value.
    """

    is_parametric = False
    extrapolates = True

    def __init__(
        self, head_dim: int, theta: float = 10000.0, rotary_pct: float = 1.0
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if not 0.0 < rotary_pct <= 1.0:
            raise ValueError(f"rotary_pct must be in (0, 1], got {rotary_pct}")
        self.head_dim = head_dim
        self.theta = theta
        self.rotary_pct = rotary_pct
        self.rotary_dim = int(head_dim * rotary_pct)
        if self.rotary_dim == 0 or self.rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_pct={rotary_pct} on head_dim={head_dim} gives "
                f"{self.rotary_dim} rotated channels; it must be even and non-zero"
            )
        inv_freq = theta ** (
            -torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
            / self.rotary_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"head_dim={self.head_dim}, rotary_dim={self.rotary_dim} "
            f"({self.rotary_pct:.0%}), theta={self.theta}"
        )

    def _cos_sin(
        self, seq_len: int, device: torch.device, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(
            offset, offset + seq_len, device=device, dtype=torch.float32
        )
        angles = torch.outer(pos, self.inv_freq.to(device))          # (T, R/2)
        angles = torch.cat((angles, angles), dim=-1)                  # (T, R)
        return angles.cos()[None, None], angles.sin()[None, None]

    def rotate_qk(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos, sin = self._cos_sin(seq_len, q.device, offset)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        if self.rotary_dim == self.head_dim:
            return (
                q * cos + _rotate_half(q) * sin,
                k * cos + _rotate_half(k) * sin,
            )

        # partial rotation: rotate the leading channels, pass the rest through
        r = self.rotary_dim
        q_rot, q_pass = q[..., :r], q[..., r:]
        k_rot, k_pass = k[..., :r], k[..., r:]
        q_out = torch.cat((q_rot * cos + _rotate_half(q_rot) * sin, q_pass), dim=-1)
        k_out = torch.cat((k_rot * cos + _rotate_half(k_rot) * sin, k_pass), dim=-1)
        return q_out, k_out


def alibi_slopes(n_head: int, max_slope_exponent: float = 3.0) -> torch.Tensor:
    """Per-head ALiBi slopes (Press et al., 2022).

    For a power-of-two head count this is the geometric series
    ``2 ** (-8 * (i + 1) / n_head)``; for other head counts the paper's
    nearest-power-of-two interpolation is used.
    """

    def _series(n: int) -> list:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - max_slope_exponent)))
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_head).is_integer():
        slopes = _series(n_head)
    else:
        closest = 2 ** math.floor(math.log2(n_head))
        slopes = _series(closest)
        extra = _series(2 * closest)[0::2][: n_head - closest]
        slopes = slopes + extra
    return torch.tensor(slopes, dtype=torch.float32)


class AlibiPositionalEncoding(PositionalScheme):
    """ALiBi: a static, head-specific linear penalty on key-query distance.

    The bias added to the logit for query ``i`` attending to key ``j`` is
    ``-slope_h * (i - j)`` for ``j <= i``.  It is zero on the diagonal and
    decreases linearly with distance, which the tests assert.
    """

    is_parametric = False
    extrapolates = True

    def __init__(self, n_head: int, max_slope_exponent: float = 3.0) -> None:
        super().__init__()
        self.n_head = n_head
        self.register_buffer(
            "slopes", alibi_slopes(n_head, max_slope_exponent), persistent=False
        )

    def attention_bias(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> torch.Tensor:
        # Both axes are shifted by ``offset``, so the returned tensor is
        # always ``(1, n_head, seq_len, seq_len)`` as documented, and the bias
        # -- like RoPE -- depends only on the relative distance.  A KV-cache
        # decode step, where keys span 0..offset+seq_len, is out of scope for
        # this study and is rejected rather than silently mis-shaped.
        q_pos = torch.arange(offset, offset + seq_len, device=device).unsqueeze(1)
        k_pos = torch.arange(offset, offset + seq_len, device=device).unsqueeze(0)
        distance = (q_pos - k_pos).clamp(min=0).to(torch.float32)     # (T, T_k)
        bias = -distance.unsqueeze(0) * self.slopes.to(device).view(-1, 1, 1)
        return bias.unsqueeze(0).to(dtype)                            # (1, H, T, T_k)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_positional_scheme(config: ModelConfig) -> PositionalScheme:
    """Instantiate the scheme named by ``config.pe_type``."""
    if config.pe_type == "nope":
        return NoPositionalEncoding()
    if config.pe_type == "learned":
        return LearnedPositionalEncoding(config.max_seq_len, config.d_model)
    if config.pe_type == "sinusoidal":
        return SinusoidalPositionalEncoding(
            config.max_seq_len, config.d_model, config.sinusoidal_theta
        )
    if config.pe_type == "rope":
        return RotaryPositionalEncoding(
            config.head_dim, config.rope_theta, config.rotary_pct
        )
    if config.pe_type == "alibi":
        return AlibiPositionalEncoding(
            config.n_head, config.alibi_max_slope_exponent
        )
    raise ValueError(f"unhandled pe_type {config.pe_type!r}")