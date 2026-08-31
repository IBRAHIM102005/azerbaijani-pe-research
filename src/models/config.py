"""Model configuration for the positional-encoding comparison.

A single :class:`ModelConfig` describes the whole model.  Every field except
``pe_type`` (and the ALiBi/RoPE hyper-parameters that only one variant reads)
is held fixed across the five experimental arms, so that positional encoding
remains the only manipulated variable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

PE_TYPES = ("learned", "sinusoidal", "rope", "alibi", "nope")


@dataclass
class ModelConfig:
    """Configuration of the shared base model.

    Attributes
    ----------
    pe_type:
        One of ``learned``, ``sinusoidal``, ``rope``, ``alibi``, ``nope``.
        This is the *only* field that is allowed to differ between the five
        experimental arms of the study.
    vocab_size:
        Must match the frozen shared SentencePiece tokenizer (16000 pieces).
    n_layer, n_head, d_model, d_ff:
        Transformer core geometry.  Identical for all arms.
    max_seq_len:
        Training context length.  Learned and sinusoidal tables are built for
        this length; RoPE/ALiBi/NoPE are length-agnostic at construction time.
    rope_theta:
        Base of the RoPE frequency schedule (read only when ``pe_type='rope'``).
    alibi_max_slope_exponent:
        Exponent controlling the ALiBi slope geometric series; the default of
        ``3.0`` reproduces the ratio ``2**(-8i/n)`` of the original paper
        (read only when ``pe_type='alibi'``).
    tie_embeddings:
        Whether the output projection shares weights with the token embedding.
        Fixed across arms; tying keeps the parameter budget comparable.
    init_seed:
        Seed for the deterministic per-parameter initialisation scheme.  Using
        the same seed for all arms guarantees that every *shared* parameter
        receives bit-identical initial values regardless of whether the arm
        also owns a learned positional table.
    """

    # --- experimental variable -------------------------------------------
    pe_type: str = "nope"

    # --- fixed transformer core ------------------------------------------
    vocab_size: int = 16000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = True

    # --- PE-specific hyper-parameters ------------------------------------
    rope_theta: float = 10000.0
    sinusoidal_theta: float = 10000.0
    alibi_max_slope_exponent: float = 3.0

    # --- reproducibility --------------------------------------------------
    init_seed: int = 2026
    init_std: float = 0.02

    # --- free-form provenance --------------------------------------------
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pe_type not in PE_TYPES:
            raise ValueError(
                f"unknown pe_type {self.pe_type!r}; expected one of {PE_TYPES}"
            )
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        if self.pe_type == "rope" and self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim, got {self.head_dim}"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def with_pe(self, pe_type: str) -> "ModelConfig":
        """Return a copy of this config with a different positional encoding."""
        payload = self.to_dict()
        payload["pe_type"] = pe_type
        return ModelConfig.from_dict(payload)
