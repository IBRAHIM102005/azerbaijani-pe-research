"""Model configuration for the positional-encoding comparison.

A single :class:`ModelConfig` describes the whole model.  Every field except
``pe_type`` (and the ALiBi/RoPE hyper-parameters that only one variant reads)
is held fixed across the five experimental arms, so that positional encoding
remains the only manipulated variable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

PE_TYPES = ("learned", "sinusoidal", "rope", "alibi", "nope")
INIT_SCHEMES = ("pythia",)

#: ``init_seed`` value meaning "this is a template, not a runnable config".
#: Production seeds come from ``configs/run_matrix.json`` via
#: :func:`src.models.run_config.resolve_run_config`.  Building a model from an
#: unresolved config raises, so a template seed can never silently become the
#: seed of a real run.
TEMPLATE_SEED = -1

#: Fields that are allowed to differ between the five experimental arms.
#: Everything else must be byte-identical; enforced by the config-contract test.
ARM_ALLOWLIST = frozenset({"pe_type"})

#: Arm-specific operational fields that are *read* by only one arm.  They are
#: identical in every shipped config (so no arm is secretly different), but are
#: listed separately because they are meaningless outside their own arm.
PE_OPERATIONAL_FIELDS = frozenset({
    "rope_theta",
    "rotary_pct",
    "sinusoidal_theta",
    "alibi_max_slope_exponent",
})


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
    rotary_pct:
        Fraction of each head's channels that RoPE rotates; the remainder is
        passed through untouched.  GPT-NeoX/Pythia use ``0.25``.  ``1.0`` is
        full rotation as in the original RoFormer paper.  Read only when
        ``pe_type='rope'``, but recorded in every config so the value is
        never implicit.
    alibi_max_slope_exponent:
        Exponent controlling the ALiBi slope geometric series; the default of
        ``3.0`` reproduces the ratio ``2**(-8i/n)`` of the original paper
        (read only when ``pe_type='alibi'``).
    tie_embeddings:
        Whether the output projection shares weights with the token embedding.
        The frozen spec requires untied embeddings (``False``), so the model
        carries a separate output head.  Fixed across arms either way.
    init_seed:
        Seed for the deterministic per-parameter initialisation scheme.  Using
        the same seed for all arms guarantees that every *shared* parameter
        receives bit-identical initial values regardless of whether the arm
        also owns a learned positional table.
    init_scheme:
        ``"pythia"`` (the frozen spec) applies **small_init** to embeddings and
        input projections and **wang_init** to the two residual output
        projections, matching GPT-NeoX/Pythia's ``init_method`` and
        ``output_layer_init_method``:

        ``small_init std = sqrt(2 / (5 * d_model))``
        ``wang_init  std = 2 / (n_layer * sqrt(d_model))``

        ``"pythia"`` is the only accepted value.  A second scheme was removed
        rather than left in place: no config used it, and a dormant
        initialisation path is exactly the kind of thing that gets selected by
        accident and silently changes what the study measures.
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
    tie_embeddings: bool = False

    # --- packed-document position policy (identical in every arm) --------
    reset_position_ids: bool = False
    reset_attention_mask: bool = False

    # --- PE-specific hyper-parameters ------------------------------------
    rope_theta: float = 10000.0
    rotary_pct: float = 0.25
    sinusoidal_theta: float = 10000.0
    alibi_max_slope_exponent: float = 3.0

    # --- reproducibility --------------------------------------------------
    init_seed: int = TEMPLATE_SEED
    data_seed: int = 2026
    init_scheme: str = "pythia"

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
        if self.init_seed != TEMPLATE_SEED and self.init_seed < 0:
            raise ValueError(
                f"init_seed must be >= 0 or the template sentinel "
                f"{TEMPLATE_SEED}, got {self.init_seed}"
            )
        if self.init_scheme not in INIT_SCHEMES:
            raise ValueError(
                f"unknown init_scheme {self.init_scheme!r}; "
                f"expected one of {INIT_SCHEMES}"
            )
        if self.pe_type == "rope":
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"RoPE requires an even head_dim, got {self.head_dim}"
                )
            if not 0.0 < self.rotary_pct <= 1.0:
                raise ValueError(
                    f"rotary_pct must be in (0, 1], got {self.rotary_pct}"
                )
            if self.rotary_dim % 2 != 0 or self.rotary_dim == 0:
                raise ValueError(
                    f"rotary_pct={self.rotary_pct} on head_dim={self.head_dim} "
                    f"gives {self.rotary_dim} rotated channels; it must be even "
                    f"and non-zero"
                )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def rotary_dim(self) -> int:
        """Number of channels per head that RoPE actually rotates."""
        return int(self.head_dim * self.rotary_pct)

    @property
    def small_init_std(self) -> float:
        """GPT-NeoX ``init_method``: Nguyen & Salazar (2019) small init."""
        return math.sqrt(2.0 / (5.0 * self.d_model))

    @property
    def wang_init_std(self) -> float:
        """GPT-NeoX ``output_layer_init_method``: Wang / GPT-J init."""
        return 2.0 / (self.n_layer * math.sqrt(self.d_model))

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

    @property
    def is_template(self) -> bool:
        """True when no production seed has been resolved into this config."""
        return self.init_seed == TEMPLATE_SEED

    def with_seed(self, init_seed: int) -> "ModelConfig":
        """Return a runnable copy with the model init seed resolved."""
        if init_seed == TEMPLATE_SEED or init_seed < 0:
            raise ValueError(f"{init_seed} is not a production seed")
        payload = self.to_dict()
        payload["init_seed"] = init_seed
        return ModelConfig.from_dict(payload)

    def with_pe(self, pe_type: str) -> "ModelConfig":
        """Return a copy of this config with a different positional encoding."""
        payload = self.to_dict()
        payload["pe_type"] = pe_type
        return ModelConfig.from_dict(payload)