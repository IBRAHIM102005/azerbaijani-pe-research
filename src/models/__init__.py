"""Model and positional-encoding stage (M2) of the Azerbaijani PE study."""

from .config import ARM_ALLOWLIST, PE_OPERATIONAL_FIELDS, PE_TYPES, TEMPLATE_SEED, ModelConfig
from .params import (
    fairness_report,
    format_fairness_table,
    parameter_report,
    shared_init_fingerprint,
)
from .positional import (
    AlibiPositionalEncoding,
    LearnedPositionalEncoding,
    NoPositionalEncoding,
    PositionalScheme,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
    alibi_slopes,
    build_positional_scheme,
    sinusoidal_table,
)
from .data_contract import DataContract, load_contract, sha256_file
from .gates import GATE_LOSS_THRESHOLD, gate_batch, gate_config, run_arm, run_gate
from .run_config import (
    ResolvedRun,
    config_sha256,
    iter_runs,
    load_run_matrix,
    resolve_run_config,
)
from .transformer import PELanguageModel, build_model, init_deterministic

__all__ = [
    "PE_TYPES",
    "ARM_ALLOWLIST",
    "PE_OPERATIONAL_FIELDS",
    "TEMPLATE_SEED",
    "ModelConfig",
    "PELanguageModel",
    "build_model",
    "init_deterministic",
    "PositionalScheme",
    "NoPositionalEncoding",
    "LearnedPositionalEncoding",
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEncoding",
    "AlibiPositionalEncoding",
    "build_positional_scheme",
    "alibi_slopes",
    "sinusoidal_table",
    "parameter_report",
    "shared_init_fingerprint",
    "fairness_report",
    "format_fairness_table",
    "DataContract",
    "load_contract",
    "sha256_file",
    "ResolvedRun",
    "resolve_run_config",
    "iter_runs",
    "load_run_matrix",
    "config_sha256",
    "gate_config",
    "gate_batch",
    "run_arm",
    "run_gate",
    "GATE_LOSS_THRESHOLD",
]
