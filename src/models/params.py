"""Parameter accounting and the cross-arm fairness check.

Two things have to be demonstrable at defence time:

1. **Parameter parity.**  The transformer core is identical in every arm.  The
   learned-absolute arm additionally owns ``max_seq_len * d_model`` positional
   parameters; that difference is intrinsic to the method, so it is reported
   in its own column instead of being buried in a single total.
2. **Initialisation parity.**  Every parameter that the arms share starts from
   bit-identical values.  ``shared_init_fingerprint`` produces a hash that is
   equal across all five arms exactly when this holds.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .config import PE_TYPES, ModelConfig
from .transformer import PELanguageModel

__all__ = [
    "parameter_report",
    "shared_init_fingerprint",
    "fairness_report",
    "format_fairness_table",
]


def _is_positional(name: str) -> bool:
    return name.startswith("pe.")


def _is_input_embedding(name: str) -> bool:
    return name == "wte.weight"


def _is_output_head(name: str) -> bool:
    return name == "lm_head.weight"


def parameter_report(model: PELanguageModel) -> Dict[str, object]:
    """Break the parameter count down into comparable buckets."""
    total = positional = input_embedding = output_head = core = 0
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if id(param) in seen:      # if tied, lm_head/wte are counted once
            continue
        seen.add(id(param))
        n = param.numel()
        total += n
        if _is_positional(name):
            positional += n
        elif _is_input_embedding(name):
            input_embedding += n
        elif _is_output_head(name):
            output_head += n
        else:
            core += n

    cfg = model.config
    return {
        "pe_type": cfg.pe_type,
        "total": total,
        "core": core,                        # blocks + final layer norm
        "input_embedding": input_embedding,
        "output_head": output_head,
        "token_embedding": input_embedding + output_head,
        "positional": positional,
        "non_embedding": total - input_embedding - output_head,
        "tied_embeddings": cfg.tie_embeddings,
        "d_model": cfg.d_model,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "max_seq_len": cfg.max_seq_len,
    }


def shared_init_fingerprint(model: nn.Module) -> str:
    """SHA-256 over every parameter the arms have in common, in name order.

    Positional parameters are excluded because only one arm has them.  Equal
    fingerprints across arms prove the initial weights differ *nowhere* except
    in the positional encoding itself.
    """
    digest = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if _is_positional(name):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(param.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()


def fairness_report(
    base_config: ModelConfig, pe_types: Iterable[str] = PE_TYPES
) -> Dict[str, object]:
    """Build every arm, compare the buckets, and flag any violation."""
    rows: List[Dict[str, object]] = []
    fingerprints: Dict[str, str] = {}

    for pe_type in pe_types:
        model = PELanguageModel(base_config.with_pe(pe_type))
        row = parameter_report(model)
        row["init_fingerprint"] = shared_init_fingerprint(model)[:16]
        fingerprints[pe_type] = row["init_fingerprint"]
        rows.append(row)
        del model

    core_counts = {r["core"] for r in rows}
    emb_counts = {r["token_embedding"] for r in rows}
    head_counts = {r["output_head"] for r in rows}
    fp_values = set(fingerprints.values())

    violations: List[str] = []
    if len(core_counts) != 1:
        violations.append(f"transformer core parameter counts differ: {core_counts}")
    if len(emb_counts) != 1:
        violations.append(f"token embedding parameter counts differ: {emb_counts}")
    if len(head_counts) != 1:
        violations.append(f"output head parameter counts differ: {head_counts}")
    if len(fp_values) != 1:
        violations.append(
            "shared-parameter initialisation differs between arms: " f"{fingerprints}"
        )

    parametric = [r["pe_type"] for r in rows if r["positional"] > 0]

    return {
        "rows": rows,
        "core_parameters": next(iter(core_counts)) if len(core_counts) == 1 else None,
        "arms_with_positional_parameters": parametric,
        "violations": violations,
        "passed": not violations,
    }


def format_fairness_table(report: Dict[str, object]) -> str:
    """Render the report as a fixed-width table for the paper / defence."""
    header = (
        f"{'PE':<12}{'total':>14}{'core':>13}{'in-emb':>12}{'out-head':>12}"
        f"{'pos':>10}{'init':>18}"
    )
    lines = [header, "-" * len(header)]
    for row in report["rows"]:                                   # type: ignore[index]
        lines.append(
            f"{row['pe_type']:<12}"
            f"{row['total']:>14,}"
            f"{row['core']:>13,}"
            f"{row['input_embedding']:>12,}"
            f"{row['output_head']:>12,}"
            f"{row['positional']:>10,}"
            f"{row['init_fingerprint']:>18}"
        )
    lines.append("")
    if report["passed"]:
        lines.append(
            "PASS: identical core and embedding budgets; identical shared init."
        )
        extra = report["arms_with_positional_parameters"]         # type: ignore[index]
        if extra:
            lines.append(
                f"Note: {', '.join(extra)} additionally owns a positional table; "
                "this is intrinsic to the method and is reported separately."
            )
    else:
        lines.append("FAIL:")
        lines.extend(f"  - {v}" for v in report["violations"])    # type: ignore[index]
    return "\n".join(lines)
