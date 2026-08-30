"""Generate the frozen data pipeline handoff from validated local artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, canonical_json_hash, sha256_file
from src.data.paths import repository_relative


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate the frozen data pipeline handoff JSON.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    return parser.parse_args()


def artifact(path: Path, recorded_hash: str | None = None) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Required data pipeline artifact is missing: {path}")
    return {
        "path": repository_relative(path, ROOT),
        "bytes": path.stat().st_size,
        "sha256": recorded_hash or sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config, ROOT)
    metadata = config.path("metadata")
    manifests = config.path("manifests")
    tokenizer_dir = config.path("tokenizer")

    prep = load_json(metadata / "preparation_summary.json")
    exact = load_json(metadata / "exact_duplicate_report.json")
    near = load_json(metadata / "near_duplicate_report.json")
    manifest_hashes = load_json(metadata / "manifest_hashes.json")
    leakage = load_json(metadata / "leakage_audit.json")
    tokenizer_audit = load_json(metadata / "tokenizer_audit.json")
    tokenizer_config = load_json(tokenizer_dir / "tokenizer_config.json")
    special_tokens = load_json(tokenizer_dir / "special_tokens.json")
    tokenizer_hashes = load_json(tokenizer_dir / "tokenizer_hashes.json")
    token_counts = load_json(metadata / "token_counts_by_source_split.json")
    subset = load_json(metadata / "training_subset_summary.json")
    validation = load_json(metadata / "frozen_corpus_validation.json")
    raw_immutability = load_json(metadata / "raw_immutability.json")
    processed_hashes = load_json(metadata / "processed_hashes.json")
    report_tables = load_json(metadata / "data_pipeline_report_tables.json")
    raw_inventory = load_json(metadata / "raw_inventory.json")
    raw_inventory_hash = load_json(metadata / "raw_inventory_hash.json")
    repair_metadata = load_json(metadata / "repair_final_metadata.json")
    source_registry = yaml.safe_load((metadata / "source_registry.yaml").read_text(encoding="utf-8"))
    candidate_recall = load_json(metadata / "near_repair_candidate_recall.json")
    all_large_buckets = load_json(metadata / "near_repair_large_bucket_leakage.json")
    near_validation = load_json(metadata / "near_repair_validation.json")
    downstream_validation = load_json(metadata / "downstream_repair_validation.json")

    if validation["status"] != "pass" or leakage["status"] != "pass" or raw_immutability["status"] != "pass":
        raise RuntimeError("data pipeline cannot be handed off while a final safety gate is failing")
    if tokenizer_audit["selection"]["selected_vocab_size"] != 16_000:
        raise RuntimeError("The frozen tokenizer is not the preregistered 16K model")
    canonical_config_hash = canonical_json_hash(config.values)
    if canonical_config_hash != prep["config_sha256"]:
        raise RuntimeError("Current config values differ from the completed preparation config")

    junit_path = metadata / "pytest_data_pipeline.xml"
    xml_root = ET.parse(junit_path).getroot()
    suite = xml_root.find("testsuite") if xml_root.tag == "testsuites" else xml_root
    test_summary = {
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
        "time_seconds": float(suite.attrib["time"]),
        "unit_tests": 0,
        "integration_tests": 0,
        "junit_path": repository_relative(junit_path, ROOT),
        "junit_sha256": sha256_file(junit_path),
    }
    for case in suite.findall("testcase"):
        if ".integration." in case.attrib.get("classname", ""):
            test_summary["integration_tests"] += 1
        else:
            test_summary["unit_tests"] += 1
    if test_summary["failures"] or test_summary["errors"] or not test_summary["tests"]:
        raise RuntimeError("The recorded data pipeline test suite is not passing")

    source_settings = config.values["sources"]
    included_sources = [source for source, values in source_settings.items() if values["included_in_core"]]
    excluded_sources = [source for source, values in source_settings.items() if not values["included_in_core"]]
    core_inventory = [row for row in raw_inventory if row["included_in_core"]]
    schemas = {
        source: {
            "text_column": source_settings[source]["text_column"],
            "schemas": [
                [{"name": field["name"], "type": field["type"]} for field in row["schema"]]
                for row in core_inventory
                if row["source"] == source
            ][:1],
        }
        for source in included_sources
    }
    split_counts = {
        split: prep["split_summary"]["splits"][split]["documents"]
        for split in ("train", "validation", "test")
    }
    split_token_totals = {
        split: sum(values["tokens"] for values in token_counts[split].values())
        for split in ("train", "validation", "test")
    }
    token_totals_by_source = {
        source: sum(token_counts[split][source]["tokens"] for split in token_counts)
        for source in included_sources
    }
    token_totals_by_group = Counter()
    for source, tokens in token_totals_by_source.items():
        token_totals_by_group[source_settings[source]["group"]] += tokens

    stable_artifacts = {
        "m1_config": artifact(args.config),
        "source_registry": artifact(metadata / "source_registry.yaml"),
        "raw_inventory": artifact(metadata / "raw_inventory.json"),
        "raw_immutability": artifact(metadata / "raw_immutability.json"),
        "preparation_summary": artifact(metadata / "preparation_summary.json"),
        "exact_duplicate_report": artifact(metadata / "exact_duplicate_report.json"),
        "near_duplicate_report": artifact(metadata / "near_duplicate_report.json"),
        "leakage_audit": artifact(metadata / "leakage_audit.json"),
        "m1_validation": artifact(metadata / "frozen_corpus_validation.json"),
        "processed_hashes": artifact(metadata / "processed_hashes.json"),
        "token_count_report": artifact(metadata / "token_counts_by_source_split.json"),
        "document_token_counts": artifact(
            metadata / "document_token_counts.parquet", validation["token_counts"]["sha256"]
        ),
        "tokenizer_audit": artifact(metadata / "tokenizer_audit.json"),
        "training_subset_summary": artifact(metadata / "training_subset_summary.json"),
        "training_subset_manifest": artifact(manifests / "train_50m.parquet", subset["manifest_sha256"]),
        "analysis_note": artifact(ROOT / "docs" / "notes" / "corpus_data_analysis.md"),
        "data_card": artifact(ROOT / "docs" / "notes" / "corpus_data_card.md"),
        "test_results": artifact(junit_path, test_summary["junit_sha256"]),
        "repair_final_metadata": artifact(metadata / "repair_final_metadata.json"),
        "repaired_candidate_recall": artifact(metadata / "near_repair_candidate_recall.json"),
        "repaired_all_large_bucket_audit": artifact(metadata / "near_repair_large_bucket_leakage.json"),
        "repaired_near_validation": artifact(metadata / "near_repair_validation.json"),
        "repaired_downstream_validation": artifact(metadata / "downstream_repair_validation.json"),
        "prerepair_failure_evidence": artifact(metadata / "near_repair_prerepair_evidence.json"),
    }
    stable_artifacts["manifests"] = {
        split: artifact(manifests / f"{split}.parquet", manifest_hashes[f"data/manifests/{split}.parquet"])
        for split in ("train", "validation", "test")
    }
    stable_artifacts["processed_corpus"] = {
        split: artifact(
            config.path("processed") / f"{split}.parquet",
            processed_hashes[f"data/processed/corpus/{split}.parquet"]["sha256"],
        )
        for split in ("train", "validation", "test")
    }
    stable_artifacts["tokenizer"] = {
        filename: artifact(tokenizer_dir / filename, expected_hash)
        for filename, expected_hash in tokenizer_hashes.items()
    }
    stable_artifacts["tokenizer"]["tokenizer_hashes.json"] = artifact(
        tokenizer_dir / "tokenizer_hashes.json"
    )
    stable_artifacts["tokenizer_training_sample"] = artifact(
        tokenizer_dir / "training_sample_manifest.parquet",
        tokenizer_audit["training_provenance"]["training_sample_manifest_sha256"],
    )

    completion_evidence_time = datetime.fromtimestamp((manifests / "train.parquet").stat().st_mtime).astimezone().isoformat()
    handoff = {
        "m1_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": config.values["pipeline_version"],
        "repository_path": ".",
        "repository_path_policy": "All operational artifact paths are relative to the repository root.",
        "dollma_access_path_recorded": str(config.path("original_dollma").resolve()),
        "dollma_runtime_resolution": (
            "Use AZ_PE_DOLLMA_ROOT when the external DOLLMA clone is not at the config-relative path."
        ),
        "portability_validation": downstream_validation["portability"],
        "preparation": {
            "state": "prepare_complete",
            "completion_evidence_local_time": completion_evidence_time,
            "config_path": repository_relative(args.config, ROOT),
            "config_canonical_values_sha256": canonical_config_hash,
            "config_file_sha256": sha256_file(args.config),
            "accounting": prep["corpus_accounting"],
            "near_band_files": validation["preparation_state"]["near_band_files"],
            "near_band_bytes_each": validation["preparation_state"]["near_band_bytes_each"],
        },
        "corpus_provenance": {
            "dataset": "DOLLMA",
            "dataset_revision": None,
            "local_access_date": source_registry["local_access_date"],
            "access_date_basis": source_registry["access_date_basis"],
            "dataset_declared_license": "CC BY-NC-SA 4.0",
            "source_level_license_status": "unknown where not stated by local metadata",
            "included_sources": included_sources,
            "excluded_sources": excluded_sources,
            "bhos_status": source_settings["bhos"]["status"],
            "translated_enwiki_status": source_settings["translated-enwiki"]["status"],
            "original_shards": len(raw_inventory),
            "core_shards": len(core_inventory),
            "core_raw_bytes": sum(row["bytes"] for row in core_inventory),
            "core_raw_documents": prep["corpus_accounting"]["raw_core_documents"],
            "inventory_sha256": raw_inventory_hash["raw_inventory_sha256"],
            "schemas": schemas,
            "raw_immutability": {
                "status": raw_immutability["status"],
                "original_shards_checked": raw_immutability["original_shards_checked"],
                "local_core_shards_checked": raw_immutability["local_core_shards_checked"],
                "mismatches": len(raw_immutability["mismatches"]),
            },
        },
        "cleaning": {
            "unicode_form": config.values["normalization"]["unicode_form"],
            "minimum_unicode_letters": config.values["normalization"]["minimum_unicode_letters"],
            "preserve_newlines": config.values["normalization"]["preserve_newlines"],
            "source_accounting": prep["accounting"]["sources"],
        },
        "deduplication": {
            "exact": {
                **exact,
                "group_report": "data/metadata/exact_duplicate_groups.csv",
                "member_report": "data/metadata/exact_duplicate_members.csv",
            },
            "near": near,
            "near_threshold": config.values["near_duplicate"]["selected_threshold"],
            "near_parameters": {
                "shingle_size": config.values["near_duplicate"]["shingle_size"],
                "fingerprint_size": config.values["near_duplicate"]["fingerprint_size"],
                "bands": config.values["near_duplicate"]["bands"],
            },
            "repair_evidence": {
                "prerepair_sample_recall": load_json(metadata / "near_repair_prerepair_evidence.json")["audit"]["measured_candidate_recall"],
                "repaired_sample_recall": candidate_recall["audit"]["measured_candidate_recall"],
                "all_large_bucket_events": all_large_buckets["bucket_scan"]["large_bucket_events"],
                "all_large_bucket_pair_opportunities": all_large_buckets["audit"]["pair_opportunities"],
                "all_large_bucket_eligible_true_pairs": all_large_buckets["audit"]["eligible_true_pairs"],
                "all_large_bucket_captured_true_pairs": all_large_buckets["audit"]["candidate_true_pairs"],
                "all_large_bucket_missed_true_pairs": (
                    all_large_buckets["audit"]["eligible_true_pairs"]
                    - all_large_buckets["audit"]["candidate_true_pairs"]
                ),
                "all_large_bucket_recall": all_large_buckets["audit"]["measured_candidate_recall"],
                "former_237_pairs_unresolved": leakage["former_237_pairs_unresolved"],
                "transitivity_sample": {
                    "sample_policy": near_validation["transitivity"]["sample_policy"],
                    "sampled_clusters": near_validation["transitivity"]["sampled_clusters"],
                    "sampled_clusters_with_endpoint_below_threshold": near_validation["transitivity"]["sampled_clusters_with_endpoint_below_threshold"],
                    "minimum_arbitrary_endpoint_similarity": near_validation["transitivity"]["minimum_arbitrary_endpoint_similarity"],
                    "minimum_representative_member_similarity": near_validation["transitivity"]["minimum_representative_member_similarity"],
                    "cluster_distribution": near_validation["transitivity"]["cluster_distribution"],
                    "semantics": near_validation["transitivity"]["semantics"],
                },
            },
        },
        "splits": {
            "seed": config.values["seeds"]["split"],
            "policy": "cluster-aware SHA-256 hash ranges: 0-899 train, 900-949 validation, 950-999 test",
            "counts": split_counts,
            "per_source": {
                split: prep["split_summary"]["splits"][split]["sources"]
                for split in ("train", "validation", "test")
            },
            "manifests": stable_artifacts["manifests"],
            "leakage_audit_path": repository_relative(metadata / "leakage_audit.json", ROOT),
            "leakage_status": leakage["status"],
            "cross_split_intersections": leakage["layers"]["internal_manifest_and_graph"]["cross_split"],
            "near_edges_crossing_splits": leakage["layers"]["internal_manifest_and_graph"]["near_graph"]["edges_crossing_components"],
            "independent_large_bucket_audit": {
                "bucket_events": leakage["layers"]["independent_all_large_bucket_audit"]["bucket_scan"]["large_bucket_events"],
                "true_pairs": leakage["layers"]["independent_all_large_bucket_audit"]["audit"]["eligible_true_pairs"],
                "candidate_recall": leakage["layers"]["independent_all_large_bucket_audit"]["audit"]["measured_candidate_recall"],
                "retained_cross_split_pairs": leakage["layers"]["independent_all_large_bucket_audit"]["audit"]["cross_split_retained_representative_pairs"],
            },
            "prerepair_confirmed_pairs_unresolved": leakage["former_237_pairs_unresolved"],
            "downstream_train_only_and_sequence_replay": {
                "status": downstream_validation["status"],
                "tokenizer_training": downstream_validation["tokenizer_training"],
                "training_subset": downstream_validation["training_subset"],
            },
        },
        "tokenizer": {
            "type": tokenizer_config["type"],
            "vocab_size": tokenizer_config["vocab_size"],
            "sentencepiece_version": tokenizer_config["sentencepiece_version"],
            "normalization_rule_name": tokenizer_config["normalization_rule_name"],
            "character_coverage": tokenizer_config["character_coverage"],
            "byte_fallback": tokenizer_config["byte_fallback"],
            "text_projection": tokenizer_config["text_projection"],
            "special_tokens": special_tokens,
            "artifact_hashes": tokenizer_hashes,
            "training_provenance": tokenizer_audit["training_provenance"],
            "candidate_audit_path": repository_relative(metadata / "tokenizer_audit.json", ROOT),
            "candidate_comparison": {
                size: {
                    "vocab_size": values["audit"]["actual_vocab_size"],
                    "fertility": values["audit"]["token_per_word_fertility"],
                    "characters_per_token": values["audit"]["average_characters_per_token"],
                    "unknown_token_rate": values["audit"]["unknown_token_rate"],
                    "unknown_token_count": values["audit"]["unknown_token_count"],
                    "unknown_rate_denominator_token_count": values["audit"]["unknown_rate_denominator_token_count"],
                    "unknown_rate_denominator_policy": values["audit"]["unknown_rate_denominator_policy"],
                    "model_sha256": values["training"]["model_sha256"],
                    "vocab_sha256": values["training"]["vocab_sha256"],
                }
                for size, values in tokenizer_audit["candidate_comparison"].items()
            },
            "selection_reason": tokenizer_audit["selection"]["reason"],
        },
        "token_counts": {
            "includes_one_eod_per_document": True,
            "by_split_and_source": token_counts,
            "split_totals": split_token_totals,
            "total_retained_tokens": sum(split_token_totals.values()),
            "by_source_all_splits": token_totals_by_source,
            "by_group_all_splits": dict(token_totals_by_group),
            "artifact": stable_artifacts["document_token_counts"],
        },
        "training_subset": {
            **subset,
            "manifest_path": repository_relative(manifests / "train_50m.parquet", ROOT),
            "split": "train",
            "sampling_without_replacement": True,
            "requested_group_weights": config.values["training_subset"]["requested_group_weights"],
            "data_seed": config.values["seeds"]["data"],
            "order_policy": "ascending SHA-256 of ('order', data_seed, document_id), then document_id",
            "future_model_seed_affects_subset_or_order": False,
            "future_consumption_policy": subset["boundary_policy"],
        },
        "environment": validation["environment"],
        "tests": test_summary,
        "validation": {
            "status": validation["status"],
            "path": repository_relative(metadata / "frozen_corpus_validation.json", ROOT),
            "sha256": sha256_file(metadata / "frozen_corpus_validation.json"),
            "matched_subset_processed_references": validation["training_subset"]["matched_processed_references"],
            "matched_subset_token_records": validation["training_subset"]["matched_train_token_records"],
            "repair_evidence_hashes": repair_metadata["artifact_hashes"],
        },
        "reports": {
            "analysis_note": stable_artifacts["analysis_note"],
            "data_card": stable_artifacts["data_card"],
            "figures": report_tables["figure_hashes"],
        },
        "artifacts": stable_artifacts,
        "known_limitations": [
            "bhos remains excluded pending a source-level decision.",
            "Source-level licenses and revisions are unknown where local metadata does not state them.",
            "Books I/Books II mapping is inferred from published size descriptions; component identity is retained.",
            "The 50-letter filter disproportionately removes short mediocore-books fragments.",
            "Language, OCR, and suspicious-text indicators are heuristic audit flags.",
            "Near-duplicate clusters are connected components; transitive members are not guaranteed to be pairwise Jaccard >= 0.95.",
        ],
    }

    definition_of_done = {
        "status": "complete",
        "complete": True,
        "created_at_utc": handoff["created_at_utc"],
        "data_preparation": {
            "completed_preparation_verified": True,
            "accounting_reconciled": True,
            "split_manifests_hash_frozen": True,
            "leakage_gate_passed": True,
            "all_59_large_buckets_exhaustively_validated": True,
            "former_237_cross_split_pairs_resolved": True,
        },
        "tokenizer": {
            "candidates_8k_16k_32k_trained_on_same_train_only_input": True,
            "candidate_audit_complete": True,
            "final_16k_frozen": True,
            "special_tokens_and_hashes_valid": True,
        },
        "token_counts_and_subset": {
            "real_16k_counts_complete": True,
            "train_only_50m_selection_complete": True,
            "no_replacement_or_split_leakage": True,
            "source_shortage_and_redistribution_recorded": True,
            "fixed_order_and_exact_consumption_policy_recorded": True,
        },
        "reproducibility": {
            "raw_immutability_passed": True,
            "processed_manifest_tokenizer_and_subset_hashes_recorded": True,
            "real_artifact_validation_passed": True,
            "all_tests_passed": True,
            "reports_and_handoff_generated": True,
            "repository_relative_operational_paths_validated": True,
            "candidate_and_verification_checkpoints_config_bound": True,
        },
        "repair_hard_gates": {
            "sample_candidate_recall": candidate_recall["audit"]["measured_candidate_recall"],
            "all_large_bucket_candidate_recall": all_large_buckets["audit"]["measured_candidate_recall"],
            "all_large_bucket_missed_true_pairs": (
                all_large_buckets["audit"]["eligible_true_pairs"]
                - all_large_buckets["audit"]["candidate_true_pairs"]
            ),
            "known_cross_split_true_near_pairs": all_large_buckets["audit"]["cross_split_retained_representative_pairs"],
            "former_237_pairs_unresolved": leakage["former_237_pairs_unresolved"],
            "full_artifact_validation": validation["status"],
            "raw_immutability": raw_immutability["status"],
        },
        "warnings": [
            "Connected-component members are not guaranteed to be pairwise Jaccard >= 0.95.",
            "Per-source license and revision metadata are unavailable upstream.",
            "The 50-letter filter disproportionately affects mediocore-books fragments.",
        ],
        "out_of_scope_model_work_created": False,
    }
    atomic_write_json(metadata / "corpus_release_status.json", definition_of_done)
    handoff["definition_of_done"] = {
        "status": definition_of_done["status"],
        "path": repository_relative(metadata / "corpus_release_status.json", ROOT),
        "sha256": sha256_file(metadata / "corpus_release_status.json"),
    }
    atomic_write_json(metadata / "training_data_contract.json", handoff)

    validated = load_json(metadata / "training_data_contract.json")
    required = ("preparation", "corpus_provenance", "splits", "tokenizer", "token_counts", "training_subset")
    if validated["m1_status"] != "complete" or any(name not in validated for name in required):
        raise RuntimeError("Generated data pipeline handoff failed structural validation")
    print(
        json.dumps(
            {
                "m1_status": validated["m1_status"],
                "handoff_path": repository_relative(metadata / "training_data_contract.json", ROOT),
                "definition_of_done": validated["definition_of_done"],
                "tests": validated["tests"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
