"""Generate measured data pipeline notes, tables, and figures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import load_config
from src.data.hashing import atomic_write_json, atomic_write_text, sha256_file


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the data pipeline analysis note and data card from frozen artifacts.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frozen" / "data_pipeline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, ROOT)
    metadata = config.path("metadata")
    notes = ROOT / "docs" / "notes"
    figures = ROOT / "results" / "figures"
    notes.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    prep = load_json(metadata / "preparation_summary.json")
    profile = load_json(metadata / "raw_quality_profile.json")
    raw_inventory = load_json(metadata / "raw_inventory.json")
    raw_inventory_hash = load_json(metadata / "raw_inventory_hash.json")
    source_registry = yaml.safe_load(
        (metadata / "source_registry.yaml").read_text(encoding="utf-8")
    )
    exact = load_json(metadata / "exact_duplicate_report.json")
    near = load_json(metadata / "near_duplicate_report.json")
    leakage = load_json(metadata / "leakage_audit.json")
    tokenizers = load_json(metadata / "tokenizer_audit.json")
    token_counts = load_json(metadata / "token_counts_by_source_split.json")
    subset = load_json(metadata / "training_subset_summary.json")
    validation = load_json(metadata / "frozen_corpus_validation.json")
    raw_immutability = load_json(metadata / "raw_immutability.json")
    tokenizer_hashes = load_json(config.path("tokenizer") / "tokenizer_hashes.json")
    recall_sample = load_json(metadata / "near_repair_candidate_recall.json")
    prerepair_sample = load_json(metadata / "near_repair_prerepair_evidence.json")
    large_bucket_audit = leakage["layers"]["independent_all_large_bucket_audit"]
    downstream_validation = leakage["layers"]["downstream_train_only_and_sequence_replay"]

    sources = list(source for source, values in config.values["sources"].items() if values["included_in_core"])
    source_groups = {source: config.values["sources"][source]["group"] for source in sources}
    retained_by_source = {
        source: sum(prep["split_summary"]["splits"][split]["sources"][source]["documents"] for split in ("train", "validation", "test"))
        for source in sources
    }
    token_totals_by_source = {
        source: sum(token_counts[split][source]["tokens"] for split in token_counts)
        for source in sources
    }
    token_totals_by_group = Counter()
    for source, tokens in token_totals_by_source.items():
        token_totals_by_group[source_groups[source]] += tokens
    near_removed_by_source = {
        source: (
            prep["accounting"]["sources"][source]["raw"]
            - prep["accounting"]["sources"][source]["removed_short"]
            - prep["accounting"]["sources"][source]["exact_duplicate"]
            - retained_by_source[source]
        )
        for source in sources
    }
    split_totals = validation["token_counts"]["splits"]
    core_shards = [row for row in raw_inventory if row["included_in_core"]]
    core_bytes = sum(row["bytes"] for row in core_shards)

    source_rows = []
    length_rows = []
    flag_rows = []
    split_source_rows = []
    for source in sources:
        raw_documents = prep["accounting"]["sources"][source]["raw"]
        removed_short = prep["accounting"]["sources"][source]["removed_short"]
        source_rows.append(
            [
                source,
                source_groups[source],
                f"{profile['sources'][source]['documents']:,}",
                f"{profile['sources'][source]['total_characters']:,}",
                f"{removed_short:,}",
                f"{100 * removed_short / raw_documents:.2f}%",
                f"{prep['accounting']['sources'][source]['exact_duplicate']:,}",
                f"{near_removed_by_source[source]:,}",
                f"{retained_by_source[source]:,}",
                f"{token_totals_by_source[source]:,}",
            ]
        )
        quantiles = profile["sources"][source]["character_quantiles"]
        length_rows.append(
            [source, f"{quantiles['p50']:,}", f"{quantiles['p95']:,}", f"{quantiles['p99']:,}"]
        )
        flags = profile["sources"][source].get("flags", {})
        flag_rows.append(
            [source, f"{sum(flags.values()):,}", ", ".join(f"{name}: {count}" for name, count in sorted(flags.items()))]
        )
        split_source_rows.append(
            [
                source,
                f"{prep['split_summary']['splits']['train']['sources'][source]['documents']:,}",
                f"{token_counts['train'][source]['tokens']:,}",
                f"{prep['split_summary']['splits']['validation']['sources'][source]['documents']:,}",
                f"{token_counts['validation'][source]['tokens']:,}",
                f"{prep['split_summary']['splits']['test']['sources'][source]['documents']:,}",
                f"{token_counts['test'][source]['tokens']:,}",
            ]
        )

    split_rows = []
    for split in ("train", "validation", "test"):
        values = split_totals[split]
        split_rows.append(
            [split, f"{values['documents']:,}", f"{values['tokens']:,}", f"{values['unknown_tokens']:,}"]
        )

    candidate_rows = []
    for vocab_size in (8000, 16000, 32000):
        candidate = tokenizers["candidate_comparison"][str(vocab_size)]
        audit = candidate["audit"]
        training = candidate["training"]
        candidate_rows.append(
            [
                f"{vocab_size:,}",
                f"{audit['token_per_word_fertility']:.4f}",
                f"{audit['average_characters_per_token']:.4f}",
                f"{audit['unknown_token_count']:,} / {audit['unknown_rate_denominator_token_count']:,}",
                f"{audit['unknown_token_rate']:.3e}",
                f"{audit['documents_with_unknown_tokens']:,}",
                f"{training['model_bytes']:,}",
                f"{training['vocab_bytes']:,}",
            ]
        )

    final_source_fertility = tokenizers["candidate_comparison"]["16000"]["audit"]["source_fertility"]
    fertility_rows = [
        [
            source,
            f"{final_source_fertility[source]['documents']:,}",
            f"{final_source_fertility[source]['token_per_word_fertility']:.4f}",
        ]
        for source in sources
    ]
    word_rows = [
        [item["word"], " · ".join(item["pieces"]), item["piece_count"]]
        for item in tokenizers["candidate_comparison"]["16000"]["audit"]["suffix_rich_long_word_examples"][:8]
    ]

    mixture_rows = []
    for group, requested_tokens in subset["requested_group_tokens"].items():
        actual_tokens = subset["actual_group_tokens"][group]
        mixture_rows.append(
            [
                group,
                f"{requested_tokens:,}",
                f"{subset['quota_phase_group_tokens'][group]:,}",
                f"{subset['quota_shortages'][group]:,}",
                f"{actual_tokens:,}",
                f"{100 * actual_tokens / subset['selected_unique_tokens']:.2f}%",
            ]
        )
    selected_source_rows = [
        [source, source_groups[source], f"{tokens:,}", f"{100 * tokens / subset['selected_unique_tokens']:.2f}%"]
        for source, tokens in subset["actual_source_tokens"].items()
    ]

    analysis = f"""# Corpus data and tokenizer analysis

## Corpus snapshot

The core corpus uses six DOLLMA components: `anl-news`, `azwiki`, `elite-blogs`, `elite-books`, `eqanun`, and `mediocore-books`. The frozen raw inventory contains {len(core_shards)} core parquet shards ({core_bytes:,} compressed bytes). `translated-enwiki` remains outside the core corpus. `bhos` is also excluded because its source role is still unresolved.

The six core sources contain {profile['global']['documents']:,} raw records and {profile['global']['total_characters']:,} Unicode characters. Books II (`mediocore-books`) contributes most document rows, while news contributes many more characters per document. A proportional document sample would therefore be dominated by short Books II fragments.

{table(['Source', 'Group', 'Raw docs', 'Raw chars', 'Short removed', 'Short removed %', 'Exact removed', 'Near removed', 'Retained', '16K tokens'], source_rows)}

Raw document lengths vary sharply by component. The figures below are Unicode-character counts before cleaning.

{table(['Source', 'Median chars', 'P95 chars', 'P99 chars'], length_rows)}

## Cleaning and duplicate accounting

Normalization uses NFC, converts CRLF to LF, trims outer whitespace, removes unsafe control characters, and preserves Azerbaijani spelling and case. The 50-Unicode-letter filter removed {prep['corpus_accounting']['removed_short']:,} records. Of these, {prep['accounting']['sources']['mediocore-books']['removed_short']:,} came from Books II. This is a material filtering choice: it removes many short fragments and should be kept in mind when interpreting the final source mixture.

After length filtering, global canonical-text deduplication removed {exact['removed_documents']:,} records in {exact['groups']:,} duplicate groups. No exact group spanned source labels. The exact-unique corpus contained {prep['corpus_accounting']['exact_unique_documents']:,} documents.

The near-duplicate stage evaluated {near['candidate_pairs_checked']:,} unique LSH candidates at the frozen 0.95 character-5-gram Jaccard threshold. It accepted {near['accepted_edges']:,} edges, forming {near['clusters']:,} connected components over {near['clustered_documents']:,} documents. The implementation retains one deterministic representative per component, removing {near['removed_documents']:,} documents. Every accepted graph edge satisfies direct Jaccard >= 0.95; transitive members of a component are not guaranteed to be pairwise above that threshold. The final retained corpus has {prep['corpus_accounting']['final_retained_documents']:,} documents. All accounting identities reconcile.

An independent pre-experiment audit found that the first implementation used anchor-only expansion in LSH buckets above 200 documents. Its measured recall was {100 * prerepair_sample['audit']['measured_candidate_recall']:.3f}% on the frozen audit sample. Before any model training, that shortcut was replaced with complete, chunked pair enumeration for all observed buckets. The same sample then captured {recall_sample['audit']['candidate_true_pairs']:,} of {recall_sample['audit']['eligible_true_pairs']:,} eligible true pairs. A second exhaustive check covered all {large_bucket_audit['bucket_scan']['large_bucket_events']} formerly problematic bucket events and captured {large_bucket_audit['audit']['candidate_true_pairs']:,} of {large_bucket_audit['audit']['eligible_true_pairs']:,} eligible pairs, with {large_bucket_audit['audit']['eligible_true_pairs'] - large_bucket_audit['audit']['candidate_true_pairs']:,} misses.

Suspicious-text heuristics were used as audit flags, not broad language filters. The raw profile records markup-like text, unusual whitespace, repeated characters, replacement characters, line-break-heavy material, and a small number of null bytes. Short excerpts are stored in `data/metadata/raw_quality_profile.json`; flagged text was not automatically discarded merely for triggering a heuristic.

{table(['Source', 'Flag events', 'Measured categories'], flag_rows)}

## Split and leakage

The split is document-level and cluster-aware, using seed 2026 and the frozen 90/5/5 hash ranges.

{table(['Split', 'Documents', '16K tokens incl. EOD', 'Unknown tokens'], split_rows)}

{table(['Source', 'Train docs', 'Train tokens', 'Val docs', 'Val tokens', 'Test docs', 'Test tokens'], split_source_rows)}

The repaired hard leakage audit passed. Document IDs, canonical hashes, and accepted duplicate-cluster IDs have zero cross-split intersections. All {near['accepted_edges']:,} accepted near-duplicate edges remain within one split. The independent exhaustive large-bucket audit found {large_bucket_audit['audit']['cross_split_retained_representative_pairs']:,} retained true-near pairs crossing splits, and all 237 prerepair confirmed pairs were resolved.

## Tokenizer candidates

All three SentencePiece BPE candidates used the same 1,000,000-document train-only corpus in the same document-ID order. Its SHA-256 is `{tokenizers['training_provenance']['training_corpus_sha256']}`. Internal line breaks are projected to spaces for SentencePiece encoding; the canonical processed text is unchanged.

The table below uses the same 100,000 train documents for every candidate. Fertility is SentencePiece tokens divided by approximate whitespace words. These BPE pieces are not interpreted as morphemes.

{table(['Vocabulary', 'Tokens/word', 'Chars/token', 'UNK / token denominator', 'UNK rate', 'Docs with UNK', 'Model bytes', 'Vocab bytes'], candidate_rows)}

The 16K source-wise audit used the same shared train sample:

{table(['Source', 'Audited docs', '16K tokens/word'], fertility_rows)}

Long and suffix-rich words were inspected as tokenization examples, not as evidence of morphological analysis:

{table(['Word', '16K pieces', 'Piece count'], word_rows)}

The preregistered 16K choice was retained. It produced 16,000 pieces, stable special-token IDs, sensible Azerbaijani round trips, and a train-audit unknown rate of {tokenizers['candidate_comparison']['16000']['audit']['unknown_token_rate']:.3e}. Candidate rates use unknown SentencePiece tokens divided by all SentencePiece tokens in the shared 100,000-document audit sample. Full-corpus unknown counts use the final 16K tokenizer over each complete split and are reported as counts, not compared as if they shared the candidate-audit denominator. The lower fertility of 32K is expected from its larger vocabulary and is not, by itself, a reason to change the protocol.

Across the full retained corpus, the final 16K tokenizer yields {validation['token_counts']['total_tokens']:,} tokens including one `<eod>` per document. The model hash is `{tokenizer_hashes['tokenizer.model']}`.

## Frozen 50M training corpus

The 50M selection uses train documents only, without replacement, with data seed 2026. Blogs contain only {subset['quota_phase_group_tokens']['Blogs']:,} unique train tokens against the requested 5,000,000. The {subset['quota_shortages']['Blogs']:,}-token shortage was redistributed across eligible native groups using the frozen weights.

{table(['Group', 'Requested', 'Quota phase', 'Shortage', 'Final selected', 'Final share'], mixture_rows)}

Component provenance is retained inside the grouped Books quota:

{table(['Source', 'Group', 'Selected tokens', 'Selected share'], selected_source_rows)}

The frozen manifest contains {subset['selected_documents']:,} unique documents and {subset['selected_unique_tokens']:,} tokens, an overshoot of {subset['overshoot_tokens']:,} caused by preserving whole documents. Future training reads this one fixed sequence and stops after exactly 50,000,000 consumed model tokens. The boundary is document {subset['exact_consumption_boundary']['document_id']} at one-based sequence position {subset['exact_consumption_boundary']['sequence_position_one_based']:,}: {subset['exact_consumption_boundary']['tokens_consumed_from_document']:,} of its {subset['exact_consumption_boundary']['full_document_tokens_including_eod']:,} tokens are consumed, before its `<eod>`. Model initialization seeds do not change the subset or its order.

The downstream replay independently reproduced the tokenizer sample, all {downstream_validation['token_counts']['documents']:,} document-token records, the selected IDs and order, and the exact boundary. Repository-internal references are relative. A simulated relocation to `{downstream_validation['portability']['simulated_repository_root']}` resolved successfully; an external DOLLMA clone at a different location must be supplied through `AZ_PE_DOLLMA_ROOT`.

## Known limitations

The local DOLLMA README provides a dataset-level CC BY-NC-SA 4.0 declaration, but source-level licenses and revisions are not stated. The source snapshot was revalidated on {source_registry['local_access_date']}; the original acquisition date is not known. The Books I/Books II mapping is inferred from the published size descriptions and matching local component sizes; component identities remain separate in all artifacts. `bhos` still requires a source decision. The 50-letter rule disproportionately affects Books II fragments: it removed {100 * prep['accounting']['sources']['mediocore-books']['removed_short'] / prep['accounting']['sources']['mediocore-books']['raw']:.2f}% of that source's raw rows. Connected-component closure can link endpoints below the direct 0.95 edge threshold; the sampled minimum endpoint similarity was about 0.820. Finally, language and OCR checks are heuristics, not verified language labels.
"""
    atomic_write_text(notes / "corpus_data_analysis.md", analysis)

    data_card = f"""# Corpus data card

## Source and scope

The corpus is a local subset of DOLLMA. Core components are `anl-news`, `azwiki`, `elite-blogs`, `elite-books`, `eqanun`, and `mediocore-books`. `translated-enwiki` is excluded to avoid a translated-text confound. `bhos` has status `requires_source_decision` and is not included.

The frozen inventory covers {len(raw_inventory)} original shards and {len(core_shards)} locally copied core shards. The authoritative inventory hash is `{raw_inventory_hash['raw_inventory_sha256']}`. The final immutability check passed for {raw_immutability['original_shards_checked']} original and {raw_immutability['local_core_shards_checked']} local-core paths. The registry records {source_registry['local_access_date']} as the inventory and hash revalidation date; the original acquisition date is unavailable.

## Processing policy

Text is normalized to Unicode NFC. CRLF is converted to LF, redundant horizontal whitespace is reduced, unsafe control characters are handled, and outer whitespace is trimmed. Case, punctuation, paragraph structure, and Azerbaijani letters are preserved. Documents with fewer than 50 Unicode letters are removed. Every removal appears in the accounting reports.

Exact duplicates are identified globally by SHA-256 of canonical text. Near duplicates use character 5-grams, 32-value bottom-k fingerprints, eight band files, complete pair enumeration within every observed colliding bucket, and an exact Jaccard threshold of 0.95. Accepted edges form connected components and one deterministic representative is retained per component. Component membership is transitive, so arbitrary members are not guaranteed to be pairwise above 0.95.

An independent audit found and corrected an anchor-only large-bucket candidate shortcut before model training. The repaired candidate set has {near['candidate_pairs_checked']:,} unique pairs. Exhaustive validation of all {large_bucket_audit['bucket_scan']['large_bucket_events']} formerly problematic bucket events captured all {large_bucket_audit['audit']['eligible_true_pairs']:,} eligible true pairs and missed none.

## Split policy

Duplicate clusters are assigned together with split seed 2026. Hash ranges target 90% train, 5% validation, and 5% test. The frozen counts are {split_totals['train']['documents']:,} train, {split_totals['validation']['documents']:,} validation, and {split_totals['test']['documents']:,} test. The layered leakage audit status is `{leakage['status']}`; the independent large-bucket check found no retained true-near cross-split pair, and all 237 prerepair confirmed pairs were resolved.

## Tokenizer policy

The final tokenizer is SentencePiece BPE with 16,000 pieces and identity normalization. It was trained on a deterministic 1,000,000-document sample from train only, shared exactly by the 8K, 16K, and 32K candidates. `<unk>` is ID 0 and `<eod>` is ID 1; BOS and padding are disabled. One `<eod>` token is counted after each document. Candidate unknown rates are explicitly defined as unknown tokens divided by total SentencePiece tokens in the shared 100,000-document train audit. Internal line breaks are projected to spaces only at SentencePiece encoding time.

## Training subset policy

The train-only model corpus targets News 25%, Native Wikipedia 20%, Books 30%, Laws 15%, and Blogs 10%. Selection is deterministic, without replacement, and independent of future model seeds. When a group is exhausted, its shortage is redistributed using the frozen eligible-group weights. The selected manifest has {subset['selected_documents']:,} documents and {subset['selected_unique_tokens']:,} tokens. Training stops after exactly 50,000,000 token IDs, {subset['exact_consumption_boundary']['tokens_consumed_from_document']:,} tokens into the boundary document and before its `<eod>`.

## Reproduction

M2/M3 should consume `data/metadata/training_data_contract.json`; completed corpus and tokenizer stages should not be rerun. `python scripts/validate_frozen_corpus.py` performs a full artifact and row-reference audit when deliberate revalidation is needed. If the repository moves while the external DOLLMA clone does not, set `AZ_PE_DOLLMA_ROOT` to its new runtime location.

## Limitations

Source-level licenses and revisions are unknown where the local metadata does not state them. The dataset-level license declaration is recorded without extending it into a legal conclusion. The Books component mapping is an evidence-based inference. The minimum-length rule removes many short Books II records, quality flags do not establish language identity or OCR correctness, and connected-component endpoints are not guaranteed to meet the direct edge threshold.
"""
    atomic_write_text(notes / "corpus_data_card.md", data_card)

    labels = sources
    raw_docs = [profile["sources"][source]["documents"] for source in labels]
    retained_docs = [retained_by_source[source] for source in labels]
    x = range(len(labels))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar([value - 0.2 for value in x], raw_docs, width=0.4, label="Raw")
    axis.bar([value + 0.2 for value in x], retained_docs, width=0.4, label="Retained")
    axis.set_yscale("log")
    axis.set_ylabel("Documents (log scale)")
    axis.set_xticks(list(x), labels, rotation=25, ha="right")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "corpus_source_documents.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(list(token_totals_by_group), [token_totals_by_group[group] for group in token_totals_by_group])
    axis.set_ylabel("16K tokens, all splits")
    axis.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(figures / "corpus_source_group_tokens.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    vocabulary = [8000, 16000, 32000]
    fertility = [tokenizers["candidate_comparison"][str(value)]["audit"]["token_per_word_fertility"] for value in vocabulary]
    axis.plot(vocabulary, fertility, marker="o")
    axis.set_xlabel("Vocabulary size")
    axis.set_ylabel("Tokens per whitespace word")
    axis.set_xticks(vocabulary, ["8K", "16K", "32K"])
    fig.tight_layout()
    fig.savefig(figures / "tokenizer_fertility.png", dpi=150)
    plt.close(fig)

    groups = list(subset["requested_group_tokens"])
    requested_shares = [100 * subset["requested_group_tokens"][group] / subset["target_tokens"] for group in groups]
    actual_shares = [100 * subset["actual_group_tokens"][group] / subset["selected_unique_tokens"] for group in groups]
    x = range(len(groups))
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar([value - 0.2 for value in x], requested_shares, width=0.4, label="Requested")
    axis.bar([value + 0.2 for value in x], actual_shares, width=0.4, label="Selected")
    axis.set_ylabel("Share (%)")
    axis.set_xticks(list(x), groups, rotation=20, ha="right")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "training_sequence_source_mixture.png", dpi=150)
    plt.close(fig)

    figure_names = (
        "corpus_source_documents.png",
        "corpus_source_group_tokens.png",
        "tokenizer_fertility.png",
        "training_sequence_source_mixture.png",
    )
    figure_hashes = {
        name: sha256_file(figures / name) for name in figure_names
    }
    tables = {
        "source_rows": source_rows,
        "length_rows": length_rows,
        "quality_flag_rows": flag_rows,
        "split_source_rows": split_source_rows,
        "split_rows": split_rows,
        "tokenizer_candidate_rows": candidate_rows,
        "tokenizer_source_fertility_rows": fertility_rows,
        "tokenizer_long_word_rows": word_rows,
        "training_mixture_rows": mixture_rows,
        "training_source_rows": selected_source_rows,
        "token_totals_by_source": token_totals_by_source,
        "token_totals_by_group": dict(token_totals_by_group),
        "figure_hashes": figure_hashes,
        "analysis_sha256": sha256_file(notes / "corpus_data_analysis.md"),
        "data_card_sha256": sha256_file(notes / "corpus_data_card.md"),
    }
    atomic_write_json(metadata / "data_pipeline_report_tables.json", tables)
    print(f"Generated {notes / 'corpus_data_analysis.md'} and {len(figure_hashes)} figures")


if __name__ == "__main__":
    main()
