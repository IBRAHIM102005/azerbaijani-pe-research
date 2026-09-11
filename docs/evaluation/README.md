# Evaluation and research results

This directory documents evaluation of five positional encodings across five model seeds.

- Entry point: `python scripts/evaluate.py --help`
- Configuration: `configs/evaluation_protocol.json`
- Dependencies: `requirements-evaluation.txt` (PyTorch is managed separately)
- Source: `src/evaluation/`
- Synthetic tests: `tests/evaluation/`
- Completed evidence and reports: `results/evaluation/`

## Completed results

Primary validation: 100/100; measured length validation: 65/65; primary test: 25/25; measured length test: 65/65. Each split also contains 10 not-applicable entries. ALiBi has the lowest observed mean test NLL (5.434068) and meets the recorded comparison rule against all four alternatives.

See `results/evaluation/report/Results_Analysis.md` and `results/evaluation/report/Slides_9_11_Content.md` for the original report and slide content.

## Preservation and scope

The results are byte-for-byte copies of the completed report export. Historical stage labels, absolute server paths, hashes, and manifest contents are intentionally retained. Paths in evidence manifests are relative to `results/evaluation/`. The exported raw document arrays are absent; use the original full result archive and original source package for reproduction.

The renamed source has a different source identity. Do not use it to resume or overwrite the completed server workspace. Naming cleanup is not a new scientific experiment.

## External prerequisites for a new evaluation

This source overlay requires the existing repository's model, tokenizer, and data code. The uploaded overlay did not include training provenance required by `prepare`: `results/manifests/m3_handoff/`, `results/manifests/m3_run_plan.json`, and per-run `results/runs/<run_id>/metadata.json` and `completed.json`. Obtain these unchanged from the original server package, together with checkpoints and input datasets, before attempting a new evaluation. In particular, the historical `m4_source_origin.json` lookup remains unchanged. These prerequisites must not be fabricated or replaced with evaluation summaries.

## Validation of this naming cleanup

Python syntax and the path-discovery unit test were checked. Full synthetic tests and GPU evaluation were not rerun in the packaging environment (PyTorch, PyArrow and SentencePiece were unavailable). The original completed result bytes and test evidence hashes were verified.

## Git workflow

Use the `evaluation-results` feature branch. This package does not include Git metadata and does not push or merge any branch. No main-branch changes are required.
