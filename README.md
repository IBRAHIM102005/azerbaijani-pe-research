# Azerbaijani Positional Encoding Research

**Research question:** Which positional encoding scheme generalizes best when a small causal language model is pretrained on a limited amount of Azerbaijani text?

This project will compare learned positional embeddings, sinusoidal encoding, RoPE, ALiBi, and NoPE. The dataset, tokenizer, model architecture, optimization setup, and training budget will be held constant as far as possible so that the positional encoding scheme remains the main experimental variable.

DOLLMA is the planned source corpus for Azerbaijani pretraining data. The repository is still at the initial setup and data-preparation stage, so no experimental results are available yet.

## Planned setup

- Azerbaijani text selected from DOLLMA
- A small Pythia-style causal decoder
- Five positional encoding variants
- Low-data pretraining with multiple random seeds
- Negative log-likelihood and perplexity evaluation

These details are provisional until the data and experiment configurations are frozen.

## Repository structure

- `data/` contains local inputs and will hold later data products and metadata.
- `src/` is reserved for the research implementation.
- `configs/` will hold positional-encoding, hardware, and frozen experiment settings.
- `experiments/`, `results/`, and `checkpoints/` separate run records from generated artifacts.
- `docs/`, `report/`, and `presentation/` contain project notes and research deliverables.

## Data

Raw DOLLMA parquet files are kept locally under `data/raw/` and are intentionally excluded from Git. See [`data/README.md`](data/README.md) for the current source selection and directory policy.

## Status

Current stage: repository bootstrap and dataset preparation. Model, tokenizer, preprocessing, training, and evaluation code have not been added.
