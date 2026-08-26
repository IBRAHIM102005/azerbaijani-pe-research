# Data

The raw corpus for this study comes from DOLLMA. It is stored locally for the planned Azerbaijani language-model pretraining experiments and is not version-controlled.

## Current local sources

The candidate directories currently included in the core corpus are:

- `anl-news`
- `azwiki`
- `elite-blogs`
- `elite-books`
- `eqanun`
- `mediocore-books`

Together, the local copy contains 14 parquet shards (about 1.85 GB, or 1.72 GiB). `translated-enwiki` is intentionally excluded from the planned native-Azerbaijani mixture. `bhos` is also excluded for now, pending a clearer source-level decision about its role in the core corpus.

## Directory layout

- `raw/` contains the unchanged local source material and is ignored by Git.
- `interim/` is reserved for temporary outputs between processing stages.
- `processed/` will contain the finalized corpus prepared for training.
- `manifests/` will record reproducible file lists, splits, and dataset versions.
- `metadata/` will hold corpus summaries and provenance notes.

Later phases will cover cleaning, deduplication, document-level splitting, manifest creation, and tokenizer preparation. None of these preprocessing steps has been run yet.

The source DOLLMA README records the dataset license as CC BY-NC-SA 4.0.
