# Reproducibility guide

1. `make setup` (Ibrahim/Yasin/Fidan own the real target; not yet wired here).
2. `make test` -- CPU-only unit + integration suite.
3. `make audit-configs` / `make audit-params` -- fairness gates.
4. `make smoke` -- fast CPU-only gate combining the above.
5. `make reproduce-headline RELEASE=<tag>` -- reconstruct headline
   tables/figures from an already-released checkpoint set (scaffolding
   only; evaluation/statistics/figure generation stages to be wired by Nihat).
6. `make reproduce-training MATRIX=<smoke|core>` -- full pretraining,
   compute-intensive, deliberately separate from the above and NOT run in
   CI (Fidan owns the real launcher; currently a stub that exits non-zero
   with a clear message).

See `docs/INTERFACE_CONTRACT.md` for exactly what this tooling expects
from Ibrahim, Yasin, Fidan, and Nihat, and `docs/ai_use.md` for the AI-assistance disclosure table.
