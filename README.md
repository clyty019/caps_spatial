# CAPS — Composition-Adjusted Programme Scores

Spatial transcriptomics lets you score a gene programme (a Hallmark set, a
cell-state signature, a cluster's markers) on every spot and look at the map.
The problem is that the map is usually a cell-type abundance map in disguise:
if the programme's genes are markers of one cell type, its score tracks where
that cell type is, and you have learned nothing about the programme's own
spatial behaviour.

CAPS removes that. It regresses **each programme gene** on the spot's cell-type
composition, keeps the residual, and averages the residuals. What survives is
the part of the programme's co-expression that the cell-type mixture does not
explain — and a `retention` statistic that tells you how much of the raw
spatial structure was real.

```
raw score     →  strongly spatial, composition R² 0.75, retention 0.42
CAPS-adjusted →  weakly spatial   (structure was mostly the composition)
```

## Install

```bash
pip install git+https://github.com/clyty019/caps_spatial.git
```

or, from a clone of that repository:

```bash
pip install -e .            # editable; or pip install . for a copy
```

Requires Python ≥ 3.9, numpy, scipy, pandas, matplotlib, anndata, h5py.

## Quick start

### Command line

```bash
caps run \
  --counts      filtered_feature_bc_matrix.h5 \
  --positions   tissue_positions_list.csv \
  --composition composition_norm.csv \
  --programmes  h.all.v2023.1.Hs.symbols.gmt \
  --outdir      caps_out/
```

`--counts` accepts a 10x `.h5`, a directory holding the `matrix.mtx.gz` triplet,
or an `.h5ad`. `--positions` can be omitted if the `.h5ad` carries
`obsm["spatial"]`. Add `--pergene HALLMARK_ANGIOGENESIS` for per-gene maps.

### Python

```python
import caps_spatial as caps

ex    = caps.load_expression("filtered_feature_bc_matrix.h5",
                             positions="tissue_positions_list.csv")
comp  = caps.load_composition("composition_norm.csv", spots=ex.spots)
progs = caps.load_programmes("h.all.v2023.1.Hs.symbols.gmt")

ei, ej, spacing = caps.build_graph(ex.coords)     # or caps.load_edges(path, ex.spots)
res = caps.run(ex, comp, progs, ei, ej)

res.summary                                    # one row per programme
res.field("HALLMARK_ANGIOGENESIS", "caps")     # per-spot adjusted score
res.field("HALLMARK_ANGIOGENESIS", "raw")      # per-spot raw score
res.genes("HALLMARK_ANGIOGENESIS")             # per-gene residual z-scores
res.scores().to_csv("scores.csv")              # spots x programmes, both channels
```

## Inputs

| input | format |
|---|---|
| `--counts` | 10x `filtered_feature_bc_matrix.h5`; a directory with `matrix.mtx(.gz)` + `barcodes` + `features`; or `.h5ad` |
| `--positions` | Space Ranger `tissue_positions_list.csv` / `tissue_positions.csv`, or any CSV with a barcode column plus two coordinate columns |
| `--composition` | spot x cell-type abundances as CSV/TSV (barcode index) or `.h5ad` — whatever your deconvolution tool emits (RCTD, cell2location, SPOTlight, …) |
| `--programmes` | standard GMT: `name <tab> description <tab> gene <tab> gene …` |
| `--edges` | optional precomputed edge table with `barcode_i`, `barcode_j` |

Genes are matched by symbol against the counts' feature names; the first
occurrence wins when a symbol repeats. Programmes left with fewer than
`--min-genes` (default 5) measured genes are skipped and listed, with the
missing symbols counted, in `summary.csv`.

Composition rows are renormalised to sum to 1 by default (`auto`) whenever they
deviate by more than 1e-3, with a message — see the definition below for why
that matters. Use `--composition-normalise never` to keep the numbers as given.

## Caps definition

The authoritative statement of the math. `src/caps_spatial/core.py` is the only
implementation of it.

With `X` the spot x gene matrix of `log1p(CPM)` (`CPM` = counts scaled to 1e4
per spot), `P` the spot x cell-type abundance matrix with rows summing to 1, and
`G` the programme's measured genes:

1. **raw score** — standardise each gene in `G` across spots, then average:
   `Y = mean_g z(X[:, g])`.
2. **CAPS score** — regress *each gene* on the composition design
   `Zc = [1, P[:, :-1]]` by OLS, standardise each gene's residual, then average:

   ```
   e_g = X[:, g] − Zc · lstsq(Zc, X[:, g])
   CAPS = mean_g (e_g − mean(e_g)) / (sd(e_g) + 1e-6)
   ```

   Averaging residuals is **not** residualising the average; the score-level
   residual is also computed, as `moran_sr`, but only to report the divergence.
3. **composition R²** — `1 − SS(score-level residual) / SS(Y − mean(Y))`: the
   fraction of the raw score's variance explained by composition. High R² means
   the raw score is largely a composition readout.
4. **retention** — `Moran(CAPS) / Moran(raw)`, defined only where
   `Moran(raw) > 0.10` (`RET_FLOOR`); NaN below, never 0 and never a clipped
   denominator. Moran's I is signed, so the ratio is meaningless near zero.

Two details worth knowing:

* **Why `P[:, :-1]`.** With rows summing to 1, the dropped column equals
  `1 − (the rest)` and is already in the span of the intercept and the kept
  columns, so the design is rank-deficient unless one column goes. *Which* one
  is dropped does not change the residuals — `[1, P[:, :-1]]` and
  `[1, P[:, 1:]]` span the same space. That equivalence holds **only** when the
  rows sum to 1, which is why the normalisation guard exists.
* **Spatial graph.** Rebuilt from coordinates by default: an undirected edge
  between two spots closer than `1.5 x` the median nearest-neighbour spacing.
  On a Visium array the first hexagonal ring sits at 1.00x and the second at
  1.73x, so this is exactly first-order adjacency. Moran's I uses the standard
  symmetric binary normalisation `(n/m) · Σ_e x̃_i x̃_j / Σ x̃²` with `m` the
  number of stored undirected edges.

## Outputs

```
<outdir>/
├── maps/<PROGRAMME>_raw_vs_caps.pdf|.png    raw and CAPS side by side, one shared colour scale
├── genes/<PROGRAMME>/<GENE>_raw_vs_caps.*   only with --pergene / --pergene-all
├── qc/retention_vs_r2.*                     retention against composition R²
├── qc/moran_raw_vs_caps.*                   per-programme Moran, raw vs adjusted
├── summary.csv                              programme, ng, moran_raw, moran_caps, moran_sr, retention, r2
└── scores.csv                               spots x programmes, both channels
```

The raw and adjusted panels always share a colour limit (`max` of each panel's
99th percentile of |value|); per-panel scaling would make a collapsed field look
as structured as the original.

## Reading the output

* **low retention + high R²** — the raw map was composition. This is the case
  CAPS exists for.
* **high retention + low R²** — genuine spatial structure in the programme.
* **low retention + low R²** — the raw score had little spatial structure to
  begin with; there is nothing to explain and nothing to remove.
* **slightly negative retention** — the adjustment reversed a weak spatial
  signal. Real, and reported as-is rather than truncated.

## Verification

Two checks were run against the companion analysis:

* **Frozen anchor.** Running the CLI on a real CRC section with the Hallmark GMT
  reproduces the companion manuscript's values for `HALLMARK_ANGIOGENESIS` —
  `ng = 33`, `r2 = 0.7546`, `moran_raw = 0.6627`, `moran_caps = 0.2784`,
  `retention = 0.4201` — to better than 1e-3, via **both** graph paths (a
  precomputed edge table, and a rebuild from spot coordinates). The rebuilt
  graph matches the stored one edge for edge: 5944 / 5944.
* **Determinism.** Two consecutive runs produce byte-identical `summary.csv`
  and `scores.csv`.

## Citation

If you use this package, please cite:

> Wei, X. (2026). *CAPS: composition-adjusted programme scores for spatial
> transcriptomics* (version 0.1.0) [Computer software].
> https://github.com/clyty019/caps_spatial

```bibtex
@software{wei2026caps,
  author  = {Wei, Xindi},
  title   = {{CAPS}: composition-adjusted programme scores for spatial transcriptomics},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/clyty019/caps_spatial}
}
```

The companion manuscript describing the method is in preparation.

## License

MIT — see `LICENSE`.
