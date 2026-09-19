"""CAPS -- composition-adjusted programme scores.

This module is the **single authoritative implementation** of the CAPS math.
Nothing else in the package may re-derive it (see ``CLAUDE.md``).

Definition
----------
Given a spot x gene matrix of library-size normalised expression ``X`` and a
spot x cell-type abundance matrix ``P`` whose rows sum to 1:

1. ``Y``  the **raw** programme score: standardise each programme gene across
   spots, then average those z-scores over the programme's genes.
2. ``CAPS`` the **adjusted** score: regress *each programme gene* on the
   composition design ``Zc = [1, P[:, :-1]]`` by OLS, standardise each gene's
   residual across spots, then average the residual z-scores over the same
   genes.

The distinction from residualising the score itself (``sr``) is the whole
point: averaging residuals is not residualising the average.  ``sr`` is
computed here too, but only to report the composition R-squared.

Why ``P[:, :-1]``: with rows summing to 1 the dropped column is
``1 - sum(others)``, i.e. already in the span of the intercept and the kept
columns, so the fit is rank-deficient unless a column is dropped. *Which*
column is dropped does not change the residuals -- the column space of
``[1, P[:, :-1]]`` equals that of ``[1, P]`` equals that of ``[1, P[:, 1:]]``.
That equivalence holds only if the rows sum to 1, which is why
``io.load_composition`` normalises by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical defaults.  cli.py reads these; do not restate the numbers there.
CPM_SCALE = 1e4      # library-size target for log1p CPM (Visium convention)
Z_EPS = 1e-6         # denominator guard in the per-gene standardisation
MIN_GENES = 5        # a programme needs this many measured genes to be scored
RET_FLOOR = 0.10     # retention is undefined where the raw Moran is <= this
BATCH = 2048         # gene columns per lstsq call


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def per_gene_z(X, eps=Z_EPS):
    """Standardise each column (gene) of ``X`` across rows (spots)."""
    X = np.asarray(X, dtype=np.float64)
    return (X - X.mean(0)) / (X.std(0) + eps)


def design_matrix(comp):
    """``[1, comp[:, :-1]]`` -- intercept plus all but the last abundance column."""
    comp = np.asarray(comp, dtype=np.float64)
    if comp.ndim != 2:
        raise ValueError(f"composition must be 2-D, got shape {comp.shape}")
    return np.column_stack([np.ones(len(comp)), comp[:, :-1]])


def _ols_residual(Zc, Y):
    """Residual of every column of ``Y`` (n x k) on ``Zc`` (n x p)."""
    Y = np.asarray(Y, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(Zc, Y, rcond=None)
    return Y - Zc @ coef


def residual_z(Xl, comp, eps=Z_EPS, batch=BATCH):
    """Per-gene OLS residual z-scores -- the CAPS gene field.

    ``Xl`` is spot x gene log1p-CPM, ``comp`` is spot x cell-type.  Returns an
    ``n_spot x n_gene`` array of residual z-scores.
    """
    Xl = np.asarray(Xl, dtype=np.float64)
    Zc = design_matrix(comp)
    out = np.empty_like(Xl)
    for a in range(0, Xl.shape[1], batch):
        e = _ols_residual(Zc, Xl[:, a:a + batch])
        out[:, a:a + batch] = per_gene_z(e, eps)
    return out


def score_level_field(Y, comp):
    """Score-level residual (``sr``): residualise the programme score itself."""
    return _ols_residual(design_matrix(comp), np.asarray(Y, dtype=np.float64))


def r2_of(Y, comp):
    """Fraction of the raw score's variance explained by composition."""
    Y = np.asarray(Y, dtype=np.float64)
    sr = score_level_field(Y, comp)
    ss = float(np.sum(sr ** 2))
    ss0 = float(np.sum((Y - Y.mean()) ** 2))
    return 1.0 - ss / max(ss0, 1e-12)


def retention(i_caps, i_raw, floor=RET_FLOOR):
    """``i_caps / i_raw``, undefined (NaN) wherever ``i_raw <= floor``.

    Moran's I is signed, so the ratio is undefined at ``i_raw = 0`` and
    sign-reversed below it.  Below the floor the result is NaN -- never 0, and
    the denominator is never clipped to the floor.  Accepts scalars or arrays.
    """
    adj = np.asarray(i_caps, dtype=float)
    raw = np.asarray(i_raw, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(raw > floor, adj / raw, np.nan)
    return float(out) if out.ndim == 0 else out


def programme_fields(Xl, comp, eps=Z_EPS):
    """Raw and CAPS programme scores for one programme's gene submatrix.

    ``Xl`` is spot x gene log1p-CPM restricted to this programme's measured
    genes.  Regressing a column subset gives exactly the same per-gene
    residuals as regressing the full matrix, because OLS columns are
    independent -- so scoring one programme at a time is not an approximation.
    """
    Xl = np.asarray(Xl, dtype=np.float64)
    if Xl.shape[1] == 0:
        raise ValueError("programme has no measured genes")
    zraw = per_gene_z(Xl, eps)
    Y = zraw.mean(1)
    zcaps = residual_z(Xl, comp, eps)
    Gw = zcaps.mean(1)
    return dict(raw=Y, caps=Gw, zraw=zraw, zcaps=zcaps,
                sr=score_level_field(Y, comp), r2=r2_of(Y, comp))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
class CapsResult:
    """Fields, per-programme statistics and the per-gene residual matrices."""

    def __init__(self, expr, comp, ei, ej, fields, skipped, per_gene):
        self.expr = expr
        self.comp = comp
        self.ei = ei
        self.ej = ej
        self.fields = fields          # programme -> dict of arrays / scalars
        self.skipped = skipped        # list of (programme, n_measured)
        self.per_gene = per_gene      # programme -> dict(raw_z, caps_z, genes)

    @property
    def programmes(self):
        return list(self.fields)

    @property
    def summary(self):
        """One row per scored programme."""
        rows = []
        for p, f in self.fields.items():
            rows.append(dict(programme=p, ng=f["ng"],
                             n_missing=len(f["missing"]),
                             moran_raw=f["moran_raw"], moran_caps=f["moran_caps"],
                             moran_sr=f["moran_sr"], retention=f["retention"],
                             r2=f["r2"]))
        return pd.DataFrame(rows)

    @property
    def spots(self):
        return np.asarray(self.expr.spots)

    def field(self, programme, which="caps"):
        return self.fields[programme][which]

    def genes(self, programme):
        """Per-gene residual z-scores as a spots x genes DataFrame."""
        pg = self.per_gene[programme]
        return pd.DataFrame(pg["caps_z"], index=self.spots, columns=pg["genes"])

    def gene_raws(self, programme):
        pg = self.per_gene[programme]
        return pd.DataFrame(pg["raw_z"], index=self.spots, columns=pg["genes"])

    def scores(self):
        """Spots x programme table holding both fields."""
        cols = {}
        for p, f in self.fields.items():
            cols[f"{p}__raw"] = f["raw"]
            cols[f"{p}__caps"] = f["caps"]
        return pd.DataFrame(cols, index=self.spots)


def run(expr, comp, programmes, ei, ej, min_genes=MIN_GENES,
        ret_floor=RET_FLOOR, want_genes=True, log=None):
    """Score every programme under raw and CAPS.

    Parameters
    ----------
    expr : Expression
        From :func:`caps_spatial.io.load_expression`.
    comp : ndarray, shape (n_spot, n_celltype)
        Rows must sum to 1 (see :func:`caps_spatial.io.load_composition`).
    programmes : dict
        programme name -> list of gene symbols.
    ei, ej : int arrays
        Undirected spatial edges over the same spots.
    min_genes : int
        Programmes with fewer measured genes are skipped and reported.
    want_genes : bool
        Keep the per-gene residual matrices (needed for per-gene figures).

    Returns
    -------
    CapsResult
    """
    from .graph import moran

    comp = np.asarray(comp, dtype=np.float64)
    sym2col = expr.sym2col
    fields, skipped, per_gene = {}, [], {}

    for name, genes in programmes.items():
        # de-duplicate while keeping order, then map to measured columns
        seen, ids, used, missing = set(), [], [], []
        for g in genes:
            if g in seen:
                continue
            seen.add(g)
            k = sym2col.get(g)
            if k is None:
                missing.append(g)
            else:
                ids.append(k)
                used.append(g)
        if len(ids) < min_genes:
            skipped.append(dict(programme=name, n_measured=len(ids),
                                n_missing=len(missing)))
            continue

        ids = np.asarray(ids, dtype=np.int64)
        f = programme_fields(expr.logx(ids), comp)
        f["ng"] = len(ids)
        f["missing"] = missing
        f["moran_raw"] = moran(f["raw"], ei, ej)
        f["moran_caps"] = moran(f["caps"], ei, ej)
        f["moran_sr"] = moran(f["sr"], ei, ej)
        f["retention"] = retention(f["moran_caps"], f["moran_raw"], ret_floor)
        f["genes"] = used
        fields[name] = f
        if want_genes:
            per_gene[name] = dict(genes=used, raw_z=f["zraw"].astype(np.float32),
                                  caps_z=f["zcaps"].astype(np.float32))
        if log:
            ret = f["retention"]
            ret_s = f"{ret:.3f}" if np.isfinite(ret) else "n/a"
            log(f"  {name}: ng={len(ids)}  Moran {f['moran_raw']:.3f} -> "
                f"{f['moran_caps']:.3f}  retention {ret_s}")

    return CapsResult(expr, comp, ei, ej, fields, skipped, per_gene)
