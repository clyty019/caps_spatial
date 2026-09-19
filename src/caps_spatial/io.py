"""Readers: expression, composition, programmes, positions.

Inputs are plain matrices -- no R, no RDS, no Seurat objects.  Anything that
already produces a spot x cell-type abundance table (RCTD, cell2location,
SPOTlight, ...) can feed :func:`load_composition`.
"""
from __future__ import annotations

import gzip
import os

import numpy as np
import pandas as pd

from .core import CPM_SCALE

# Symbols that feature files use to mean "no symbol"; treated as unmeasured.
_NULL_SYM = {"", "nan", "NA", "."}


class Expression:
    """Library-size normalised expression, oriented spots x genes.

    ``logX`` is sparse ``log1p(scale * counts / library)``; ``sym2col`` maps a
    gene symbol to its column.  Slicing a programme's columns returns dense
    float64 in the same dtype the canonical pipeline uses.
    """

    def __init__(self, spots, symbols, logX, positions=None):
        self.spots = np.asarray(spots, dtype=str)
        self.symbols = np.asarray(symbols, dtype=str)
        self.logX = logX.tocsr() if hasattr(logX, "tocsr") else logX
        self.positions = positions
        self.sym2col = {s: i for i, s in enumerate(self.symbols)}

    def __len__(self):
        return len(self.spots)

    @property
    def shape(self):
        return (len(self.spots), len(self.symbols))

    @property
    def coords(self):
        """Spot coordinates as an (n, 2) array, or None if not loaded."""
        if self.positions is None:
            return None
        return self.positions[["x", "y"]].to_numpy(dtype=float)

    def logx(self, cols):
        """Dense log1p-CPM submatrix for the given gene columns."""
        cols = np.asarray(cols, dtype=np.int64)
        return np.asarray(self.logX[:, cols].todense(), dtype=np.float64)


def _unique_symbols(sym):
    """Keep the first occurrence of each usable symbol; return the column mask."""
    sym = np.asarray(sym, dtype=str)
    ok = np.array([s not in _NULL_SYM for s in sym])
    seen, keep = set(), np.zeros(len(sym), dtype=bool)
    for i in np.flatnonzero(ok):
        if sym[i] not in seen:
            seen.add(sym[i])
            keep[i] = True
    return sym[keep], keep


def _normalise_counts(C, scale):
    """Row-normalise a spot x gene count matrix to ``scale`` then log1p."""
    from scipy import sparse

    C = sparse.csr_matrix(C).astype(np.float64)
    lib = np.asarray(C.sum(1)).ravel()
    f = np.where(lib > 0, scale / np.maximum(lib, 1.0), 0.0)
    return C.multiply(f[:, None]).tocsr().log1p()


def _read_10x_h5(path):
    import h5py

    with h5py.File(path, "r") as h:
        g = h["matrix"] if "matrix" in h else h
        shape = tuple(int(v) for v in g["shape"][:])
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        feat = g["features"] if "features" in g else g["genes"]
        sym = np.array([s.decode() if isinstance(s, bytes) else str(s)
                        for s in feat["name"][:]])
        bc = np.array([s.decode() if isinstance(s, bytes) else str(s)
                       for s in g["barcodes"][:]])
    from scipy import sparse

    # 10x stores genes x spots
    M = sparse.csc_matrix((data, indices, indptr), shape=shape)
    return sparse.csr_matrix(M.T), bc, sym


def _read_mtx_dir(path):
    import scipy.io as sio
    from scipy import sparse

    def pick(*names):
        for n in names:
            p = os.path.join(path, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"{path}: none of {names} found")

    mtx = pick("matrix.mtx.gz", "matrix.mtx")
    opener = gzip.open if mtx.endswith(".gz") else open
    with opener(mtx, "rb") as fh:
        M = sio.mmread(fh)                       # genes x spots
    bc = pd.read_csv(pick("barcodes.tsv.gz", "barcodes.tsv"),
                     header=None, sep="\t")[0].astype(str).to_numpy()
    fdf = pd.read_csv(pick("features.tsv.gz", "features.tsv",
                           "genes.tsv.gz", "genes.tsv"),
                      header=None, sep="\t")
    sym = fdf[1].astype(str).to_numpy() if fdf.shape[1] > 1 \
        else fdf[0].astype(str).to_numpy()
    return sparse.csr_matrix(M.T), bc, sym


def _read_h5ad(path):
    import anndata as ad

    a = ad.read_h5ad(path)
    C = a.X
    coords = None
    if "spatial" in a.obsm:
        coords = np.asarray(a.obsm["spatial"], dtype=float)
    return C, np.asarray(a.obs_names, dtype=str), \
        np.asarray(a.var_names, dtype=str), coords


def load_expression(counts, positions=None, scale=CPM_SCALE):
    """Read a Visium (or any spot x gene) count matrix.

    Parameters
    ----------
    counts : str
        ``.h5`` (10x Cell Ranger), a directory holding the ``matrix.mtx.gz``
        triplet, or ``.h5ad``.
    positions : str, optional
        Spot coordinates.  Required unless ``counts`` is an ``.h5ad`` carrying
        ``obsm["spatial"]``.
    scale : float
        Library-size target; counts are normalised per spot to this, then
        ``log1p``.  Default 1e4 (CPM).

    Returns
    -------
    Expression
    """
    coords = None
    if str(counts).endswith(".h5ad"):
        C, bc, sym, coords = _read_h5ad(counts)
    elif str(counts).endswith(".h5"):
        C, bc, sym = _read_10x_h5(counts)
    elif os.path.isdir(counts):
        C, bc, sym = _read_mtx_dir(counts)
    else:
        raise ValueError(
            f"unrecognised counts input {counts!r}: expected .h5, .h5ad, or a "
            "directory containing matrix.mtx.gz + barcodes + features")

    sym, keep = _unique_symbols(sym)
    from scipy import sparse

    C = sparse.csr_matrix(C)[:, keep]
    if C.shape[0] != len(bc):
        raise ValueError(
            f"counts matrix has {C.shape[0]} rows but {len(bc)} barcodes; "
            "expected spots x genes after orientation")

    pos = None
    if positions is not None:
        pos = load_positions(positions, spots=bc)
    elif coords is not None:
        pos = pd.DataFrame(coords[:, :2], index=bc, columns=["x", "y"])

    return Expression(bc, sym, _normalise_counts(C, scale), positions=pos)


def load_positions(path, spots=None):
    """Read spot coordinates as a DataFrame indexed by barcode with x / y.

    Accepts ``tissue_positions.csv`` (header) and
    ``tissue_positions_list.csv`` (headerless) from Space Ranger, or any CSV
    with a barcode column plus two coordinate columns.
    """
    # The first line must be inspected as text: a headerless Space Ranger
    # file's first data row also starts with a barcode.
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        first = fh.readline().rstrip("\n").lstrip("﻿")
    cells = [c.strip().strip('"').lower()
             for c in (first.split(",") if "," in first else first.split("\t"))]

    if "barcode" in cells:
        d = pd.read_csv(path)
    else:
        raw = pd.read_csv(path, header=None, dtype=str)
        n = raw.shape[1]
        if n >= 6:      # Space Ranger list: barcode, in_tissue, row, col, y, x
            d = raw.copy()
            d.columns = ["barcode", "in_tissue", "array_row", "array_col",
                         "y", "x"][:n]
        elif n == 3:
            d = raw.copy()
            d.columns = ["barcode", "x", "y"]
        else:
            raise ValueError(
                f"{path}: expected a barcode column plus two coordinate "
                f"columns, or a 6-column Space Ranger positions file "
                f"(found {n} columns)")

    lower = {c.lower(): c for c in d.columns}
    bcol = lower.get("barcode", d.columns[0])
    xcol = next((lower[c] for c in ("x", "pxl_col_in_fullres", "pxl_col",
                                    "col", "array_col") if c in lower), None)
    ycol = next((lower[c] for c in ("y", "pxl_row_in_fullres", "pxl_row",
                                    "row", "array_row") if c in lower), None)
    if xcol is None or ycol is None:
        raise ValueError(f"{path}: no coordinate columns found in {list(d.columns)}")

    # build from arrays: passing Series alongside an explicit index would make
    # pandas align on the series' own RangeIndex and blank every coordinate
    out = pd.DataFrame(
        {"x": pd.to_numeric(d[xcol], errors="coerce").to_numpy(float),
         "y": pd.to_numeric(d[ycol], errors="coerce").to_numpy(float)},
        index=d[bcol].astype(str).to_numpy())
    if out.isna().any().any():
        raise ValueError(f"{path}: coordinates contain non-numeric values")
    if spots is not None:
        miss = [s for s in spots if s not in out.index]
        if miss:
            raise ValueError(
                f"{path}: {len(miss)} of {len(spots)} expression barcodes have "
                f"no coordinates, e.g. {miss[:3]}")
        out = out.loc[list(spots)]
    return out


def load_composition(path, spots=None, normalise="auto", log=print):
    """Read a spot x cell-type abundance matrix.

    Parameters
    ----------
    path : str
        CSV / TSV with a barcode index column, or ``.h5ad`` whose ``obs``
        carries the abundances.
    spots : sequence of str, optional
        Expression barcodes.  The composition is reindexed onto exactly this
        order, so the two matrices are aligned by barcode rather than by row
        number.
    normalise : {"auto", "always", "never"}
        ``auto`` renormalises rows onto the simplex when the row sums deviate
        from 1 by more than 1e-3 (and says so); ``always`` always
        renormalises; ``never`` takes the numbers as given.  Row sums of 1 are
        what makes the dropped design column irrelevant, so this is a
        correctness guard, not cosmetics.

    Returns
    -------
    pandas.DataFrame
    """
    if str(path).endswith(".h5ad"):
        import anndata as ad

        a = ad.read_h5ad(path)
        d = pd.DataFrame(np.asarray(a.X), index=np.asarray(a.obs_names, dtype=str),
                         columns=np.asarray(a.var_names, dtype=str))
    else:
        sep = "\t" if str(path).endswith((".tsv", ".tsv.gz")) else ","
        d = pd.read_csv(path, sep=sep, index_col=0)
    d.index = d.index.astype(str)
    d = d.apply(pd.to_numeric, errors="raise").astype(float)

    if spots is not None:
        spots = [str(s) for s in spots]
        miss = [s for s in spots if s not in d.index]
        if miss:
            raise ValueError(
                f"{path}: {len(miss)} of {len(spots)} expression barcodes are "
                f"missing from the composition, e.g. {miss[:3]}")
        d = d.loc[spots]

    s = d.to_numpy().sum(1)
    if normalise == "always" or (normalise == "auto"
                                 and not np.allclose(s, 1.0, atol=1e-3)):
        if not np.all(s > 0):
            raise ValueError(
                f"{path}: {(s <= 0).sum()} spots have zero total abundance; "
                "cannot renormalise (pass normalise='never' to keep as-is)")
        if normalise == "auto":
            log(f"  composition rows deviated from 1 (max |sum-1| = "
                f"{np.abs(s - 1).max():.3g}); renormalised")
        d = d.div(s, axis=0)
    elif normalise == "never":
        pass
    return d


def load_programmes(path):
    """Read a GMT file: ``name <tab> description <tab> gene <tab> gene ...``.

    Returns ``{programme: [gene, ...]}``.  Lines with fewer than three fields
    are skipped, matching the canonical parser.
    """
    sets = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            name = p[0].strip()
            genes = [g.strip() for g in p[2:] if g.strip()]
            if name and genes:
                sets[name] = genes
    if not sets:
        raise ValueError(f"{path}: no gene sets parsed (is this a GMT file?)")
    return sets
