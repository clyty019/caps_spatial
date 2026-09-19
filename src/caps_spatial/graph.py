"""Spatial graph construction and Moran's I.

Two ways in:

* :func:`build_graph` -- rebuild the graph from spot coordinates using the
  distance-gate rule (undirected edge between two spots closer than
  ``gate_factor x`` the median nearest-neighbour spacing).  On a Visium array
  this is equivalent to first-order hexagonal adjacency: the 6 nearest
  neighbours sit at 1.00x the spacing, the second ring at sqrt(3) = 1.73x, so a
  1.5x gate keeps the first ring and nothing else.
* :func:`load_edge_list` -- read a precomputed edge table (barcode_i,
  barcode_j, ...), if you already have one.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

GATE_FACTOR = 1.5


def build_graph(coords, gate_factor=GATE_FACTOR):
    """Distance-gated spatial graph.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Spot centre coordinates in any linear unit (pixels are fine).
    gate_factor : float
        Edge iff ``dist < gate_factor * median(nearest-neighbour distance)``.

    Returns
    -------
    ei, ej : int arrays
        Undirected edges, each stored exactly once.
    spacing : float
        The median nearest-neighbour distance used for the gate.
    """
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got {xy.shape}")
    tree = cKDTree(xy)
    nn, _ = tree.query(xy, k=2)          # column 0 is the point itself
    spacing = float(np.median(nn[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("degenerate coordinates: median NN spacing is not positive")
    pairs = tree.query_pairs(gate_factor * spacing, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError(
            f"no edges at gate {gate_factor} x spacing ({spacing:.4g}); "
            "check that the coordinates are spot centres on a regular array")
    ei = pairs[:, 0].astype(np.int64)
    ej = pairs[:, 1].astype(np.int64)
    return ei, ej, spacing


def load_edge_list(path, spots):
    """Read an edge table, return ``(ei, ej)`` indexed into ``spots``.

    Expected columns: two barcode columns named ``barcode_i`` / ``barcode_j``
    (extra columns are ignored).  Accepts .csv or .csv.gz.
    """
    import pandas as pd

    d = pd.read_csv(path)
    if not {"barcode_i", "barcode_j"}.issubset(d.columns):
        raise ValueError(
            f"{path}: edge table needs columns 'barcode_i' and 'barcode_j', "
            f"found {list(d.columns)}")
    idx = {b: i for i, b in enumerate(spots)}
    try:
        ei = np.fromiter((idx[b] for b in d["barcode_i"].astype(str)), np.int64,
                         len(d))
        ej = np.fromiter((idx[b] for b in d["barcode_j"].astype(str)), np.int64,
                         len(d))
    except KeyError as e:
        raise ValueError(f"{path}: edge endpoint {e} not in the expression spots") from None
    return ei, ej


def moran(x, ei, ej):
    """Moran's I of ``x`` on an undirected edge list.

    Standard symmetric binary weights, edge list storing each undirected edge
    once.  ``S0 = sum_ij w_ij = 2m`` while ``sum_ij w_ij z_i z_j = 2 * sum_edges
    z_i z_j``; the two factors of 2 cancel, so the normalisation is ``n / m``
    with ``m`` the number of stored edges.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0 or len(ei) == 0:
        return 0.0
    xb = x - x.mean()
    den = float((xb ** 2).sum())
    if den <= 0:
        return 0.0
    num = float((xb[ei] * xb[ej]).sum())
    return (n / float(len(ei))) * (num / den)
