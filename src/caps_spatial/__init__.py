"""CAPS -- composition-adjusted programme scores for spatial transcriptomics.

Quick start::

    import caps_spatial as caps

    ex    = caps.load_expression("filtered_feature_bc_matrix.h5",
                                 positions="tissue_positions_list.csv")
    comp  = caps.load_composition("composition_norm.csv", spots=ex.spots)
    progs = caps.load_programmes("h.all.v2023.1.Hs.symbols.gmt")
    res   = caps.run(ex, comp, progs)

    res.summary                                  # Moran / retention / R2 per programme
    res.field("HALLMARK_ANGIOGENESIS", "caps")   # per-spot adjusted score
    res.genes("HALLMARK_ANGIOGENESIS")           # per-gene residual z-scores

Or from the command line::

    caps run --counts ... --composition ... --programmes ... --outdir out/
"""
from .core import (CPM_SCALE, MIN_GENES, RET_FLOOR, Z_EPS, CapsResult,
                   programme_fields, residual_z, retention, run)
from .graph import build_graph, load_edge_list as load_edges, moran
from .io import (Expression, load_composition, load_expression, load_positions,
                 load_programmes)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_expression", "load_composition", "load_programmes", "load_positions",
    "Expression", "run", "CapsResult",
    "build_graph", "load_edges", "moran",
    "residual_z", "programme_fields", "retention",
    "CPM_SCALE", "MIN_GENES", "RET_FLOOR", "Z_EPS",
]
