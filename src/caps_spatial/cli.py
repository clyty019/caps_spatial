"""Command line entry point: ``caps run ...``.

Orchestration only.  Every number that means something lives in
:mod:`caps_spatial.core`; the defaults here are read from there.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .core import CPM_SCALE, MIN_GENES, RET_FLOOR, run
from .graph import build_graph, load_edge_list
from .io import load_composition, load_expression, load_programmes


def _build_parser():
    p = argparse.ArgumentParser(
        prog="caps",
        description="Composition-adjusted programme scores for spatial "
                    "transcriptomics.")
    p.add_argument("--version", action="version", version=f"caps-spatial {__version__}")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="score programmes on one section")
    r.add_argument("--counts", required=True,
                   help="10x .h5, an mtx triplet directory, or .h5ad")
    r.add_argument("--composition", required=True,
                   help="spot x cell-type abundance matrix (CSV/TSV/h5ad)")
    r.add_argument("--programmes", required=True, help="GMT gene sets")
    r.add_argument("--outdir", required=True, help="output directory")
    r.add_argument("--positions", default=None,
                   help="spot coordinates (Space Ranger positions CSV); "
                        "omit if the .h5ad carries obsm['spatial']")
    r.add_argument("--edges", default=None,
                   help="precomputed edge table with barcode_i / barcode_j; "
                        "takes precedence over rebuilding from coordinates")
    r.add_argument("--scale", type=float, default=CPM_SCALE,
                   help=f"library-size normalisation target (default {CPM_SCALE:g})")
    r.add_argument("--ret-floor", type=float, default=RET_FLOOR,
                   help="raw Moran at or below this -> retention is NaN "
                        f"(default {RET_FLOOR:g})")
    r.add_argument("--min-genes", type=int, default=MIN_GENES,
                   help="skip programmes with fewer measured genes "
                        f"(default {MIN_GENES})")
    r.add_argument("--composition-normalise", choices=("auto", "always", "never"),
                   default="auto",
                   help="renormalise abundances to rows summing to 1; the "
                        "dropped design column is only irrelevant when they do")
    r.add_argument("--pergene", nargs="*", default=[], metavar="PROGRAMME",
                   help="also write per-gene maps for these programmes")
    r.add_argument("--pergene-all", action="store_true",
                   help="per-gene maps for every scored programme (many files)")
    r.add_argument("--no-maps", action="store_true",
                   help="write only summary.csv / scores.csv")
    r.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    return p


def _resolve_edges(args, expr):
    """Edges from an explicit table, else rebuilt from spot coordinates."""
    if args.edges:
        print(f"edges: reading {args.edges}")
        return load_edge_list(args.edges, expr.spots)
    if expr.coords is None:
        raise SystemExit(
            "no spatial graph: pass --edges <table> or --positions <csv>")
    ei, ej, spacing = build_graph(expr.coords)
    print(f"edges: rebuilt {len(ei)} edges from coordinates "
          f"(median NN spacing {spacing:.3g}, gate x1.5)")
    return ei, ej


def _run(args):
    os.makedirs(args.outdir, exist_ok=True)

    print(f"counts      : {args.counts}")
    expr = load_expression(args.counts, positions=args.positions,
                           scale=args.scale)
    print(f"expression  : {expr.shape[0]} spots x {expr.shape[1]} unique genes")
    if expr.positions is None:
        print("positions   : none loaded -- maps will be skipped")

    print(f"composition : {args.composition}")
    comp = load_composition(args.composition, spots=expr.spots,
                            normalise=args.composition_normalise)
    print(f"              {comp.shape[0]} spots x {comp.shape[1]} cell types")

    progs = load_programmes(args.programmes)
    print(f"programmes  : {len(progs)} from {args.programmes}")

    ei, ej = _resolve_edges(args, expr)
    print("scoring:")
    res = run(expr, comp, progs, ei, ej, min_genes=args.min_genes,
              ret_floor=args.ret_floor, log=lambda s: print(s))

    for s in res.skipped:
        print(f"  skipped {s['programme']}: only {s['n_measured']} of its genes "
              f"are measured ({s['n_missing']} not in the panel)")

    if not res.programmes:
        raise SystemExit("no programme had enough measured genes; nothing written")

    summary = os.path.join(args.outdir, "summary.csv")
    res.summary.to_csv(summary, index=False, float_format="%.6g")
    scores = os.path.join(args.outdir, "scores.csv")
    res.scores().to_csv(scores, index_label="barcode", float_format="%.6g")
    print(f"  -> {summary}\n  -> {scores}")

    if args.no_maps:
        return 0

    from . import plots

    if expr.positions is None:
        print("maps skipped: no spot coordinates (pass --positions)")
        return 0

    for prog in res.programmes:
        plots.pair_map(res, prog, args.outdir, dpi=args.dpi)
    plots.qc_maps(res, args.outdir, dpi=args.dpi)

    want = list(res.programmes) if args.pergene_all else list(args.pergene)
    for prog in want:
        if prog not in res.fields:
            print(f"  per-gene maps skipped for {prog!r}: not scored "
                  f"(check --min-genes / gene symbols)")
            continue
        plots.pergene_maps(res, prog, args.outdir, dpi=args.dpi)
    return 0


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command != "run":
        _build_parser().print_help()
        return 2
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
