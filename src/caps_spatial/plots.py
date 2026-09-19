"""Figures: raw-vs-CAPS maps, per-gene maps, and QC diagnostics.

Spot fields follow one convention: a flat tissue backdrop, the value layer on
a diverging colour map centred at zero, and -- for every side-by-side pair --
**one shared colour limit for both panels**.  Scaling each panel to its own
range would make an adjusted field that collapsed to noise look as structured
as the raw one.  The limit comes from :func:`field_vmax`.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import patheffects as mpe                           # noqa: E402

TISSUE = "#D7CFC1"          # backdrop spot colour
CMAP_FIELD = "RdBu_r"       # diverging, centred at 0
FIELD_PCT = 99.0            # percentile of |value| used for the colour limit
TISSUE_S = 1.2              # backdrop marker size
VALUE_S = 2.6               # value marker size


def _halo(px=2.6):
    return [mpe.withStroke(linewidth=px, foreground="white")]


def _axis_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def field_vmax(*values, pct=FIELD_PCT):
    """Shared colour limit for a raw/CAPS pair.

    ``max`` over the panels of the ``pct``-th percentile of ``|value|``: large
    enough that neither panel clips, still driven by the bulk of the
    distribution rather than a single outlier spot.
    """
    return max(float(np.percentile(np.abs(np.asarray(v, dtype=float)), pct))
               for v in values) + 1e-9


def draw_field(ax, px, py, val, vmax):
    """One spot field on a colour scale supplied by the caller."""
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    ax.scatter(px, py, s=TISSUE_S, c=TISSUE, linewidth=0, zorder=1)
    ax.scatter(px, py, s=VALUE_S, c=np.asarray(val, float), cmap=CMAP_FIELD,
               vmin=-vmax, vmax=vmax, linewidth=0, zorder=2)
    ax.set_xlim(px.min(), px.max())
    ax.set_ylim(py.min(), py.max())
    ax.set_aspect("equal", adjustable="box")
    _axis_off(ax)


def assert_data_in_bounds(fig, tol=1e-9):
    """Every scatter point must lie inside its own axes' limits.

    A hardcoded ``set_xlim``/``set_ylim`` silently drops points when the data
    change scale.  Cheap guard, run before every save.
    """
    bad = []
    for i, ax in enumerate(fig.axes):
        for col in ax.collections:
            off = getattr(col, "get_offsets", lambda: None)()
            if off is None or len(off) == 0:
                continue
            xy = np.asarray(off, dtype=float)
            x, y = xy[:, 0], xy[:, 1]
            x0, x1 = sorted(ax.get_xlim())
            y0, y1 = sorted(ax.get_ylim())
            n_out = int(((x < x0 - tol) | (x > x1 + tol)
                         | (y < y0 - tol) | (y > y1 + tol)).sum())
            if n_out:
                bad.append((i, ax.get_title()[:45], n_out, len(x)))
    if bad:
        for i, t, n_out, tot in bad:
            print(f"  [DATA] axes#{i} {t!r}: {n_out}/{tot} points outside limits")
        raise AssertionError(f"{len(bad)} axes with clipped data")


def _save(fig, stem, dpi):
    """Write <stem>.pdf and <stem>.png."""
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    print(f"  -> {stem}.pdf / .png")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _fmt(v, spec=".2f"):
    return "n/a" if v is None or not np.isfinite(v) else format(v, spec)


def _coords(res, programme=None):
    pos = res.expr.positions
    if pos is None:
        raise ValueError(
            "no spot coordinates loaded -- pass --positions (or --edges plus "
            "positions) to draw maps")
    return pos["x"].to_numpy(float), pos["y"].to_numpy(float)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def pair_map(res, programme, outdir, dpi=300):
    """``maps/<PROGRAMME>_raw_vs_caps.{pdf,png}`` -- raw and CAPS, one scale."""
    px, py = _coords(res)
    y = res.field(programme, "raw")
    g = res.field(programme, "caps")
    vmax = field_vmax(y, g)
    f = res.fields[programme]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.1))
    for ax, val, lab, mor in ((axes[0], y, "raw score", f["moran_raw"]),
                              (axes[1], g, "CAPS-adjusted", f["moran_caps"])):
        draw_field(ax, px, py, val, vmax)
        ax.set_title(f"{lab}   Moran {mor:+.3f}", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=CMAP_FIELD,
                               norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05,
                      pad=0.03, aspect=45)
    cb.set_label("programme score (mean per-gene z)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(programme, fontsize=10.5, y=1.0)
    fig.text(0.005, -0.10,
             f"composition R² {f['r2']:.2f}    retention "
             f"{_fmt(f['retention'])}    {f['ng']} genes measured",
             fontsize=7.5, color="0.3")
    assert_data_in_bounds(fig)
    _save(fig, os.path.join(outdir, "maps",
                            f"{_safe(programme)}_raw_vs_caps"), dpi)


def pergene_maps(res, programme, outdir, dpi=300):
    """``genes/<PROGRAMME>/<GENE>_raw_vs_caps.pdf`` for every measured gene."""
    px, py = _coords(res)
    raws = res.gene_raws(programme)
    caps = res.genes(programme)
    out = os.path.join(outdir, "genes", _safe(programme))
    for gene in raws.columns:
        a = raws[gene].to_numpy(float)
        b = caps[gene].to_numpy(float)
        vmax = field_vmax(a, b)
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.9))
        for ax, val, lab in ((axes[0], a, "raw (log1p CPM z)"),
                             (axes[1], b, "CAPS residual z")):
            draw_field(ax, px, py, val, vmax)
            ax.set_title(lab, fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=CMAP_FIELD,
                                   norm=plt.Normalize(-vmax, vmax))
        cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05,
                          pad=0.03, aspect=45)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"{programme}  ·  {gene}", fontsize=10)
        assert_data_in_bounds(fig)
        _save(fig, os.path.join(out, f"{_safe(gene)}_raw_vs_caps"), dpi)


def qc_maps(res, outdir, dpi=300):
    """``qc/retention_vs_r2`` and ``qc/moran_raw_vs_caps``."""
    d = res.summary
    keep = d[np.isfinite(d["retention"])].copy()

    # -- composition R^2 vs retained spatial structure -------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    if len(keep):
        ax.scatter(keep["r2"], keep["retention"], s=16, c="#3B5B78",
                   alpha=0.85, linewidth=0)
        if len(keep) >= 8:
            bins = np.linspace(keep["r2"].min(), keep["r2"].max(), 6)
            ctr = 0.5 * (bins[1:] + bins[:-1])
            which = np.digitize(keep["r2"], bins)
            med = [keep["retention"][which == k].median() for k in range(1, 6)]
            ax.plot(ctr, med, "-o", color="#AE5F3E", lw=1.6, ms=4,
                    label="binned median")
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.text(ax.get_xlim()[0], 1.0, "  fully retained", fontsize=7.5,
            color="0.45", va="bottom")
    ax.set_xlabel("composition R² of the raw score", fontsize=8.5)
    ax.set_ylabel("retention  (Moran CAPS / Moran raw)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    dropped = int((~np.isfinite(d["retention"])).sum())
    fig.suptitle("Does retained structure shrink as the score becomes "
                 "composition-driven?", fontsize=9.5)
    if dropped:
        fig.text(0.005, -0.02, f"{dropped} programme(s) omitted: raw Moran at "
                 f"or below the 0.10 floor (retention undefined)",
                 fontsize=7, color="0.35")
    _save(fig, os.path.join(outdir, "qc", "retention_vs_r2"), dpi)

    # -- per-programme Moran, raw vs CAPS ---------------------------------
    d = d.sort_values("moran_raw", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(d) + 1.4))
    yy = np.arange(len(d))
    ax.hlines(yy, d["moran_caps"], d["moran_raw"], color="0.75", lw=1.2,
              zorder=1)
    ax.scatter(d["moran_raw"], yy, s=22, c="#3B5B78", zorder=2, label="raw")
    ax.scatter(d["moran_caps"], yy, s=22, c="#AE5F3E", zorder=3,
               label="CAPS-adjusted")
    ax.axvline(0.20, color="0.6", lw=0.8, ls="--")
    ax.text(0.20, len(d) - 0.4, " 0.20", fontsize=7.5, color="0.45")
    ax.set_yticks(yy)
    ax.set_yticklabels(d["programme"], fontsize=6.5)
    ax.set_ylim(-0.8, max(len(d) - 0.2, 1))
    ax.set_xlabel("Moran's I of the programme field", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    # legend above the axes: inside the frame it collides with the dumbbells
    # whenever the programme count is small
    ax.legend(fontsize=7.5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.suptitle("Spatial structure before and after composition adjustment",
                 fontsize=10, y=1.10)
    _save(fig, os.path.join(outdir, "qc", "moran_raw_vs_caps"), dpi)


def _safe(name):
    """Filesystem-safe stem for a programme or gene name."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))
