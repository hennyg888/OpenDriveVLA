"""Plotting helpers for attention heatmaps + embedding distributions.

All functions are matplotlib-only (no seaborn / no GPU dependence). Each
saves to ``out_path`` and closes the figure. Dimensions are tuned for
on-screen readability rather than print quality.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------
# Attention heatmaps
# ---------------------------------------------------------------------------


def plot_attention_heatmap_per_layer(
    attn: np.ndarray,
    group_names: List[str],
    out_path: str,
    title: str = "",
    layers_to_show: Optional[Iterable[int]] = None,
    row_normalize: bool = True,
):
    """attn: (L, O, G). Produces a grid of layer-wise heatmaps."""
    L, O, G = attn.shape
    if layers_to_show is None:
        # Pick at most 9 evenly-spaced layers.
        n = min(9, L)
        layers_to_show = list(np.linspace(0, L - 1, n).round().astype(int))
    layers_to_show = list(layers_to_show)
    n = len(layers_to_show)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.2))
    axes = np.atleast_2d(axes).reshape(rows, cols)
    vmin, vmax = 0.0, 1.0 if row_normalize else attn.max()

    for i, L_idx in enumerate(layers_to_show):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        mat = attn[L_idx].copy()  # (O, G)
        if row_normalize:
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mat = mat / row_sums
        # Display as (G, O) so groups are on y-axis.
        im = ax.imshow(mat.T, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"Layer {L_idx}", fontsize=10)
        ax.set_yticks(range(G))
        ax.set_yticklabels(group_names, fontsize=8)
        ax.set_xlabel("output token idx", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    # Hide unused subplots.
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_attention_aggregate_heatmap(
    attn: np.ndarray,
    group_names: List[str],
    out_path: str,
    title: str = "",
    row_normalize: bool = True,
):
    """attn: (L, G). Single heatmap, layers x groups."""
    mat = attn.copy()
    if row_normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums
    L, G = mat.shape
    fig, ax = plt.subplots(figsize=(max(10, L * 0.4), max(4, G * 0.35)))
    im = ax.imshow(mat.T, aspect="auto", cmap="viridis")
    ax.set_xticks(range(L))
    ax.set_xticklabels([str(i) for i in range(L)], fontsize=7)
    ax.set_yticks(range(G))
    ax.set_yticklabels(group_names, fontsize=9)
    ax.set_xlabel("layer", fontsize=10)
    ax.set_ylabel("group", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_layer_curves(
    attn: np.ndarray,
    group_names: List[str],
    out_path: str,
    title: str = "",
):
    """attn: (L, G). Line plot, x = layer, one line per group."""
    L, G = attn.shape
    mat = attn.copy()
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(max(10, L * 0.35), 5))
    cmap = plt.get_cmap("tab20")
    for gi, name in enumerate(group_names):
        ax.plot(range(L), mat[:, gi], marker=".", label=name, color=cmap(gi % 20))
    ax.set_xlabel("layer")
    ax.set_ylabel("attention mass (row-normalized)")
    ax.set_xticks(range(L))
    ax.set_xticklabels([str(i) for i in range(L)], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_tf_vs_ar_bar(
    tf_attn: np.ndarray,
    ar_attn: np.ndarray,
    group_names: List[str],
    out_path: str,
    title: str = "",
):
    """tf_attn: (L, O, G), ar_attn: (L, K, G+ext). Bars: mean over (L, output)
    of normalized per-row distribution.

    NOTE: ar_attn's last group ("output") column doubles as gen_so_far.
    For a fair comparison we drop the gen_so_far / output column from
    BOTH tensors before plotting.
    """
    G = len(group_names)
    tf = tf_attn[..., :G]
    ar = ar_attn[..., :G]

    def _mean_dist(a):
        # a: (L, *, G). Row-normalize across G first, then mean.
        row_sums = a.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        a = a / row_sums
        return a.mean(axis=(0, 1))

    tf_mean = _mean_dist(tf)
    ar_mean = _mean_dist(ar)

    x = np.arange(G)
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, G * 0.7), 4.5))
    ax.bar(x - width / 2, tf_mean, width, label="teacher-forced")
    ax.bar(x + width / 2, ar_mean, width, label="autoregressive")
    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean attention mass")
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Embedding distribution scatter
# ---------------------------------------------------------------------------


def _confidence_ellipse(ax, x, y, n_std: float = 1.96, **kwargs):
    """Add a confidence ellipse for a 2D point cloud."""
    if x.size < 2:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[0] <= 0:
        return
    angle = float(np.degrees(np.arctan2(*vecs[:, 0][::-1])))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    e = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        **kwargs,
    )
    ax.add_patch(e)


def plot_embedding_scatter(
    points_by_group: dict,
    out_path: str,
    title: str = "",
    method_name: str = "PCA",
    explained_var: Optional[Tuple[float, float]] = None,
):
    """points_by_group: {group_name: (n, 2) array}. Draws a scatter with
    one color per group and a 95%-confidence ellipse for each.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab20")
    for gi, (name, pts) in enumerate(points_by_group.items()):
        if pts.size == 0:
            continue
        color = cmap(gi % 20)
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.55, label=f"{name} (n={len(pts)})", color=color)
        _confidence_ellipse(
            ax, pts[:, 0], pts[:, 1], n_std=1.96,
            facecolor="none", edgecolor=color, lw=1.2, alpha=0.8,
        )
    if explained_var is not None:
        ax.set_xlabel(f"{method_name}-1 ({explained_var[0]*100:.1f}%)")
        ax.set_ylabel(f"{method_name}-2 ({explained_var[1]*100:.1f}%)")
    else:
        ax.set_xlabel(f"{method_name}-1")
        ax.set_ylabel(f"{method_name}-2")
    ax.legend(fontsize=8, loc="best")
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
