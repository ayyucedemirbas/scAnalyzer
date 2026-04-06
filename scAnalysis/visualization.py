from __future__ import annotations

from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns

from .core import SingleCellDataset

sns.set_theme(style="white", context="paper")

_PVAL_FLOOR = 1e-300


def _get_color_data(
    data: SingleCellDataset, color_key: str
) -> Tuple[np.ndarray, bool, str]:
    if color_key in data.obs.columns:
        vals = data.obs[color_key].values
        is_cat = pd.api.types.is_categorical_dtype(
            data.obs[color_key]
        ) or pd.api.types.is_object_dtype(data.obs[color_key])
        return vals, is_cat, color_key

    if color_key in data.var.index:
        gi = data.var.index.get_loc(color_key)
        if sp.issparse(data.X):
            vals = data.X[:, gi].toarray().flatten()
        else:
            vals = np.asarray(data.X[:, gi]).flatten()
        return vals, False, color_key

    obs_cols = list(data.obs.columns)
    n_genes = data.n_vars
    raise ValueError(
        f"Key '{color_key}' not found in obs columns or gene names. "
        f"obs columns: {obs_cols[:10]}{'…' if len(obs_cols) > 10 else ''}; "
        f"dataset has {n_genes:,} genes."
    )


def _maybe_show_or_save(
    fig: plt.Figure,
    save: Optional[str],
    ax_provided: bool,
) -> None:
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
        print(f"Saved → {save}")
    if not save and not ax_provided:
        plt.show()
    plt.close(fig)


def plot_embedding(
    data: SingleCellDataset,
    basis: str = "X_umap",
    color: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    s: int = 10,
    alpha: float = 0.8,
    figsize: Tuple[int, int] = (6, 6),
    legend_loc: str = "right margin",
    ax: Optional[plt.Axes] = None,
    save: Optional[str] = None,
) -> plt.Axes:
    if basis not in data.obsm:
        available = list(data.obsm.keys())
        raise ValueError(f"'{basis}' not found in obsm. Available: {available}")

    coords = data.obsm[basis]
    x, y = coords[:, 0], coords[:, 1]

    ax_provided = ax is not None
    if not ax_provided:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if color:
        values, is_cat, label = _get_color_data(data, color)

        if is_cat:
            df_plot = pd.DataFrame({"x": x, "y": y, "cat": values})
            sns.scatterplot(
                data=df_plot,
                x="x",
                y="y",
                hue="cat",
                s=s,
                alpha=alpha,
                ax=ax,
                palette="tab20",
                linewidth=0,
            )
            if legend_loc == "right margin":
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                    frameon=False,
                )
            elif legend_loc == "on data":
                ax.legend().remove()
                for cat in np.unique(values):
                    mask = values == cat
                    cx, cy = np.mean(x[mask]), np.mean(y[mask])
                    txt = ax.text(
                        cx,
                        cy,
                        str(cat),
                        fontsize=9,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        color="black",
                    )
                    txt.set_path_effects(
                        [PathEffects.withStroke(linewidth=3, foreground="white")]
                    )
            else:
                ax.legend(loc=legend_loc, frameon=False)
        else:
            sc = ax.scatter(x, y, c=values, s=s, cmap=cmap, alpha=alpha, linewidths=0)
            plt.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
    else:
        ax.scatter(x, y, s=s, alpha=alpha, c="steelblue", linewidths=0)

    ax.set_xlabel(f"{basis} 1", labelpad=4)
    ax.set_ylabel(f"{basis} 2", labelpad=4)
    ax.set_title(title or (f"{basis}" + (f" — {color}" if color else "")))
    sns.despine(ax=ax)

    if not ax_provided:
        _maybe_show_or_save(fig, save, ax_provided)

    return ax


def plot_umap(data: SingleCellDataset, **kwargs) -> plt.Axes:
    return plot_embedding(data, basis="X_umap", **kwargs)


def plot_tsne(data: SingleCellDataset, **kwargs) -> plt.Axes:
    return plot_embedding(data, basis="X_tsne", **kwargs)


def plot_pca(data: SingleCellDataset, **kwargs) -> plt.Axes:
    return plot_embedding(data, basis="X_pca", **kwargs)


def plot_violin(
    data: SingleCellDataset,
    keys: Union[str, List[str]],
    groupby: str,
    rotation: int = 90,
    figsize: Optional[Tuple[int, int]] = None,
    save: Optional[str] = None,
) -> plt.Figure:
    if isinstance(keys, str):
        keys = [keys]

    groups = data.obs[groupby].values
    rows = []
    for key in keys:
        vals, _, _ = _get_color_data(data, key)
        rows.append(pd.DataFrame({"Expression": vals, "Group": groups, "Feature": key}))

    plot_df = pd.concat(rows, ignore_index=True)

    if figsize is None:
        figsize = (max(4, len(np.unique(groups)) * len(keys) * 0.6 + 1), 5)

    fig, ax = plt.subplots(figsize=figsize)

    hue_col = "Feature" if len(keys) > 1 else None
    sns.violinplot(
        data=plot_df,
        x="Group",
        y="Expression",
        hue=hue_col,
        inner="quartile",
        density_norm="width",
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    ax.set_title(f"{', '.join(keys)} by {groupby}")
    sns.despine(ax=ax)
    fig.tight_layout()

    _maybe_show_or_save(fig, save, ax_provided=False)
    return fig


def plot_heatmap(
    data: SingleCellDataset,
    var_names: List[str],
    groupby: str,
    use_raw: bool = False,
    standard_scale: str = "var",
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None,
    save: Optional[str] = None,
) -> plt.Figure:
    if groupby not in data.obs:
        raise ValueError(f"'{groupby}' not in obs. Available: {list(data.obs.columns)}")

    valid = [v for v in var_names if v in data.var.index]
    if not valid:
        raise ValueError("None of the requested var_names found in data.")
    idx = [data.var.index.get_loc(v) for v in valid]

    src = (
        (data.raw.X if hasattr(data.raw, "X") else data.raw)
        if (use_raw and data.raw is not None)
        else data.X
    )
    X_sub = src[:, idx]
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()

    groups = data.obs[groupby]
    ug = np.sort(groups.unique())
    means = np.vstack([X_sub[(groups == g).values].mean(axis=0) for g in ug])
    df_hm = pd.DataFrame(means, index=ug, columns=valid)

    if standard_scale == "var":
        mu = df_hm.mean()
        sd = df_hm.std().replace(0, 1)
        df_hm = (df_hm - mu) / sd

    if figsize is None:
        figsize = (max(4, len(valid) * 0.5 + 2), max(3, len(ug) * 0.5 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_hm, cmap=cmap, ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title(f"Mean expression by {groupby}")
    ax.set_xlabel("Genes")
    ax.set_ylabel(groupby)
    fig.tight_layout()

    _maybe_show_or_save(fig, save, ax_provided=False)
    return fig


def plot_dotplot(
    data: SingleCellDataset,
    var_names: List[str],
    groupby: str,
    cmap: str = "Reds",
    standard_scale: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    save: Optional[str] = None,
) -> plt.Figure:
    if groupby not in data.obs:
        raise ValueError(f"'{groupby}' not in obs.")

    valid = [v for v in var_names if v in data.var.index]
    idx = [data.var.index.get_loc(v) for v in valid]
    X_sub = data.X[:, idx]
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()

    groups = data.obs[groupby]
    ug = np.sort(groups.unique())

    rows = []
    for g in ug:
        mask = (groups == g).values
        gd = X_sub[mask]
        frac = (gd > 0).mean(axis=0)
        mu = gd.mean(axis=0)
        for j, v in enumerate(valid):
            rows.append({"Group": g, "Gene": v, "Fraction": frac[j], "MeanExpr": mu[j]})

    df = pd.DataFrame(rows)

    if standard_scale:
        df["MeanExpr"] = df.groupby("Gene")["MeanExpr"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )

    if figsize is None:
        figsize = (max(4, len(valid) * 0.8 + 1), max(3, len(ug) * 0.5 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        df["Gene"],
        df["Group"],
        s=df["Fraction"] * 400,
        c=df["MeanExpr"],
        cmap=cmap,
        alpha=0.9,
        linewidths=0.3,
        edgecolors="grey",
    )
    plt.colorbar(scatter, ax=ax, label="Scaled mean expr", fraction=0.03, pad=0.04)
    ax.set_xlabel("Genes")
    ax.set_ylabel(groupby)
    ax.set_title(f"Dotplot — {groupby}")
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()

    _maybe_show_or_save(fig, save, ax_provided=False)
    return fig


def volcano_plot(
    data: SingleCellDataset,
    group: str,
    key: str = "rank_genes_groups",
    pval_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    top_n_genes: int = 10,
    figsize: Tuple[int, int] = (8, 6),
    save: Optional[str] = None,
) -> plt.Figure:
    if key not in data.uns:
        raise ValueError(f"'{key}' not found. Run rank_genes_groups() first.")
    group = str(group)
    if group not in data.uns[key]:
        raise ValueError(f"Group '{group}' not found in '{key}'.")

    df = data.uns[key][group].copy()
    df["neg_log10_pval"] = -np.log10(df["pvals_adj"].clip(lower=_PVAL_FLOOR))

    cat = np.where(
        (df["pvals_adj"] < pval_threshold) & (df["logfoldchanges"] > lfc_threshold),
        "Up",
        np.where(
            (df["pvals_adj"] < pval_threshold)
            & (df["logfoldchanges"] < -lfc_threshold),
            "Down",
            "NS",
        ),
    )
    df["category"] = cat

    colours = {"NS": "#AAAAAA", "Up": "#E74C3C", "Down": "#3498DB"}

    fig, ax = plt.subplots(figsize=figsize)
    for c, col in colours.items():
        sub = df[df["category"] == c]
        ax.scatter(
            sub["logfoldchanges"],
            sub["neg_log10_pval"],
            c=col,
            s=8,
            alpha=0.6,
            label=c,
            linewidths=0,
        )

    ax.axhline(-np.log10(pval_threshold), color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(lfc_threshold, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(-lfc_threshold, color="black", lw=0.8, ls="--", alpha=0.5)

    sig = df[df["category"] != "NS"].nsmallest(top_n_genes, "pvals_adj")
    for _, row in sig.iterrows():
        ax.text(
            row["logfoldchanges"] + 0.02,
            row["neg_log10_pval"],
            row["names"],
            fontsize=7,
            alpha=0.85,
        )

    ax.set_xlabel("Log₂ Fold Change")
    ax.set_ylabel("−log₁₀ Adjusted P-value")
    ax.set_title(f"Volcano — group '{group}'")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()

    _maybe_show_or_save(fig, save, ax_provided=False)
    return fig


def plot_qc_violin(
    data: SingleCellDataset,
    metrics: Optional[List[str]] = None,
    groupby: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    save: Optional[str] = None,
) -> plt.Figure:
    if metrics is None:
        pct_cols = [c for c in data.obs.columns if c.startswith("pct_counts_")]
        metrics = [
            c
            for c in ["n_genes_by_counts", "total_counts"] + pct_cols
            if c in data.obs.columns
        ]
    if not metrics:
        raise ValueError("No QC metrics found. Run calculate_qc_metrics() first.")

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if groupby and groupby in data.obs.columns:
            pdata = pd.DataFrame({"v": data.obs[metric], "g": data.obs[groupby]})
            sns.violinplot(data=pdata, x="g", y="v", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            sns.violinplot(y=data.obs[metric], ax=ax, color="lightblue")
        ax.set_title(metric)
        ax.set_ylabel("")
        ax.set_xlabel("")
        sns.despine(ax=ax)

    fig.tight_layout()
    _maybe_show_or_save(fig, save, ax_provided=False)
    return fig


def plot_highest_expr_genes(
    data: SingleCellDataset,
    n_top: int = 20,
    figsize: Tuple[int, int] = (6, 8),
    save: Optional[str] = None,
) -> plt.Figure:
    X = data.X
    gene_means = np.ravel(X.mean(axis=0)) if sp.issparse(X) else X.mean(axis=0)

    top_idx = np.argsort(gene_means)[::-1][:n_top]
    top_genes = data.var.index[top_idx]
    top_vals = gene_means[top_idx]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(top_genes))
    ax.barh(y, top_vals, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(top_genes)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Expression")
    ax.set_title(f"Top {n_top} Highest Expressed Genes")
    sns.despine(ax=ax)
    fig.tight_layout()

    _maybe_show_or_save(fig, save, ax_provided=False)
    return fig
