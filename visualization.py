from typing import List, Optional, Tuple, Union

import matplotlib.patheffects as PathEffects  # <--- Added explicit import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns

from core import SingleCellDataset

# Set default aesthetic
sns.set_theme(style="white", context="paper")


def _get_color_data(data: SingleCellDataset, color_key: str):
    """
    Helper to extract color data (gene expression or metadata) for plotting.
    Returns: values (array), is_categorical (bool), label (str)
    """
    # 1. Check in obs (metadata/clusters)
    if color_key in data.obs.columns:
        values = data.obs[color_key].values
        is_categorical = pd.api.types.is_categorical_dtype(
            data.obs[color_key]
        ) or pd.api.types.is_object_dtype(data.obs[color_key])
        return values, is_categorical, color_key

    # 2. Check in var (gene expression)
    if color_key in data.var.index:
        # Locate gene index
        gene_idx = data.var.index.get_loc(color_key)

        # Extract column
        if sp.issparse(data.X):
            values = data.X[:, gene_idx].toarray().flatten()
        else:
            values = data.X[:, gene_idx]

        return values, False, color_key  # Gene expression is continuous

    raise ValueError(f"Key '{color_key}' not found in obs or var_names.")


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
):
    """
    Generic plotter for 2D embeddings (UMAP, t-SNE, PCA).
    """
    if basis not in data.obsm:
        raise ValueError(f"{basis} not found in data.obsm.")

    # Get coordinates
    coords = data.obsm[basis]
    x, y = coords[:, 0], coords[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Handle coloring
    if color:
        values, is_cat, label = _get_color_data(data, color)

        if is_cat:
            # Categorical plot (e.g., clusters)
            df_plot = pd.DataFrame({"x": x, "y": y, "category": values})
            sns.scatterplot(
                data=df_plot,
                x="x",
                y="y",
                hue="category",
                s=s,
                alpha=alpha,
                ax=ax,
                palette="tab20",
                edgecolor=None,
            )
            if legend_loc == "right margin":
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            elif legend_loc == "on data":
                ax.legend().remove()
                for cat in np.unique(values):
                    mask = values == cat
                    cx, cy = np.mean(x[mask]), np.mean(y[mask])
                    # Add text with white outline for readability
                    txt = ax.text(
                        cx,
                        cy,
                        str(cat),
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        color="black",
                    )
                    # Use the explicitly imported PathEffects
                    txt.set_path_effects(
                        [PathEffects.withStroke(linewidth=3, foreground="white")]
                    )

        else:
            # Continuous plot (e.g., gene expression)
            sc = ax.scatter(x, y, c=values, s=s, cmap=cmap, alpha=alpha, edgecolor=None)
            plt.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)

    else:
        # No color
        ax.scatter(x, y, s=s, alpha=alpha, c="gray")

    ax.set_xlabel(f"{basis}_1")
    ax.set_ylabel(f"{basis}_2")
    ax.set_title(
        title if title else f"{basis} colored by {color if color else 'index'}"
    )

    # Clean spines
    sns.despine(ax=ax)

    # Save logic
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {save}")

    if ax is None:
        plt.show()


def plot_umap(data, **kwargs):
    plot_embedding(data, basis="X_umap", **kwargs)


def plot_tsne(data, **kwargs):
    plot_embedding(data, basis="X_tsne", **kwargs)


def plot_pca(data, **kwargs):
    plot_embedding(data, basis="X_pca", **kwargs)


def plot_violin(
    data: SingleCellDataset,
    keys: Union[str, List[str]],
    groupby: str,
    rotation: int = 90,
    save: Optional[str] = None,
):
    """
    Violin plot of gene expression or metadata per group.
    """
    if isinstance(keys, str):
        keys = [keys]

    # Prepare data
    plot_data = []
    groups = data.obs[groupby].values

    for key in keys:
        vals, _, _ = _get_color_data(data, key)
        df_temp = pd.DataFrame({"Expression": vals, "Group": groups, "Gene": key})
        plot_data.append(df_temp)

    final_df = pd.concat(plot_data)

    plt.figure(figsize=(len(keys) * 2 + 2, 6))

    sns.violinplot(
        data=final_df,
        x="Group",
        y="Expression",
        hue="Gene",
        split=False,
        inner="quartile",
        density_norm="width",
    )

    plt.xticks(rotation=rotation)
    plt.title(f"Expression of {', '.join(keys)} by {groupby}")
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {save}")

    plt.show()


def plot_heatmap(
    data: SingleCellDataset,
    var_names: List[str],
    groupby: str,
    use_raw: bool = False,
    standard_scale: str = "var",
    cmap: str = "viridis",
    save: Optional[str] = None,
):
    """
    Plots a heatmap of the mean expression per group.
    """
    if groupby not in data.obs:
        raise ValueError(f"Group {groupby} not found.")

    valid_vars = [v for v in var_names if v in data.var.index]
    var_indices = [data.var.index.get_loc(v) for v in valid_vars]

    if use_raw and data.raw is not None:
        raw_X = data.raw.X if hasattr(data.raw, "X") else data.raw
        X_subset = raw_X[:, var_indices]
    else:
        X_subset = data.X[:, var_indices]

    if sp.issparse(X_subset):
        X_subset = X_subset.toarray()

    groups = data.obs[groupby]
    unique_groups = np.sort(groups.unique())

    mean_expression = []

    for g in unique_groups:
        mask = (groups == g).values
        mean_expr = np.mean(X_subset[mask, :], axis=0)
        mean_expression.append(mean_expr)

    heatmap_data = np.array(mean_expression)
    df_heatmap = pd.DataFrame(heatmap_data, index=unique_groups, columns=valid_vars)

    if standard_scale == "var":
        df_heatmap = (df_heatmap - df_heatmap.mean()) / df_heatmap.std()

    plt.figure(figsize=(len(valid_vars) * 0.5 + 2, len(unique_groups) * 0.5 + 2))
    sns.heatmap(df_heatmap, cmap=cmap, xticklabels=True, yticklabels=True)
    plt.title(f"Mean expression by {groupby}")
    plt.xlabel("Genes")
    plt.ylabel(groupby)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {save}")

    plt.show()


def plot_dotplot(
    data: SingleCellDataset,
    var_names: List[str],
    groupby: str,
    cmap: str = "Reds",
    standard_scale: bool = True,
    save: Optional[str] = None,
):
    """
    Dotplot visualization.
    """
    if groupby not in data.obs:
        raise ValueError(f"Group {groupby} not found.")

    valid_vars = [v for v in var_names if v in data.var.index]
    var_indices = [data.var.index.get_loc(v) for v in valid_vars]

    X_subset = data.X[:, var_indices]
    if sp.issparse(X_subset):
        X_subset = X_subset.toarray()

    groups = data.obs[groupby]
    unique_groups = np.sort(groups.unique())

    rows, cols, fraction, mean_expr = [], [], [], []

    for i, g in enumerate(unique_groups):
        mask = (groups == g).values
        group_data = X_subset[mask, :]

        frac = np.count_nonzero(group_data, axis=0) / group_data.shape[0]
        mu = np.mean(group_data, axis=0)

        for j, v in enumerate(valid_vars):
            rows.append(g)
            cols.append(v)
            fraction.append(frac[j])
            mean_expr.append(mu[j])

    df_dot = pd.DataFrame(
        {"Group": rows, "Gene": cols, "Fraction": fraction, "MeanExpression": mean_expr}
    )

    if standard_scale:
        df_dot["MeanExpression"] = df_dot.groupby("Gene")["MeanExpression"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )

    plt.figure(figsize=(len(valid_vars) * 0.8 + 1, len(unique_groups) * 0.5 + 1))

    sns.scatterplot(
        data=df_dot,
        x="Gene",
        y="Group",
        size="Fraction",
        hue="MeanExpression",
        sizes=(20, 200),
        palette=cmap,
        marker="o",
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title(f"Dotplot by {groupby}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
        print(f"Saved plot to {save}")

    plt.show()


def volcano_plot(
    data: SingleCellDataset,
    group: str,
    key: str = 'rank_genes_groups',
    pval_threshold: float = 0.05,
    lfc_threshold: float = 0.5,
    top_n_genes: int = 10,
    figsize: Tuple[int, int] = (8, 6),
    save: Optional[str] = None,
):
    """
    Volcano plot for differential expression results.
    
    Parameters
    ----------
    data : SingleCellDataset
        Dataset with DE results.
    group : str
        Which group to plot results for.
    key : str, default: 'rank_genes_groups'
        Key in uns containing DE results.
    pval_threshold : float, default: 0.05
        Significance threshold for adjusted p-value.
    lfc_threshold : float, default: 0.5
        Log fold change threshold.
    top_n_genes : int, default: 10
        Number of top genes to label.
    figsize : tuple, default: (8, 6)
        Figure size.
    save : str, optional
        Path to save figure.
    """
    
    if key not in data.uns:
        raise ValueError(f"{key} not found. Run rank_genes_groups() first.")
    
    if group not in data.uns[key]:
        raise ValueError(f"Group {group} not found in {key}.")
    
    df = data.uns[key][group].copy()
    
    # Prepare data
    df['-log10(pval)'] = -np.log10(df['pvals_adj'] + 1e-300)  # Avoid log(0)
    
    # Color categories
    df['category'] = 'Not significant'
    df.loc[
        (df['pvals_adj'] < pval_threshold) & (df['logfoldchanges'] > lfc_threshold),
        'category'
    ] = 'Up-regulated'
    df.loc[
        (df['pvals_adj'] < pval_threshold) & (df['logfoldchanges'] < -lfc_threshold),
        'category'
    ] = 'Down-regulated'
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'Not significant': 'gray', 'Up-regulated': 'red', 'Down-regulated': 'blue'}
    
    for cat in ['Not significant', 'Down-regulated', 'Up-regulated']:
        subset = df[df['category'] == cat]
        ax.scatter(
            subset['logfoldchanges'],
            subset['-log10(pval)'],
            c=colors[cat],
            alpha=0.6,
            s=10,
            label=cat
        )
    
    # Add threshold lines
    ax.axhline(-np.log10(pval_threshold), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(lfc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(-lfc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Label top genes
    significant = df[df['category'] != 'Not significant'].sort_values('pvals_adj')
    for i, row in significant.head(top_n_genes).iterrows():
        ax.text(
            row['logfoldchanges'],
            row['-log10(pval)'],
            row['names'],
            fontsize=8,
            alpha=0.8
        )
    
    ax.set_xlabel('Log Fold Change', fontsize=12)
    ax.set_ylabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title(f'Volcano Plot: {group}', fontsize=14)
    ax.legend()
    
    sns.despine()
    plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save}")
    
    plt.show()


def plot_qc_violin(
    data: SingleCellDataset,
    metrics: List[str] = None,
    groupby: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    save: Optional[str] = None,
):
    """
    Violin plots for QC metrics.
    
    Parameters
    ----------
    data : SingleCellDataset
        Dataset with QC metrics calculated.
    metrics : List[str], optional
        List of metrics to plot. If None, uses default metrics.
    groupby : str, optional
        Group by this column (e.g., 'batch').
    figsize : tuple, default: (12, 4)
        Figure size.
    save : str, optional
        Path to save figure.
    """
    
    if metrics is None:
        # Default metrics
        available = ['n_genes_by_counts', 'total_counts']
        pct_cols = [c for c in data.obs.columns if c.startswith('pct_counts_')]
        metrics = available + pct_cols
        metrics = [m for m in metrics if m in data.obs.columns]
    
    if len(metrics) == 0:
        raise ValueError("No metrics found. Run calculate_qc_metrics() first.")
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if groupby and groupby in data.obs.columns:
            # Grouped violin
            plot_data = pd.DataFrame({
                'value': data.obs[metric],
                'group': data.obs[groupby]
            })
            sns.violinplot(data=plot_data, x='group', y='value', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        else:
            # Single violin
            sns.violinplot(y=data.obs[metric], ax=ax, color='lightblue')
        
        ax.set_title(metric)
        ax.set_ylabel('Value')
        ax.set_xlabel('')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save}")
    
    plt.show()


def plot_highest_expr_genes(
    data: SingleCellDataset,
    n_top: int = 20,
    figsize: Tuple[int, int] = (6, 8),
    save: Optional[str] = None,
):
    """
    Bar plot of genes with highest expression.
    
    Parameters
    ----------
    data : SingleCellDataset
        Annotated data matrix.
    n_top : int, default: 20
        Number of top genes to show.
    figsize : tuple, default: (6, 8)
        Figure size.
    save : str, optional
        Path to save figure.
    """
    
    X = data.X
    
    # Calculate mean expression per gene
    if sp.issparse(X):
        gene_means = np.ravel(X.mean(axis=0))
    else:
        gene_means = X.mean(axis=0)
    
    # Get top genes
    top_indices = np.argsort(gene_means)[::-1][:n_top]
    top_genes = data.var.index[top_indices]
    top_values = gene_means[top_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_genes))
    ax.barh(y_pos, top_values, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_genes)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Expression')
    ax.set_title(f'Top {n_top} Highest Expressed Genes')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {save}")
    
    plt.show()
