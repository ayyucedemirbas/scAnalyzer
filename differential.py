from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from core import SingleCellDataset


def rank_genes_groups(
    data: SingleCellDataset,
    groupby: str,
    groups: Union[str, list] = "all",
    reference: str = "rest",
    method: str = "t-test",
    n_genes: Optional[int] = None,
    key_added: str = "rank_genes_groups",
    use_raw: bool = True,
) -> SingleCellDataset:
    """
    Rank genes for characterizing groups.

    Args:
        data: The SingleCellDataset.
        groupby: The key of the observations grouping to consider.
        groups: Subset of groups, e.g. ['g1', 'g2', 'g3'], to which comparison shall be restricted.
        reference: If 'rest', compare each group to the union of the rest of the group.
        method: 't-test', 'wilcoxon', or 'logreg'.
        n_genes: The number of genes that appear in the returned tables.
    """

    if groupby not in data.obs:
        raise ValueError(f"groupby key '{groupby}' not found in obs.")

    # Get labels
    labels = data.obs[groupby]
    unique_labels = labels.unique()

    if groups != "all":
        unique_labels = [g for g in unique_labels if g in groups]

    # FIX: Handle data.raw being a SingleCellDataset object
    if use_raw and data.raw is not None:
        if hasattr(data.raw, "X"):  # It's a SingleCellDataset
            X = data.raw.X
            # Use var names from raw if possible, else fall back
            var_names = (
                data.raw.var.index if hasattr(data.raw, "var") else data.var.index
            )
        else:  # It's just a matrix
            X = data.raw
            var_names = data.var.index
    else:
        X = data.X
        var_names = data.var.index

    # Initialize results storage
    results = {}

    print(
        f"Differential: Ranking genes for {len(unique_labels)} groups using {method}..."
    )

    # Iterate over each group
    for group in unique_labels:
        print(f"  ... processing group {group}")

        # Create masks
        group_mask = (labels == group).values

        if reference == "rest":
            rest_mask = ~group_mask
        else:
            if reference not in unique_labels:
                raise ValueError(f"Reference group {reference} not found.")
            rest_mask = (labels == reference).values

        # Split data
        X_group = X[group_mask, :]
        X_rest = X[rest_mask, :]

        n_group = X_group.shape[0]
        n_rest = X_rest.shape[0]

        # Calculate basic stats (Mean, Pct)
        if sp.issparse(X):
            mean_group = np.ravel(X_group.mean(axis=0))
            mean_rest = np.ravel(X_rest.mean(axis=0))
        else:
            mean_group = np.mean(X_group, axis=0)
            mean_rest = np.mean(X_rest, axis=0)

        logfoldchanges = mean_group - mean_rest

        # Percentage of cells expressing gene
        if sp.issparse(X):
            pct_group = np.ravel((X_group > 0).sum(axis=0)) / n_group
            pct_rest = np.ravel((X_rest > 0).sum(axis=0)) / n_rest
        else:
            pct_group = np.count_nonzero(X_group, axis=0) / n_group
            pct_rest = np.count_nonzero(X_rest, axis=0) / n_rest

        # Statistical Tests
        if method == "t-test":
            if sp.issparse(X):
                mean_sq_group = np.ravel(X_group.power(2).mean(axis=0))
                mean_sq_rest = np.ravel(X_rest.power(2).mean(axis=0))
                var_group = mean_sq_group - mean_group**2
                var_rest = mean_sq_rest - mean_rest**2
            else:
                var_group = np.var(X_group, axis=0)
                var_rest = np.var(X_rest, axis=0)

            var_group[var_group == 0] = 1e-12
            var_rest[var_rest == 0] = 1e-12

            denominator = np.sqrt((var_group / n_group) + (var_rest / n_rest))
            t_scores = (mean_group - mean_rest) / denominator

            pvals = 2 * stats.t.sf(np.abs(t_scores), df=(n_group + n_rest - 2))

        elif method == "wilcoxon":
            X_csc = X.tocsc() if sp.issparse(X) else X
            pvals = np.zeros(len(var_names))
            scores = np.zeros(len(var_names))

            for i in range(len(var_names)):
                g_col = X_csc[:, i]
                if sp.issparse(g_col):
                    g_col = g_col.toarray().flatten()

                x = g_col[group_mask]
                y = g_col[rest_mask]

                try:
                    s, p = stats.ranksums(x, y)
                    scores[i] = s
                    pvals[i] = p
                except ValueError:
                    scores[i] = 0
                    pvals[i] = 1.0

        else:
            raise ValueError("Method must be 't-test' or 'wilcoxon'.")

        pvals[np.isnan(pvals)] = 1.0
        _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

        # Organise Result
        df = pd.DataFrame(
            {
                "names": var_names,
                "scores": t_scores if method == "t-test" else scores,
                "logfoldchanges": logfoldchanges,
                "pvals": pvals,
                "pvals_adj": pvals_adj,
                "pct_in": pct_group,
                "pct_out": pct_rest,
            }
        )

        df = df.sort_values("scores", ascending=False)

        if n_genes:
            df = df.head(n_genes)

        results[group] = df

    data.uns[key_added] = results
    return data


def get_marker_genes(
    data: SingleCellDataset,
    group: str,
    key: str = "rank_genes_groups",
    pval_cutoff: float = 0.05,
    lfc_cutoff: float = 0.5,
) -> pd.DataFrame:
    """
    Retrieves the table of marker genes for a specific group, filtered by thresholds.
    """
    if key not in data.uns:
        raise ValueError(f"Key {key} not found in uns. Run rank_genes_groups first.")

    df = data.uns[key][group]

    mask = (df["pvals_adj"] < pval_cutoff) & (df["logfoldchanges"] > lfc_cutoff)
    return df[mask]
