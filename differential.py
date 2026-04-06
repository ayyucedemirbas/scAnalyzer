from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from core import SingleCellDataset


def rank_genes_groups(
    data: SingleCellDataset,
    groupby: str,
    groups: Union[str, List[str]] = "all",
    reference: str = "rest",
    method: str = "t-test",
    n_genes: Optional[int] = None,
    key_added: str = "rank_genes_groups",
    use_raw: bool = True,
) -> SingleCellDataset:
    if groupby not in data.obs:
        raise ValueError(
            f"Column '{groupby}' not found in obs. "
            f"Available: {list(data.obs.columns)}"
        )

    labels = data.obs[groupby]
    unique_labels = np.asarray(labels.unique())

    if groups != "all":
        groups = list(groups)
        missing = [g for g in groups if g not in unique_labels]
        if missing:
            raise ValueError(f"Groups not found in '{groupby}': {missing}")
        unique_labels = np.asarray([g for g in unique_labels if g in groups])

    if use_raw and data.raw is not None:
        X_use = data.raw.X if hasattr(data.raw, "X") else data.raw
        var_names = (
            data.raw.var.index
            if hasattr(data.raw, "var") and not callable(data.raw.var)
            else data.var.index
        )
    else:
        X_use = data.X
        var_names = data.var.index

    n_genes_total = X_use.shape[1]
    is_sparse = sp.issparse(X_use)

    print(
        f"Differential: Ranking genes for {len(unique_labels)} groups "
        f"using {method} (use_raw={use_raw}) …"
    )

    results: Dict[str, pd.DataFrame] = {}

    for group in unique_labels:
        print(f"  … processing group '{group}'")

        group_mask = (labels == group).values
        if reference == "rest":
            rest_mask = ~group_mask
        else:
            if reference not in unique_labels and reference not in labels.values:
                raise ValueError(
                    f"Reference group '{reference}' not found in '{groupby}'."
                )
            rest_mask = (labels == reference).values

        X_g = X_use[group_mask, :]
        X_r = X_use[rest_mask, :]
        n_g = X_g.shape[0]
        n_r = X_r.shape[0]

        if is_sparse:
            mean_g = np.ravel(X_g.mean(axis=0))
            mean_r = np.ravel(X_r.mean(axis=0))
            pct_g = np.ravel((X_g > 0).sum(axis=0)) / n_g
            pct_r = np.ravel((X_r > 0).sum(axis=0)) / n_r
        else:
            mean_g = X_g.mean(axis=0)
            mean_r = X_r.mean(axis=0)
            pct_g = (X_g > 0).mean(axis=0)
            pct_r = (X_r > 0).mean(axis=0)

        lfc = mean_g - mean_r

        if method == "t-test":
            scores, pvals = _vectorised_ttest(
                X_g, X_r, mean_g, mean_r, is_sparse, n_g, n_r
            )

        elif method == "wilcoxon":
            scores, pvals = _vectorised_wilcoxon(X_g, X_r, is_sparse)

        else:
            raise ValueError(f"method must be 't-test' or 'wilcoxon', got '{method}'.")

        pvals = np.where(np.isnan(pvals), 1.0, pvals)

        _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

        df = pd.DataFrame(
            {
                "names": var_names,
                "scores": scores,
                "logfoldchanges": lfc,
                "pvals": pvals,
                "pvals_adj": pvals_adj,
                "pct_in": pct_g,
                "pct_out": pct_r,
            }
        )
        df = df.sort_values("scores", ascending=False).reset_index(drop=True)

        if n_genes is not None:
            df = df.head(n_genes)

        results[str(group)] = df

    data.uns[key_added] = results
    return data


def _vectorised_ttest(X_g, X_r, mean_g, mean_r, is_sparse: bool, n_g: int, n_r: int):
    if is_sparse:
        var_g = np.ravel(X_g.power(2).mean(axis=0)) - mean_g**2
        var_r = np.ravel(X_r.power(2).mean(axis=0)) - mean_r**2
    else:
        var_g = np.var(X_g, axis=0)
        var_r = np.var(X_r, axis=0)

    var_g = np.clip(var_g, 1e-12, None)
    var_r = np.clip(var_r, 1e-12, None)

    se = np.sqrt(var_g / n_g + var_r / n_r)
    t_scores = (mean_g - mean_r) / se

    df_welch = (var_g / n_g + var_r / n_r) ** 2 / (
        (var_g / n_g) ** 2 / (n_g - 1) + (var_r / n_r) ** 2 / max(n_r - 1, 1)
    )
    pvals = 2 * stats.t.sf(np.abs(t_scores), df=df_welch)
    return t_scores, pvals


def _vectorised_wilcoxon(X_g, X_r, is_sparse: bool):
    if is_sparse:
        g_arr = X_g.toarray()
        r_arr = X_r.toarray()
    else:
        g_arr = np.asarray(X_g)
        r_arr = np.asarray(X_r)

    n_g, n_genes = g_arr.shape
    n_r = r_arr.shape[0]

    combined = np.vstack([g_arr, r_arr])

    from scipy.stats import rankdata

    ranks = np.apply_along_axis(rankdata, 0, combined)

    rank_sum_g = ranks[:n_g].sum(axis=0)
    expected = n_g * (n_g + n_r + 1) / 2.0
    std_dev = np.sqrt(n_g * n_r * (n_g + n_r + 1) / 12.0)

    z_scores = (rank_sum_g - expected) / std_dev
    pvals = 2 * stats.norm.sf(np.abs(z_scores))

    return z_scores, pvals


def get_marker_genes(
    data: SingleCellDataset,
    group: str,
    key: str = "rank_genes_groups",
    pval_cutoff: float = 0.05,
    lfc_cutoff: float = 0.5,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    if key not in data.uns:
        raise ValueError(
            f"Key '{key}' not found in uns. " "Run rank_genes_groups() first."
        )
    group = str(group)
    if group not in data.uns[key]:
        available = list(data.uns[key].keys())
        raise ValueError(f"Group '{group}' not found. Available: {available}")

    df = data.uns[key][group]
    mask = (df["pvals_adj"] < pval_cutoff) & (df["logfoldchanges"] > lfc_cutoff)
    result = df[mask]

    if top_n is not None:
        result = result.head(top_n)

    return result.reset_index(drop=True)
