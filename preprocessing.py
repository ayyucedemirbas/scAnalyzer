from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from core import SingleCellDataset


def calculate_qc_metrics(
    data: SingleCellDataset,
    qc_vars: Optional[List[str]] = None,
    inplace: bool = True,
) -> Optional[SingleCellDataset]:
    if not inplace:
        data = data.copy()

    X = data.X

    if sp.issparse(X):
        n_genes = X.getnnz(axis=1)
        total = np.ravel(X.sum(axis=1))
    else:
        n_genes = np.count_nonzero(X, axis=1)
        total = np.ravel(X.sum(axis=1))

    data.obs["n_genes_by_counts"] = n_genes
    data.obs["total_counts"] = total

    if qc_vars:
        for prefix in qc_vars:
            gene_mask = data.var.index.str.startswith(prefix)
            if gene_mask.sum() == 0:
                data.obs[f"pct_counts_{prefix}"] = 0.0
                continue
            if sp.issparse(X):
                prefix_counts = np.ravel(X[:, gene_mask].sum(axis=1))
            else:
                prefix_counts = X[:, gene_mask].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(total > 0, prefix_counts / total * 100, 0.0)
            data.obs[f"pct_counts_{prefix}"] = pct

    return None if inplace else data


def filter_cells(
    data: SingleCellDataset,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    max_pct_mito: Optional[float] = None,
) -> SingleCellDataset:
    mask = np.ones(data.n_obs, dtype=bool)

    def _require(col: str) -> None:
        if col not in data.obs.columns:
            raise ValueError(
                f"Column '{col}' not found — run calculate_qc_metrics() first."
            )

    if min_counts is not None:
        _require("total_counts")
        mask &= data.obs["total_counts"].values >= min_counts
    if max_counts is not None:
        _require("total_counts")
        mask &= data.obs["total_counts"].values <= max_counts
    if min_genes is not None:
        _require("n_genes_by_counts")
        mask &= data.obs["n_genes_by_counts"].values >= min_genes
    if max_genes is not None:
        _require("n_genes_by_counts")
        mask &= data.obs["n_genes_by_counts"].values <= max_genes

    if max_pct_mito is not None:
        mito_col = next(
            (
                c
                for c in data.obs.columns
                if c.startswith("pct_counts_MT") or c.startswith("pct_counts_mt")
            ),
            None,
        )
        if mito_col is None:
            raise ValueError(
                "No mitochondrial percentage column found. "
                "Run calculate_qc_metrics(qc_vars=['MT-']) first."
            )
        mask &= data.obs[mito_col].values <= max_pct_mito

    n_keep = int(mask.sum())
    print(f"filter_cells: keeping {n_keep:,} / {data.n_obs:,} cells.")
    return data[mask, :]


def filter_genes(
    data: SingleCellDataset,
    min_cells: int = 3,
) -> SingleCellDataset:
    X = data.X
    n_cells = X.getnnz(axis=0) if sp.issparse(X) else np.count_nonzero(X, axis=0)
    data.var["n_cells"] = n_cells

    mask = n_cells >= min_cells
    print(f"filter_genes: keeping {int(mask.sum()):,} / {data.n_vars:,} genes.")
    return data[:, mask]


def normalize_total(
    data: SingleCellDataset,
    target_sum: float = 1e4,
    inplace: bool = True,
) -> Optional[SingleCellDataset]:
    if not inplace:
        data = data.copy()

    if data.raw is None:
        data.raw = data.X.copy()

    X = data.X

    if sp.issparse(X):
        counts = np.ravel(X.sum(axis=1))
        counts = np.where(counts == 0, 1.0, counts)
        scale = target_sum / counts
        from scipy.sparse import diags

        data.X = diags(scale, 0) @ X
    else:
        counts = X.sum(axis=1).reshape(-1, 1)
        counts[counts == 0] = 1.0
        data.X = (X / counts) * target_sum

    return None if inplace else data


def log1p(
    data: SingleCellDataset,
    inplace: bool = True,
) -> Optional[SingleCellDataset]:
    if not inplace:
        data = data.copy()

    if sp.issparse(data.X):
        data.X = data.X.log1p()
    else:
        data.X = np.log1p(data.X)

    return None if inplace else data


def highly_variable_genes(
    data: SingleCellDataset,
    n_top_genes: int = 2000,
    n_bins: int = 20,
    inplace: bool = True,
) -> Optional[SingleCellDataset]:
    if not inplace:
        data = data.copy()

    X = data.X

    if sp.issparse(X):
        mean = np.ravel(X.mean(axis=0))
        mean_sq = np.ravel(X.power(2).mean(axis=0))
        var = mean_sq - mean**2
    else:
        mean = X.mean(axis=0)
        var = X.var(axis=0)

    mean_safe = np.where(mean == 0, 1e-12, mean)
    var_safe = np.where(var == 0, 1e-12, var)
    dispersion = var_safe / mean_safe

    data.var["means"] = mean
    data.var["dispersions"] = dispersion

    log_mean = np.log10(mean_safe)

    actual_bins = min(n_bins, max(1, len(mean) // 2))

    bin_edges = np.percentile(log_mean, np.linspace(0, 100, actual_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6
    bin_labels = np.digitize(log_mean, bin_edges) - 1
    bin_labels = np.clip(bin_labels, 0, actual_bins - 1)

    disp_norm = np.zeros_like(dispersion)
    for b in range(actual_bins):
        idx = bin_labels == b
        if idx.sum() < 2:
            continue
        d = dispersion[idx]
        mu, sigma = d.mean(), d.std()
        disp_norm[idx] = (d - mu) / (sigma if sigma > 0 else 1.0)

    data.var["dispersions_norm"] = disp_norm

    top_idx = np.argsort(disp_norm)[::-1][:n_top_genes]
    data.var["highly_variable"] = False
    data.var.iloc[top_idx, data.var.columns.get_loc("highly_variable")] = True

    print(f"HVG: identified {n_top_genes:,} highly variable genes.")
    return None if inplace else data


def scale(
    data: SingleCellDataset,
    max_value: Optional[float] = 10.0,
    zero_center: bool = True,
    inplace: bool = True,
) -> Optional[SingleCellDataset]:
    if not inplace:
        data = data.copy()

    X = data.X

    if sp.issparse(X):
        mean = np.ravel(X.mean(axis=0))
        mean_sq = np.ravel(X.power(2).mean(axis=0))
        std = np.sqrt(np.clip(mean_sq - mean**2, 0, None))
    else:
        mean = X.mean(axis=0)
        std = X.std(axis=0)

    std = np.where(std == 0, 1.0, std)

    if zero_center:
        if sp.issparse(X):
            import warnings

            warnings.warn(
                "scale(zero_center=True) densifies the sparse matrix. "
                "Consider zero_center=False to preserve sparsity.",
                UserWarning,
                stacklevel=2,
            )
            X = X.toarray()
        X = (X - mean) / std
    else:
        if sp.issparse(X):
            from scipy.sparse import diags

            X = X @ diags(1.0 / std)
        else:
            X = X / std

    if max_value is not None:
        X = np.clip(X, -max_value, max_value) if not sp.issparse(X) else X

    data.X = X
    return None if inplace else data
