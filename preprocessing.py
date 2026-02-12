from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Import the class for type hinting
from core import SingleCellDataset


def calculate_qc_metrics(
    data: SingleCellDataset, qc_vars: Optional[List[str]] = None, inplace: bool = True
) -> Optional[pd.DataFrame]:
    """
    Calculates basic QC metrics:
    - n_genes_by_counts: Number of genes with at least 1 count in a cell.
    - total_counts: Total number of counts per cell.
    - pct_counts_{qc_var}: Percentage of counts in specific gene sets (e.g., 'MT-').
    """
    if not inplace:
        data = data.copy()

    X = data.X
    obs = data.obs
    var = data.var

    # 1. Count number of genes per cell ( > 0)
    if sp.issparse(X):
        n_genes = X.getnnz(axis=1)
        total_counts = np.ravel(X.sum(axis=1))
    else:
        n_genes = np.count_nonzero(X, axis=1)
        total_counts = np.ravel(X.sum(axis=1))

    obs["n_genes_by_counts"] = n_genes
    obs["total_counts"] = total_counts

    # 2. Calculate percentage for QC vars (e.g., Mitochondrial)
    if qc_vars:
        for qc_var in qc_vars:
            # Assumes gene names/symbols are in the index or a column
            # We search in the index for this implementation
            gene_mask = var.index.str.startswith(qc_var)

            if sp.issparse(X):
                qc_counts = np.ravel(X[:, gene_mask].sum(axis=1))
            else:
                qc_counts = np.ravel(X[:, gene_mask].sum(axis=1))

            obs[f"pct_counts_{qc_var}"] = (qc_counts / total_counts) * 100
            # Handle division by zero
            obs[f"pct_counts_{qc_var}"] = obs[f"pct_counts_{qc_var}"].fillna(0)

    if not inplace:
        return data


def filter_cells(
    data: SingleCellDataset,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    max_pct_mito: Optional[float] = None,
) -> SingleCellDataset:
    """
    Filters cells based on QC metrics. Returns a new filtered SingleCellDataset.
    """
    mask = np.ones(data.n_obs, dtype=bool)

    if min_counts is not None:
        mask &= data.obs["total_counts"] >= min_counts
    if max_counts is not None:
        mask &= data.obs["total_counts"] <= max_counts
    if min_genes is not None:
        mask &= data.obs["n_genes_by_counts"] >= min_genes
    if max_genes is not None:
        mask &= data.obs["n_genes_by_counts"] <= max_genes

    # Check for mitochondrial filter if column exists
    # Assuming 'MT-' was passed to calculate_qc_metrics previously
    mito_col = [
        c for c in data.obs.columns if "pct_counts_MT" in c or "pct_counts_mt" in c
    ]
    if max_pct_mito is not None and mito_col:
        mask &= data.obs[mito_col[0]] <= max_pct_mito

    print(f"Filtering cells: Keeping {np.sum(mask)} out of {data.n_obs} cells.")
    return data[mask, :]


def filter_genes(data: SingleCellDataset, min_cells: int = 3) -> SingleCellDataset:
    """
    Filters genes that are detected in fewer than `min_cells`.
    """
    X = data.X
    if sp.issparse(X):
        # Count non-zeros down columns
        n_cells = X.getnnz(axis=0)
    else:
        n_cells = np.count_nonzero(X, axis=0)

    mask = n_cells >= min_cells
    data.var["n_cells"] = n_cells  # Store this info

    print(f"Filtering genes: Keeping {np.sum(mask)} out of {data.n_vars} genes.")
    return data[:, mask]


def normalize_total(
    data: SingleCellDataset, target_sum: float = 1e4, inplace: bool = True
) -> Optional[SingleCellDataset]:
    """
    Normalizes counts per cell so that every cell sums to `target_sum`.
    """
    if not inplace:
        data = data.copy()

    X = data.X

    # Save raw counts if not already saved
    if data.raw is None:
        data.raw = data.X.copy()

    if sp.issparse(X):
        counts_per_cell = np.ravel(X.sum(axis=1))
        # Avoid division by zero
        counts_per_cell[counts_per_cell == 0] = 1

        # Sparse matrix multiplication for broadcasting division
        # reshape counts to (n_cells, 1)
        scale_factor = target_sum / counts_per_cell.reshape(-1, 1)

        # Multiply: strictly speaking, for CSR, row scaling is efficiently done
        # by multiplying the data array directly if we iterate properly,
        # but scipy.sparse.diags is cleaner.
        from scipy.sparse import diags

        d = diags(np.ravel(scale_factor), 0)
        data.X = d @ X

    else:
        counts_per_cell = X.sum(axis=1).reshape(-1, 1)
        counts_per_cell[counts_per_cell == 0] = 1
        data.X = (X / counts_per_cell) * target_sum

    if not inplace:
        return data


def log1p(data: SingleCellDataset, inplace: bool = True) -> Optional[SingleCellDataset]:
    """
    Logarithmizes the data: X = log(X + 1).
    """
    if not inplace:
        data = data.copy()

    if sp.issparse(data.X):
        data.X = data.X.log1p()
    else:
        data.X = np.log1p(data.X)

    return data if not inplace else None


def highly_variable_genes(
    data: SingleCellDataset, n_top_genes: int = 2000, inplace: bool = True
) -> Optional[SingleCellDataset]:
    """
    Identifies highly variable genes (HVGs) based on dispersion.
    Implementation of the 'Seurat' method (log(mean) vs log(variance/mean)).
    """
    if not inplace:
        data = data.copy()

    X = data.X

    # Calculate Mean and Variance
    # Note: If X is CSR, conversion to CSC or dense might be faster for col-wise stats,
    # but we stick to standard operations.

    if sp.issparse(X):
        mean = np.ravel(X.mean(axis=0))
        # Var = E[X^2] - (E[X])^2
        mean_sq = np.ravel(X.power(2).mean(axis=0))
        var = mean_sq - mean**2
    else:
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0)

    # Filter out genes with zero mean to avoid logs of zero
    mean[mean == 0] = 1e-12
    var[var == 0] = 1e-12

    # Calculate dispersion
    dispersion = var / mean

    # Log transform for stability and plotting (Seurat style)
    data.var["means"] = mean
    data.var["dispersions"] = dispersion
    data.var["mean_bin"] = pd.cut(data.var["means"], bins=20)

    # Select top N genes by dispersion (normalized dispersion is better, but raw rank works for simple cases)
    # Here we simply sort by dispersion.
    # A more robust way is binning and z-scoring dispersion within bins,
    # but strict sorting is acceptable for a lightweight toolkit.

    # robust z-score of dispersion
    dispersion_norm = (dispersion - dispersion.mean()) / dispersion.std()
    data.var["dispersions_norm"] = dispersion_norm

    # specific selection
    indices = np.argsort(dispersion_norm)[::-1][:n_top_genes]

    data.var["highly_variable"] = False
    data.var.iloc[indices, data.var.columns.get_loc("highly_variable")] = True

    print(f"HVG: Identified {n_top_genes} highly variable genes.")

    return data if not inplace else None


def scale(
    data: SingleCellDataset, max_value: Optional[float] = 10.0, zero_center: bool = True
):
    """
    Scales data to unit variance and zero mean.

    CRITICAL: This densifies the matrix if zero_center=True.
    """
    X = data.X

    # 1. Calculate stats
    if sp.issparse(X):
        mean = np.ravel(X.mean(axis=0))
        # Standard deviation
        mean_sq = np.ravel(X.power(2).mean(axis=0))
        var = mean_sq - mean**2
        std = np.sqrt(var)
    else:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

    std[std == 0] = 1  # Avoid division by zero

    # 2. Scale
    if zero_center:
        # If sparse, we must densify to center
        if sp.issparse(X):
            print("Warning: Scaling with zero_center=True densifies the matrix.")
            X = X.toarray()

        X = (X - mean) / std
    else:
        # If not centering, we can keep sparsity
        if sp.issparse(X):
            from scipy.sparse import diags

            # multiply columns by 1/std
            d = diags(1 / std, 0)
            X = X @ d
        else:
            X = X / std

    # 3. Clip values
    if max_value is not None:
        X[X > max_value] = max_value
        X[X < -max_value] = -max_value

    data.X = X
    return None
