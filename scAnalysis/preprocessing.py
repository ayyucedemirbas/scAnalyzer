from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
import statsmodels.api as sm

from .core import SingleCellDataset

"""
It counts: 
    1. How many total RNA molecules does this cell have?
    2. How many UNIQUE genes were found in this cell?
    3. What percentage of the RNA comes from Mitochondria (MT-)? 
       (Too much MT = the cell is dead/dying and leaking its nuclear RNA).
"""
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

"""
    - Too few counts/genes? It's probably an empty droplet, not a real cell.
    - Too many counts/genes? It's probably two cells stuck together (doublet).
    - Too much mitochondria? The cell is dead.
"""
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

"""
If a gene is only seen in 1 or 2 cells out of 10000, it's statistically 
useless. It just takes up RAM and adds noise. We throw these rare genes away.
"""

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

"""
The sequencing machine reads some cells, for example, 5000 times and others 20000 times.
    If cell A has more "CD3E" gene counts than cell B, is it because it's a T-cell, 
    or just because the machine read cell A more?
    We force every cell to have exactly the same total amount of counts (e.g., 10000).
"""
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

def normalize_scran_pooling(
    data: SingleCellDataset, 
    n_pools: int = 50, 
    target_sum: float = 1e4,
    inplace: bool = True
) -> Optional[SingleCellDataset]:
    """
    scran-like pooling-based normalization.
    Groups cells into pools to mitigate the noise caused by high dropout rates (zeros),
    calculating more robust size factors than single-cell level scaling.
    """
    if not inplace:
        data = data.copy()

    if data.raw is None:
        data.raw = data.X.copy()

    X = data.X
    n_cells = X.shape[0]
    
    # 1. Coarsely cluster cells into pools using K-Means
    # True scran uses a rolling window on a KNN graph, but K-Means on total counts
    # is a fast and effective approximation for grouping cells of similar depths.
    n_clusters = min(n_pools, max(1, n_cells // 10))
    
    if sp.issparse(X):
        total_counts = np.ravel(X.sum(axis=1))
    else:
        total_counts = X.sum(axis=1)
        
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=3)
    clusters = km.fit_predict(total_counts.reshape(-1, 1))

    # 2. Calculate a pseudo-size factor for each pool
    size_factors = np.zeros(n_cells)
    
    for i in range(n_clusters):
        mask = clusters == i
        pool_cells = mask.sum()
        if pool_cells == 0:
            continue
            
        if sp.issparse(X):
            pool_sum = np.ravel(X[mask, :].sum(axis=0))
        else:
            pool_sum = X[mask, :].sum(axis=0)
            
        # The median expression of the pool serves as the pool size factor
        pool_size_factor = np.median(pool_sum[pool_sum > 0])
        if np.isnan(pool_size_factor) or pool_size_factor == 0:
            pool_size_factor = 1.0
            
        # Distribute the pool factor back to individual cells based on their fraction of the pool's total reads
        cell_totals = total_counts[mask]
        cell_factors = pool_size_factor * (cell_totals / (cell_totals.sum() + 1e-12))
        size_factors[mask] = cell_factors

    # 3. Correct any zero or negative size factors
    size_factors = np.where(size_factors <= 0, 1.0, size_factors)
    
    # 4. Apply the normalization scale
    scale = target_sum / size_factors
    
    if sp.issparse(X):
        from scipy.sparse import diags
        data.X = diags(scale, 0) @ X
    else:
        data.X = X * scale[:, np.newaxis]

    print("normalize_scran_pooling: Robust size factors applied.")
    return None if inplace else data

def normalize_sctransform(
    data: SingleCellDataset, 
    inplace: bool = True
) -> Optional[SingleCellDataset]:
    """
    sctransform-like normalization using regularized Negative Binomial regression.
    Models the variance-mean relationship for each gene and regresses out the effect 
    of sequencing depth, returning Pearson residuals as the normalized values.
    """
    if not inplace:
        data = data.copy()

    if data.raw is None:
        data.raw = data.X.copy()

    X = data.X
    n_cells, n_genes = X.shape

    # For speed, we convert to dense. For massive datasets, you would want to 
    # iterate over sparse columns directly without full densification.
    if sp.issparse(X):
        total_counts = np.ravel(X.sum(axis=1))
        X_dense = X.toarray() 
    else:
        total_counts = X.sum(axis=1)
        X_dense = X.copy()

    total_counts = np.where(total_counts == 0, 1.0, total_counts)
    log_totals = np.log10(total_counts)
    
    # Exogenous variable (Predictor: log10 of sequencing depth)
    exog = sm.add_constant(log_totals)
    
    normalized_X = np.zeros_like(X_dense, dtype=float)
    
    print(f"normalize_sctransform: Fitting NB models for {n_genes} genes. This may take a while...")
    
    for i in range(n_genes):
        y = X_dense[:, i]
        
        # Only fit models for genes that are expressed in at least a few cells
        if np.count_nonzero(y) < 3:
            continue
            
        try:
            # GLM with Negative Binomial family
            model = sm.GLM(y, exog, family=sm.families.NegativeBinomial())
            result = model.fit(disp=0)
            
            # Expected values (mu)
            mu = result.mu
            
            # Variance estimate: var = mu + alpha * mu^2 (assuming alpha=1 for simple dispersion)
            var = mu + (mu ** 2) 
            
            # Calculate Pearson Residuals: (Observed - Expected) / sqrt(Variance)
            residuals = (y - mu) / np.sqrt(var + 1e-12)
            
            # Clip extreme outliers
            max_val = np.sqrt(n_cells)
            residuals = np.clip(residuals, -max_val, max_val)
            
            normalized_X[:, i] = residuals
            
        except Exception:
            # If the model fails to converge, leave the residuals as 0
            pass

    # Overwrite the original matrix with the Pearson residuals
    if sp.issparse(data.X):
        data.X = sp.csr_matrix(normalized_X)
    else:
        data.X = normalized_X
        
    print("normalize_sctransform: Completed. X matrix now contains Pearson residuals.")

    return None if inplace else data

"""

Biological signals grow exponentially. One gene might have 1 count, 
another might have 10,000. If we give this to Machine Learning (PCA), the 
10,000 gene will crush everything else. 
Applying log(x+1) compresses these huge gaps, making the data curve more 'normal'.
"""

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

"""
Out of 20,000 genes, most are genes that maintain a static, 
invariant expression profile. They do basic cell maintenance and look exactly 
the same in every cell type. We don't care about them. We want the genes that 
fluctuate wildly between cells (Highly Variable Genes), 
because these are the markers that define different cell identities
"""
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

"""
Even among these feature-selected informative genes, some naturally exhibit 
high-magnitude expression (e.g., 500 units) while others operate at low-abundance 
levels (e.g., 10 units). By converting them to Z-scores, we standardize the data so that 
every gene has a mean of 0 and a standard deviation of 1. Consequently, downstream 
algorithms like PCA evaluate each gene based on its relative variance and deviation from 
its own baseline, rather than being biased by its absolute transcriptional magnitude.
"""

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