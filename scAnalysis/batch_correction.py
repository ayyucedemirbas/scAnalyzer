from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .core import SingleCellDataset


def harmony_integrate(
    data: SingleCellDataset,
    batch_key: str = "batch",
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    theta: float = 2.0,
    lamb: float = 1.0,
    sigma: float = 0.1,
    max_iter_harmony: int = 10,
    epsilon_harmony: float = 1e-4,
    random_state: int = 0,
    verbose: bool = True,
) -> SingleCellDataset:
    _check_inputs(data, basis=basis, batch_key=batch_key)

    np.random.seed(random_state)

    Z = data.obsm[basis].copy().astype(float)
    batch_labels = data.obs[batch_key].values

    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    batch_idx = np.searchsorted(unique_batches, batch_labels)

    n_cells, n_pcs = Z.shape

    if verbose:
        print(f"Harmony: {n_cells:,} cells · {n_pcs} PCs · " f"{n_batches} batches")

    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z /= norms

    Phi = np.zeros((n_cells, n_batches))
    Phi[np.arange(n_cells), batch_idx] = 1.0

    n_clusters = min(100, max(10, n_cells // 30))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    km.fit(Z)
    Y = km.cluster_centers_.T  # (n_pcs × n_clusters)

    Z_orig = Z.copy()
    Z_corr = Z.copy()

    for it in range(max_iter_harmony):
        Z_prev = Z_corr.copy()

        dists = (
            np.einsum("ij,ij->i", Z_corr, Z_corr)[:, None]
            + np.einsum("ij,ij->i", Y.T, Y.T)[None, :]  # <--- Change ->j to ->i here
            - 2.0 * Z_corr @ Y
        )
        R = np.exp(-dists / sigma)

        for k in range(n_clusters):
            batch_counts = Phi.T @ R[:, k]
            total = batch_counts.sum()
            if total == 0:
                continue
            batch_freqs = batch_counts / total
            freq_per_cell = batch_freqs[batch_idx]
            R[:, k] *= (1.0 / np.maximum(freq_per_cell, 0.01)) ** theta

        row_sums = R.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        R /= row_sums

        cluster_masses = R.sum(axis=0)
        cluster_masses[cluster_masses == 0] = 1.0
        Y = (Z_corr.T @ R) / cluster_masses[None, :]

        Z_corr = Z_orig.copy()
        for k in range(n_clusters):
            w = R[:, k]  # (n_cells,)
            center_k = Y[:, k]  # (n_pcs,)

            for b in range(n_batches):
                batch_mask = batch_idx == b
                w_b = w[batch_mask]
                mass_b = w_b.sum()
                if mass_b < 1e-8:
                    continue
                mu_b = (Z_orig[batch_mask].T @ w_b) / mass_b
                correction = center_k - mu_b
                Z_corr[batch_mask] += lamb * np.outer(w_b, correction)

        change = float(np.mean(np.abs(Z_corr - Z_prev)))
        if verbose:
            print(f"  iter {it + 1:2d}/{max_iter_harmony}  Δ={change:.6f}")
        if change < epsilon_harmony:
            if verbose:
                print(f"  converged.")
            break

    data.obsm[adjusted_basis] = Z_corr
    if verbose:
        print(f"Harmony: done — corrected embedding in obsm['{adjusted_basis}']")

    return data


def combat(
    data: SingleCellDataset,
    batch_key: str = "batch",
    covariates: Optional[List[str]] = None,
    inplace: bool = True,
) -> SingleCellDataset:
    if batch_key not in data.obs.columns:
        raise ValueError(
            f"'{batch_key}' not found in obs. " f"Available: {list(data.obs.columns)}"
        )

    if not inplace:
        data = data.copy()

    X = data.X
    if sp.issparse(X):
        X = X.toarray()
    else:
        X = X.copy()

    X = X.astype(float)

    batch_labels = data.obs[batch_key].values
    unique_batches = np.unique(batch_labels)
    print(f"ComBat: correcting {len(unique_batches)} batches …")

    # Global per-gene stats
    grand_mean = X.mean(axis=0)  # (n_genes,)
    grand_std = X.std(axis=0)  # (n_genes,)
    grand_std = np.where(grand_std == 0, 1.0, grand_std)

    X_corrected = X.copy()

    for batch in unique_batches:
        mask = batch_labels == batch
        X_b = X[mask, :]  # (n_b, n_genes)

        batch_mean = X_b.mean(axis=0)  # (n_genes,)
        batch_std = X_b.std(axis=0)
        batch_std = np.where(batch_std == 0, 1.0, batch_std)

        # Standardise within batch, then restore global distribution
        X_corrected[mask, :] = ((X_b - batch_mean) / batch_std) * grand_std + grand_mean

    # Write back — preserve original sparsity format if possible
    if sp.issparse(data.X):
        data.X = sp.csr_matrix(X_corrected)
    else:
        data.X = X_corrected

    print("ComBat: batch correction complete.")
    return data


def mnn_correct(
    datasets: List[SingleCellDataset],
    batch_key: str = "batch",
    k: int = 20,
    sigma: float = 1.0,
) -> SingleCellDataset:
    from utils import merge

    batch_keys = [f"batch_{i}" for i in range(len(datasets))]
    return merge(datasets, batch_keys=batch_keys, batch_category=batch_key)


def _check_inputs(
    data: SingleCellDataset,
    basis: Optional[str] = None,
    batch_key: Optional[str] = None,
) -> None:
    if basis is not None and basis not in data.obsm:
        raise ValueError(
            f"Embedding '{basis}' not found in obsm. "
            "Run dimensionality.run_pca() first."
        )
    if batch_key is not None and batch_key not in data.obs.columns:
        raise ValueError(
            f"Batch column '{batch_key}' not found in obs. "
            f"Available: {list(data.obs.columns)}"
        )
