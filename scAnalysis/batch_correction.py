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
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import cdist
    import numpy as np
    import scipy.sparse as sp

    if len(datasets) < 2:
        print("MNN: Less than 2 datasets provided. Nothing to correct.")
        return datasets[0].copy() if datasets else None

    ref_data = datasets[0].copy()
    corrected_datasets = [ref_data]

    for i in range(1, len(datasets)):
        target_data = datasets[i].copy()
        
        X_ref = ref_data.X.toarray() if sp.issparse(ref_data.X) else np.asarray(ref_data.X)
        X_target = target_data.X.toarray() if sp.issparse(target_data.X) else np.asarray(target_data.X)
        
        nn_ref = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_ref)
        _, indices_target_to_ref = nn_ref.kneighbors(X_target)
        
        nn_target = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_target)
        _, indices_ref_to_target = nn_target.kneighbors(X_ref)
        
        mnn_t = []
        mnn_r = []
        
        for t_idx in range(X_target.shape[0]):
            for r_idx in indices_target_to_ref[t_idx]:
                if t_idx in indices_ref_to_target[r_idx]:
                    mnn_t.append(t_idx)
                    mnn_r.append(r_idx)
        
        mnn_t = np.array(mnn_t)
        mnn_r = np.array(mnn_r)
        
        if len(mnn_t) == 0:
            print(f"     Warning: No MNNs found between batches. Merging without correction.")
            corrected_datasets.append(target_data)
            # Update reference to include this uncorrected batch
            ref_data = merge([ref_data, target_data], join="inner")
            continue
            
        print(f"     Found {len(mnn_t)} mutual anchors.")

        correction_vectors = X_ref[mnn_r] - X_target[mnn_t] # Shape: (n_anchors, n_genes)
        
        anchor_target_coords = X_target[mnn_t]
        dist_sq = cdist(X_target, anchor_target_coords, metric='sqeuclidean')
        
        weights = np.exp(-dist_sq / sigma)
        
        weight_sums = weights.sum(axis=1, keepdims=True)
        weight_sums[weight_sums == 0] = 1.0
        weights_normalized = weights / weight_sums
        
        target_correction = weights_normalized @ correction_vectors
        
        X_target_corrected = X_target + target_correction
        
        if sp.issparse(target_data.X):
            target_data.X = sp.csr_matrix(X_target_corrected)
        else:
            target_data.X = X_target_corrected
            
        corrected_datasets.append(target_data)
        
        ref_data = merge(
            [ref_data, target_data], 
            batch_keys=["ref", f"b{i}"], 
            batch_category="_temp_batch", 
            join="inner"
        )
        
    
    batch_keys_final = [f"batch_{i}" for i in range(len(datasets))]
    final_merged = merge(
        corrected_datasets, 
        batch_keys=batch_keys_final, 
        batch_category=batch_key, 
        join="inner"
    )
    
    return final_merged


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
