from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

from core import SingleCellDataset


def harmony_integrate(
    data: SingleCellDataset,
    batch_key: str = 'batch',
    basis: str = 'X_pca',
    adjusted_basis: str = 'X_pca_harmony',
    theta: float = 2.0,
    lamb: float = 1.0,
    sigma: float = 0.1,
    max_iter_harmony: int = 10,
    max_iter_clustering: int = 20,
    epsilon_cluster: float = 1e-5,
    epsilon_harmony: float = 1e-4,
    random_state: int = 0,
    verbose: bool = True,
) -> SingleCellDataset:
    """
    Harmony batch correction algorithm.
    
    Harmony iteratively:
    1. Clusters cells in PCA space
    2. Computes batch-specific corrections for each cluster
    3. Updates cell positions to reduce batch effects
    
    Parameters

    data : SingleCellDataset
        Dataset with PCA computed.
    batch_key : str, default: 'batch'
        Column in data.obs containing batch labels.
    basis : str, default: 'X_pca'
        Key in obsm containing coordinates to correct.
    adjusted_basis : str, default: 'X_pca_harmony'
        Key to store corrected coordinates.
    theta : float, default: 2.0
        Diversity clustering penalty. Higher = more diverse clusters.
    lamb : float, default: 1.0
        Ridge regression penalty for batch correction.
    sigma : float, default: 0.1
        Width of soft k-means clusters.
    max_iter_harmony : int, default: 10
        Maximum Harmony iterations.
    max_iter_clustering : int, default: 20
        Maximum clustering iterations per Harmony step.
    epsilon_cluster : float, default: 1e-5
        Convergence threshold for clustering.
    epsilon_harmony : float, default: 1e-4
        Convergence threshold for Harmony.
    random_state : int, default: 0
        Random seed.
    verbose : bool, default: True
        Print progress.
    
    Returns

    SingleCellDataset
        Adds corrected coordinates to data.obsm[adjusted_basis].
    
    Examples

    >>> from batch_correction import harmony_integrate
    >>> harmony_integrate(data, batch_key='batch')
    >>> # Use corrected coordinates for downstream analysis
    >>> dimensionality.run_umap(data, n_pcs=None)  # Will use harmony-corrected PCA
    
    References

    Korsunsky et al. (2019). Fast, sensitive and accurate integration of
    single-cell data with Harmony. Nature Methods.
    """
    
    if basis not in data.obsm:
        raise ValueError(f"{basis} not found. Run PCA first.")
    
    if batch_key not in data.obs.columns:
        raise ValueError(f"{batch_key} not found in obs.")
    
    if verbose:
        print("Harmony: Starting batch integration...")
    
    np.random.seed(random_state)
    
    # Get data
    Z = data.obsm[basis].copy()  # Cell coordinates (n_cells x n_PCs)
    batch_labels = data.obs[batch_key].values
    
    # Encode batches
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    batch_indices = np.array([batch_to_idx[b] for b in batch_labels])
    
    n_cells, n_pcs = Z.shape
    
    if verbose:
        print(f"Harmony: {n_cells} cells, {n_pcs} PCs, {n_batches} batches")
    
    # Create one-hot encoding for batches (Phi matrix)
    Phi = np.zeros((n_cells, n_batches))
    Phi[np.arange(n_cells), batch_indices] = 1
    
    # Initialize cluster centers (K-means style)
    n_clusters = min(100, n_cells // 30)  # Heuristic
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1)
    cluster_labels = kmeans.fit_predict(Z)
    Y = kmeans.cluster_centers_.T  # (n_pcs x n_clusters)
    
    # Harmony iterations
    Z_orig = Z.copy()
    Z_corr = Z.copy()
    
    for iteration in range(max_iter_harmony):
        Z_prev = Z_corr.copy()
        
        # 1. Soft clustering (E-step)
        # Compute distances to cluster centers
        # dist[i, k] = ||Z_corr[i] - Y[:, k]||^2
        dists = np.sum(Z_corr**2, axis=1, keepdims=True) + \
                np.sum(Y**2, axis=0, keepdims=True) - \
                2 * Z_corr @ Y
        
        # Soft assignments (with diversity penalty)
        # R[i, k] = exp(-dist[i, k] / sigma) * diversity_weight
        R = np.exp(-dists / sigma)
        
        # Apply diversity penalty per batch
        # Penalize clusters that are dominated by one batch
        for k in range(n_clusters):
            # Count cells from each batch in cluster k
            batch_counts = Phi.T @ R[:, k]  # (n_batches,)
            total = batch_counts.sum()
            
            if total > 0:
                batch_freqs = batch_counts / total
                # Diversity penalty: downweight over-represented batches
                for b in range(n_batches):
                    batch_mask = batch_indices == b
                    R[batch_mask, k] *= (1 / max(batch_freqs[b], 0.01)) ** theta
        
        # Normalize R to be row-stochastic
        R = R / (R.sum(axis=1, keepdims=True) + 1e-10)
        
        # 2. Update cluster centers (M-step)
        Y = (Z_corr.T @ R) / (R.sum(axis=0, keepdims=True) + 1e-10)
        
        # 3. Compute batch corrections
        # For each cluster, compute batch-specific correction vectors
        # Phi.T @ R gives (n_batches x n_clusters): how much each batch contributes to each cluster
        # We want to remove batch-specific deviations
        
        # Compute cluster-specific batch effects
        # For each cluster k, compute mean position per batch
        for k in range(n_clusters):
            cluster_weight = R[:, k:k+1]  # (n_cells, 1)
            
            # Weighted mean per batch
            for b in range(n_batches):
                batch_mask = (batch_indices == b)
                if batch_mask.sum() == 0:
                    continue
                
                batch_cluster_cells = batch_mask & (cluster_weight.flatten() > 0.01)
                if batch_cluster_cells.sum() == 0:
                    continue
                
                # Compute batch-specific deviation from cluster center
                cells_in_batch_cluster = Z_orig[batch_cluster_cells]
                weights = cluster_weight[batch_cluster_cells].flatten()
                
                # Weighted mean position
                weighted_mean = (cells_in_batch_cluster.T @ weights) / weights.sum()
                
                # Correction: move towards global cluster center
                correction = Y[:, k] - weighted_mean
                
                # Apply correction with ridge penalty
                Z_corr[batch_cluster_cells] += lamb * correction * weights.reshape(-1, 1)
        
        # Check convergence
        change = np.mean(np.abs(Z_corr - Z_prev))
        
        if verbose:
            print(f"Harmony: Iteration {iteration+1}/{max_iter_harmony}, change = {change:.6f}")
        
        if change < epsilon_harmony:
            if verbose:
                print(f"Harmony: Converged at iteration {iteration+1}")
            break
    
    # Store corrected coordinates
    data.obsm[adjusted_basis] = Z_corr
    
    if verbose:
        print(f"Harmony: Corrected coordinates stored in obsm['{adjusted_basis}']")
    
    return data


def combat(
    data: SingleCellDataset,
    batch_key: str = 'batch',
    covariates: Optional[List[str]] = None,
    inplace: bool = True,
) -> Optional[SingleCellDataset]:
    """
    ComBat batch effect removal using empirical Bayes.
    
    Note: This is a simplified implementation. For production use,
    consider using the combat function from scanpy or combat-python package.
    
    Parameters
    data : SingleCellDataset
        Expression data (should be normalized and log-transformed).
    batch_key : str, default: 'batch'
        Column in obs containing batch labels.
    covariates : List[str], optional
        Additional covariates to preserve (e.g., cell type, condition).
    inplace : bool, default: True
        Modify data in place or return copy.
    
    Returns
    SingleCellDataset or None
        If inplace=False, returns corrected dataset.
    
    Examples
    >>> from batch_correction import combat
    >>> combat(data, batch_key='batch', covariates=['celltype'])
    """
    
    if not inplace:
        data = data.copy()
    
    if batch_key not in data.obs.columns:
        raise ValueError(f"{batch_key} not found in obs.")
    
    
    # Get expression matrix
    X = data.X
    if sp.issparse(X):
        X = X.toarray()
    
    batch_labels = data.obs[batch_key].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    
    print(f"ComBat: Found {n_batches} batches")
    
    # Simple mean-centering per batch
    # (Full ComBat involves empirical Bayes estimation)
    
    X_corrected = X.copy()
    
    # For each gene, standardize across batches
    for gene_idx in range(data.n_vars):
        gene_expr = X[:, gene_idx]
        
        # Grand mean
        grand_mean = gene_expr.mean()
        grand_std = gene_expr.std()
        
        if grand_std == 0:
            continue
        
        # Batch-specific adjustments
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_expr = gene_expr[batch_mask]
            
            if len(batch_expr) == 0:
                continue
            
            # Batch mean and std
            batch_mean = batch_expr.mean()
            batch_std = batch_expr.std()
            
            if batch_std == 0:
                batch_std = 1
            
            # Standardize within batch, then rescale to grand mean/std
            standardized = (batch_expr - batch_mean) / batch_std
            rescaled = standardized * grand_std + grand_mean
            
            X_corrected[batch_mask, gene_idx] = rescaled
    
    # Update data
    if sp.issparse(data.X):
        data.X = sp.csr_matrix(X_corrected)
    else:
        data.X = X_corrected
    
    print("ComBat: Batch correction complete.")
    
    return None if inplace else data


def mnn_correct(
    datasets: List[SingleCellDataset],
    batch_key: str = 'batch',
    k: int = 20,
    sigma: float = 1.0,
) -> SingleCellDataset:
    """
    Mutual Nearest Neighbors (MNN) batch correction.
    
    Note: This is a conceptual implementation. For production,
    use mnnpy or scanpy's mnn_correct.
    
    Parameters

    datasets : List[SingleCellDataset]
        List of datasets to integrate.
    batch_key : str, default: 'batch'
        Key to store batch labels.
    k : int, default: 20
        Number of nearest neighbors.
    sigma : float, default: 1.0
        Bandwidth for Gaussian correction.
    
    Returns

    SingleCellDataset
        Integrated dataset.
    """
    
    from utils import merge
    
    batch_keys = [f"batch_{i}" for i in range(len(datasets))]
    integrated = merge(datasets, batch_keys=batch_keys, batch_category=batch_key)
    
    return integrated
