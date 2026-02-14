"""
Quality Control Module: Doublet Detection and Advanced QC
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.neighbors import NearestNeighbors

from core import SingleCellDataset


def scrublet(
    data: SingleCellDataset,
    expected_doublet_rate: float = 0.06,
    sim_doublet_ratio: float = 2.0,
    n_neighbors: int = 30,
    min_counts: int = 2,
    min_cells: int = 3,
    min_gene_variability_pctl: float = 85.0,
    n_prin_comps: int = 30,
    random_state: int = 0,
    verbose: bool = True,
) -> SingleCellDataset:
    """
    Detect doublets using Scrublet algorithm.
    
    Scrublet works by:
    1. Simulating artificial doublets by adding random cell pairs
    2. Embedding real and simulated cells in PCA space
    3. For each cell, calculating proportion of simulated doublets in neighborhood
    4. Thresholding to predict doublets
    
    Parameters
    ----------
    data : SingleCellDataset
        Annotated data matrix.
    expected_doublet_rate : float, default: 0.06
        Expected fraction of doublets in the data (typically 0.05-0.10).
    sim_doublet_ratio : float, default: 2.0
        Number of simulated doublets = sim_doublet_ratio × n_cells.
    n_neighbors : int, default: 30
        Number of neighbors for KNN graph.
    min_counts : int, default: 2
        Minimum UMI counts for gene filtering.
    min_cells : int, default: 3
        Minimum cells expressing a gene.
    min_gene_variability_pctl : float, default: 85.0
        Keep genes above this variability percentile.
    n_prin_comps : int, default: 30
        Number of principal components.
    random_state : int, default: 0
        Random seed for reproducibility.
    verbose : bool, default: True
        Print progress messages.
    
    Returns
    -------
    SingleCellDataset
        Updates data.obs with:
        - 'doublet_score': Doublet score for each cell
        - 'predicted_doublet': Boolean doublet prediction
    
    Examples
    --------
    >>> from quality_control import scrublet
    >>> scrublet(data, expected_doublet_rate=0.08)
    >>> doublets = data.obs[data.obs['predicted_doublet']].index
    >>> print(f"Detected {len(doublets)} doublets")
    """
    
    if verbose:
        print("Scrublet: Starting doublet detection...")
    
    np.random.seed(random_state)
    
    # Get expression matrix
    X = data.X.copy()
    if sp.issparse(X):
        X = X.tocsr()
    
    n_obs = X.shape[0]
    
    # 1. Filter genes (basic QC on genes)
    if verbose:
        print(f"Scrublet: Filtering genes...")
    
    if sp.issparse(X):
        gene_counts = np.ravel(X.sum(axis=0))
        gene_cells = X.getnnz(axis=0)
    else:
        gene_counts = X.sum(axis=0)
        gene_cells = np.count_nonzero(X, axis=0)
    
    gene_mask = (gene_counts >= min_counts) & (gene_cells >= min_cells)
    X_filtered = X[:, gene_mask]
    
    if verbose:
        print(f"Scrublet: Kept {gene_mask.sum()}/{len(gene_mask)} genes")
    
    # 2. Select variable genes
    if sp.issparse(X_filtered):
        gene_mean = np.ravel(X_filtered.mean(axis=0))
        gene_var = np.ravel(X_filtered.power(2).mean(axis=0)) - gene_mean**2
    else:
        gene_mean = X_filtered.mean(axis=0)
        gene_var = X_filtered.var(axis=0)
    
    gene_mean[gene_mean == 0] = 1e-12
    gene_dispersion = gene_var / gene_mean
    
    # Select genes above variability threshold
    disp_threshold = np.percentile(gene_dispersion, min_gene_variability_pctl)
    hvg_mask = gene_dispersion >= disp_threshold
    
    X_hvg = X_filtered[:, hvg_mask]
    
    if verbose:
        print(f"Scrublet: Selected {hvg_mask.sum()} variable genes")
    
    # 3. Simulate doublets
    n_sim = int(n_obs * sim_doublet_ratio)
    
    if verbose:
        print(f"Scrublet: Simulating {n_sim} doublets...")
    
    # Randomly pair cells
    parent1_idx = np.random.choice(n_obs, n_sim, replace=True)
    parent2_idx = np.random.choice(n_obs, n_sim, replace=True)
    
    # Create synthetic doublets by adding expression
    if sp.issparse(X_hvg):
        X_sim = X_hvg[parent1_idx, :] + X_hvg[parent2_idx, :]
    else:
        X_sim = X_hvg[parent1_idx, :] + X_hvg[parent2_idx, :]
    
    # 4. Combine real and simulated cells
    if sp.issparse(X_hvg):
        X_combined = sp.vstack([X_hvg, X_sim])
    else:
        X_combined = np.vstack([X_hvg, X_sim])
    
    # Labels: 0 = real cell, 1 = simulated doublet
    labels = np.concatenate([np.zeros(n_obs), np.ones(n_sim)])
    
    # 5. Normalize (simple log normalization)
    if verbose:
        print("Scrublet: Normalizing data...")
    
    if sp.issparse(X_combined):
        counts_per_cell = np.ravel(X_combined.sum(axis=1))
        counts_per_cell[counts_per_cell == 0] = 1
        from scipy.sparse import diags
        scale = diags(1 / counts_per_cell, 0)
        X_norm = scale @ X_combined
        X_norm = X_norm.log1p()
    else:
        counts_per_cell = X_combined.sum(axis=1).reshape(-1, 1)
        counts_per_cell[counts_per_cell == 0] = 1
        X_norm = X_combined / counts_per_cell
        X_norm = np.log1p(X_norm)
    
    # 6. PCA
    if verbose:
        print(f"Scrublet: Computing {n_prin_comps} PCs...")
    
    # Limit n_components to available features
    n_features = X_norm.shape[1]
    n_components_actual = min(n_prin_comps, n_features)
    
    if n_components_actual < n_prin_comps:
        if verbose:
            print(f"Scrublet: Warning - Only {n_features} features available, using {n_components_actual} components")
    
    if sp.issparse(X_norm):
        pca = TruncatedSVD(n_components=n_components_actual, random_state=random_state)
    else:
        pca = PCA(n_components=n_components_actual, random_state=random_state)
    
    X_pca = pca.fit_transform(X_norm)
    
    # 7. Calculate doublet scores using KNN
    if verbose:
        print("Scrublet: Calculating doublet scores...")
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nbrs.fit(X_pca)
    
    # For each cell, find neighbors
    _, indices = nbrs.kneighbors(X_pca[:n_obs])  # Only score real cells
    
    # Doublet score = fraction of neighbors that are simulated doublets
    doublet_scores = np.zeros(n_obs)
    for i in range(n_obs):
        neighbor_labels = labels[indices[i]]
        doublet_scores[i] = neighbor_labels.mean()
    
    # 8. Determine threshold
    # Use simulated doublets' scores to set threshold
    sim_scores = doublet_scores  # Could be refined by looking at simulated doublet neighborhood
    
    # Simple threshold: use expected doublet rate
    threshold = np.percentile(doublet_scores, 100 * (1 - expected_doublet_rate))
    
    # Alternative: automatic threshold using histogram
    # (more sophisticated: fit bimodal distribution)
    
    predicted_doublets = doublet_scores > threshold
    
    # 9. Store results
    data.obs['doublet_score'] = doublet_scores
    data.obs['predicted_doublet'] = predicted_doublets
    
    if verbose:
        n_doublets = predicted_doublets.sum()
        pct_doublets = 100 * n_doublets / n_obs
        print(f"Scrublet: Detected {n_doublets} doublets ({pct_doublets:.1f}%)")
        print(f"Scrublet: Threshold = {threshold:.3f}")
    
    return data


def filter_doublets(
    data: SingleCellDataset,
    doublet_score_threshold: Optional[float] = None,
    use_prediction: bool = True
) -> SingleCellDataset:
    """
    Remove predicted doublets from dataset.
    
    Parameters
    ----------
    data : SingleCellDataset
        Dataset with doublet predictions.
    doublet_score_threshold : float, optional
        Manual threshold for doublet score. If None, uses 'predicted_doublet' column.
    use_prediction : bool, default: True
        Use boolean prediction if available.
    
    Returns
    -------
    SingleCellDataset
        Filtered dataset with doublets removed.
    """
    
    if doublet_score_threshold is not None:
        if 'doublet_score' not in data.obs.columns:
            raise ValueError("doublet_score not found. Run scrublet() first.")
        mask = data.obs['doublet_score'] <= doublet_score_threshold
    elif use_prediction and 'predicted_doublet' in data.obs.columns:
        mask = ~data.obs['predicted_doublet']
    else:
        raise ValueError("No doublet information found.")
    
    n_removed = (~mask).sum()
    print(f"Removing {n_removed} doublets, keeping {mask.sum()} cells")
    
    return data[mask, :]


def detect_outliers(
    data: SingleCellDataset,
    metric: str = 'total_counts',
    n_mads: float = 5.0,
    method: str = 'both'
) -> SingleCellDataset:
    """
    Detect outlier cells using MAD (Median Absolute Deviation).
    
    Parameters
    ----------
    data : SingleCellDataset
        Dataset with QC metrics calculated.
    metric : str, default: 'total_counts'
        Which metric to use for outlier detection.
    n_mads : float, default: 5.0
        Number of MADs from median to consider as outlier.
    method : {'both', 'lower', 'upper'}, default: 'both'
        Detect outliers on both tails, or just lower/upper.
    
    Returns
    -------
    SingleCellDataset
        Adds 'is_outlier_{metric}' column to obs.
    """
    
    if metric not in data.obs.columns:
        raise ValueError(f"{metric} not found in obs. Run calculate_qc_metrics() first.")
    
    values = data.obs[metric].values
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    if mad == 0:
        mad = 1e-10  # Avoid division by zero
    
    # Calculate MAD scores
    mad_scores = np.abs(values - median) / mad
    
    # Detect outliers
    if method == 'both':
        outliers = mad_scores > n_mads
    elif method == 'lower':
        outliers = (values < median) & (mad_scores > n_mads)
    elif method == 'upper':
        outliers = (values > median) & (mad_scores > n_mads)
    else:
        raise ValueError("method must be 'both', 'lower', or 'upper'")
    
    data.obs[f'is_outlier_{metric}'] = outliers
    
    n_outliers = outliers.sum()
    print(f"Detected {n_outliers} outlier cells based on {metric} (>{n_mads} MADs)")
    
    return data
