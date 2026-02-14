from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

# Third-party libraries
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# Handle optional UMAP dependency
try:
    import umap
except ImportError:
    umap = None

from core import SingleCellDataset


def run_pca(
    data: SingleCellDataset,
    n_components: int = 50,
    use_highly_variable: bool = True,
    svd_solver: str = "arpack",
    random_state: int = 0,
) -> SingleCellDataset:
    """
    Computes PCA (Principal Component Analysis).

    Stores:
        - Cell coordinates in data.obsm['X_pca']
        - Gene loadings in data.varm['PCs']
        - Variance info in data.uns['pca']
    """
    # Select features
    if use_highly_variable and "highly_variable" in data.var.columns:
        print("PCA: Using highly variable genes.")
        mask = data.var["highly_variable"].values
        X_subset = data.X[:, mask]
    else:
        X_subset = data.X

    # Choose solver based on sparsity
    if sp.issparse(X_subset):
        print("PCA: Input is sparse, using TruncatedSVD.")
        pca = TruncatedSVD(
            n_components=n_components, algorithm=svd_solver, random_state=random_state
        )
    else:
        pca = PCA(
            n_components=n_components, svd_solver=svd_solver, random_state=random_state
        )

    # Fit and Transform
    X_pca = pca.fit_transform(X_subset)

    # Store results
    data.obsm["X_pca"] = X_pca

    # Store loadings (Project back to full gene space if subset was used)
    # TruncatedSVD components_: (n_components, n_features)
    loadings = np.zeros((data.n_vars, n_components))

    if use_highly_variable and "highly_variable" in data.var.columns:
        loadings[mask, :] = pca.components_.T
    else:
        loadings = pca.components_.T

    data.varm["PCs"] = loadings

    # Store variance ratio
    data.uns["pca"] = {
        "variance": pca.explained_variance_,
        "variance_ratio": pca.explained_variance_ratio_,
    }

    print(f"PCA: Computed {n_components} components.")
    return data


def neighbors(
    data: SingleCellDataset,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    metric: str = "euclidean",
    random_state: int = 0,
) -> SingleCellDataset:
    """
    Computes a neighborhood graph of observations.

    Stores:
        - Distances in data.uns['neighbors']['distances']
        - Connectivities in data.uns['neighbors']['connectivities']
    """
    if "X_pca" not in data.obsm:
        raise ValueError("Please run PCA before computing neighbors.")

    X = data.obsm["X_pca"]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    print(f"Neighbors: Computing kNN graph with k={n_neighbors}...")

    # Fit Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto")
    nbrs.fit(X)

    # Returns (distances, indices)
    distances, indices = nbrs.kneighbors(X)

    # Construct sparse matrices for graph representation
    # We create an adjacency matrix (n_obs x n_obs)

    n_obs = data.n_obs
    row_indices = np.repeat(np.arange(n_obs), n_neighbors)
    col_indices = indices.flatten()
    data_vals = distances.flatten()

    # Distance matrix
    dist_matrix = sp.csr_matrix(
        (data_vals, (row_indices, col_indices)), shape=(n_obs, n_obs)
    )

    # Connectivity matrix (binary or weighted by distance, usually 1 for kNN)
    # Standard practice is to use connectivities weighted by Gaussian kernel (UMAP style),
    # but for simple Louvain/Leiden, a binary kNN graph often suffices or 1/dist.
    # Here we stick to a simple binary connectivity for the toolkit's simplicity.
    conn_vals = np.ones_like(data_vals)
    conn_matrix = sp.csr_matrix(
        (conn_vals, (row_indices, col_indices)), shape=(n_obs, n_obs)
    )

    data.uns["neighbors"] = {
        "params": {"n_neighbors": n_neighbors, "metric": metric},
        "distances": dist_matrix,
        "connectivities": conn_matrix,
    }

    return data


def run_tsne(
    data: SingleCellDataset,
    n_pcs: Optional[int] = None,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: float = 200.0,
    random_state: int = 0,
) -> SingleCellDataset:
    """
    Computes t-SNE embedding.

    Stores:
        - Coordinates in data.obsm['X_tsne']
    """
    if "X_pca" not in data.obsm:
        raise ValueError("Please run PCA before running t-SNE.")

    X = data.obsm["X_pca"]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    print(f"t-SNE: Running with perplexity={perplexity}...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        random_state=random_state,
        init="pca",
        n_jobs=-1,
    )

    X_tsne = tsne.fit_transform(X)
    data.obsm["X_tsne"] = X_tsne

    return data


def run_umap(
    data: SingleCellDataset,
    n_pcs: Optional[int] = None,
    min_dist: float = 0.5,
    spread: float = 1.0,
    random_state: int = 0,
) -> SingleCellDataset:
    """
    Computes UMAP embedding.
    Requires 'umap-learn' package.

    Stores:
        - Coordinates in data.obsm['X_umap']
    """
    if umap is None:
        raise ImportError(
            "umap-learn is not installed. Please install it via `pip install umap-learn`."
        )

    if "X_pca" not in data.obsm:
        raise ValueError("Please run PCA before running UMAP.")

    X = data.obsm["X_pca"]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    print(f"UMAP: Running with min_dist={min_dist}...")

    reducer = umap.UMAP(
        n_components=2,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
        metric="euclidean",
    )

    X_umap = reducer.fit_transform(X)
    data.obsm["X_umap"] = X_umap

    return data
