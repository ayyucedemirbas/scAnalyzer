from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .core import SingleCellDataset


def impute_wnid(
    data: SingleCellDataset,
    k: int = 15,
    dropout_thresh: float = 0.5,
    n_pcs: int = 30,
    inplace: bool = True,
    random_state: int = 0,
) -> Optional[SingleCellDataset]:
    
    if not inplace:
        data = data.copy()

    X = data.X
    is_sparse = sp.issparse(X)
    
    if is_sparse:
        X_dense = X.toarray()
    else:
        X_dense = X.copy()

    if "X_pca" in data.obsm:
        pca_emb = data.obsm["X_pca"][:, :n_pcs]
    else:
        n_comp = min(n_pcs, X_dense.shape[1] - 1)
        pca = PCA(n_components=n_comp, random_state=random_state)
        pca_emb = pca.fit_transform(np.log1p(X_dense))

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nbrs.fit(pca_emb)
    distances, indices = nbrs.kneighbors(pca_emb)

    gene_means = X_dense.mean(axis=0)
    gene_vars = X_dense.var(axis=0)

    mu = np.where(gene_means == 0, 1e-12, gene_means)
    dispersion = gene_vars / mu
    
    dropout_prob = np.exp(-mu / np.maximum(dispersion, 1e-12))
    
    dropout_mask = (X_dense == 0) & (dropout_prob > dropout_thresh)

    X_imputed = X_dense.copy()

    for i in range(X_dense.shape[0]):
        cell_dropouts = dropout_mask[i]
        
        if not np.any(cell_dropouts):
            continue
        
        neighbor_idx = indices[i, 1:]
        dists = distances[i, 1:]
        
        weights = np.exp(-dists)
        weights_sum = weights.sum()
        
        if weights_sum == 0:
            continue
            
        weights /= weights_sum
        
        neighbor_expr = X_dense[neighbor_idx][:, cell_dropouts]
        imputed_values = np.dot(weights, neighbor_expr)
        
        X_imputed[i, cell_dropouts] = imputed_values

    if is_sparse:
        data.X = sp.csr_matrix(X_imputed)
    else:
        data.X = X_imputed

    return None if inplace else data