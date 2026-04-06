from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

try:
    import umap as _umap_mod
except ImportError:
    _umap_mod = None

try:
    import phate as _phate_mod
except ImportError:
    _phate_mod = None

from core import SingleCellDataset


def run_pca(
    data: SingleCellDataset,
    n_components: int = 50,
    use_highly_variable: bool = True,
    svd_solver: str = "arpack",
    random_state: int = 0,
) -> SingleCellDataset:
    if use_highly_variable and "highly_variable" in data.var.columns:
        mask = data.var["highly_variable"].values
        X_sub = data.X[:, mask]
        print(f"PCA: using {int(mask.sum()):,} HVGs.")
    else:
        mask = None
        X_sub = data.X

    n_feat = X_sub.shape[1]
    n_comp = min(n_components, n_feat - 1)  # sklearn constraint
    if n_comp < n_components:
        warnings.warn(
            f"n_components={n_components} capped to {n_comp} "
            f"(only {n_feat} features after HVG selection).",
            UserWarning,
            stacklevel=2,
        )

    if sp.issparse(X_sub):
        print("PCA: sparse input → TruncatedSVD.")
        pca = TruncatedSVD(
            n_components=n_comp, algorithm=svd_solver, random_state=random_state
        )
    else:
        pca = PCA(n_components=n_comp, svd_solver=svd_solver, random_state=random_state)

    X_pca = pca.fit_transform(X_sub)
    data.obsm["X_pca"] = X_pca

    loadings = np.zeros((data.n_vars, n_comp))
    if mask is not None:
        loadings[mask, :] = pca.components_.T
    else:
        loadings = pca.components_.T
    data.varm["PCs"] = loadings

    data.uns["pca"] = {
        "variance": pca.explained_variance_,
        "variance_ratio": pca.explained_variance_ratio_,
    }

    cum_var = float(pca.explained_variance_ratio_.sum() * 100)
    print(f"PCA: computed {n_comp} components " f"({cum_var:.1f}% variance explained).")
    return data


def neighbors(
    data: SingleCellDataset,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    metric: str = "euclidean",
    random_state: int = 0,
) -> SingleCellDataset:
    if "X_pca" not in data.obsm:
        raise ValueError(
            "PCA embedding not found. " "Run dimensionality.run_pca() first."
        )

    X = data.obsm["X_pca"]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    print(f"Neighbors: k={n_neighbors}, metric='{metric}' …")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm="auto")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)  # (n_obs, k)

    n_obs = data.n_obs
    rows = np.repeat(np.arange(n_obs), n_neighbors)
    cols = indices.flatten()
    dists = distances.flatten()

    dist_matrix = sp.csr_matrix((dists, (rows, cols)), shape=(n_obs, n_obs))

    sigma = distances[:, -1]
    sigma = np.maximum(sigma, 1e-10)
    sigma_i = np.repeat(sigma, n_neighbors)
    weights = np.exp(-(dists**2) / (sigma_i**2))

    conn_matrix = sp.csr_matrix((weights, (rows, cols)), shape=(n_obs, n_obs))
    conn_matrix = conn_matrix.maximum(conn_matrix.T)

    data.uns["neighbors"] = {
        "params": {
            "n_neighbors": n_neighbors,
            "metric": metric,
            "n_pcs": n_pcs,
        },
        "distances": dist_matrix,
        "connectivities": conn_matrix,
    }

    print(f"Neighbors: graph built ({n_obs:,} cells).")
    return data


def run_tsne(
    data: SingleCellDataset,
    n_pcs: Optional[int] = None,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: Union[float, str] = "auto",
    n_iter: int = 1000,
    random_state: int = 0,
) -> SingleCellDataset:
    if "X_pca" not in data.obsm:
        raise ValueError("Run run_pca() before run_tsne().")

    X = data.obsm["X_pca"]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    max_perplexity = (X.shape[0] - 1) / 3.0
    if perplexity > max_perplexity:
        warnings.warn(
            f"perplexity={perplexity} too large for {X.shape[0]} cells; "
            f"clamped to {max_perplexity:.1f}.",
            UserWarning,
            stacklevel=2,
        )
        perplexity = max_perplexity

    print(f"t-SNE: perplexity={perplexity:.1f}, n_iter={n_iter} …")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=n_iter,  # <--- Change n_iter=n_iter to max_iter=n_iter
        random_state=random_state,
        init="pca",
        n_jobs=-1,
    )
    data.obsm["X_tsne"] = tsne.fit_transform(X)
    return data


def run_umap(
    data: SingleCellDataset,
    n_pcs: Optional[int] = None,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
) -> SingleCellDataset:
    if _umap_mod is None:
        raise ImportError(
            "umap-learn is required. Install with: pip install umap-learn"
        )
    if "X_pca" not in data.obsm:
        raise ValueError("Run run_pca() before run_umap().")

    X = data.obsm["X_pca"]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    print(f"UMAP: min_dist={min_dist}, n_components={n_components} …")
    reducer = _umap_mod.UMAP(
        n_components=n_components,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
        metric="euclidean",
    )
    data.obsm["X_umap"] = reducer.fit_transform(X)
    return data


def run_diffmap(
    data: SingleCellDataset,
    n_components: int = 15,
    alpha: float = 0.5,
) -> SingleCellDataset:
    if "neighbors" not in data.uns:
        raise ValueError("Run neighbors() before run_diffmap().")

    K = data.uns["neighbors"]["connectivities"].astype(float)

    if alpha > 0:
        q = np.ravel(K.sum(axis=1))
        q = np.maximum(q, 1e-10)
        D_alpha_inv = sp.diags(q ** (-alpha))
        K = D_alpha_inv @ K @ D_alpha_inv

    row_sums = np.ravel(K.sum(axis=1))
    row_sums = np.maximum(row_sums, 1e-10)
    T = sp.diags(1.0 / row_sums) @ K

    print(f"Diffusion map: computing {n_components} components ...")

    d_sqrt = np.sqrt(row_sums)
    d_sqrt_inv = 1.0 / d_sqrt
    M = sp.diags(d_sqrt) @ T @ sp.diags(d_sqrt_inv)

    from scipy.sparse.linalg import eigsh

    k = min(n_components + 1, M.shape[0] - 1)
    eigenvalues, eigenvectors = eigsh(M.tocsr(), k=k, which="LM")

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    phi = sp.diags(d_sqrt_inv).dot(eigenvectors)

    data.obsm["X_diffmap"] = phi[:, 1 : n_components + 1].real
    data.uns["diffmap_evals"] = eigenvalues[1 : n_components + 1].real

    print(
        f"Diffusion map: done.  "
        f"Top eigenvalue gap: "
        f"{eigenvalues[1]:.4f} → {eigenvalues[2]:.4f}."
    )
    return data


def run_phate(
    data: SingleCellDataset,
    n_pcs: Optional[int] = None,
    n_components: int = 2,
    knn: int = 5,
    random_state: int = 0,
) -> SingleCellDataset:
    if _phate_mod is None:
        raise ImportError("phate is required. Install with: pip install phate")

    if "X_pca" not in data.obsm:
        raise ValueError("Run run_pca() before run_phate().")

    X = data.obsm["X_pca"]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    print(f"PHATE: n_components={n_components}, knn={knn} …")
    phate_op = _phate_mod.PHATE(
        n_components=n_components,
        knn=knn,
        random_state=random_state,
        n_jobs=-1,
        verbose=False,
    )
    data.obsm["X_phate"] = phate_op.fit_transform(X)
    return data


from typing import Union
