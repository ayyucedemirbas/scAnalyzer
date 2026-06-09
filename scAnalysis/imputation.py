from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from .core import SingleCellDataset


def _to_dense(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    return X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)


def _validate_data(data: SingleCellDataset) -> None:
    if data.n_obs == 0 or data.n_vars == 0:
        raise ValueError("Dataset is empty (0 cells or 0 genes).")


def _get_pca_embedding(
    data: SingleCellDataset,
    n_pcs: int,
    random_state: int,
) -> np.ndarray:
    """Return PCA coords — re-uses obsm['X_pca'] when available."""
    if "X_pca" in data.obsm:
        emb = data.obsm["X_pca"]
        n_use = min(n_pcs, emb.shape[1])
        if n_use < n_pcs:
            warnings.warn(
                f"Requested {n_pcs} PCs but obsm['X_pca'] only has "
                f"{emb.shape[1]}; using {n_use}.",
                UserWarning,
                stacklevel=3,
            )
        return emb[:, :n_use]

    X = data.X
    n_comp = min(n_pcs, data.n_vars - 1, data.n_obs - 1)
    if n_comp < n_pcs:
        warnings.warn(
            f"n_pcs={n_pcs} capped to {n_comp} "
            f"(dataset: {data.n_obs} cells × {data.n_vars} genes).",
            UserWarning,
            stacklevel=3,
        )

    # log1p on a copy, do not touch data.X
    if sp.issparse(X):
        X_log = X.log1p()
        reducer = TruncatedSVD(n_components=n_comp, random_state=random_state)
    else:
        X_log = np.log1p(X)
        reducer = PCA(n_components=n_comp, random_state=random_state)

    return reducer.fit_transform(X_log)


def _build_knn(
    embedding: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
    nbrs.fit(embedding)
    dists, inds = nbrs.kneighbors(embedding)
    return dists[:, 1:], inds[:, 1:]


def _compute_weights(
    dists: np.ndarray,
    weight_method: str,
    bandwidth: Optional[float],
) -> np.ndarray:
    if weight_method == "gaussian":
        if bandwidth is not None:
            h = bandwidth
        else:
            # per-cell adaptive bandwidth: median distance across neighbours
            h = np.median(dists, axis=1, keepdims=True) + 1e-10  # (B, 1)
        w = np.exp(-(dists ** 2) / (h ** 2)) # (B, k)
    elif weight_method == "inverse":
        w = 1.0 / (dists + 1e-10) # (B, k)
    else:  # uniform
        w = np.ones_like(dists) # (B, k)

    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return w / row_sums # (B, k)


def _dropout_mask(
    X_dense: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Boolean mask (n_cells × n_genes): True where a zero is likely a technical
    dropout rather than true absence.

    Uses a CV²-based dropout probability: P(dropout) ≈ exp(−μ / CV²).
    High CV² (bursty, expressed gene) → higher estimated dropout probability.
    """
    eps = 1e-12

    mu_all = X_dense.mean(axis=0) # (n_genes,)
    sq_mean = (X_dense ** 2).mean(axis=0)
    var_all = np.maximum(sq_mean - mu_all ** 2, 0.0) # (n_genes,)

    mu_safe = np.where(mu_all == 0, eps, mu_all)
    cv2 = var_all / (mu_safe ** 2)#(n_genes,)

    dropout_prob = np.exp(-mu_safe / np.maximum(cv2, eps))
    dropout_prob = np.clip(dropout_prob, 0.0, 1.0)

    return (X_dense == 0) & (dropout_prob[np.newaxis, :] > threshold)


def impute_wnid(
    data: SingleCellDataset,
    k: int = 15,
    dropout_thresh: float = 0.5,
    n_pcs: int = 30,
    weight_method: str = "gaussian",
    bandwidth: Optional[float] = None,
    clip_pct: float = 99.0,
    batch_size: int = 64,
    inplace: bool = True,
    random_state: int = 0,
    verbose: bool = True,
) -> Optional[SingleCellDataset]:

    _validate_data(data)

    if not (0.0 <= dropout_thresh <= 1.0):
        raise ValueError("dropout_thresh must be in [0, 1].")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if weight_method not in {"gaussian", "uniform", "inverse"}:
        raise ValueError("weight_method must be 'gaussian', 'uniform', or 'inverse'.")

    if not inplace:
        data = data.copy()

    if verbose:
        print(f"WNID: {data.n_obs:,} cells × {data.n_vars:,} genes  "
              f"(k={k}, dropout_thresh={dropout_thresh}, batch={batch_size}) …")

    X_dense = _to_dense(data.X)

    mask = _dropout_mask(X_dense, dropout_thresh)
    n_dropout = int(mask.sum())
    if verbose:
        pct = 100.0 * n_dropout / X_dense.size
        print(f"WNID: {n_dropout:,} candidate dropout entries ({pct:.2f}% of matrix).")

    if n_dropout == 0:
        if verbose:
            print("WNID: no dropout entries found — nothing to impute.")
        return None if inplace else data

    emb = _get_pca_embedding(data, n_pcs, random_state)
    dists, inds = _build_knn(emb, k)

    nonzero_vals = X_dense[X_dense > 0]
    global_clip = float(np.percentile(nonzero_vals, clip_pct)) if nonzero_vals.size else np.inf


    X_imp = X_dense.copy() #vectorized
    n_obs = data.n_obs

    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        B = end - start

        batch_mask  = mask[start:end] # (B, n_genes)
        batch_dists = dists[start:end]  # (B, k)
        batch_inds  = inds[start:end]  # (B, k)

        if not batch_mask.any():
            continue

        w = _compute_weights(batch_dists, weight_method, bandwidth)

        nb_expr = X_dense[batch_inds]

        weighted = np.einsum("bi,bij->bj", w, nb_expr) # (B, n_genes)

        weighted = np.minimum(weighted, global_clip)
        X_imp[start:end] = np.where(batch_mask, weighted, X_imp[start:end])

        if verbose and (start // batch_size) % 20 == 0 and start > 0:
            print(f"  … {end:,}/{n_obs:,} cells processed")

    if sp.issparse(data.X):
        data.X = sp.csr_matrix(X_imp)
    else:
        data.X = X_imp

    if verbose:
        print("WNID: imputation complete.")
    return None if inplace else data


def impute_knn_smooth(
    data: SingleCellDataset,
    k: int = 10,
    weight_method: str = "gaussian",
    n_pcs: int = 30,
    batch_size: int = 64,
    inplace: bool = True,
    random_state: int = 0,
    verbose: bool = True,
) -> Optional[SingleCellDataset]:
    _validate_data(data)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if weight_method not in {"gaussian", "uniform", "inverse"}:
        raise ValueError("weight_method must be 'gaussian', 'uniform', or 'inverse'.")

    if not inplace:
        data = data.copy()

    if verbose:
        print(f"kNN-smooth: {data.n_obs:,} cells × {data.n_vars:,} genes  "
              f"(k={k}, weight='{weight_method}', batch={batch_size}) …")

    emb = _get_pca_embedding(data, n_pcs, random_state)
    dists, inds = _build_knn(emb, k)  # (n_obs, k) each

    X_dense = _to_dense(data.X) # (n_obs, n_genes)
    X_smooth = np.empty_like(X_dense)
    n_obs = data.n_obs

    for start in range(0, n_obs, batch_size):
        end = min(start + batch_size, n_obs)
        B = end - start

        batch_dists = dists[start:end]                           # (B, k)
        batch_inds  = inds[start:end]                            # (B, k)

        w_nb = _compute_weights(batch_dists, weight_method, bandwidth=None)  # (B, k)

        w_self = w_nb.mean(axis=1, keepdims=True)                # (B, 1)
        w_all  = np.concatenate([w_self, w_nb], axis=1)         # (B, k+1)
        w_all /= w_all.sum(axis=1, keepdims=True) 


        self_inds = np.arange(start, end)[:, np.newaxis]        # (B, 1)
        all_inds  = np.concatenate([self_inds, batch_inds], axis=1)  # (B, k+1)

        nb_expr = X_dense[all_inds]                             # (B, k+1, n_genes)

        X_smooth[start:end] = np.einsum("bi,bij->bj", w_all, nb_expr)  # (B, n_genes)

        if verbose and (start // batch_size) % 20 == 0 and start > 0:
            print(f"  … {end:,}/{n_obs:,} cells processed")

    if sp.issparse(data.X):
        data.X = sp.csr_matrix(X_smooth)
    else:
        data.X = X_smooth

    if verbose:
        print("kNN-smooth: complete.")
    return None if inplace else data


def impute_diffusion(
    data: SingleCellDataset,
    t: int = 3,
    n_pcs: int = 30,
    k: int = 10,
    alpha: float = 1.0,
    use_prebuilt_graph: bool = True,
    inplace: bool = True,
    random_state: int = 0,
    verbose: bool = True,
) -> Optional[SingleCellDataset]:

    _validate_data(data)
    if t < 1:
        raise ValueError("t must be >= 1.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    if not inplace:
        data = data.copy()

    if use_prebuilt_graph and "neighbors" in data.uns:
        if verbose:
            print("Diffusion imputation: reusing prebuilt neighbour graph.")
        K = data.uns["neighbors"]["connectivities"].astype(float)
    else:
        if verbose:
            print(f"Diffusion imputation: building kNN graph "
                  f"(k={k}, n_pcs={n_pcs}) …")
        emb = _get_pca_embedding(data, n_pcs, random_state)
        dists_arr, inds_arr = _build_knn(emb, k)

        n_obs = data.n_obs
        rows = np.repeat(np.arange(n_obs), k)
        cols = inds_arr.flatten()
        sigma = np.maximum(dists_arr[:, -1], 1e-10)
        weights = np.exp(-(dists_arr.flatten() ** 2) / np.repeat(sigma ** 2, k))

        K = sp.csr_matrix((weights, (rows, cols)), shape=(n_obs, n_obs))
        K = K.maximum(K.T)          # symmetrise

    if alpha > 0:
        d = np.ravel(K.sum(axis=1))
        d = np.maximum(d, 1e-10)
        D_alpha = sp.diags(d ** (-alpha))
        K = D_alpha @ K @ D_alpha

    row_sums = np.ravel(K.sum(axis=1))
    row_sums = np.maximum(row_sums, 1e-10)
    T = sp.diags(1.0 / row_sums) @ K     # row-stochastic

    if verbose:
        print(f"Diffusion imputation: diffusing for t={t} step(s) …")


    X_dense = _to_dense(data.X)

    # T^t X  (applied iteratively to avoid materialising T^t)
    X_diff = X_dense.copy()
    for step in range(t):
        # T is sparse (n_obs × n_obs); X_diff is dense (n_obs × n_genes)
        X_diff = T @ X_diff
        if verbose and t > 1:
            print(f"  step {step + 1}/{t}")


    if sp.issparse(data.X):
        data.X = sp.csr_matrix(X_diff)
    else:
        data.X = X_diff

    if verbose:
        print("Diffusion imputation: complete.")
    return None if inplace else data


def dropout_stats(
    data: SingleCellDataset,
    top_n: int = 10,
    verbose: bool = True,
) -> dict:

    _validate_data(data)

    X = data.X
    if sp.issparse(X):
        n_zeros_per_gene = data.n_obs - np.diff(X.tocsc().indptr)
        n_zeros_per_cell = data.n_vars - np.diff(X.tocsr().indptr)
    else:
        n_zeros_per_gene = (X == 0).sum(axis=0)
        n_zeros_per_cell = (X == 0).sum(axis=1)

    per_gene = n_zeros_per_gene / data.n_obs
    per_cell = n_zeros_per_cell / data.n_vars
    global_rate = float(per_gene.mean())

    order = np.argsort(per_gene)[::-1][:top_n]
    top_genes = [
        (str(data.var.index[i]), float(per_gene[i])) for i in order
    ]

    if verbose:
        print(f"Dropout statistics  ({data.n_obs:,} cells × {data.n_vars:,} genes)")
        print(f"  Global dropout rate : {global_rate * 100:.1f}%")
        print(f"  Per-cell  median    : {float(np.median(per_cell)) * 100:.1f}%")
        print(f"  Per-gene  median    : {float(np.median(per_gene)) * 100:.1f}%")
        print(f"  Top {top_n} highest-dropout genes:")
        for name, rate in top_genes:
            print(f"    {name:<20s}  {rate * 100:5.1f}%")

    return {
        "global_rate": global_rate,
        "per_gene": np.asarray(per_gene),
        "per_cell": np.asarray(per_cell),
        "top_genes": top_genes,
    }


def compare_imputation(
    before: SingleCellDataset,
    after: SingleCellDataset,
    genes: Optional[list] = None,
) -> None:

    if before.shape != after.shape:
        raise ValueError(
            f"Shape mismatch: before={before.shape}, after={after.shape}."
        )

    if genes is not None:
        missing = [g for g in genes if g not in before.var.index]
        if missing:
            warnings.warn(f"{len(missing)} gene(s) not found and skipped.", UserWarning)
        genes = [g for g in genes if g in before.var.index]
        idx = [before.var.index.get_loc(g) for g in genes]
        Xb = _to_dense(before.X)[:, idx]
        Xa = _to_dense(after.X)[:, idx]
    else:
        Xb = _to_dense(before.X)
        Xa = _to_dense(after.X)

    def _stats(X: np.ndarray) -> dict:
        return {
            "mean": float(X.mean()),
            "std": float(X.std()),
            "sparsity": float((X == 0).mean()),
            "max": float(X.max()),
        }

    sb, sa = _stats(Xb), _stats(Xa)

    header = f"{'Metric':<20s}  {'Before':>12s}  {'After':>12s}  {'Δ':>12s}"
    print(header)
    for key in ("mean", "std", "sparsity", "max"):
        delta = sa[key] - sb[key]
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<18s}  {sb[key]:>12.4f}  {sa[key]:>12.4f}  "
              f"{sign}{delta:>11.4f}")

