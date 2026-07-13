from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

from .core import SingleCellDataset


def _require_spatial(data: SingleCellDataset, key: str = "spatial") -> np.ndarray:
    if key not in data.obsm:
        raise ValueError(
            f"'{key}' not found in obsm. Spatial coordinates must be stored "
            f"in data.obsm['{key}'] before using spatial functions."
        )
    return data.obsm[key]


def spatial_neighbors(
    data: SingleCellDataset,
    coord_key: str = "spatial",
    n_neighbors: int = 6,
    method: str = "knn",
    radius: Optional[float] = None,
) -> SingleCellDataset:
    """
    Build a spatial neighbor graph from physical coordinates.

    method:
      - 'knn'       : k-nearest-neighbors in physical space.
      - 'radius'    : all cells within `radius` distance.
      - 'delaunay'  : Delaunay triangulation (good for regular grids like Visium).
    """
    coords = _require_spatial(data, coord_key)
    n_obs = data.n_obs

    print(f"Spatial neighbors: method='{method}' on {n_obs:,} cells …")

    if method == "knn":
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
        nbrs.fit(coords)
        dists, indices = nbrs.kneighbors(coords)
        # drop self (first column)
        dists, indices = dists[:, 1:], indices[:, 1:]

        rows = np.repeat(np.arange(n_obs), n_neighbors)
        cols = indices.flatten()
        vals = dists.flatten()

        dist_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
        conn = sp.csr_matrix((np.ones_like(vals), (rows, cols)), shape=(n_obs, n_obs))
        conn = conn.maximum(conn.T)

    elif method == "radius":
        if radius is None:
            raise ValueError("radius must be provided when method='radius'.")
        nbrs = NearestNeighbors(radius=radius, algorithm="auto")
        nbrs.fit(coords)
        dist_lists, ind_lists = nbrs.radius_neighbors(coords)

        rows, cols, vals = [], [], []
        for i, (d_row, i_row) in enumerate(zip(dist_lists, ind_lists)):
            for d, j in zip(d_row, i_row):
                if j == i:
                    continue
                rows.append(i)
                cols.append(j)
                vals.append(d)

        dist_matrix = sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_obs, n_obs)
        )
        conn = sp.csr_matrix(
            (np.ones(len(vals)), (rows, cols)), shape=(n_obs, n_obs)
        )
        conn = conn.maximum(conn.T)

    elif method == "delaunay":
        tri = Delaunay(coords)
        rows, cols = [], []
        for simplex in tri.simplices:
            for a in range(len(simplex)):
                for b in range(len(simplex)):
                    if a != b:
                        rows.append(simplex[a])
                        cols.append(simplex[b])

        rows = np.array(rows)
        cols = np.array(cols)
        vals = np.linalg.norm(coords[rows] - coords[cols], axis=1)

        dist_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
        conn = sp.csr_matrix(
            (np.ones_like(vals), (rows, cols)), shape=(n_obs, n_obs)
        )
        conn = conn.maximum(conn.T)
        conn.data[:] = 1.0

    else:
        raise ValueError("method must be 'knn', 'radius', or 'delaunay'.")

    data.uns["spatial_neighbors"] = {
        "params": {"method": method, "n_neighbors": n_neighbors, "radius": radius},
        "distances": dist_matrix,
        "connectivities": conn,
    }

    avg_deg = conn.nnz / n_obs
    print(f"Spatial neighbors: graph built (avg degree ≈ {avg_deg:.1f}).")
    return data


def moran_i(
    data: SingleCellDataset,
    genes: Optional[List[str]] = None,
    graph_key: str = "spatial_neighbors",
    n_perms: int = 100,
    random_state: int = 0,
) -> pd.DataFrame:

    # Moran's I spatial autocorrelation statistic per gene, with a permutation-based p-value. High I = expression is spatially clustered (neighbors have similar expression); low/negative I = dispersed.

    if graph_key not in data.uns:
        raise ValueError(f"'{graph_key}' not found. Run spatial_neighbors() first.")

    W = data.uns[graph_key]["connectivities"].astype(float)
    n = data.n_obs
    S0 = W.sum()
    if S0 == 0:
        raise ValueError("Spatial weight matrix has no edges.")

    if genes is None:
        gene_idx = np.arange(data.n_vars)
        gene_names = data.var.index.values
    else:
        missing = [g for g in genes if g not in data.var.index]
        if missing:
            warnings.warn(f"{len(missing)} gene(s) not found and skipped: {missing}")
        gene_names = [g for g in genes if g in data.var.index]
        gene_idx = [data.var.index.get_loc(g) for g in gene_names]
        gene_names = np.array(gene_names)

    X = data.X[:, gene_idx]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    print(f"Moran's I: computing for {len(gene_idx):,} genes …")

    x_centered = X - X.mean(axis=0, keepdims=True)
    denom = np.sum(x_centered**2, axis=0)
    denom = np.where(denom == 0, 1e-12, denom)

    # numerator: sum_ij w_ij * z_i * z_j  == diag(z^T W z)
    Wz = W @ x_centered  # (n, n_genes)
    numer = np.sum(x_centered * Wz, axis=0)

    I_obs = (n / S0) * (numer / denom)

    rng = np.random.default_rng(random_state)
    pvals = np.ones(len(gene_idx))
    if n_perms > 0:
        perm_I = np.zeros((n_perms, len(gene_idx)))
        for p in range(n_perms):
            perm = rng.permutation(n)
            x_perm = x_centered[perm]
            Wz_perm = W @ x_perm
            numer_perm = np.sum(x_perm * Wz_perm, axis=0)
            perm_I[p] = (n / S0) * (numer_perm / denom)

        pvals = (np.sum(np.abs(perm_I) >= np.abs(I_obs), axis=0) + 1) / (n_perms + 1)

    df = pd.DataFrame(
        {"gene": gene_names, "moran_I": I_obs, "pval": pvals}
    ).sort_values("moran_I", ascending=False).reset_index(drop=True)

    print(f"Moran's I: done. Top gene = '{df.iloc[0]['gene']}' "
          f"(I={df.iloc[0]['moran_I']:.3f}).")
    return df


def neighborhood_enrichment(
    data: SingleCellDataset,
    cluster_key: str,
    graph_key: str = "spatial_neighbors",
    n_perms: int = 500,
    random_state: int = 0,
) -> Dict[str, np.ndarray]:
  #  Permutation-based cluster-cluster spatial neighborhood enrichment. Positive z = cluster pair co-localizes more than expected by chance; negative z = clusters avoid each other.
    if cluster_key not in data.obs.columns:
        raise ValueError(f"'{cluster_key}' not in obs.")
    if graph_key not in data.uns:
        raise ValueError(f"'{graph_key}' not found. Run spatial_neighbors() first.")

    W = data.uns[graph_key]["connectivities"]
    labels = data.obs[cluster_key].astype(str).values
    categories = np.sort(np.unique(labels))
    n_cat = len(categories)

    label_idx = np.searchsorted(categories, labels)

    def _count_matrix(idx: np.ndarray) -> np.ndarray:
        counts = np.zeros((n_cat, n_cat))
        coo = W.tocoo()
        src_cat = idx[coo.row]
        dst_cat = idx[coo.col]
        np.add.at(counts, (src_cat, dst_cat), coo.data)
        return counts

    print(f"Neighborhood enrichment: {n_cat} groups, {n_perms} permutations …")

    obs_counts = _count_matrix(label_idx)

    rng = np.random.default_rng(random_state)
    perm_counts = np.zeros((n_perms, n_cat, n_cat))
    for p in range(n_perms):
        perm_idx = rng.permutation(label_idx)
        perm_counts[p] = _count_matrix(perm_idx)

    perm_mean = perm_counts.mean(axis=0)
    perm_std = perm_counts.std(axis=0)
    perm_std = np.where(perm_std == 0, 1e-12, perm_std)

    zscore = (obs_counts - perm_mean) / perm_std

    print("Neighborhood enrichment: done.")
    return {
        "categories": categories,
        "zscore": zscore,
        "counts": obs_counts,
    }


def spatial_domains(
    data: SingleCellDataset,
    n_clusters: int = 10,
    use_rep: str = "X_pca",
    graph_key: str = "spatial_neighbors",
    spatial_weight: float = 0.5,
    random_state: int = 0,
    key_added: str = "spatial_domain",
) -> SingleCellDataset:

    # Identify spatial domains by clustering a blend of transcriptional
    # similarity and spatial proximity: cells that are both molecularly similar AND physically close get pulled into the same domain.

    # spatial_weight in [0, 1]: 0 = pure expression clustering,
    # 1 = pure spatial smoothing of expression clusters.

    from sklearn.cluster import KMeans

    if use_rep not in data.obsm:
        raise ValueError(f"'{use_rep}' not found in obsm. Run run_pca() first.")
    if graph_key not in data.uns:
        raise ValueError(f"'{graph_key}' not found. Run spatial_neighbors() first.")
    if not (0.0 <= spatial_weight <= 1.0):
        raise ValueError("spatial_weight must be in [0, 1].")

    emb = data.obsm[use_rep].copy().astype(float)
    W = data.uns[graph_key]["connectivities"].astype(float)

    row_sums = np.ravel(W.sum(axis=1))
    row_sums = np.maximum(row_sums, 1e-10)
    W_norm = sp.diags(1.0 / row_sums) @ W

    print(f"Spatial domains: blending expression + spatial context "
          f"(spatial_weight={spatial_weight}) …")

    emb_smoothed = (1 - spatial_weight) * emb + spatial_weight * (W_norm @ emb)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(emb_smoothed)

    data.obs[key_added] = pd.Categorical(labels.astype(str))
    data.obsm[f"{use_rep}_spatial_smoothed"] = emb_smoothed

    print(f"Spatial domains: found {n_clusters} domains → obs['{key_added}'].")
    return data


def spatial_variable_genes(
    data: SingleCellDataset,
    graph_key: str = "spatial_neighbors",
    n_top_genes: int = 2000,
    n_perms: int = 50,
    random_state: int = 0,
) -> SingleCellDataset:

    #Convenience wrapper: run Moran's I on all genes and flag the top spatially-variable genes in var['spatially_variable'].
    df = moran_i(data, genes=None, graph_key=graph_key, n_perms=n_perms,
                 random_state=random_state)

    top_genes = set(df.head(n_top_genes)["gene"])
    data.var["moran_I"] = data.var.index.map(
        dict(zip(df["gene"], df["moran_I"]))
    ).astype(float)
    data.var["spatially_variable"] = data.var.index.isin(top_genes)

    print(f"Spatial HVGs: flagged {len(top_genes):,} spatially variable genes "
          f"in var['spatially_variable'].")
    return data


def co_occurrence(
    data: SingleCellDataset,
    cluster_key: str,
    coord_key: str = "spatial",
    interval: Optional[np.ndarray] = None,
    n_steps: int = 10,
) -> Dict[str, np.ndarray]:

    # Distance-binned co-occurrence probability between cluster pairs: P(cluster b at distance d | cluster a at center), relative to the baseline frequency of cluster b overall.

    coords = _require_spatial(data, coord_key)
    if cluster_key not in data.obs.columns:
        raise ValueError(f"'{cluster_key}' not in obs.")

    labels = data.obs[cluster_key].astype(str).values
    categories = np.sort(np.unique(labels))
    n_cat = len(categories)
    label_idx = np.searchsorted(categories, labels)

    from scipy.spatial.distance import pdist, squareform

    dist_mat = squareform(pdist(coords))

    if interval is None:
        max_d = dist_mat[dist_mat > 0].max()
        interval = np.linspace(0, max_d, n_steps + 1)

    print(f"Co-occurrence: {n_cat} clusters × {len(interval) - 1} distance bins …")

    baseline = np.array([(label_idx == c).mean() for c in range(n_cat)])

    result = np.zeros((n_cat, n_cat, len(interval) - 1))

    for b in range(len(interval) - 1):
        lo, hi = interval[b], interval[b + 1]
        in_bin = (dist_mat > lo) & (dist_mat <= hi)

        for ca in range(n_cat):
            a_mask = label_idx == ca
            if a_mask.sum() == 0:
                continue
            sub = in_bin[a_mask]
            n_neighbors_total = sub.sum()
            if n_neighbors_total == 0:
                continue
            for cb in range(n_cat):
                b_mask = label_idx == cb
                n_b_in_bin = sub[:, b_mask].sum()
                p_b_given_a = n_b_in_bin / n_neighbors_total
                result[ca, cb, b] = p_b_given_a / max(baseline[cb], 1e-12)

    print("Co-occurrence: done.")
    return {
        "categories": categories,
        "interval": interval,
        "occ": result,
    }


def plot_spatial(
    data: SingleCellDataset,
    color: Optional[str] = None,
    coord_key: str = "spatial",
    cmap: str = "viridis",
    s: int = 20,
    alpha: float = 0.9,
    figsize: Tuple[int, int] = (7, 7),
    invert_y: bool = True,
    title: Optional[str] = None,
    save: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .visualization import _get_color_data, _maybe_show_or_save

    coords = _require_spatial(data, coord_key)
    x, y = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    if color:
        values, is_cat, label = _get_color_data(data, color)
        if is_cat:
            df_plot = pd.DataFrame({"x": x, "y": y, "cat": values})
            sns.scatterplot(
                data=df_plot, x="x", y="y", hue="cat", s=s, alpha=alpha,
                ax=ax, palette="tab20", linewidth=0,
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                      borderaxespad=0.0, frameon=False)
        else:
            sc = ax.scatter(x, y, c=values, s=s, cmap=cmap, alpha=alpha, linewidths=0)
            plt.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)
    else:
        ax.scatter(x, y, s=s, alpha=alpha, c="steelblue", linewidths=0)

    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("spatial 1")
    ax.set_ylabel("spatial 2")
    ax.set_title(title or (f"Spatial" + (f" — {color}" if color else "")))
    sns.despine(ax=ax)

    _maybe_show_or_save(fig, save, ax_provided=False)
    return ax