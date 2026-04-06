from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.metrics import silhouette_score

try:
    import igraph as ig
    import leidenalg
except ImportError:
    leidenalg = None
    ig = None

try:
    import louvain as _louvain_mod
except ImportError:
    _louvain_mod = None

from core import SingleCellDataset


def _get_rep(data: SingleCellDataset, use_rep: str) -> np.ndarray:
    if use_rep not in data.obsm:
        available = list(data.obsm.keys())
        raise ValueError(
            f"Embedding '{use_rep}' not found in obsm. "
            f"Available: {available}. "
            "Run dimensionality.run_pca() first."
        )
    return data.obsm[use_rep]


def _get_adjacency(data: SingleCellDataset) -> sp.csr_matrix:
    if "neighbors" not in data.uns:
        raise ValueError(
            "Neighbor graph not found. " "Run dimensionality.neighbors() first."
        )
    return data.uns["neighbors"]["connectivities"]


def _store_labels(data: SingleCellDataset, labels: np.ndarray, key: str) -> None:
    data.obs[key] = pd.Categorical(labels.astype(str))


def cluster_kmeans(
    data: SingleCellDataset,
    n_clusters: Optional[int] = 10,
    random_state: int = 0,
    use_rep: str = "X_pca",
    key_added: str = "kmeans",
    auto_select_k: bool = False,
    k_range: Tuple[int, int] = (2, 20),
) -> SingleCellDataset:
    X = _get_rep(data, use_rep)

    if auto_select_k:
        lo, hi = k_range
        best_k, best_score = lo, -1.0
        for k in range(lo, hi + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            lbl = km.fit_predict(X)
            if len(np.unique(lbl)) < 2:
                continue
            s = silhouette_score(X, lbl, sample_size=min(len(X), 2000))
            if s > best_score:
                best_score, best_k = s, k
        n_clusters = best_k
        print(f"K-Means auto-select: k={n_clusters} " f"(silhouette={best_score:.3f})")

    print(f"Clustering: K-Means k={n_clusters} on '{use_rep}' …")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    _store_labels(data, labels, key_added)
    return data


def cluster_leiden(
    data: SingleCellDataset,
    resolution: float = 1.0,
    random_state: int = 0,
    n_iterations: int = -1,
    key_added: str = "leiden",
) -> SingleCellDataset:
    if leidenalg is None or ig is None:
        raise ImportError(
            "leidenalg and python-igraph are required. "
            "Install with: pip install leidenalg igraph"
        )

    adj = _get_adjacency(data)
    sources, targets = adj.nonzero()
    weights = np.asarray(adj[sources, targets]).flatten()

    g = ig.Graph(
        n=adj.shape[0],
        edges=list(zip(sources.tolist(), targets.tolist())),
        edge_attrs={"weight": weights.tolist()},
    )

    print(f"Clustering: Leiden resolution={resolution} …")
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        n_iterations=n_iterations,
        resolution_parameter=resolution,
        seed=random_state,
    )
    labels = np.array(partition.membership)
    _store_labels(data, labels, key_added)
    n_clust = len(np.unique(labels))
    print(f"Leiden: found {n_clust} clusters.")
    return data


def cluster_louvain(
    data: SingleCellDataset,
    resolution: float = 1.0,
    random_state: int = 0,
    key_added: str = "louvain",
) -> SingleCellDataset:
    if _louvain_mod is None or ig is None:
        raise ImportError(
            "louvain and python-igraph are required. "
            "Install with: pip install louvain igraph"
        )

    adj = _get_adjacency(data)
    sources, targets = adj.nonzero()
    weights = np.asarray(adj[sources, targets]).flatten()

    g = ig.Graph(
        n=adj.shape[0],
        edges=list(zip(sources.tolist(), targets.tolist())),
        edge_attrs={"weight": weights.tolist()},
    )

    print(f"Clustering: Louvain resolution={resolution} …")
    partition = _louvain_mod.find_partition(
        g,
        _louvain_mod.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )
    labels = np.array(partition.membership)
    _store_labels(data, labels, key_added)
    return data


def cluster_hierarchical(
    data: SingleCellDataset,
    n_clusters: int = 10,
    linkage: str = "ward",
    use_rep: str = "X_pca",
    key_added: str = "hierarchical",
) -> SingleCellDataset:
    X = _get_rep(data, use_rep)
    print(f"Clustering: Hierarchical ({linkage}) k={n_clusters} …")
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    _store_labels(data, hc.fit_predict(X), key_added)
    return data


def cluster_dbscan(
    data: SingleCellDataset,
    eps: float = 0.5,
    min_samples: int = 5,
    use_rep: str = "X_pca",
    key_added: str = "dbscan",
) -> SingleCellDataset:
    X = _get_rep(data, use_rep)
    print(f"Clustering: DBSCAN eps={eps} on '{use_rep}' …")
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    _store_labels(data, labels, key_added)
    n_noise = int((labels == -1).sum())
    n_clust = len(np.unique(labels[labels != -1]))
    print(f"DBSCAN: {n_clust} clusters, {n_noise} noise points.")
    return data


def cluster_spectral(
    data: SingleCellDataset,
    n_clusters: int = 10,
    use_rep: str = "X_pca",
    random_state: int = 0,
    key_added: str = "spectral",
    affinity: str = "rbf",
    gamma: float = 1.0,
) -> SingleCellDataset:
    print(f"Clustering: Spectral k={n_clusters}, affinity='{affinity}' …")

    if affinity == "precomputed":
        adj = _get_adjacency(data)
        adj_sym = (adj + adj.T) / 2
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=random_state,
        )
        labels = sc.fit_predict(adj_sym.toarray())
    else:
        X = _get_rep(data, use_rep)
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            gamma=gamma,
            random_state=random_state,
            n_jobs=-1,
        )
        labels = sc.fit_predict(X)

    _store_labels(data, labels, key_added)
    return data


def cluster_stats(
    data: SingleCellDataset,
    cluster_key: str,
    var_names: Optional[List[str]] = None,
    use_raw: bool = False,
) -> pd.DataFrame:
    if cluster_key not in data.obs.columns:
        raise ValueError(
            f"'{cluster_key}' not in obs. Available: {list(data.obs.columns)}"
        )

    if use_raw and data.raw is not None:
        X = data.raw.X if hasattr(data.raw, "X") else data.raw
        vnames = data.raw.var.index if hasattr(data.raw, "var") else data.var.index
    else:
        X = data.X
        vnames = data.var.index

    if var_names is not None:
        idx = [vnames.get_loc(g) for g in var_names if g in vnames]
        X = X[:, idx]
        vnames = vnames[idx]

    if sp.issparse(X):
        X = X.toarray()

    labels = data.obs[cluster_key].values
    unique = np.sort(pd.Series(labels).unique())
    n_total = len(labels)

    rows = {}
    for g in unique:
        mask = labels == g
        rows[str(g)] = np.mean(X[mask], axis=0)

    stats_df = pd.DataFrame(rows, index=vnames).T
    stats_df.index.name = cluster_key

    counts = pd.Series(labels).value_counts()
    stats_df.insert(0, "n_cells", [counts.get(g, 0) for g in stats_df.index])
    stats_df.insert(1, "pct_of_total", (stats_df["n_cells"] / n_total * 100).round(2))

    return stats_df
