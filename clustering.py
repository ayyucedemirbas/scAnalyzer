from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Third-party libraries for geometric clustering
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import kneighbors_graph

# Optional dependencies for graph clustering
# These are standard in single-cell but require specific installations
try:
    import igraph as ig
    import leidenalg
except ImportError:
    leidenalg = None
    ig = None

try:
    import louvain
except ImportError:
    louvain = None

from core import SingleCellDataset


def cluster_kmeans(
    data: SingleCellDataset,
    n_clusters: int = 10,
    random_state: int = 0,
    use_rep: str = "X_pca",
    key_added: str = "kmeans",
) -> SingleCellDataset:
    """
    Performs K-Means clustering on a low-dimensional representation.
    """
    if use_rep not in data.obsm:
        raise ValueError(f"Representation {use_rep} not found in obsm.")

    X = data.obsm[use_rep]

    print(f"Clustering: Running K-Means with k={n_clusters} on {use_rep}...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)

    # Store as categorical for easier plotting later
    data.obs[key_added] = pd.Categorical(labels.astype(str))

    return data


def cluster_leiden(
    data: SingleCellDataset,
    resolution: float = 1.0,
    random_state: int = 0,
    n_iterations: int = -1,
    key_added: str = "leiden",
) -> SingleCellDataset:
    """
    Performs Leiden community detection on the neighbor graph.
    Preferred over Louvain for better partition guarantees.
    Requires 'leidenalg' and 'python-igraph'.
    """
    if leidenalg is None or ig is None:
        raise ImportError("Please install 'leidenalg' and 'python-igraph'.")

    if "neighbors" not in data.uns:
        raise ValueError("Run neighbors() first to compute the graph.")

    adjacency = data.uns["neighbors"]["connectivities"]
    sources, targets = adjacency.nonzero()
    weights = adjacency.data

    # Create graph
    g = ig.Graph(
        n=adjacency.shape[0],
        edges=list(zip(sources, targets)),
        edge_attrs={"weight": weights},
    )

    print(f"Clustering: Running Leiden with resolution={resolution}...")

    # Run Leiden
    # RBConfigurationVertexPartition is standard for optimizing modularity with resolution parameter
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        n_iterations=n_iterations,
        resolution_parameter=resolution,
        seed=random_state,
    )

    # Store results
    labels = np.array(partition.membership)
    data.obs[key_added] = pd.Categorical(labels.astype(str))

    return data


def cluster_louvain(
    data: SingleCellDataset,
    resolution: float = 1.0,
    random_state: int = 0,
    key_added: str = "louvain",
) -> SingleCellDataset:
    """
    Performs Louvain community detection.
    Requires 'louvain' and 'python-igraph'.
    """
    if louvain is None or ig is None:
        raise ImportError("Please install 'louvain' and 'python-igraph'.")

    if "neighbors" not in data.uns:
        raise ValueError("Run neighbors() first to compute the graph.")

    adjacency = data.uns["neighbors"]["connectivities"]
    sources, targets = adjacency.nonzero()
    weights = adjacency.data
    g = ig.Graph(
        n=adjacency.shape[0],
        edges=list(zip(sources, targets)),
        edge_attrs={"weight": weights},
    )

    print(f"Clustering: Running Louvain with resolution={resolution}...")

    partition = louvain.find_partition(
        g,
        louvain.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )

    labels = np.array(partition.membership)
    data.obs[key_added] = pd.Categorical(labels.astype(str))

    return data


def cluster_hierarchical(
    data: SingleCellDataset,
    n_clusters: int = 10,
    linkage: str = "ward",
    use_rep: str = "X_pca",
    key_added: str = "hierarchical",
) -> SingleCellDataset:
    """
    Performs Agglomerative Hierarchical Clustering.
    """
    if use_rep not in data.obsm:
        raise ValueError(f"Representation {use_rep} not found in obsm.")

    X = data.obsm[use_rep]

    print(f"Clustering: Running Hierarchical ({linkage}) with k={n_clusters}...")

    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hc.fit_predict(X)

    data.obs[key_added] = pd.Categorical(labels.astype(str))

    return data


def cluster_dbscan(
    data: SingleCellDataset,
    eps: float = 0.5,
    min_samples: int = 5,
    use_rep: str = "X_umap",
    key_added: str = "dbscan",
) -> SingleCellDataset:
    """
    Performs DBSCAN clustering.
    Good for non-linear shapes and detecting outliers (labeled as -1).
    Usually run on UMAP coordinates rather than high-dim PCA.
    """
    if use_rep not in data.obsm:
        raise ValueError(f"Representation {use_rep} not found in obsm.")

    X = data.obsm[use_rep]

    print(f"Clustering: Running DBSCAN on {use_rep} with eps={eps}...")

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    data.obs[key_added] = pd.Categorical(labels.astype(str))

    return data
