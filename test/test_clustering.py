import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import clustering
from core import SingleCellDataset

try:
    import igraph
    import leidenalg

    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

try:
    import igraph
    import louvain

    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


class TestClustering(unittest.TestCase):

    def setUp(self):
        n_obs = 20
        n_vars = 10

        X = sp.csr_matrix(np.random.rand(n_obs, n_vars))
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

        self.data = SingleCellDataset(X, obs, var)

        blob1 = np.random.normal(loc=0, scale=0.5, size=(10, 5))
        blob2 = np.random.normal(loc=10, scale=0.5, size=(10, 5))
        pca_embedding = np.vstack([blob1, blob2])

        self.data.obsm["X_pca"] = pca_embedding
        self.data.obsm["X_umap"] = pca_embedding[:, :2]  # Mock UMAP

        row_ind = np.arange(n_obs)
        col_ind = (np.arange(n_obs) + 1) % n_obs
        data_val = np.ones(n_obs)

        adjacency = sp.csr_matrix((data_val, (row_ind, col_ind)), shape=(n_obs, n_obs))

        self.data.uns["neighbors"] = {
            "connectivities": adjacency,
            "distances": adjacency,  # Mock distances
        }

    def test_kmeans(self):
        clustering.cluster_kmeans(
            self.data, n_clusters=2, use_rep="X_pca", key_added="kmeans_test"
        )

        self.assertIn("kmeans_test", self.data.obs.columns)
        unique_labels = self.data.obs["kmeans_test"].unique()
        self.assertEqual(len(unique_labels), 2)
        self.assertTrue(pd.api.types.is_categorical_dtype(self.data.obs["kmeans_test"]))

    def test_hierarchical(self):
        clustering.cluster_hierarchical(
            self.data, n_clusters=2, use_rep="X_pca", key_added="hclust"
        )

        self.assertIn("hclust", self.data.obs.columns)
        unique_labels = self.data.obs["hclust"].unique()
        self.assertEqual(len(unique_labels), 2)

    def test_dbscan(self):
        clustering.cluster_dbscan(
            self.data, eps=2.0, min_samples=2, use_rep="X_umap", key_added="dbscan"
        )

        self.assertIn("dbscan", self.data.obs.columns)
        self.assertTrue(len(self.data.obs["dbscan"]) == self.data.n_obs)

    @unittest.skipUnless(HAS_LEIDEN, "leidenalg or igraph not installed")
    def test_leiden(self):
        clustering.cluster_leiden(self.data, resolution=1.0, key_added="leiden_test")

        self.assertIn("leiden_test", self.data.obs.columns)
        self.assertTrue(len(self.data.obs["leiden_test"]) > 0)

    @unittest.skipUnless(HAS_LOUVAIN, "louvain or igraph not installed")
    def test_louvain(self):
        clustering.cluster_louvain(self.data, resolution=1.0, key_added="louvain_test")

        self.assertIn("louvain_test", self.data.obs.columns)
        self.assertTrue(len(self.data.obs["louvain_test"]) > 0)

    def test_missing_representation_error(self):
        with self.assertRaises(ValueError):
            clustering.cluster_kmeans(self.data, use_rep="X_nonexistent")

    def test_missing_neighbors_error(self):
        del self.data.uns["neighbors"]

        if HAS_LEIDEN:
            with self.assertRaises(ValueError):
                clustering.cluster_leiden(self.data)


if __name__ == "__main__":
    unittest.main()
