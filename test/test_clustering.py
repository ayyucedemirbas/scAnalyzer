import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import clustering
from core import SingleCellDataset

# Check optional dependencies for skipping tests if necessary
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
        """
        Create a synthetic dataset with embeddings and neighbor graph.
        """
        n_obs = 20
        n_vars = 10

        # 1. Basic Data
        X = sp.csr_matrix(np.random.rand(n_obs, n_vars))
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

        self.data = SingleCellDataset(X, obs, var)

        # 2. Add Embeddings (for K-Means, Hierarchical, DBSCAN)
        # Create 2 distinct blobs for testing
        blob1 = np.random.normal(loc=0, scale=0.5, size=(10, 5))
        blob2 = np.random.normal(loc=10, scale=0.5, size=(10, 5))
        pca_embedding = np.vstack([blob1, blob2])

        self.data.obsm["X_pca"] = pca_embedding
        self.data.obsm["X_umap"] = pca_embedding[:, :2]  # Mock UMAP

        # 3. Add Neighbor Graph (for Leiden, Louvain)
        # Create a simple sparse adjacency matrix (ring topology for simplicity)
        # Cell i connects to i+1
        row_ind = np.arange(n_obs)
        col_ind = (np.arange(n_obs) + 1) % n_obs
        data_val = np.ones(n_obs)

        adjacency = sp.csr_matrix((data_val, (row_ind, col_ind)), shape=(n_obs, n_obs))

        self.data.uns["neighbors"] = {
            "connectivities": adjacency,
            "distances": adjacency,  # Mock distances
        }

    def test_kmeans(self):
        """Test K-Means clustering."""
        clustering.cluster_kmeans(
            self.data, n_clusters=2, use_rep="X_pca", key_added="kmeans_test"
        )

        self.assertIn("kmeans_test", self.data.obs.columns)
        # Should have 2 unique labels ('0', '1')
        unique_labels = self.data.obs["kmeans_test"].unique()
        self.assertEqual(len(unique_labels), 2)
        # Check dtype is categorical
        self.assertTrue(pd.api.types.is_categorical_dtype(self.data.obs["kmeans_test"]))

    def test_hierarchical(self):
        """Test Agglomerative Hierarchical clustering."""
        clustering.cluster_hierarchical(
            self.data, n_clusters=2, use_rep="X_pca", key_added="hclust"
        )

        self.assertIn("hclust", self.data.obs.columns)
        unique_labels = self.data.obs["hclust"].unique()
        self.assertEqual(len(unique_labels), 2)

    def test_dbscan(self):
        """Test DBSCAN clustering."""
        # Use a large epsilon so points cluster, or small to find noise.
        # Our data has two blobs far apart (dist ~10).
        # eps=0.5 should separate them or find noise, eps=15 should merge them.
        clustering.cluster_dbscan(
            self.data, eps=2.0, min_samples=2, use_rep="X_umap", key_added="dbscan"
        )

        self.assertIn("dbscan", self.data.obs.columns)
        # Just check it ran and produced labels
        self.assertTrue(len(self.data.obs["dbscan"]) == self.data.n_obs)

    @unittest.skipUnless(HAS_LEIDEN, "leidenalg or igraph not installed")
    def test_leiden(self):
        """Test Leiden graph clustering."""
        clustering.cluster_leiden(self.data, resolution=1.0, key_added="leiden_test")

        self.assertIn("leiden_test", self.data.obs.columns)
        # Ensure labels are created
        self.assertTrue(len(self.data.obs["leiden_test"]) > 0)

    @unittest.skipUnless(HAS_LOUVAIN, "louvain or igraph not installed")
    def test_louvain(self):
        """Test Louvain graph clustering."""
        clustering.cluster_louvain(self.data, resolution=1.0, key_added="louvain_test")

        self.assertIn("louvain_test", self.data.obs.columns)
        self.assertTrue(len(self.data.obs["louvain_test"]) > 0)

    def test_missing_representation_error(self):
        """Test error when obsm key is missing."""
        with self.assertRaises(ValueError):
            clustering.cluster_kmeans(self.data, use_rep="X_nonexistent")

    def test_missing_neighbors_error(self):
        """Test error when neighbor graph is missing for graph clustering."""
        # Remove neighbors
        del self.data.uns["neighbors"]

        # Only run if libraries are present, otherwise ImportError might hide the ValueError
        if HAS_LEIDEN:
            with self.assertRaises(ValueError):
                clustering.cluster_leiden(self.data)


if __name__ == "__main__":
    unittest.main()
