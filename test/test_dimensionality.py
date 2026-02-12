import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import dimensionality
from core import SingleCellDataset

# Check for UMAP dependency for conditional skipping
try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class TestDimensionality(unittest.TestCase):

    def setUp(self):
        """
        Create a synthetic dataset.
        50 cells, 20 genes.
        """
        n_obs = 50
        n_vars = 20

        # Create random data
        X = np.random.rand(n_obs, n_vars)
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

        self.data = SingleCellDataset(X, obs, var)

        # Create sparse version
        self.data_sparse = SingleCellDataset(sp.csr_matrix(X), obs.copy(), var.copy())

    def test_pca_dense(self):
        """Test PCA on dense matrix."""
        n_comps = 5
        dimensionality.run_pca(
            self.data, n_components=n_comps, use_highly_variable=False
        )

        # Check obsm (Coordinates)
        self.assertIn("X_pca", self.data.obsm)
        self.assertEqual(self.data.obsm["X_pca"].shape, (50, n_comps))

        # Check varm (Loadings)
        self.assertIn("PCs", self.data.varm)
        self.assertEqual(self.data.varm["PCs"].shape, (20, n_comps))

        # Check uns (Variance info)
        self.assertIn("pca", self.data.uns)
        self.assertIn("variance_ratio", self.data.uns["pca"])

    def test_pca_sparse(self):
        """Test PCA (TruncatedSVD) on sparse matrix."""
        n_comps = 5
        dimensionality.run_pca(
            self.data_sparse, n_components=n_comps, use_highly_variable=False
        )

        self.assertIn("X_pca", self.data_sparse.obsm)
        self.assertEqual(self.data_sparse.obsm["X_pca"].shape, (50, n_comps))

    def test_pca_hvg(self):
        """Test PCA using only highly variable genes."""
        # Mark first 10 genes as HVG
        self.data.var["highly_variable"] = False
        self.data.var.iloc[:10, self.data.var.columns.get_loc("highly_variable")] = True

        n_comps = 3
        dimensionality.run_pca(
            self.data, n_components=n_comps, use_highly_variable=True
        )

        # Loadings shape should still be (n_vars, n_comps) but non-HVG rows should be 0
        loadings = self.data.varm["PCs"]
        self.assertEqual(loadings.shape, (20, n_comps))

        # Check that a non-HVG gene (index 15) has 0 loading
        self.assertTrue(np.all(loadings[15, :] == 0))
        # Check that an HVG gene (index 0) has non-zero loading
        self.assertFalse(np.all(loadings[0, :] == 0))

    def test_neighbors(self):
        """Test neighbor graph construction."""
        # Neighbors requires PCA first
        dimensionality.run_pca(self.data, n_components=10)

        k = 5
        dimensionality.neighbors(self.data, n_neighbors=k)

        self.assertIn("neighbors", self.data.uns)
        self.assertIn("distances", self.data.uns["neighbors"])
        self.assertIn("connectivities", self.data.uns["neighbors"])

        # Check shape of adjacency matrix
        adj = self.data.uns["neighbors"]["connectivities"]
        self.assertEqual(adj.shape, (50, 50))

        # Check that each row has k neighbors
        self.assertEqual(adj[0, :].getnnz(), k)

    def test_neighbors_no_pca_error(self):
        """Test error if neighbors run without PCA."""
        with self.assertRaises(ValueError):
            dimensionality.neighbors(self.data)

    def test_tsne(self):
        """Test t-SNE."""
        dimensionality.run_pca(self.data, n_components=10)

        # Use low perplexity because n_obs is small (50)
        dimensionality.run_tsne(self.data, perplexity=5.0)

        self.assertIn("X_tsne", self.data.obsm)
        self.assertEqual(self.data.obsm["X_tsne"].shape, (50, 2))

    def test_tsne_no_pca_error(self):
        """Test error if t-SNE run without PCA."""
        with self.assertRaises(ValueError):
            dimensionality.run_tsne(self.data)

    @unittest.skipUnless(HAS_UMAP, "umap-learn not installed")
    def test_umap(self):
        """Test UMAP."""
        dimensionality.run_pca(self.data, n_components=10)
        dimensionality.run_umap(self.data)

        self.assertIn("X_umap", self.data.obsm)
        self.assertEqual(self.data.obsm["X_umap"].shape, (50, 2))

    @unittest.skipUnless(HAS_UMAP, "umap-learn not installed")
    def test_umap_no_pca_error(self):
        """Test error if UMAP run without PCA."""
        with self.assertRaises(ValueError):
            dimensionality.run_umap(self.data)


if __name__ == "__main__":
    unittest.main()
