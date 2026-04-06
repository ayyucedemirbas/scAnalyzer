import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import dimensionality
from core import SingleCellDataset

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class TestDimensionality(unittest.TestCase):

    def setUp(self):
        n_obs = 50
        n_vars = 20

        X = np.random.rand(n_obs, n_vars)
        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

        self.data = SingleCellDataset(X, obs, var)

        self.data_sparse = SingleCellDataset(sp.csr_matrix(X), obs.copy(), var.copy())

    def test_pca_dense(self):
        n_comps = 5
        dimensionality.run_pca(
            self.data, n_components=n_comps, use_highly_variable=False
        )

        self.assertIn("X_pca", self.data.obsm)
        self.assertEqual(self.data.obsm["X_pca"].shape, (50, n_comps))

        self.assertIn("PCs", self.data.varm)
        self.assertEqual(self.data.varm["PCs"].shape, (20, n_comps))

        self.assertIn("pca", self.data.uns)
        self.assertIn("variance_ratio", self.data.uns["pca"])

    def test_pca_sparse(self):
        n_comps = 5
        dimensionality.run_pca(
            self.data_sparse, n_components=n_comps, use_highly_variable=False
        )

        self.assertIn("X_pca", self.data_sparse.obsm)
        self.assertEqual(self.data_sparse.obsm["X_pca"].shape, (50, n_comps))

    def test_pca_hvg(self):
        self.data.var["highly_variable"] = False
        self.data.var.iloc[:10, self.data.var.columns.get_loc("highly_variable")] = True

        n_comps = 3
        dimensionality.run_pca(
            self.data, n_components=n_comps, use_highly_variable=True
        )

        loadings = self.data.varm["PCs"]
        self.assertEqual(loadings.shape, (20, n_comps))

        self.assertTrue(np.all(loadings[15, :] == 0))
        self.assertFalse(np.all(loadings[0, :] == 0))

    def test_neighbors(self):
        dimensionality.run_pca(self.data, n_components=10)

        k = 5
        dimensionality.neighbors(self.data, n_neighbors=k)

        self.assertIn("neighbors", self.data.uns)
        self.assertIn("distances", self.data.uns["neighbors"])
        self.assertIn("connectivities", self.data.uns["neighbors"])

        adj = self.data.uns["neighbors"]["connectivities"]
        self.assertEqual(adj.shape, (50, 50))

        self.assertEqual(adj[0, :].getnnz(), k)

    def test_neighbors_no_pca_error(self):
        with self.assertRaises(ValueError):
            dimensionality.neighbors(self.data)

    def test_tsne(self):
        dimensionality.run_pca(self.data, n_components=10)

        dimensionality.run_tsne(self.data, perplexity=5.0)

        self.assertIn("X_tsne", self.data.obsm)
        self.assertEqual(self.data.obsm["X_tsne"].shape, (50, 2))

    def test_tsne_no_pca_error(self):
        with self.assertRaises(ValueError):
            dimensionality.run_tsne(self.data)

    @unittest.skipUnless(HAS_UMAP, "umap-learn not installed")
    def test_umap(self):
        dimensionality.run_pca(self.data, n_components=10)
        dimensionality.run_umap(self.data)

        self.assertIn("X_umap", self.data.obsm)
        self.assertEqual(self.data.obsm["X_umap"].shape, (50, 2))

    @unittest.skipUnless(HAS_UMAP, "umap-learn not installed")
    def test_umap_no_pca_error(self):
        with self.assertRaises(ValueError):
            dimensionality.run_umap(self.data)


if __name__ == "__main__":
    unittest.main()
