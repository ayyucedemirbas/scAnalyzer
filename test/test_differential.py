import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import differential
from core import SingleCellDataset


class TestDifferential(unittest.TestCase):

    def setUp(self):
        """
        Create a synthetic dataset with clear group differences.
        Group A: High expression of Gene 0.
        Group B: High expression of Gene 1.
        Gene 2: Random noise.
        """
        n_obs = 20
        n_vars = 3

        # Create counts
        # First 10 cells -> Group A
        # Next 10 cells -> Group B
        X = np.zeros((n_obs, n_vars))

        # Group A (Cells 0-9): High Gene 0 (Mean 5), Low Gene 1 (Mean 0)
        X[:10, 0] = np.random.normal(5, 1, 10)
        X[:10, 1] = np.random.normal(0, 0.1, 10)

        # Group B (Cells 10-19): Low Gene 0 (Mean 0), High Gene 1 (Mean 5)
        X[10:, 0] = np.random.normal(0, 0.1, 10)
        X[10:, 1] = np.random.normal(5, 1, 10)

        # Gene 2: Random everywhere
        X[:, 2] = np.random.normal(2, 1, 20)

        # Ensure no negative values (it's expression counts)
        X[X < 0] = 0

        obs = pd.DataFrame(
            {"group": ["A"] * 10 + ["B"] * 10},
            index=[f"cell_{i}" for i in range(n_obs)],
        )

        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

        self.data = SingleCellDataset(X, obs, var)
        self.data_sparse = SingleCellDataset(sp.csr_matrix(X), obs, var)

    def test_ttest_dense(self):
        """Test T-test identifying markers in dense matrix."""
        differential.rank_genes_groups(self.data, groupby="group", method="t-test")

        # Check Group A results
        res_A = self.data.uns["rank_genes_groups"]["A"]

        # Gene 0 should be the top marker for A
        self.assertEqual(res_A.iloc[0]["names"], "gene_0")
        self.assertTrue(res_A.iloc[0]["logfoldchanges"] > 0)
        self.assertTrue(res_A.iloc[0]["pvals_adj"] < 0.05)

        # Check Group B results
        res_B = self.data.uns["rank_genes_groups"]["B"]

        # Gene 1 should be the top marker for B
        self.assertEqual(res_B.iloc[0]["names"], "gene_1")
        self.assertTrue(res_B.iloc[0]["logfoldchanges"] > 0)

    def test_ttest_sparse(self):
        """Test T-test identifying markers in sparse matrix."""
        differential.rank_genes_groups(
            self.data_sparse, groupby="group", method="t-test"
        )

        res_A = self.data_sparse.uns["rank_genes_groups"]["A"]
        self.assertEqual(res_A.iloc[0]["names"], "gene_0")
        self.assertTrue(res_A.iloc[0]["pvals_adj"] < 0.05)

    def test_wilcoxon(self):
        """Test Wilcoxon rank-sum test."""
        differential.rank_genes_groups(self.data, groupby="group", method="wilcoxon")

        res_A = self.data.uns["rank_genes_groups"]["A"]
        # Gene 0 should still be top for A
        self.assertEqual(res_A.iloc[0]["names"], "gene_0")

    def test_use_raw_matrix(self):
        """Test using data.raw (when raw is just a matrix)."""
        # Set main X to zeros to prove we are using raw
        self.data.raw = self.data.X.copy()
        self.data.X[:] = 0

        differential.rank_genes_groups(self.data, groupby="group", use_raw=True)

        res_A = self.data.uns["rank_genes_groups"]["A"]
        # Should still find gene_0 because it looked at raw
        self.assertEqual(res_A.iloc[0]["names"], "gene_0")

        # If we didn't use raw, scores would be 0 or NaN
        differential.rank_genes_groups(self.data, groupby="group", use_raw=False)
        res_A_no_raw = self.data.uns["rank_genes_groups"]["A"]
        # Since X is zero, stats are likely 0/NaN
        self.assertTrue(
            np.isnan(res_A_no_raw.iloc[0]["scores"])
            or res_A_no_raw.iloc[0]["scores"] == 0
        )

    def test_use_raw_object(self):
        """Test using data.raw (when raw is a SingleCellDataset object)."""
        # Create a raw object
        raw_data = self.data.copy()
        self.data.raw = raw_data  # AnnData style

        # Zero out main X
        self.data.X[:] = 0

        differential.rank_genes_groups(self.data, groupby="group", use_raw=True)
        res_A = self.data.uns["rank_genes_groups"]["A"]
        self.assertEqual(res_A.iloc[0]["names"], "gene_0")

    def test_get_marker_genes_filter(self):
        """Test filtering logic for retrieving markers."""
        differential.rank_genes_groups(self.data, groupby="group")

        # Filter: strictly only significant genes
        markers = differential.get_marker_genes(
            self.data, group="A", pval_cutoff=0.01, lfc_cutoff=1.0
        )

        # Should contain gene_0 but likely not gene_2 (noise) or gene_1 (negative LFC)
        self.assertIn("gene_0", markers["names"].values)
        self.assertNotIn("gene_1", markers["names"].values)

    def test_missing_group_error(self):
        """Test error when groupby key is missing."""
        with self.assertRaises(ValueError):
            differential.rank_genes_groups(self.data, groupby="nonexistent_group")


if __name__ == "__main__":
    unittest.main()
