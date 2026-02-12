import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import preprocessing
from core import SingleCellDataset


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """
        Create a small synthetic dataset for testing.
        5 cells, 4 genes.
        """
        # Counts matrix:
        # Cell 0: [10, 0,  0, 0] -> 1 gene, 10 counts
        # Cell 1: [ 1, 1,  0, 0] -> 2 genes, 2 counts
        # Cell 2: [ 0, 0, 10, 0] -> 1 gene, 10 counts (MT high)
        # Cell 3: [ 2, 2,  2, 0] -> 3 genes, 6 counts
        # Cell 4: [ 0, 0,  0, 0] -> 0 genes, 0 counts (Empty)

        self.X = np.array(
            [[10, 0, 0, 0], [1, 1, 0, 0], [0, 0, 10, 0], [2, 2, 2, 0], [0, 0, 0, 0]],
            dtype=np.float32,
        )

        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(5)])

        # Gene 2 is "Mitochondrial" (MT-Gene)
        var = pd.DataFrame(index=["Gene1", "Gene2", "MT-Gene", "Gene4"])

        self.data = SingleCellDataset(sp.csr_matrix(self.X), obs, var)

    def test_calculate_qc_metrics(self):
        """Test calculation of basic QC metrics."""
        preprocessing.calculate_qc_metrics(self.data, qc_vars=["MT-"])

        # Check n_genes_by_counts
        expected_n_genes = [1, 2, 1, 3, 0]
        np.testing.assert_array_equal(
            self.data.obs["n_genes_by_counts"], expected_n_genes
        )

        # Check total_counts
        expected_total = [10, 2, 10, 6, 0]
        np.testing.assert_array_equal(self.data.obs["total_counts"], expected_total)

        # Check MT percentage
        # Cell 0: 0/10 = 0%
        # Cell 2: 10/10 = 100%
        # Cell 3: 2/6 = 33.33%
        self.assertEqual(self.data.obs.loc["cell_0", "pct_counts_MT-"], 0.0)
        self.assertEqual(self.data.obs.loc["cell_2", "pct_counts_MT-"], 100.0)
        self.assertAlmostEqual(
            self.data.obs.loc["cell_3", "pct_counts_MT-"], 33.333333, places=4
        )

    def test_filter_cells(self):
        """Test cell filtering logic."""
        # Pre-calc metrics so filter works
        preprocessing.calculate_qc_metrics(self.data, qc_vars=["MT-"])

        # Filter:
        # min_counts=3 -> drops cell 1 (2), cell 4 (0)
        # max_pct_mito=50 -> drops cell 2 (100%)
        # Remaining: Cell 0 (10 counts, 0% MT), Cell 3 (6 counts, 33% MT)

        filtered = preprocessing.filter_cells(
            self.data, min_counts=3, max_pct_mito=50.0
        )

        self.assertEqual(filtered.n_obs, 2)
        self.assertIn("cell_0", filtered.obs.index)
        self.assertIn("cell_3", filtered.obs.index)
        self.assertNotIn("cell_2", filtered.obs.index)

    def test_filter_genes(self):
        """Test gene filtering."""
        # Gene counts (cells > 0):
        # Gene1: 3 cells (0, 1, 3)
        # Gene2: 2 cells (1, 3)
        # MT-Gene: 2 cells (2, 3)
        # Gene4: 0 cells

        # Keep genes in at least 3 cells -> Only Gene1
        filtered = preprocessing.filter_genes(self.data, min_cells=3)

        self.assertEqual(filtered.n_vars, 1)
        self.assertEqual(filtered.var.index[0], "Gene1")

    def test_normalize_total(self):
        """Test total count normalization."""
        target_sum = 100.0
        preprocessing.normalize_total(self.data, target_sum=target_sum)

        # Check that rows sum to 100 (excluding the empty cell)
        # We perform on dense version for easier checking
        X_norm = self.data.X.toarray()
        row_sums = X_norm.sum(axis=1)

        # Cell 0, 1, 2, 3 should be 100
        np.testing.assert_allclose(row_sums[0:4], target_sum, rtol=1e-5)

        # Check raw is saved
        self.assertIsNotNone(self.data.raw)

    def test_log1p(self):
        """Test log transformation."""
        # X = log(X + 1)
        # Value 10 becomes log(11) approx 2.39
        preprocessing.log1p(self.data)

        X_log = self.data.X.toarray()
        self.assertAlmostEqual(X_log[0, 0], np.log1p(10), places=4)

    def test_highly_variable_genes(self):
        """Test HVG identification."""
        # Create a dataset where Gene 0 is highly variable (0, 100, 0, 100...)
        # and Gene 1 is constant (5, 5, 5, 5...)
        X_hvg = np.array([[0, 5], [100, 5], [0, 5], [100, 5]])
        data_hvg = SingleCellDataset(
            sp.csr_matrix(X_hvg),
            pd.DataFrame(index=[f"c{i}" for i in range(4)]),
            pd.DataFrame(index=["Variable", "Constant"]),
        )

        preprocessing.highly_variable_genes(data_hvg, n_top_genes=1)

        self.assertTrue(data_hvg.var.loc["Variable", "highly_variable"])
        self.assertFalse(data_hvg.var.loc["Constant", "highly_variable"])

    def test_scale(self):
        """Test scaling (z-score)."""
        # Dense input for scaling test
        self.data.X = self.data.X.toarray()

        # Scale to unit variance and zero mean
        preprocessing.scale(self.data, max_value=None)

        X_scaled = self.data.X

        # Check column means approx 0
        col_means = X_scaled.mean(axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-5)

        # Check column std approx 1 (for columns with variance)
        # Gene4 was all 0s, so std is 0 (handled in code to avoid div/0, usually becomes 0)
        col_std = X_scaled.std(axis=0)
        self.assertAlmostEqual(col_std[0], 1.0, places=4)

    def test_scale_clips_values(self):
        """Test that scaling clips values."""
        self.data.X = self.data.X.toarray()
        # Force a value to be an outlier after scaling
        preprocessing.scale(self.data, max_value=0.1)

        max_val = self.data.X.max()
        self.assertLessEqual(max_val, 0.1)


if __name__ == "__main__":
    unittest.main()
