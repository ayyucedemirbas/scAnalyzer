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

        self.X = np.array(
            [[10, 0, 0, 0], [1, 1, 0, 0], [0, 0, 10, 0], [2, 2, 2, 0], [0, 0, 0, 0]],
            dtype=np.float32,
        )

        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(5)])

        var = pd.DataFrame(index=["Gene1", "Gene2", "MT-Gene", "Gene4"])

        self.data = SingleCellDataset(sp.csr_matrix(self.X), obs, var)

    def test_calculate_qc_metrics(self):
        preprocessing.calculate_qc_metrics(self.data, qc_vars=["MT-"])

        expected_n_genes = [1, 2, 1, 3, 0]
        np.testing.assert_array_equal(
            self.data.obs["n_genes_by_counts"], expected_n_genes
        )

        expected_total = [10, 2, 10, 6, 0]
        np.testing.assert_array_equal(self.data.obs["total_counts"], expected_total)

        self.assertEqual(self.data.obs.loc["cell_0", "pct_counts_MT-"], 0.0)
        self.assertEqual(self.data.obs.loc["cell_2", "pct_counts_MT-"], 100.0)
        self.assertAlmostEqual(
            self.data.obs.loc["cell_3", "pct_counts_MT-"], 33.333333, places=4
        )

    def test_filter_cells(self):
        preprocessing.calculate_qc_metrics(self.data, qc_vars=["MT-"])

        filtered = preprocessing.filter_cells(
            self.data, min_counts=3, max_pct_mito=50.0
        )

        self.assertEqual(filtered.n_obs, 2)
        self.assertIn("cell_0", filtered.obs.index)
        self.assertIn("cell_3", filtered.obs.index)
        self.assertNotIn("cell_2", filtered.obs.index)

    def test_filter_genes(self):
        filtered = preprocessing.filter_genes(self.data, min_cells=3)

        self.assertEqual(filtered.n_vars, 1)
        self.assertEqual(filtered.var.index[0], "Gene1")

    def test_normalize_total(self):
        target_sum = 100.0
        preprocessing.normalize_total(self.data, target_sum=target_sum)

        X_norm = self.data.X.toarray()
        row_sums = X_norm.sum(axis=1)

        np.testing.assert_allclose(row_sums[0:4], target_sum, rtol=1e-5)

        self.assertIsNotNone(self.data.raw)

    def test_log1p(self):
        preprocessing.log1p(self.data)

        X_log = self.data.X.toarray()
        self.assertAlmostEqual(X_log[0, 0], np.log1p(10), places=4)

    def test_highly_variable_genes(self):
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
        self.data.X = self.data.X.toarray()

        preprocessing.scale(self.data, max_value=None)

        X_scaled = self.data.X

        col_means = X_scaled.mean(axis=0)
        np.testing.assert_allclose(col_means, 0, atol=1e-5)

        col_std = X_scaled.std(axis=0)
        self.assertAlmostEqual(col_std[0], 1.0, places=4)

    def test_scale_clips_values(self):
        self.data.X = self.data.X.toarray()
        preprocessing.scale(self.data, max_value=0.1)

        max_val = self.data.X.max()
        self.assertLessEqual(max_val, 0.1)


if __name__ == "__main__":
    unittest.main()
