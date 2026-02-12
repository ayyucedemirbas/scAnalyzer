import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import utils
from core import SingleCellDataset


class TestUtils(unittest.TestCase):

    def setUp(self):
        """
        Create two small datasets for merging tests.
        """
        # Dataset 1: 2 cells, Genes [A, B, C]
        X1 = np.array([[1, 2, 3], [4, 5, 6]])
        obs1 = pd.DataFrame(index=["c1", "c2"])
        var1 = pd.DataFrame(index=["GeneA", "GeneB", "GeneC"])
        self.data1 = SingleCellDataset(sp.csr_matrix(X1), obs1, var1)

        # Dataset 2: 2 cells, Genes [B, C, D]
        # Overlap: B, C. Unique: A (in 1), D (in 2)
        X2 = np.array([[7, 8, 9], [10, 11, 12]])
        obs2 = pd.DataFrame(index=["c3", "c4"])
        var2 = pd.DataFrame(index=["GeneB", "GeneC", "GeneD"])
        self.data2 = SingleCellDataset(sp.csr_matrix(X2), obs2, var2)

    def test_merge_inner(self):
        """Test inner join (intersection of genes)."""
        merged = utils.merge(
            [self.data1, self.data2], join="inner", batch_keys=["d1", "d2"]
        )

        # Should have 4 cells (2+2)
        self.assertEqual(merged.n_obs, 4)

        # Should have 2 genes (intersection: B, C)
        self.assertEqual(merged.n_vars, 2)
        self.assertListEqual(sorted(merged.var.index.tolist()), ["GeneB", "GeneC"])

        # Check batch labels
        self.assertIn("batch", merged.obs.columns)
        self.assertEqual(merged.obs["batch"].value_counts()["d1"], 2)
        self.assertEqual(merged.obs["batch"].value_counts()["d2"], 2)

    def test_merge_outer(self):
        """Test outer join (union of genes)."""
        merged = utils.merge(
            [self.data1, self.data2], join="outer", batch_keys=["d1", "d2"]
        )

        # Should have 4 cells
        self.assertEqual(merged.n_obs, 4)

        # Should have 4 genes (union: A, B, C, D)
        self.assertEqual(merged.n_vars, 4)
        expected_genes = ["GeneA", "GeneB", "GeneC", "GeneD"]
        self.assertListEqual(sorted(merged.var.index.tolist()), expected_genes)

        # Check if missing values are 0 (sparse default)
        # Cell c3 (from d2) should have 0 for GeneA
        # Find index of c3 in merged
        # Note: indices are renamed to "d2_c3"
        idx = merged.obs.index.get_loc("d2_c3")
        geneA_idx = merged.var.index.get_loc("GeneA")

        val = merged.X[idx, geneA_idx]
        self.assertEqual(val, 0)

    def test_merge_dense(self):
        """Test merging dense matrices."""
        # Convert to dense
        self.data1.X = self.data1.X.toarray()
        self.data2.X = self.data2.X.toarray()

        merged = utils.merge([self.data1, self.data2], join="inner")

        # Result should be dense (numpy array)
        self.assertIsInstance(merged.X, np.ndarray)
        self.assertEqual(merged.shape, (4, 2))

    def test_subsample_n(self):
        """Test subsampling by number."""
        # data1 has 2 cells
        sub = utils.subsample(self.data1, n=1)
        self.assertEqual(sub.n_obs, 1)

    def test_subsample_fraction(self):
        """Test subsampling by fraction."""
        # Create larger dataset
        X = np.zeros((100, 5))
        data = SingleCellDataset(X)

        sub = utils.subsample(data, fraction=0.5)
        self.assertEqual(sub.n_obs, 50)

    def test_subsample_error(self):
        """Test subsampling input validation."""
        with self.assertRaises(ValueError):
            utils.subsample(self.data1, fraction=1.5)  # Invalid fraction

        with self.assertRaises(ValueError):
            utils.subsample(self.data1)  # No args

    def test_get_mean_var(self):
        """Test mean and variance calculation."""
        # Simple matrix:
        # [[1, 2],
        #  [3, 4]]
        # Col means: [2, 3]
        # Col var: [1, 1] (Population variance: ((1-2)^2 + (3-2)^2)/2 = 1)
        X = np.array([[1, 2], [3, 4]])
        data = SingleCellDataset(sp.csr_matrix(X))

        mean, var = utils.get_mean_var(data, axis=0)

        np.testing.assert_array_equal(mean, [2, 3])
        np.testing.assert_array_equal(var, [1, 1])


if __name__ == "__main__":
    unittest.main()
