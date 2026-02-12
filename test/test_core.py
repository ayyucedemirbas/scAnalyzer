import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from core import SingleCellDataset


class TestSingleCellDataset(unittest.TestCase):

    def setUp(self):
        """
        Create a small synthetic dataset before each test.
        3 cells, 4 genes.
        """
        # Dense matrix
        self.X_dense = np.array([[1, 0, 2, 0], [0, 5, 0, 1], [3, 0, 0, 4]])

        # Sparse matrix (CSR)
        self.X_sparse = sp.csr_matrix(self.X_dense)

        # Metadata
        self.obs = pd.DataFrame(
            {"batch": ["A", "A", "B"]}, index=["cell_1", "cell_2", "cell_3"]
        )
        self.var = pd.DataFrame(
            {"symbol": ["Gene1", "Gene2", "Gene3", "Gene4"]},
            index=["g1", "g2", "g3", "g4"],
        )

        self.scd_dense = SingleCellDataset(self.X_dense, self.obs, self.var)
        self.scd_sparse = SingleCellDataset(self.X_sparse, self.obs, self.var)

    def test_initialization(self):
        """Test if the object initializes correctly."""
        # Check shapes
        self.assertEqual(self.scd_dense.shape, (3, 4))
        self.assertEqual(self.scd_sparse.shape, (3, 4))

        # Check n_obs and n_vars properties
        self.assertEqual(self.scd_dense.n_obs, 3)
        self.assertEqual(self.scd_dense.n_vars, 4)

        # Check if sparse input remains sparse
        self.assertTrue(sp.issparse(self.scd_sparse.X))

        # Check if metadata is stored correctly
        pd.testing.assert_frame_equal(self.scd_dense.obs, self.obs)
        pd.testing.assert_frame_equal(self.scd_dense.var, self.var)

    def test_initialization_mismatch(self):
        """Test if initialization raises error on shape mismatch."""
        # Wrong obs length
        bad_obs = pd.DataFrame({"batch": ["A", "B"]}, index=["c1", "c2"])  # Only 2 rows
        with self.assertRaises(ValueError):
            SingleCellDataset(self.X_dense, bad_obs, self.var)

        # Wrong var length
        bad_var = pd.DataFrame(index=["g1", "g2"])  # Only 2 rows
        with self.assertRaises(ValueError):
            SingleCellDataset(self.X_dense, self.obs, bad_var)

    def test_slicing_basic(self):
        """Test basic integer slicing."""
        # Slice first cell
        subset = self.scd_dense[0, :]
        self.assertEqual(subset.shape, (1, 4))

        # Slice first two cells and first two genes
        subset = self.scd_dense[0:2, 0:2]
        self.assertEqual(subset.shape, (2, 2))

        # Check values
        expected_X = self.X_dense[0:2, 0:2]
        np.testing.assert_array_equal(subset.X, expected_X)

    def test_slicing_with_names(self):
        pass

    def test_slicing_boolean(self):
        """Test slicing with boolean masks."""
        # Mask for batch 'A'
        mask = (self.scd_dense.obs["batch"] == "A").values
        subset = self.scd_dense[mask, :]

        self.assertEqual(subset.shape, (2, 4))
        self.assertTrue(all(subset.obs["batch"] == "A"))

    def test_copy(self):
        """Test deep copying."""
        scd_copy = self.scd_dense.copy()

        # Modify copy
        scd_copy.obs["new_col"] = 100

        # Original should not change
        self.assertNotIn("new_col", self.scd_dense.obs.columns)

        # Modify X in copy
        scd_copy.X[0, 0] = 999
        self.assertEqual(self.scd_dense.X[0, 0], 1)  # Original should be 1

    def test_obsm_varm_storage(self):
        """Test storage of embeddings."""
        pca_dummy = np.random.rand(3, 2)
        self.scd_dense.obsm["X_pca"] = pca_dummy

        # Check retrieval
        np.testing.assert_array_equal(self.scd_dense.obsm["X_pca"], pca_dummy)

        # Check slicing handles obsm
        subset = self.scd_dense[0:1, :]
        self.assertEqual(subset.obsm["X_pca"].shape, (1, 2))
        np.testing.assert_array_equal(subset.obsm["X_pca"], pca_dummy[0:1])

    def test_raw_attribute(self):
        """Test the raw attribute storage."""
        self.scd_dense.raw = self.scd_dense.X.copy()

        # Modify main X
        self.scd_dense.X = self.scd_dense.X * 2

        # Raw should be unchanged
        self.assertEqual(self.scd_dense.raw[0, 0], 1)
        self.assertEqual(self.scd_dense.X[0, 0], 2)


if __name__ == "__main__":
    unittest.main()
