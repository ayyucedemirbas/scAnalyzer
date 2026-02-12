import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import sc_io
from core import SingleCellDataset


class TestIO(unittest.TestCase):

    def setUp(self):
        """
        Create a temporary directory and a small dataset for testing.
        """
        self.test_dir = tempfile.mkdtemp()

        # Create a small dataset
        # 3 cells, 2 genes
        self.X_dense = np.array([[1, 0], [0, 2], [3, 0]])
        self.X_sparse = sp.csr_matrix(self.X_dense)

        self.obs = pd.DataFrame(
            {"batch": ["A", "B", "A"], "score": [0.1, 0.5, 0.9]},
            index=["cell_1", "cell_2", "cell_3"],
        )
        self.var = pd.DataFrame({"symbol": ["Gene1", "Gene2"]}, index=["g1", "g2"])

        self.data = SingleCellDataset(self.X_sparse, self.obs, self.var)
        self.data.obsm["X_pca"] = np.random.rand(3, 2)
        self.data.uns["params"] = {"n_neighbors": 10}

    def tearDown(self):
        """Cleanup temporary files."""
        shutil.rmtree(self.test_dir)

    def test_make_unique(self):
        """Test the unique naming helper."""
        names = np.array(["A", "B", "A", "C", "B", "A"])
        unique_names = sc_io._make_unique(names)

        expected = np.array(["A", "B", "A-1", "C", "B-1", "A-2"])
        np.testing.assert_array_equal(unique_names, expected)

    def test_write_read_h5ad_sparse(self):
        """Test writing and reading H5AD with sparse matrix."""
        filename = os.path.join(self.test_dir, "test_sparse.h5ad")

        # Write
        sc_io.write_h5ad(self.data, filename)

        # Read
        loaded_data = sc_io.read_h5ad(filename)

        # Check X
        self.assertTrue(sp.issparse(loaded_data.X))
        np.testing.assert_array_equal(loaded_data.X.toarray(), self.X_dense)

        # Check Obs/Var
        pd.testing.assert_frame_equal(loaded_data.obs, self.obs)
        pd.testing.assert_frame_equal(loaded_data.var, self.var)

        # Check obsm
        np.testing.assert_array_equal(
            loaded_data.obsm["X_pca"], self.data.obsm["X_pca"]
        )

    def test_write_read_h5ad_dense(self):
        """Test writing and reading H5AD with dense matrix."""
        filename = os.path.join(self.test_dir, "test_dense.h5ad")

        # Convert to dense
        self.data.X = self.data.X.toarray()

        # Write
        sc_io.write_h5ad(self.data, filename)

        # Read
        loaded_data = sc_io.read_h5ad(filename)

        # Check X
        self.assertFalse(sp.issparse(loaded_data.X))
        np.testing.assert_array_equal(loaded_data.X, self.X_dense)

    def test_csv_io(self):
        """Test CSV reading and writing."""
        prefix = os.path.join(self.test_dir, "test_csv")

        # Write
        sc_io.write_csvs(self.data, prefix)

        # Check files exist
        self.assertTrue(os.path.exists(f"{prefix}_X.csv"))
        self.assertTrue(os.path.exists(f"{prefix}_obs.csv"))
        self.assertTrue(os.path.exists(f"{prefix}_var.csv"))

        # Read back (basic check)
        # Note: read_csv reads just the matrix usually
        loaded_data = sc_io.read_csv(f"{prefix}_X.csv")
        np.testing.assert_array_equal(loaded_data.X.toarray(), self.X_dense)

    def test_h5ad_categorical_obs(self):
        """Test H5AD handles categorical columns in obs correctly."""
        # Convert 'batch' to categorical
        self.data.obs["batch"] = self.data.obs["batch"].astype("category")

        filename = os.path.join(self.test_dir, "test_cat.h5ad")
        sc_io.write_h5ad(self.data, filename)

        loaded_data = sc_io.read_h5ad(filename)

        # Check values
        pd.testing.assert_series_equal(self.data.obs["batch"], loaded_data.obs["batch"])

        # Check dtype is categorical
        self.assertTrue(pd.api.types.is_categorical_dtype(loaded_data.obs["batch"]))


if __name__ == "__main__":
    unittest.main()
