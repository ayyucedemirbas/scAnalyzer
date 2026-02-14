import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import batch_correction as bc
from core import SingleCellDataset


class TestBatchCorrection(unittest.TestCase):

    def setUp(self):
        self.n_cells = 100
        self.n_genes = 50
        self.n_pcs = 20
        self.n_batches = 2

        rng = np.random.default_rng(42)
        X = rng.random((self.n_cells, self.n_genes))

        X[50:, :] += 2.0

        batch_labels = ["batch_1"] * 50 + ["batch_2"] * 50
        obs = pd.DataFrame(
            {
                "batch": batch_labels,
                "cell_type": ["A"] * 25 + ["B"] * 25 + ["A"] * 25 + ["B"] * 25,
            },
            index=[f"cell_{i}" for i in range(self.n_cells)],
        )

        var = pd.DataFrame(index=[f"gene_{i}" for i in range(self.n_genes)])

        self.data = SingleCellDataset(X, obs, var)

        X_pca = rng.random((self.n_cells, self.n_pcs))
        X_pca[50:, :] += 1.0
        self.data.obsm["X_pca"] = X_pca

    def test_harmony_integrate_runs(self):
        bc.harmony_integrate(
            self.data,
            batch_key="batch",
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
            verbose=False,
            max_iter_harmony=3,
        )

        self.assertIn("X_pca_harmony", self.data.obsm)

        corrected = self.data.obsm["X_pca_harmony"]
        self.assertEqual(corrected.shape, (self.n_cells, self.n_pcs))

        original = self.data.obsm["X_pca"]
        self.assertFalse(np.allclose(corrected, original))

    def test_harmony_missing_keys_raises_error(self):
        del self.data.obsm["X_pca"]
        with self.assertRaises(ValueError):
            bc.harmony_integrate(self.data, basis="X_pca")

        self.setUp()

        with self.assertRaises(ValueError):
            bc.harmony_integrate(self.data, batch_key="non_existent_batch")

    def test_combat_runs_inplace(self):
        original_X = self.data.X.copy()

        bc.combat(self.data, batch_key="batch", inplace=True)

        self.assertFalse(np.allclose(self.data.X, original_X))

        self.assertEqual(self.data.X.shape, (self.n_cells, self.n_genes))

    def test_combat_returns_new_object(self):
        original_X = self.data.X.copy()

        new_data = bc.combat(self.data, batch_key="batch", inplace=False)

        self.assertIsInstance(new_data, SingleCellDataset)
        self.assertIsNot(new_data, self.data)

        self.assertTrue(np.allclose(self.data.X, original_X))

        self.assertFalse(np.allclose(new_data.X, original_X))

    def test_combat_sparse_matrix(self):
        self.data.X = sp.csr_matrix(self.data.X)

        bc.combat(self.data, batch_key="batch", inplace=True)

        self.assertTrue(sp.issparse(self.data.X) or isinstance(self.data.X, np.ndarray))

    def test_mnn_correct_structure(self):
        data1 = self.data[0:50, :]
        data2 = self.data[50:100, :]

        integrated = bc.mnn_correct([data1, data2], batch_key="batch_label")

        self.assertIsInstance(integrated, SingleCellDataset)
        self.assertEqual(integrated.n_obs, 100)

        self.assertIn("batch_label", integrated.obs.columns)
        unique_batches = integrated.obs["batch_label"].unique()
        self.assertEqual(len(unique_batches), 2)


if __name__ == "__main__":
    unittest.main()
