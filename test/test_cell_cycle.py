import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import cell_cycle as cc
from core import SingleCellDataset


class TestCellCycle(unittest.TestCase):

    def setUp(self):
        self.n_cells = 100
        self.n_genes = 200

        rng = np.random.default_rng(42)
        # Use random float data (not poisson) to ensure variance for binning
        X = rng.uniform(0, 5, size=(self.n_cells, self.n_genes))

        self.s_genes = ["MCM5", "PCNA", "TYMS"]
        self.g2m_genes = ["HMGB2", "CDK1", "NUSAP1"]

        # Ensure marker genes are at the end of the list
        gene_names = (
            [f"Gene_{i}" for i in range(self.n_genes - 6)]
            + self.s_genes
            + self.g2m_genes
        )

        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(self.n_cells)])
        var = pd.DataFrame(index=gene_names)

        self.data = SingleCellDataset(X, obs, var)

    def test_score_genes_basic(self):
        scores = cc.score_genes(
            self.data, gene_list=self.s_genes, ctrl_size=5, n_bins=5, random_state=42
        )
        self.assertEqual(len(scores), self.n_cells)

    def test_score_cell_cycle_human(self):
        X = self.data.X

        # Massive signal boost to guarantee S phase classification
        s_indices = [self.data.var.index.get_loc(g) for g in self.s_genes]
        X[0:10, s_indices] += 50.0

        # Massive signal boost for G2M
        g2m_indices = [self.data.var.index.get_loc(g) for g in self.g2m_genes]
        X[10:20, g2m_indices] += 50.0

        self.data.X = X

        cc.score_cell_cycle(
            self.data,
            s_genes=self.s_genes,
            g2m_genes=self.g2m_genes,
            organism="human",
            ctrl_size=5,
        )

        self.assertIn("S_score", self.data.obs.columns)
        self.assertIn("phase", self.data.obs.columns)

        # Check first 10 cells are S phase
        s_cells = self.data.obs.iloc[0:10]
        self.assertTrue((s_cells["phase"] == "S").sum() >= 8)

    def test_regress_out_cell_cycle_diff(self):
        self.data.obs["S_score"] = np.random.rand(self.n_cells)
        self.data.obs["G2M_score"] = np.random.rand(self.n_cells)

        original_X = self.data.X.copy()
        cc.regress_out_cell_cycle(self.data, difference_only=True)

        # Check that data changed
        self.assertFalse(np.allclose(self.data.X, original_X))

    def test_sparse_matrix_support(self):
        # Convert to sparse and re-test
        self.data.X = sp.csr_matrix(self.data.X)

        cc.score_cell_cycle(
            self.data, s_genes=self.s_genes, g2m_genes=self.g2m_genes, ctrl_size=5
        )
        self.assertIn("phase", self.data.obs.columns)


if __name__ == "__main__":
    unittest.main()
