import os
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import enrichment as enrich
from core import SingleCellDataset


class TestEnrichment(unittest.TestCase):

    def setUp(self):
        self.n_cells = 100
        self.n_genes = 200

        rng = np.random.default_rng(42)
        X = rng.poisson(2, size=(self.n_cells, self.n_genes)).astype(float)

        self.gene_set = ["Gene_10", "Gene_11", "Gene_12"]

        X[0:10, 10:13] += 5.0

        obs = pd.DataFrame(index=[f"cell_{i}" for i in range(self.n_cells)])
        var = pd.DataFrame(index=[f"Gene_{i}" for i in range(self.n_genes)])

        self.data = SingleCellDataset(X, obs, var)

    def test_gene_set_score_basic(self):
        enrich.gene_set_score(
            self.data,
            gene_list=self.gene_set,
            score_name="test_score",
            ctrl_size=5,
            n_bins=5,
            random_state=42,
        )

        self.assertIn("test_score", self.data.obs.columns)

        high_score_cells = self.data.obs.iloc[0:10]["test_score"]
        low_score_cells = self.data.obs.iloc[10:]["test_score"]
        self.assertGreater(high_score_cells.mean(), low_score_cells.mean())

    def test_score_multiple_gene_sets(self):
        gene_sets = {"set1": ["Gene_10", "Gene_11"], "set2": ["Gene_20", "Gene_21"]}

        enrich.score_multiple_gene_sets(self.data, gene_sets, ctrl_size=5)

        self.assertIn("set1_score", self.data.obs.columns)
        self.assertIn("set2_score", self.data.obs.columns)

    def test_rank_genes_groups_by_enrichment_hypergeometric(self):
        de_df = pd.DataFrame(
            {
                "names": ["Gene_10", "Gene_11", "Gene_12", "Gene_50", "Gene_60"],
                "pvals_adj": [0.01, 0.01, 0.01, 0.9, 0.9],
                "logfoldchanges": [2.0, 2.0, 2.0, 0.1, 0.1],
            }
        )

        self.data.uns["rank_genes_groups"] = {"cluster0": de_df}

        gene_sets = {"target_set": ["Gene_10", "Gene_11", "Gene_12"]}

        results = enrich.rank_genes_groups_by_enrichment(
            self.data, gene_sets, groupby="leiden", method="hypergeometric"
        )

        self.assertIn("cluster0", results)
        res_df = results["cluster0"]

        target_row = res_df[res_df["gene_set"] == "target_set"].iloc[0]
        self.assertEqual(target_row["overlap_size"], 3)
        self.assertLess(target_row["pval"], 0.05)

    def test_gsea_preranked(self):
        ranked_genes = pd.DataFrame(
            {
                "names": [f"Gene_{i}" for i in range(50)],
                "scores": sorted(np.random.normal(0, 1, 50), reverse=True),
            }
        )

        ranked_genes.loc[0:4, "names"] = [
            "Gene_A",
            "Gene_B",
            "Gene_C",
            "Gene_D",
            "Gene_E",
        ]
        gene_set = ["Gene_A", "Gene_B", "Gene_C"]

        res = enrich.gsea_preranked(ranked_genes, gene_set, nperm=10, random_state=42)

        self.assertIn("ES", res)
        self.assertIn("NES", res)
        self.assertIn("pval", res)
        self.assertGreater(res["ES"], 0)  # Should be positive enrichment

    def test_load_gene_sets_placeholder(self):
        sets = enrich.load_gene_sets()
        self.assertIsInstance(sets, dict)
        self.assertIn("HALLMARK_HYPOXIA", sets)


if __name__ == "__main__":
    unittest.main()
