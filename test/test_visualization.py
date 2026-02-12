import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import matplotlib

import visualization
from core import SingleCellDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestVisualization(unittest.TestCase):

    def setUp(self):
        """
        Create a small dataset and a temporary directory for saving plots.
        """
        self.test_dir = tempfile.mkdtemp()

        # 20 cells, 5 genes
        n_obs = 20
        n_vars = 5

        X = np.random.rand(n_obs, n_vars)
        obs = pd.DataFrame(
            {"group": ["A"] * 10 + ["B"] * 10, "batch": ["1", "2"] * 10},
            index=[f"cell_{i}" for i in range(n_obs)],
        )
        obs["group"] = obs["group"].astype("category")

        var = pd.DataFrame(index=[f"Gene{i}" for i in range(n_vars)])

        self.data = SingleCellDataset(sp.csr_matrix(X), obs, var)

        # Add mock embeddings
        self.data.obsm["X_pca"] = np.random.rand(n_obs, 2)
        self.data.obsm["X_umap"] = np.random.rand(n_obs, 2)
        self.data.obsm["X_tsne"] = np.random.rand(n_obs, 2)

    def tearDown(self):
        """Cleanup temporary files."""
        shutil.rmtree(self.test_dir)
        plt.close("all")  # Close open figures

    def test_plot_embedding_categorical(self):
        """Test plotting UMAP with categorical color."""
        save_path = os.path.join(self.test_dir, "umap_cat.png")

        visualization.plot_umap(
            self.data, color="group", save=save_path, legend_loc="on data"
        )

        self.assertTrue(os.path.exists(save_path))

    def test_plot_embedding_continuous(self):
        """Test plotting PCA with continuous color (gene expression)."""
        save_path = os.path.join(self.test_dir, "pca_cont.png")

        visualization.plot_pca(self.data, color="Gene0", save=save_path)

        self.assertTrue(os.path.exists(save_path))

    def test_plot_tsne(self):
        """Test t-SNE plotting wrapper."""
        save_path = os.path.join(self.test_dir, "tsne.png")
        visualization.plot_tsne(self.data, save=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_violin(self):
        """Test violin plot."""
        save_path = os.path.join(self.test_dir, "violin.png")

        visualization.plot_violin(
            self.data, keys=["Gene0", "Gene1"], groupby="group", save=save_path
        )

        self.assertTrue(os.path.exists(save_path))

    def test_plot_heatmap(self):
        """Test heatmap."""
        save_path = os.path.join(self.test_dir, "heatmap.png")

        visualization.plot_heatmap(
            self.data,
            var_names=["Gene0", "Gene1", "Gene2"],
            groupby="group",
            save=save_path,
        )

        self.assertTrue(os.path.exists(save_path))

    def test_plot_dotplot(self):
        """Test dotplot."""
        save_path = os.path.join(self.test_dir, "dotplot.png")

        visualization.plot_dotplot(
            self.data, var_names=["Gene0", "Gene1"], groupby="group", save=save_path
        )

        self.assertTrue(os.path.exists(save_path))

    def test_plot_error_missing_key(self):
        """Test error when coloring by missing key."""
        with self.assertRaises(ValueError):
            visualization.plot_umap(self.data, color="NonExistentGene")

    def test_plot_error_missing_basis(self):
        """Test error when basis is missing."""
        del self.data.obsm["X_umap"]
        with self.assertRaises(ValueError):
            visualization.plot_umap(self.data)


if __name__ == "__main__":
    unittest.main()
