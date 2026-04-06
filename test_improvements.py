import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import batch_correction
import clustering
import differential
import dimensionality
import preprocessing
import sc_io
import trajectory
import utils
import visualization
from core import SingleCellDataset


def _make_data(n_cells=60, n_genes=20, sparse=True, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, n_genes))
    X[: n_cells // 2, 0] += 5.0
    X[n_cells // 2 :, 1] += 5.0
    if sparse:
        X = sp.csr_matrix(X)
    half = n_cells // 2
    obs = pd.DataFrame(
        {
            "group": ["A"] * half + ["B"] * half,
            "batch": ["b1"] * (half // 2)
            + ["b2"] * (half - half // 2)
            + ["b1"] * (half // 2)
            + ["b2"] * (half - half // 2),
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    obs["group"] = obs["group"].astype("category")
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    d = SingleCellDataset(X, obs, var)
    return d


class TestCore(unittest.TestCase):

    def setUp(self):
        self.d = _make_data()

    def test_len(self):
        self.assertEqual(len(self.d), 60)

    def test_contains_gene(self):
        self.assertIn("g0", self.d)
        self.assertNotIn("FAKE_GENE", self.d)

    def test_contains_obs_col(self):
        self.assertIn("group", self.d)

    def test_iter_yields_cell_names(self):
        names = list(self.d)
        self.assertEqual(len(names), 60)
        self.assertEqual(names[0], "c0")

    def test_to_df_shape(self):
        df = self.d.to_df()
        self.assertEqual(df.shape, (60, 20))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns[:3]), ["g0", "g1", "g2"])

    def test_summary_returns_string(self):
        s = self.d.summary()
        self.assertIsInstance(s, str)
        self.assertIn("60", s)

    def test_obs_names_var_names(self):
        self.assertEqual(len(self.d.obs_names()), 60)
        self.assertEqual(len(self.d.var_names()), 20)

    def test_copy_is_independent(self):
        c = self.d.copy()
        c.obs["new_col"] = 999
        self.assertNotIn("new_col", self.d.obs.columns)

    def test_x_setter_shape_guard(self):
        with self.assertRaises(ValueError):
            self.d.X = np.zeros((10, 10))


class TestDifferential(unittest.TestCase):

    def setUp(self):
        self.d = _make_data(sparse=False)

    def test_wilcoxon_no_name_error(self):
        differential.rank_genes_groups(self.d, groupby="group", method="wilcoxon")
        res = self.d.uns["rank_genes_groups"]["A"]
        self.assertIn("scores", res.columns)
        self.assertIn("pvals_adj", res.columns)
        self.assertFalse(res["scores"].isna().any())

    def test_wilcoxon_finds_correct_marker(self):
        differential.rank_genes_groups(self.d, groupby="group", method="wilcoxon")
        res = differential.get_marker_genes(self.d, group="A", lfc_cutoff=1.0)
        self.assertEqual(res.iloc[0]["names"], "g0")

    def test_ttest_finds_correct_marker(self):
        differential.rank_genes_groups(self.d, groupby="group", method="t-test")
        res = differential.get_marker_genes(self.d, group="B", lfc_cutoff=1.0)
        self.assertEqual(res.iloc[0]["names"], "g1")

    def test_top_n_parameter(self):
        differential.rank_genes_groups(self.d, groupby="group")
        res = differential.get_marker_genes(self.d, group="A", top_n=1)
        self.assertLessEqual(len(res), 1)

    def test_missing_group_raises(self):
        with self.assertRaises(ValueError):
            differential.rank_genes_groups(self.d, groupby="nonexistent")

    def test_sparse_matrix(self):
        d_sp = _make_data(sparse=True)
        differential.rank_genes_groups(d_sp, groupby="group", method="t-test")
        res = d_sp.uns["rank_genes_groups"]["A"]
        self.assertFalse(res["scores"].isna().any())


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        X = sp.csr_matrix(rng.poisson(2, (40, 15)).astype(float))
        obs = pd.DataFrame(index=[f"c{i}" for i in range(40)])
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(15)])
        self.d = SingleCellDataset(X, obs, var)

    def test_log1p_inplace_returns_none(self):
        result = preprocessing.log1p(self.d, inplace=True)
        self.assertIsNone(result)

    def test_log1p_not_inplace_returns_data(self):
        orig = self.d.copy()
        result = preprocessing.log1p(self.d, inplace=False)
        self.assertIsInstance(result, SingleCellDataset)
        np.testing.assert_array_equal(self.d.X.toarray(), orig.X.toarray())

    def test_normalize_total_raw_on_correct_object(self):
        orig_raw = self.d.raw
        new_d = preprocessing.normalize_total(self.d, inplace=False)
        self.assertIsNone(orig_raw)
        self.assertIsNotNone(new_d.raw)

    def test_scale_inplace_false(self):
        d2 = self.d.copy()
        d2.X = d2.X.toarray()
        result = preprocessing.scale(d2, inplace=False)
        self.assertIsInstance(result, SingleCellDataset)

    def test_hvg_bin_normalised(self):
        preprocessing.highly_variable_genes(self.d, n_top_genes=5)
        self.assertIn("highly_variable", self.d.var.columns)
        self.assertEqual(self.d.var["highly_variable"].sum(), 5)
        self.assertIn("dispersions_norm", self.d.var.columns)


class TestBatchCorrection(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        n_cells, n_genes = 80, 30
        X = rng.random((n_cells, n_genes))
        X[40:, :] += 2.0
        obs = pd.DataFrame(
            {"batch": ["b1"] * 40 + ["b2"] * 40},
            index=[f"c{i}" for i in range(n_cells)],
        )
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
        self.d = SingleCellDataset(X, obs, var)

    def test_combat_inplace_modifies_x(self):
        orig = self.d.X.copy()
        batch_correction.combat(self.d, batch_key="batch", inplace=True)
        self.assertFalse(np.allclose(self.d.X, orig))

    def test_combat_inplace_false_returns_copy(self):
        orig = self.d.X.copy()
        new_d = batch_correction.combat(self.d, batch_key="batch", inplace=False)
        self.assertIsInstance(new_d, SingleCellDataset)
        self.assertIsNot(new_d, self.d)
        np.testing.assert_array_almost_equal(self.d.X, orig)

    def test_combat_sparse(self):
        d2 = self.d.copy()
        d2.X = sp.csr_matrix(d2.X)
        batch_correction.combat(d2, batch_key="batch", inplace=True)
        self.assertEqual(d2.X.shape, (80, 30))

    def test_harmony_runs(self):
        d2 = self.d.copy()
        dimensionality.run_pca(d2, n_components=10, use_highly_variable=False)
        batch_correction.harmony_integrate(
            d2, batch_key="batch", max_iter_harmony=2, verbose=False
        )
        self.assertIn("X_pca_harmony", d2.obsm)
        self.assertEqual(d2.obsm["X_pca_harmony"].shape, (80, 10))

    def test_harmony_missing_basis_raises(self):
        with self.assertRaises(ValueError):
            batch_correction.harmony_integrate(self.d, basis="X_nonexistent")


class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.d = _make_data(sparse=True)
        self.d.obsm["X_umap"] = np.random.rand(60, 2)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)
        plt.close("all")

    def test_save_creates_file(self):
        path = os.path.join(self.tmpdir, "umap.png")
        visualization.plot_umap(self.d, color="group", save=path)
        self.assertTrue(os.path.exists(path))

    def test_no_figure_leak(self):
        before = len(plt.get_fignums())
        for _ in range(5):
            path = os.path.join(self.tmpdir, "tmp.png")
            visualization.plot_umap(self.d, color="group", save=path)
        after = len(plt.get_fignums())
        self.assertEqual(after - before, 0)

    def test_missing_key_error_message(self):
        with self.assertRaises(ValueError) as ctx:
            visualization.plot_umap(self.d, color="FAKE")
        self.assertIn("not found", str(ctx.exception))

    def test_missing_basis_error(self):
        with self.assertRaises(ValueError):
            visualization.plot_embedding(self.d, basis="X_nonexistent")

    def test_plot_dotplot_saves(self):
        path = os.path.join(self.tmpdir, "dot.png")
        visualization.plot_dotplot(
            self.d, var_names=["g0", "g1"], groupby="group", save=path
        )
        self.assertTrue(os.path.exists(path))

    def test_plot_heatmap_saves(self):
        path = os.path.join(self.tmpdir, "hm.png")
        visualization.plot_heatmap(
            self.d, var_names=["g0", "g1", "g2"], groupby="group", save=path
        )
        self.assertTrue(os.path.exists(path))

    def test_volcano_plot(self):
        differential.rank_genes_groups(
            self.d, groupby="group", method="t-test", use_raw=False
        )
        path = os.path.join(self.tmpdir, "volcano.png")
        visualization.volcano_plot(self.d, group="A", save=path)
        self.assertTrue(os.path.exists(path))


class TestIO(unittest.TestCase):

    def setUp(self):
        X = sp.csr_matrix(np.eye(4))
        obs = pd.DataFrame(
            {
                "batch": pd.Categorical(["A", "B", "A", "B"]),
                "score": [0.1, 0.2, 0.3, 0.4],
            },
            index=["c1", "c2", "c3", "c4"],
        )
        var = pd.DataFrame(index=["g1", "g2", "g3", "g4"])
        self.d = SingleCellDataset(X, obs, var)
        self.d.obsm["X_pca"] = np.random.rand(4, 2)
        self.d.uns["n_neighbors"] = 10

    def test_h5ad_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            path = f.name
        try:
            sc_io.write_h5ad(self.d, path)
            d2 = sc_io.read_h5ad(path)
            np.testing.assert_array_equal(d2.X.toarray(), self.d.X.toarray())
            self.assertEqual(list(d2.obs.columns), list(self.d.obs.columns))
            self.assertTrue(pd.api.types.is_categorical_dtype(d2.obs["batch"]))
        finally:
            os.unlink(path)

    def test_make_unique(self):
        names = np.array(["A", "B", "A", "C", "B", "A"])
        u = sc_io._make_unique(names)
        self.assertEqual(list(u), ["A", "B", "A-1", "C", "B-1", "A-2"])

    def test_make_unique_non_string(self):
        names = np.array([1, 2, 1, 3])
        u = sc_io._make_unique(names)
        self.assertEqual(len(u), 4)


class TestDimensionality(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(2)
        X = rng.random((50, 15))
        obs = pd.DataFrame(index=[f"c{i}" for i in range(50)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(15)])
        self.d = SingleCellDataset(X, obs, var)

    def test_run_pca_basic(self):
        dimensionality.run_pca(self.d, n_components=5, use_highly_variable=False)
        self.assertIn("X_pca", self.d.obsm)
        self.assertEqual(self.d.obsm["X_pca"].shape, (50, 5))

    def test_run_pca_caps_components(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dimensionality.run_pca(self.d, n_components=100, use_highly_variable=False)
            self.assertTrue(
                any("capped" in str(warning.message).lower() for warning in w)
                or self.d.obsm["X_pca"].shape[1] < 100
            )

    def test_neighbors_gaussian_weights(self):
        dimensionality.run_pca(self.d, n_components=5, use_highly_variable=False)
        dimensionality.neighbors(self.d, n_neighbors=5)
        conn = self.d.uns["neighbors"]["connectivities"]
        data_vals = conn.data
        self.assertTrue(np.all(data_vals > 0))
        self.assertTrue(np.all(data_vals <= 1.0 + 1e-9))

    def test_neighbors_no_pca_raises(self):
        with self.assertRaises(ValueError):
            dimensionality.neighbors(self.d)

    def test_diffmap(self):
        dimensionality.run_pca(self.d, n_components=5, use_highly_variable=False)
        dimensionality.neighbors(self.d, n_neighbors=5)
        dimensionality.run_diffmap(self.d, n_components=5)
        self.assertIn("X_diffmap", self.d.obsm)
        self.assertEqual(self.d.obsm["X_diffmap"].shape[0], 50)


class TestClustering(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(3)
        X1 = rng.normal(0, 0.5, (25, 5))
        X2 = rng.normal(10, 0.5, (25, 5))
        pca = np.vstack([X1, X2])
        obs = pd.DataFrame(index=[f"c{i}" for i in range(50)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
        self.d = SingleCellDataset(np.random.rand(50, 5), obs, var)
        self.d.obsm["X_pca"] = pca

    def test_kmeans_finds_two_clusters(self):
        clustering.cluster_kmeans(self.d, n_clusters=2, use_rep="X_pca")
        labels = self.d.obs["kmeans"].astype(str)
        self.assertEqual(len(labels.unique()), 2)

    def test_spectral(self):
        clustering.cluster_spectral(
            self.d, n_clusters=2, use_rep="X_pca", key_added="spectral"
        )
        self.assertIn("spectral", self.d.obs.columns)
        self.assertEqual(len(self.d.obs["spectral"].unique()), 2)

    def test_cluster_stats(self):
        clustering.cluster_kmeans(self.d, n_clusters=2, use_rep="X_pca")
        stats = clustering.cluster_stats(self.d, cluster_key="kmeans")
        self.assertIn("n_cells", stats.columns)
        self.assertIn("pct_of_total", stats.columns)
        self.assertEqual(stats["n_cells"].sum(), 50)

    def test_missing_rep_error(self):
        with self.assertRaises(ValueError):
            clustering.cluster_kmeans(self.d, use_rep="X_nonexistent")

    def test_dbscan_default_rep_is_pca(self):
        clustering.cluster_dbscan(self.d, eps=3.0, min_samples=2)
        self.assertIn("dbscan", self.d.obs.columns)


class TestUtils(unittest.TestCase):

    def setUp(self):
        X1 = sp.csr_matrix(np.array([[1.0, 2, 3], [4, 5, 6]]))
        obs1 = pd.DataFrame(index=["c1", "c2"])
        var1 = pd.DataFrame(index=["GA", "GB", "GC"])
        self.d1 = SingleCellDataset(X1, obs1, var1)

        X2 = sp.csr_matrix(np.array([[7.0, 8, 9], [10, 11, 12]]))
        obs2 = pd.DataFrame(index=["c3", "c4"])
        var2 = pd.DataFrame(index=["GB", "GC", "GD"])
        self.d2 = SingleCellDataset(X2, obs2, var2)

    def test_merge_inner(self):
        m = utils.merge([self.d1, self.d2], join="inner")
        self.assertEqual(m.n_obs, 4)
        self.assertEqual(sorted(m.var.index.tolist()), ["GB", "GC"])

    def test_merge_outer(self):
        m = utils.merge([self.d1, self.d2], join="outer")
        self.assertEqual(m.n_obs, 4)
        self.assertEqual(sorted(m.var.index.tolist()), ["GA", "GB", "GC", "GD"])

    def test_merge_outer_missing_values_zero(self):
        m = utils.merge([self.d1, self.d2], join="outer", batch_keys=["d1", "d2"])
        # c3/c4 from d2 should have 0 for GA
        ga_idx = list(m.var.index).index("GA")
        c3_idx = list(m.obs.index).index("d2_c3")
        val = m.X[c3_idx, ga_idx]
        if sp.issparse(m.X):
            val = m.X.toarray()[c3_idx, ga_idx]
        self.assertEqual(val, 0.0)

    def test_subsample_no_global_seed_side_effect(self):
        np.random.seed(99)
        state_before = np.random.get_state()[1][0]
        utils.subsample(self.d1, n=1, random_state=42)
        state_after = np.random.get_state()[1][0]
        self.assertEqual(state_before, state_after)

    def test_subsample_stratified(self):
        rng = np.random.default_rng(0)
        X = sp.csr_matrix(np.ones((100, 3)))
        obs = pd.DataFrame(
            {"group": ["A"] * 80 + ["B"] * 20},
            index=[f"c{i}" for i in range(100)],
        )
        var = pd.DataFrame(index=["g0", "g1", "g2"])
        d = SingleCellDataset(X, obs, var)
        sub = utils.subsample(d, n=20, stratify="group", random_state=0)
        counts = sub.obs["group"].value_counts()
        self.assertIn("A", counts)
        self.assertIn("B", counts)

    def test_filter_obs_callable(self):
        d = _make_data()
        sub = utils.filter_obs(d, lambda obs: obs["group"] == "A")
        self.assertTrue((sub.obs["group"] == "A").all())

    def test_filter_var_mask(self):
        d = _make_data()
        mask = np.array([True, False] * 10)
        sub = utils.filter_var(d, mask)
        self.assertEqual(sub.n_vars, 10)

    def test_rename_obs(self):
        d = _make_data()
        utils.rename_obs(d, {"group": "cell_type"})
        self.assertIn("cell_type", d.obs.columns)
        self.assertNotIn("group", d.obs.columns)

    def test_get_mean_var_axis1(self):
        d = _make_data(sparse=False)
        mean, var = utils.get_mean_var(d, axis=1)
        self.assertEqual(len(mean), 60)


class TestTrajectory(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(5)
        n = 80
        X = rng.random((n, 10))
        obs = pd.DataFrame(
            {"cluster": (["0"] * 20 + ["1"] * 20 + ["2"] * 20 + ["3"] * 20)},
            index=[f"c{i}" for i in range(n)],
        )
        var = pd.DataFrame(index=[f"g{i}" for i in range(10)])
        self.d = SingleCellDataset(X, obs, var)
        dimensionality.run_pca(self.d, n_components=8, use_highly_variable=False)
        dimensionality.neighbors(self.d, n_neighbors=8)

    def test_select_root_cell_extreme(self):
        root = trajectory.select_root_cell(
            self.d, cluster_key="cluster", root_cluster="0", strategy="extreme"
        )
        self.assertIsInstance(root, int)
        self.assertGreaterEqual(root, 0)
        self.assertLess(root, self.d.n_obs)

    def test_select_root_cell_medoid(self):
        root = trajectory.select_root_cell(
            self.d, cluster_key="cluster", root_cluster="0", strategy="medoid"
        )
        self.assertIsInstance(root, int)

    def test_diffusion_pseudotime(self):
        root = trajectory.select_root_cell(
            self.d, cluster_key="cluster", root_cluster="0"
        )
        trajectory.diffusion_pseudotime(self.d, root_cell=root)
        pt = self.d.obs["dpt_pseudotime"]
        self.assertFalse(pt.isna().any())
        self.assertAlmostEqual(float(pt.iloc[root]), 0.0, places=5)
        self.assertTrue((pt >= 0).all() and (pt <= 1).all())

    def test_gene_trends_shape(self):
        root = trajectory.select_root_cell(
            self.d, cluster_key="cluster", root_cluster="0"
        )
        trajectory.diffusion_pseudotime(self.d, root_cell=root)
        trends = trajectory.gene_trends(
            self.d, genes=["g0", "g1", "g2"], n_bins=10, use_raw=False
        )
        self.assertEqual(trends.shape, (10, 3))

    def test_branching_detection(self):
        root = trajectory.select_root_cell(
            self.d, cluster_key="cluster", root_cluster="0"
        )
        trajectory.diffusion_pseudotime(self.d, root_cell=root, n_branchings=2)
        self.assertIn("dpt_groups", self.d.obs.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
