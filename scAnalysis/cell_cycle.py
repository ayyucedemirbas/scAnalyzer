from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from .core import SingleCellDataset

S_GENES_HUMAN = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "MLF1IP",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "POLA1",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]

G2M_GENES_HUMAN = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
    "TMPO",
    "CENPF",
    "TACC3",
    "FAM64A",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "HN1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPA",
]

S_GENES_MOUSE = [g.capitalize() for g in S_GENES_HUMAN]
G2M_GENES_MOUSE = [g.capitalize() for g in G2M_GENES_HUMAN]


def score_genes(
    data: SingleCellDataset,
    gene_list: List[str],
    ctrl_size: int = 50,
    n_bins: int = 25,
    random_state: int = 0,
    use_raw: bool = False,
) -> np.ndarray:
    np.random.seed(random_state)

    # 1. Get expression matrix
    if use_raw and data.raw is not None:
        if hasattr(data.raw, "X"):
            X = data.raw.X
            var_names = data.raw.var.index
        else:
            X = data.raw
            var_names = data.var.index
    else:
        X = data.X
        var_names = data.var.index

    gene_list_found = [g for g in gene_list if g in var_names]

    if len(gene_list_found) == 0:
        return np.zeros(data.n_obs)

    gene_indices = [var_names.get_loc(g) for g in gene_list_found]

    if sp.issparse(X):
        gene_means = np.ravel(X.mean(axis=0))
    else:
        gene_means = X.mean(axis=0)

    if np.std(gene_means) < 1e-6:
        gene_bins = np.zeros(len(gene_means), dtype=int)
    else:
        try:
            gene_bins = pd.qcut(gene_means, q=n_bins, labels=False, duplicates="drop")
        except:
            gene_bins = pd.cut(gene_means, bins=n_bins, labels=False)

        if np.any(pd.isna(gene_bins)):
            gene_bins = np.nan_to_num(gene_bins, nan=0).astype(int)

    ctrl_genes = []
    all_indices = np.arange(len(var_names))
    non_target_indices = np.setdiff1d(all_indices, gene_indices)

    if len(non_target_indices) == 0:
        return np.zeros(data.n_obs)

    for gene_idx in gene_indices:
        if gene_idx >= len(gene_bins):
            continue

        gene_bin = gene_bins[gene_idx]

        candidates = non_target_indices[gene_bins[non_target_indices] == gene_bin]

        if len(candidates) == 0:
            candidates = non_target_indices[
                np.abs(gene_bins[non_target_indices] - gene_bin) <= 1
            ]

        if len(candidates) == 0:
            candidates = non_target_indices

        if len(candidates) > 0:
            n_select = min(ctrl_size, len(candidates))
            selected = np.random.choice(candidates, n_select, replace=False)
            ctrl_genes.extend(selected)

    if len(ctrl_genes) == 0:
        return np.zeros(data.n_obs)

    ctrl_genes = np.array(ctrl_genes, dtype=int)

    if sp.issparse(X):
        mean_target = X[:, gene_indices].mean(axis=1).A1
        mean_ctrl = X[:, ctrl_genes].mean(axis=1).A1
    else:
        mean_target = X[:, gene_indices].mean(axis=1)
        mean_ctrl = X[:, ctrl_genes].mean(axis=1)

    score = mean_target - mean_ctrl
    return score


def score_cell_cycle(
    data: SingleCellDataset,
    s_genes: Optional[List[str]] = None,
    g2m_genes: Optional[List[str]] = None,
    organism: str = "human",
    ctrl_size: int = 50,
    random_state: int = 0,
    use_raw: bool = False,
) -> SingleCellDataset:

    if s_genes is None:
        s_genes = S_GENES_HUMAN if organism == "human" else S_GENES_MOUSE
    if g2m_genes is None:
        g2m_genes = G2M_GENES_HUMAN if organism == "human" else G2M_GENES_MOUSE

    s_score = score_genes(
        data, s_genes, ctrl_size=ctrl_size, random_state=random_state, use_raw=use_raw
    )
    g2m_score = score_genes(
        data, g2m_genes, ctrl_size=ctrl_size, random_state=random_state, use_raw=use_raw
    )

    data.obs["S_score"] = s_score
    data.obs["G2M_score"] = g2m_score

    phases = np.array(["G1"] * data.n_obs)

    s_mask = (s_score > 0) & (s_score > g2m_score)
    phases[s_mask] = "S"

    g2m_mask = (g2m_score > 0) & (g2m_score >= s_score)
    phases[g2m_mask] = "G2M"

    data.obs["phase"] = pd.Categorical(phases, categories=["G1", "S", "G2M"])

    return data


def regress_out_cell_cycle(
    data: SingleCellDataset,
    s_genes: Optional[List[str]] = None,
    g2m_genes: Optional[List[str]] = None,
    organism: str = "human",
    difference_only: bool = True,
) -> SingleCellDataset:

    if "S_score" not in data.obs.columns:
        score_cell_cycle(data, s_genes, g2m_genes, organism)

    s_score = data.obs["S_score"].values
    g2m_score = data.obs["G2M_score"].values

    X = data.X
    if sp.issparse(X):
        X = X.toarray()

    for gene_idx in range(data.n_vars):
        y = X[:, gene_idx]

        if difference_only:
            diff = s_score - g2m_score
            if np.std(diff) > 1e-12:
                beta = np.cov(y, diff)[0, 1] / np.var(diff)
                X[:, gene_idx] = y - beta * diff
        else:
            if np.std(s_score) > 1e-12 and np.std(g2m_score) > 1e-12:
                from sklearn.linear_model import LinearRegression

                reg = LinearRegression()
                predictors = np.column_stack([s_score, g2m_score])
                reg.fit(predictors, y)
                X[:, gene_idx] = y - reg.predict(predictors) + reg.intercept_

    if sp.issparse(data.X):
        data.X = sp.csr_matrix(X)
    else:
        data.X = X

    return data
