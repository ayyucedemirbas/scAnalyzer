from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from core import SingleCellDataset


def select_root_cell(
    data: SingleCellDataset,
    cluster_key: str,
    root_cluster: Union[str, int],
    embedding_key: str = "X_pca",
    strategy: str = "extreme",
) -> int:
    if cluster_key not in data.obs.columns:
        raise ValueError(f"'{cluster_key}' not in obs.")

    mask = data.obs[cluster_key].astype(str) == str(root_cluster)
    if mask.sum() == 0:
        raise ValueError(
            f"No cells found in cluster '{root_cluster}'. "
            f"Available: {data.obs[cluster_key].unique().tolist()}"
        )

    if embedding_key not in data.obsm:
        raise ValueError(
            f"Embedding '{embedding_key}' not in obsm. "
            f"Run dimensionality.run_pca() first."
        )

    cluster_idx = np.where(mask)[0]
    X_clust = data.obsm[embedding_key][cluster_idx]

    if strategy == "extreme":
        overall_mean = data.obsm[embedding_key].mean(axis=0)
        dists = np.linalg.norm(X_clust - overall_mean, axis=1)
        local_idx = int(np.argmax(dists))
    elif strategy == "medoid":
        centroid = X_clust.mean(axis=0)
        dists = np.linalg.norm(X_clust - centroid, axis=1)
        local_idx = int(np.argmin(dists))
    else:
        raise ValueError("strategy must be 'extreme' or 'medoid'.")

    root = int(cluster_idx[local_idx])
    print(
        f"trajectory: root cell selected = index {root} "
        f"(obs '{data.obs.index[root]}', cluster='{root_cluster}', "
        f"strategy='{strategy}')."
    )
    return root


def diffusion_pseudotime(
    data: SingleCellDataset,
    root_cell: int,
    n_dcs: int = 10,
    key_added: str = "dpt_pseudotime",
    n_branchings: int = 0,
) -> SingleCellDataset:
    if "neighbors" not in data.uns:
        raise ValueError(
            "Run dimensionality.neighbors() before diffusion_pseudotime()."
        )

    if "X_diffmap" in data.obsm:
        print(f"DPT: using pre-computed diffusion map.")
        dc = data.obsm["X_diffmap"][:, :n_dcs]
        evals = data.uns.get("diffmap_evals", np.ones(n_dcs))[:n_dcs]
    else:
        print(f"DPT: computing diffusion map on the fly …")
        _quick_diffmap(data, n_components=n_dcs)
        dc = data.obsm["X_diffmap"][:, :n_dcs]
        evals = data.uns.get("diffmap_evals", np.ones(n_dcs))[:n_dcs]

    evals_safe = np.where(np.abs(evals) < 1e-10, 1e-10, evals)
    dc_scaled = dc / evals_safe[: dc.shape[1]]

    root_vec = dc_scaled[root_cell]
    sq_dist = np.sum((dc_scaled - root_vec) ** 2, axis=1)
    pseudotime = np.sqrt(np.maximum(sq_dist, 0.0))

    pt_min, pt_max = pseudotime.min(), pseudotime.max()
    if pt_max > pt_min:
        pseudotime = (pseudotime - pt_min) / (pt_max - pt_min)

    data.obs[key_added] = pseudotime
    print(
        f"DPT: pseudotime stored in obs['{key_added}']. "
        f"Root cell = {root_cell} (t=0)."
    )

    if n_branchings > 0:
        _detect_branches(data, pseudotime, dc_scaled, n_branchings)

    return data


def gene_trends(
    data: SingleCellDataset,
    genes: List[str],
    pseudotime_key: str = "dpt_pseudotime",
    n_bins: int = 50,
    use_raw: bool = True,
) -> pd.DataFrame:
    if pseudotime_key not in data.obs.columns:
        raise ValueError(
            f"'{pseudotime_key}' not in obs. " "Run diffusion_pseudotime() first."
        )

    pt = data.obs[pseudotime_key].values
    valid = ~np.isnan(pt)

    if use_raw and data.raw is not None:
        X = data.raw.X if hasattr(data.raw, "X") else data.raw
        vnames = data.raw.var.index if hasattr(data.raw, "var") else data.var.index
    else:
        X = data.X
        vnames = data.var.index

    missing = [g for g in genes if g not in vnames]
    if missing:
        print(f"gene_trends: {len(missing)} genes not found and skipped: {missing}")
    genes = [g for g in genes if g in vnames]
    if not genes:
        raise ValueError("None of the requested genes were found.")

    gene_idx = [vnames.get_loc(g) for g in genes]
    X_sub = X[valid, :][:, gene_idx]
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()

    pt_valid = pt[valid]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(pt_valid, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rows = []
    midpoints = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            rows.append(np.full(len(genes), np.nan))
        else:
            rows.append(X_sub[mask].mean(axis=0))
        midpoints.append((bins[b] + bins[b + 1]) / 2)

    df = pd.DataFrame(rows, index=midpoints, columns=genes)
    df.index.name = pseudotime_key
    return df


def _quick_diffmap(data: SingleCellDataset, n_components: int = 15) -> None:
    K = data.uns["neighbors"]["connectivities"].astype(float)
    row_sums = np.ravel(K.sum(axis=1))
    row_sums = np.maximum(row_sums, 1e-10)
    T = sp.diags(1.0 / row_sums) @ K

    d_sqrt = np.sqrt(row_sums)
    d_sqrt_inv = 1.0 / d_sqrt
    M = sp.diags(d_sqrt) @ T @ sp.diags(d_sqrt_inv)

    k = min(n_components + 1, M.shape[0] - 1)
    evals, evecs = eigsh(M.tocsr(), k=k, which="LM")
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    phi = sp.diags(d_sqrt_inv).dot(evecs)
    data.obsm["X_diffmap"] = phi[:, 1:].real
    data.uns["diffmap_evals"] = evals[1:].real


def _detect_branches(
    data: SingleCellDataset,
    pseudotime: np.ndarray,
    dc_scaled: np.ndarray,
    n_branchings: int,
) -> None:
    late_mask = pseudotime > np.median(pseudotime)
    late_idx = np.where(late_mask)[0]

    if len(late_idx) < n_branchings + 1:
        print("DPT branches: too few late cells for branching detection.")
        return

    tips = [int(late_idx[np.argmax(pseudotime[late_idx])])]
    for _ in range(n_branchings):
        dists_to_tips = np.min(
            np.stack(
                [np.sum((dc_scaled - dc_scaled[t]) ** 2, axis=1) for t in tips],
                axis=1,
            ),
            axis=1,
        )
        dists_to_tips[~late_mask] = -1
        tips.append(int(np.argmax(dists_to_tips)))

    tip_coords = dc_scaled[tips]
    dists = np.sum((dc_scaled[:, None, :] - tip_coords[None, :, :]) ** 2, axis=2)
    branch_labels = np.argmin(dists, axis=1).astype(str)
    data.obs["dpt_groups"] = pd.Categorical(branch_labels)

    print(
        f"DPT: detected {n_branchings} branching(s), " f"stored in obs['dpt_groups']."
    )
