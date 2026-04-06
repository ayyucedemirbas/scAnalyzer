from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .core import SingleCellDataset


def merge(
    datasets: List[SingleCellDataset],
    batch_keys: Optional[List[str]] = None,
    batch_category: str = "batch",
    join: str = "inner",
) -> SingleCellDataset:
    if not datasets:
        raise ValueError("datasets must not be empty.")
    if len(datasets) == 1:
        return datasets[0]

    if batch_keys is None:
        batch_keys = [str(i) for i in range(len(datasets))]
    if len(batch_keys) != len(datasets):
        raise ValueError(
            f"batch_keys has {len(batch_keys)} entries but "
            f"{len(datasets)} datasets were provided."
        )

    print(f"merge: {len(datasets)} datasets, join='{join}' …")

    var_indices = [d.var.index for d in datasets]
    if join == "inner":
        final_genes = var_indices[0]
        for vi in var_indices[1:]:
            final_genes = final_genes.intersection(vi)
        final_genes = final_genes.sort_values()
    elif join == "outer":
        final_genes = var_indices[0]
        for vi in var_indices[1:]:
            final_genes = final_genes.union(vi)
        final_genes = final_genes.sort_values()
    else:
        raise ValueError(f"join must be 'inner' or 'outer', got '{join}'.")

    print(f"merge: final gene set size = {len(final_genes):,}")

    X_parts: list = []
    obs_parts: list = []

    for batch_label, d in zip(batch_keys, datasets):
        obs_copy = d.obs.copy()
        obs_copy[batch_category] = batch_label
        obs_copy.index = f"{batch_label}_" + obs_copy.index.astype(str)
        obs_parts.append(obs_copy)

        if sp.issparse(d.X):
            # col_map[j] = position of d.var.index[j] in final_genes (-1 = absent)
            col_map = final_genes.get_indexer(d.var.index)
            coo = d.X.tocoo()
            new_col_idx = col_map[coo.col]
            valid = new_col_idx >= 0
            X_new = sp.coo_matrix(
                (coo.data[valid], (coo.row[valid], new_col_idx[valid])),
                shape=(d.n_obs, len(final_genes)),
            ).tocsr()
            X_parts.append(X_new)
        else:
            col_map = final_genes.get_indexer(d.var.index)
            X_new = np.zeros((d.n_obs, len(final_genes)), dtype=float)
            for src_col, dst_col in enumerate(col_map):
                if dst_col >= 0:
                    X_new[:, dst_col] = d.X[:, src_col]
            X_parts.append(X_new)

    is_sparse = sp.issparse(X_parts[0])
    if is_sparse:
        X_final = sp.vstack(X_parts, format="csr")
    else:
        X_final = np.vstack(X_parts)

    obs_final = pd.concat(obs_parts)
    var_final = pd.DataFrame(index=final_genes)

    print(
        f"merge: result = {X_final.shape[0]:,} cells × " f"{X_final.shape[1]:,} genes."
    )
    return SingleCellDataset(X=X_final, obs=obs_final, var=var_final)


def concat(
    datasets: List[SingleCellDataset],
    batch_keys: Optional[List[str]] = None,
    batch_category: str = "batch",
) -> SingleCellDataset:
    return merge(
        datasets, batch_keys=batch_keys, batch_category=batch_category, join="inner"
    )


def subsample(
    data: SingleCellDataset,
    n: Optional[int] = None,
    fraction: Optional[float] = None,
    random_state: int = 0,
    stratify: Optional[str] = None,
) -> SingleCellDataset:
    rng = np.random.default_rng(random_state)
    n_obs = data.n_obs

    if n is None and fraction is None:
        raise ValueError("Provide either n or fraction.")
    if fraction is not None and not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")

    if fraction is not None:
        n = max(1, int(n_obs * fraction))

    if n >= n_obs:
        print(f"subsample: n={n} ≥ n_obs={n_obs}; returning full dataset.")
        return data

    if stratify is not None:
        if stratify not in data.obs.columns:
            raise ValueError(f"stratify column '{stratify}' not in obs.")
        groups = data.obs[stratify].values
        unique_groups = np.unique(groups)
        indices_list = []
        for g in unique_groups:
            g_idx = np.where(groups == g)[0]
            g_n = max(1, int(round(n * len(g_idx) / n_obs)))
            g_n = min(g_n, len(g_idx))
            indices_list.append(rng.choice(g_idx, g_n, replace=False))
        indices = np.concatenate(indices_list)
    else:
        indices = rng.choice(n_obs, n, replace=False)

    indices = np.sort(indices)
    print(f"subsample: kept {len(indices):,} / {n_obs:,} cells.")
    return data[indices, :]


def get_mean_var(
    data: SingleCellDataset,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    X = data.X
    if sp.issparse(X):
        mean = np.ravel(X.mean(axis=axis))
        mean_sq = np.ravel(X.power(2).mean(axis=axis))
        var = np.maximum(mean_sq - mean**2, 0.0)
    else:
        mean = np.mean(X, axis=axis)
        var = np.var(X, axis=axis)
    return mean, var


def filter_obs(
    data: SingleCellDataset,
    mask: Union[np.ndarray, Callable[[pd.DataFrame], np.ndarray]],
) -> SingleCellDataset:
    if callable(mask):
        mask = mask(data.obs)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (data.n_obs,):
        raise ValueError(f"mask length {len(mask)} does not match n_obs={data.n_obs}.")
    print(f"filter_obs: keeping {int(mask.sum()):,} / {data.n_obs:,} cells.")
    return data[mask, :]


def filter_var(
    data: SingleCellDataset,
    mask: Union[np.ndarray, Callable[[pd.DataFrame], np.ndarray]],
) -> SingleCellDataset:
    if callable(mask):
        mask = mask(data.var)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (data.n_vars,):
        raise ValueError(
            f"mask length {len(mask)} does not match n_vars={data.n_vars}."
        )
    print(f"filter_var: keeping {int(mask.sum()):,} / {data.n_vars:,} genes.")
    return data[:, mask]


def rename_obs(
    data: SingleCellDataset,
    mapping: dict,
    inplace: bool = True,
) -> Optional[SingleCellDataset]:
    if not inplace:
        data = data.copy()
    data.obs.rename(columns=mapping, inplace=True)
    return None if inplace else data


def rename_var(
    data: SingleCellDataset,
    mapping: dict,
    inplace: bool = True,
) -> Optional[SingleCellDataset]:
    if not inplace:
        data = data.copy()
    data.var.index = data.var.index.map(lambda x: mapping.get(x, x))
    return None if inplace else data


def describe_obs(data: SingleCellDataset, col: str) -> None:
    if col not in data.obs:
        raise ValueError(f"'{col}' not in obs. Available: {list(data.obs.columns)}")
    s = data.obs[col]
    if pd.api.types.is_numeric_dtype(s):
        print(s.describe().to_string())
    else:
        print(s.value_counts().to_string())
