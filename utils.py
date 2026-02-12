from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from core import SingleCellDataset


def merge(
    datasets: List[SingleCellDataset],
    batch_keys: Optional[List[str]] = None,
    batch_category: str = "batch",
    join: str = "inner",
) -> SingleCellDataset:
    """
    Merges multiple SingleCellDataset objects into one.

    Args:
        datasets: List of SingleCellDataset objects.
        batch_keys: List of strings to label each dataset (e.g., ['Control', 'Treated']).
        batch_category: Name of the column in obs to store the batch labels.
        join: 'inner' (intersection of genes) or 'outer' (union of genes).
    """
    if len(datasets) == 0:
        raise ValueError("No datasets provided for merging.")
    if len(datasets) == 1:
        return datasets[0]

    print(f"Utils: Merging {len(datasets)} datasets with join='{join}'...")

    # 1. Align Genes (Var)
    # Get all gene names
    var_names_list = [d.var.index for d in datasets]

    if join == "inner":
        common_vars = set(var_names_list[0])
        for v in var_names_list[1:]:
            common_vars &= set(v)
        final_vars = sorted(list(common_vars))
    elif join == "outer":
        all_vars = set(var_names_list[0])
        for v in var_names_list[1:]:
            all_vars |= set(v)
        final_vars = sorted(list(all_vars))
    else:
        raise ValueError("Join must be 'inner' or 'outer'.")

    print(f"Utils: Final gene count: {len(final_vars)}")

    # 2. Concatenate Data (X and Obs)
    X_list = []
    obs_list = []

    for i, data in enumerate(datasets):
        # Handle Batch Labeling
        current_obs = data.obs.copy()
        if batch_keys:
            batch_label = batch_keys[i]
        else:
            batch_label = str(i)
        current_obs[batch_category] = batch_label

        # Make indices unique to avoid collisions
        current_obs.index = f"{batch_label}_" + current_obs.index.astype(str)
        obs_list.append(current_obs)

        # Handle X (Reorder/Pad columns)
        # We need to map current genes to final_vars

        if join == "inner":
            # Subset columns
            # Get indices of final_vars in current data
            # Assuming unique indices for simplicity
            indexer = data.var.index.get_indexer(final_vars)
            # -1 indicates missing, but inner join implies all present
            # However, safety check:
            if np.any(indexer == -1):
                raise ValueError("Inner join failed: missing genes.")

            X_subset = data.X[:, indexer]
            X_list.append(X_subset)

        elif join == "outer":
            # Create empty matrix of shape (n_cells, n_final_vars)
            n_cells = data.n_obs
            n_genes = len(final_vars)

            # Map existing genes
            # This is slow if done naively.
            # Build a column map: {gene: final_col_idx}
            final_var_map = {name: j for j, name in enumerate(final_vars)}

            # Current genes indices in final matrix
            # current_col_indices = [final_var_map[g] for g in data.var.index if g in final_var_map]
            # But simpler: use pandas reindexing logic on DataFrame if dense,
            # or sparse matrix construction if sparse.

            # Constructing sparse matrix directly
            # 1. Get COO format of current X
            if sp.issparse(data.X):
                coo = data.X.tocoo()
            else:
                coo = sp.coo_matrix(data.X)

            # 2. Map old column indices to new column indices
            current_genes = data.var.index
            # Array where index i contains the new column index for old column i
            col_map = np.full(data.n_vars, -1, dtype=int)

            for old_idx, gene in enumerate(current_genes):
                if gene in final_var_map:
                    col_map[old_idx] = final_var_map[gene]

            # Filter out genes not in final (shouldn't happen in outer, but good practice)
            valid_mask = col_map[coo.col] != -1
            new_rows = coo.row[valid_mask]
            new_cols = col_map[coo.col[valid_mask]]
            new_data = coo.data[valid_mask]

            # Create new sparse matrix
            X_new = sp.coo_matrix(
                (new_data, (new_rows, new_cols)), shape=(n_cells, n_genes)
            ).tocsr()
            X_list.append(X_new)

    # 3. Stack
    if sp.issparse(X_list[0]):
        X_final = sp.vstack(X_list)
    else:
        X_final = np.vstack(X_list)

    obs_final = pd.concat(obs_list)
    var_final = pd.DataFrame(index=final_vars)

    return SingleCellDataset(X=X_final, obs=obs_final, var=var_final)


def subsample(
    data: SingleCellDataset,
    n: Optional[int] = None,
    fraction: Optional[float] = None,
    random_state: int = 0,
) -> SingleCellDataset:
    """
    Subsamples cells from the dataset.
    Provide either n (number of cells) or fraction.
    """
    n_obs = data.n_obs
    np.random.seed(random_state)

    if n is not None:
        if n > n_obs:
            print(
                f"Warning: Requested n={n} is larger than dataset size {n_obs}. Returning full dataset."
            )
            return data
        indices = np.random.choice(n_obs, n, replace=False)
    elif fraction is not None:
        if fraction > 1.0 or fraction < 0.0:
            raise ValueError("Fraction must be between 0 and 1.")
        n = int(n_obs * fraction)
        indices = np.random.choice(n_obs, n, replace=False)
    else:
        raise ValueError("Must provide either n or fraction.")

    indices.sort()
    print(f"Utils: Subsampled to {len(indices)} cells.")

    # Use slicing from core.py
    return data[indices, :]


def get_mean_var(data: SingleCellDataset, axis: int = 0):
    """
    Computes mean and variance efficiently for sparse or dense matrices.
    axis=0: per gene (across cells)
    axis=1: per cell (across genes)
    """
    X = data.X

    if sp.issparse(X):
        mean = np.ravel(X.mean(axis=axis))
        mean_sq = np.ravel(X.power(2).mean(axis=axis))
        var = mean_sq - mean**2
        return mean, var
    else:
        mean = np.mean(X, axis=axis)
        var = np.var(X, axis=axis)
        return mean, var


def describe_obs(data: SingleCellDataset, col: str):
    """
    Prints summary statistics for a column in obs.
    """
    if col not in data.obs:
        print(f"Column {col} not found.")
        return

    series = data.obs[col]
    if pd.api.types.is_numeric_dtype(series):
        print(series.describe())
    else:
        print(series.value_counts())
