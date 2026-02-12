import gzip
import os
import shutil
from typing import Dict, Union

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from core import SingleCellDataset

# --- 10x Genomics (Matrix Market) ---


def read_10x_mtx(
    path: str, var_names_col: int = 1, cache: bool = False
) -> SingleCellDataset:
    """
    Reads 10x Genomics output (matrix.mtx, barcodes.tsv, features.tsv).

    Args:
        path: Directory containing the .mtx and .tsv files.
        var_names_col: Column index in features.tsv to use as gene names (0=ID, 1=Symbol).
    """
    path = path.rstrip("/")

    # define filenames
    mtx_file = os.path.join(path, "matrix.mtx")
    if not os.path.exists(mtx_file):
        mtx_file += ".gz"

    barcodes_file = os.path.join(path, "barcodes.tsv")
    if not os.path.exists(barcodes_file):
        barcodes_file += ".gz"

    features_file = os.path.join(path, "features.tsv")
    if not os.path.exists(features_file):
        # Fallback for older 10x versions
        features_file = os.path.join(path, "genes.tsv")
        if not os.path.exists(features_file):
            features_file += ".gz"

    print(f"IO: Reading 10x data from {path}...")

    # 1. Read Matrix
    # 10x matrix is Genes x Cells. We need Cells x Genes.
    X = sio.mmread(mtx_file).T.tocsr()

    # 2. Read Barcodes (obs)
    obs = pd.read_csv(barcodes_file, header=None, sep="\t")
    obs.columns = ["barcode"]
    obs.index = obs["barcode"]

    # 3. Read Features (var)
    var = pd.read_csv(features_file, header=None, sep="\t")
    # Standard 10x: Col 0 = ID, Col 1 = Symbol, Col 2 = Type
    if var.shape[1] > var_names_col:
        var_names = var.iloc[:, var_names_col].values
        # Ensure unique names
        var_names = _make_unique(var_names)
        var.index = var_names
        var.columns = ["gene_ids", "gene_symbols", "feature_types"][: var.shape[1]]
    else:
        var.index = var.iloc[:, 0].values

    return SingleCellDataset(X=X, obs=obs, var=var)


# --- CSV Format ---


def read_csv(
    filename: str, delimiter: str = ",", first_column_names: bool = True
) -> SingleCellDataset:
    """
    Reads a dense matrix from CSV.
    Assumes rows = cells, columns = genes.
    """
    print(f"IO: Reading CSV from {filename}...")
    df = pd.read_csv(
        filename, sep=delimiter, index_col=0 if first_column_names else None
    )

    X = sp.csr_matrix(df.values)
    obs = pd.DataFrame(index=df.index)
    var = pd.DataFrame(index=df.columns)

    return SingleCellDataset(X=X, obs=obs, var=var)


def write_csvs(data: SingleCellDataset, prefix: str = "output"):
    """
    Writes data to multiple CSVs: {prefix}_X.csv, {prefix}_obs.csv, {prefix}_var.csv.
    """
    # Write Obs
    data.obs.to_csv(f"{prefix}_obs.csv")

    # Write Var
    data.var.to_csv(f"{prefix}_var.csv")

    # Write X
    if sp.issparse(data.X):
        # Warning: dense write might be huge
        X_df = pd.DataFrame(
            data.X.toarray(), index=data.obs.index, columns=data.var.index
        )
    else:
        X_df = pd.DataFrame(data.X, index=data.obs.index, columns=data.var.index)

    X_df.to_csv(f"{prefix}_X.csv")
    print(f"IO: Wrote CSVs to {prefix}_*.csv")


# --- H5AD (HDF5) Format ---
# Implementing a subset of the AnnData H5AD spec manually using h5py.


def _write_dataframe_to_hdf5(group, df):
    """Writes a pandas DataFrame to an HDF5 group."""
    # Write Index
    # Handle string index
    if df.index.dtype == "object" or df.index.dtype == "U":
        dt = h5py.special_dtype(vlen=str)
        group.create_dataset("_index", data=df.index.values.astype(object), dtype=dt)
    else:
        group.create_dataset("_index", data=df.index.values)

    # Write Columns
    for col in df.columns:
        vals = df[col].values

        # String columns
        if vals.dtype == "object" or vals.dtype.type is np.str_:
            # Convert to fixed-length or variable length strings for HDF5
            try:
                # Try simple conversion
                vals = vals.astype("S")
            except:
                # Fallback to variable length
                dt = h5py.special_dtype(vlen=str)
                group.create_dataset(col, data=vals, dtype=dt)
                continue

        # Categorical columns (convert to codes + categories)
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            cat = df[col].values
            codes = cat.codes
            categories = cat.categories.values.astype("S")

            dset = group.create_dataset(col, data=codes)
            dset.attrs["categories"] = categories
            dset.attrs["encoding-type"] = "categorical"
        else:
            # Numeric columns
            group.create_dataset(col, data=vals)

    group.attrs["_index"] = "_index"
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["column-order"] = df.columns.values.astype("S")


def _read_dataframe_from_hdf5(group) -> pd.DataFrame:
    """Reads a pandas DataFrame from an HDF5 group."""
    index_key = group.attrs.get("_index", "_index")

    # Read Index
    if index_key in group:
        index = group[index_key][:]
        if isinstance(index[0], bytes):
            index = index.astype(str)
    else:
        index = None

    data_dict = {}

    # Read Columns
    for key in group.keys():
        if key == index_key:
            continue

        dset = group[key]
        vals = dset[:]

        # Handle Categorical
        if "categories" in dset.attrs:
            categories = dset.attrs["categories"]
            if isinstance(categories[0], bytes):
                categories = categories.astype(str)
            vals = pd.Categorical.from_codes(vals, categories=categories)

        # Handle Bytes to String
        elif isinstance(vals.flat[0], bytes):
            vals = vals.astype(str)

        data_dict[key] = vals

    df = pd.DataFrame(data_dict, index=index)

    # Reorder columns if order is saved
    if "column-order" in group.attrs:
        order = group.attrs["column-order"]
        if isinstance(order[0], bytes):
            order = order.astype(str)
        # Only keep columns that actually exist (safety)
        existing_order = [c for c in order if c in df.columns]
        df = df[existing_order]

    return df


def write_h5ad(data: SingleCellDataset, filename: str):
    """
    Writes SingleCellDataset to .h5ad format compatible with AnnData.
    """
    print(f"IO: Writing H5AD to {filename}...")

    with h5py.File(filename, "w") as f:
        # 1. Write X (Sparse or Dense)
        if sp.issparse(data.X):
            # Write CSR
            X_group = f.create_group("X")
            fmt = "csr_matrix" if sp.isspmatrix_csr(data.X) else "csc_matrix"
            X_group.attrs["encoding-type"] = fmt
            X_group.attrs["shape"] = data.X.shape

            X_group.create_dataset("data", data=data.X.data)
            X_group.create_dataset("indices", data=data.X.indices)
            X_group.create_dataset("indptr", data=data.X.indptr)
        else:
            # Write Dense
            f.create_dataset("X", data=data.X)

        # 2. Write obs
        obs_group = f.create_group("obs")
        _write_dataframe_to_hdf5(obs_group, data.obs)

        # 3. Write var
        var_group = f.create_group("var")
        _write_dataframe_to_hdf5(var_group, data.var)

        # 4. Write obsm (Dictionaries of arrays)
        if data.obsm:
            obsm_group = f.create_group("obsm")
            for key, val in data.obsm.items():
                obsm_group.create_dataset(key, data=val)

        # 5. Write varm
        if data.varm:
            varm_group = f.create_group("varm")
            for key, val in data.varm.items():
                varm_group.create_dataset(key, data=val)

        # 6. Write uns (Simplified: only dicts and arrays)
        # Writing complex nested dicts to HDF5 is hard.
        # We perform a shallow write of the first level.
        if data.uns:
            uns_group = f.create_group("uns")
            for key, val in data.uns.items():
                try:
                    if isinstance(val, dict):
                        # Create subgroup
                        sub = uns_group.create_group(key)
                        for k, v in val.items():
                            if isinstance(v, (np.ndarray, list)):
                                sub.create_dataset(k, data=v)
                            # Handle simple values
                            elif isinstance(v, (int, float, str)):
                                sub.attrs[k] = v
                    elif isinstance(val, (np.ndarray, list)):
                        uns_group.create_dataset(key, data=val)
                except Exception as e:
                    print(f"Warning: Could not write uns['{key}']: {e}")


def read_h5ad(filename: str) -> SingleCellDataset:
    """
    Reads .h5ad file.
    """
    print(f"IO: Reading H5AD from {filename}...")

    with h5py.File(filename, "r") as f:
        # 1. Read X
        if isinstance(f["X"], h5py.Group):
            # Sparse
            g = f["X"]
            data = g["data"][:]
            indices = g["indices"][:]
            indptr = g["indptr"][:]
            shape = tuple(g.attrs["shape"])

            if g.attrs["encoding-type"] == "csr_matrix":
                X = sp.csr_matrix((data, indices, indptr), shape=shape)
            else:
                X = sp.csc_matrix((data, indices, indptr), shape=shape)
        else:
            # Dense
            X = f["X"][:]

        # 2. Read Obs/Var
        obs = _read_dataframe_from_hdf5(f["obs"])
        var = _read_dataframe_from_hdf5(f["var"])

        # 3. Read obsm
        obsm = {}
        if "obsm" in f:
            for key in f["obsm"].keys():
                obsm[key] = f["obsm"][key][:]

        # 4. Read varm
        varm = {}
        if "varm" in f:
            for key in f["varm"].keys():
                varm[key] = f["varm"][key][:]

        # 5. Read uns (Simplified)
        uns = {}
        if "uns" in f:
            for key in f["uns"].keys():
                item = f["uns"][key]
                if isinstance(item, h5py.Group):
                    # Reconstruct simple dict
                    d = {}
                    # Read datasets
                    for k in item.keys():
                        d[k] = item[k][:]
                    # Read attrs
                    for k in item.attrs.keys():
                        d[k] = item.attrs[k]
                    uns[key] = d
                # Note: Reading scalar datasets or top-level attrs for uns is omitted for brevity

    return SingleCellDataset(X, obs, var, uns=uns, obsm=obsm, varm=varm)


def _make_unique(names: np.ndarray) -> np.ndarray:
    """Makes a list of strings unique by appending -1, -2, etc."""
    new_names = []
    seen = {}
    for name in names:
        if name in seen:
            seen[name] += 1
            new_names.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            new_names.append(name)
    return np.array(new_names)
