from __future__ import annotations

import gzip
import os
import shutil
from typing import Dict, Optional, Union

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from .core import SingleCellDataset


def read_10x_mtx(
    path: str,
    var_names_col: int = 1,
) -> SingleCellDataset:
    path = path.rstrip("/")

    """
    1. matrix.mtx: The actual numbers (how many times a gene was seen in a cell).
    2. barcodes.tsv: The names/IDs of the cells (rows).
    3. features.tsv (or genes.tsv): The names of the genes (columns).

    https://kb.10xgenomics.com/s/article/115000794686-How-is-the-MEX-format-used-for-the-gene-barcode-matrices
    https://www.10xgenomics.com/support/software/cell-ranger-atac/latest/analysis/outputs/feature-barcode-matrices
    """

    def _resolve(base: str) -> str:
        for candidate in (base, base + ".gz"):
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Could not find {base} or {base}.gz")

    mtx_file = _resolve(os.path.join(path, "matrix.mtx"))
    barcodes_file = _resolve(os.path.join(path, "barcodes.tsv"))

    features_base = os.path.join(path, "features.tsv")
    if not os.path.exists(features_base) and not os.path.exists(features_base + ".gz"):
        features_base = os.path.join(path, "genes.tsv")
    features_file = _resolve(features_base)

    print(f"IO: data from '{path}' …")

    # in 10x Genomics data, cells are columns and genes are rows, we need to Transpose it
    # so cells are rows and genes are columns.

    X = sio.mmread(mtx_file).T.tocsr()

    obs = pd.read_csv(barcodes_file, header=None, sep="\t", names=["barcode"])
    obs.index = obs["barcode"]

    var = pd.read_csv(features_file, header=None, sep="\t")
    n_cols = var.shape[1]
    col_names = ["gene_ids", "gene_symbols", "feature_types"][:n_cols]
    var.columns = col_names

    use_col = min(var_names_col, n_cols - 1)
    var_names = _make_unique(var.iloc[:, use_col].values.astype(str))
    var.index = var_names

    print(f"IO: Loaded {X.shape[0]:,} cells × {X.shape[1]:,} genes.")
    return SingleCellDataset(X=X, obs=obs, var=var)


def read_csv(
    filename: str,
    delimiter: str = ",",
    first_column_names: bool = True,
) -> SingleCellDataset:
    print(f"IO: Reading CSV from '{filename}' …")
    df = pd.read_csv(
        filename,
        sep=delimiter,
        index_col=0 if first_column_names else None,
    )
    X = sp.csr_matrix(df.values)
    obs = pd.DataFrame(index=df.index)
    var = pd.DataFrame(index=df.columns)
    return SingleCellDataset(X=X, obs=obs, var=var)


def read_text(
    filename: str,
    delimiter: str = "\t",
    first_column_names: bool = True,
) -> SingleCellDataset:
    return read_csv(
        filename, delimiter=delimiter, first_column_names=first_column_names
    )


def write_csvs(data: SingleCellDataset, prefix: str = "output") -> None:
    data.obs.to_csv(f"{prefix}_obs.csv")
    data.var.to_csv(f"{prefix}_var.csv")

    X_df = pd.DataFrame(
        data.X.toarray() if sp.issparse(data.X) else data.X,
        index=data.obs.index,
        columns=data.var.index,
    )
    X_df.to_csv(f"{prefix}_X.csv")
    print(f"IO: Wrote CSVs → {prefix}_{{X,obs,var}}.csv")


def _write_dataframe_to_hdf5(group: h5py.Group, df: pd.DataFrame) -> None:
    dt_str = h5py.string_dtype()

    group.create_dataset(
        "_index",
        data=df.index.values.astype(str).astype(object),
        dtype=dt_str,
    )
    group.attrs["_index"] = "_index"
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["column-order"] = np.array(df.columns.tolist(), dtype="S")

    for col in df.columns:
        vals = df[col]
        if isinstance(vals.dtype, pd.CategoricalDtype):
            codes = vals.cat.codes.values
            cats = vals.cat.categories.astype(str).tolist()
            dset = group.create_dataset(col, data=codes)
            dset.attrs["categories"] = np.array(cats, dtype=object)
            dset.attrs["encoding-type"] = "categorical"
        elif vals.dtype == object or pd.api.types.is_string_dtype(vals):
            group.create_dataset(
                col, data=vals.astype(str).values.astype(object), dtype=dt_str
            )
        else:
            group.create_dataset(col, data=vals.values)


def _read_dataframe_from_hdf5(group: h5py.Group) -> pd.DataFrame:
    index_key = group.attrs.get("_index", "_index")

    index: Optional[np.ndarray] = None
    if index_key in group:
        raw = group[index_key][:]
        index = raw.astype(str) if raw.dtype.kind in ("S", "O", "U") else raw

    col_order_raw = group.attrs.get("column-order", np.array([], dtype=object))
    if hasattr(col_order_raw, "tolist"):
        col_order = [c.decode() if isinstance(c, bytes) else c for c in col_order_raw]
    else:
        col_order = []

    data_dict: Dict[str, np.ndarray] = {}
    for key in group.keys():
        if key == index_key:
            continue
            
        dset = group[key]
        
        if isinstance(dset, h5py.Group):
            if "categories" in dset and "codes" in dset:
                cats_raw = dset["categories"][:]
                codes = dset["codes"][:]
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in cats_raw]
                data_dict[key] = pd.Categorical.from_codes(codes, categories=cats)
            else:
                continue
        else:
            raw = dset[:]

            if "categories" in dset.attrs:
                cats_raw = dset.attrs["categories"]
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in cats_raw]
                data_dict[key] = pd.Categorical.from_codes(raw, categories=cats)
            elif raw.dtype.kind in ("S", "O"):
                data_dict[key] = raw.astype(str)
            else:
                data_dict[key] = raw

    df = pd.DataFrame(data_dict, index=index)

    existing_order = [c for c in col_order if c in df.columns]
    if existing_order:
        df = df[existing_order]

    return df


def write_h5ad(data: SingleCellDataset, filename: str) -> None:
    print(f"IO: Writing H5AD -> '{filename}' ...")

    with h5py.File(filename, "w") as f:
        if sp.issparse(data.X):
            xg = f.create_group("X")
            fmt = "csr_matrix" if sp.isspmatrix_csr(data.X) else "csc_matrix"
            xg.attrs["encoding-type"] = fmt
            xg.attrs["shape"] = data.X.shape
            xg.create_dataset("data", data=data.X.data)
            xg.create_dataset("indices", data=data.X.indices)
            xg.create_dataset("indptr", data=data.X.indptr)
        else:
            f.create_dataset("X", data=data.X)

        _write_dataframe_to_hdf5(f.create_group("obs"), data.obs)
        _write_dataframe_to_hdf5(f.create_group("var"), data.var)

        if data.obsm:
            og = f.create_group("obsm")
            for k, v in data.obsm.items():
                og.create_dataset(k, data=v)

        if data.varm:
            vg = f.create_group("varm")
            for k, v in data.varm.items():
                vg.create_dataset(k, data=v)

        if data.uns:
            ug = f.create_group("uns")
            for k, v in data.uns.items():
                try:
                    if isinstance(v, dict):
                        sg = ug.create_group(k)
                        for sk, sv in v.items():
                            if isinstance(sv, (np.ndarray, list)):
                                sg.create_dataset(sk, data=np.asarray(sv))
                            elif isinstance(sv, (int, float, str, bool)):
                                sg.attrs[sk] = sv
                    elif isinstance(v, (np.ndarray, list)):
                        ug.create_dataset(k, data=np.asarray(v))
                    elif isinstance(v, (int, float, str, bool)):
                        ug.attrs[k] = v
                except Exception as exc:
                    print(f"  Warning: could not write uns['{k}']: {exc}")

    print(f"IO: Wrote {data.n_obs:,} cells × {data.n_vars:,} genes.")


def read_h5ad(filename: str) -> SingleCellDataset:
    print(f"IO: Reading H5AD from '{filename}' ...")

    with h5py.File(filename, "r") as f:
        x_item = f["X"]
        if isinstance(x_item, h5py.Group):
            d = x_item["data"][:]
            indices = x_item["indices"][:]
            indptr = x_item["indptr"][:]
            shape = tuple(x_item.attrs["shape"])
            if x_item.attrs.get("encoding-type") == "csr_matrix":
                X: Union[sp.csr_matrix, np.ndarray] = sp.csr_matrix(
                    (d, indices, indptr), shape=shape
                )
            else:
                X = sp.csc_matrix((d, indices, indptr), shape=shape)
        else:
            X = x_item[:]

        obs = _read_dataframe_from_hdf5(f["obs"])
        var = _read_dataframe_from_hdf5(f["var"])

        obsm: Dict[str, np.ndarray] = {}
        if "obsm" in f:
            for k in f["obsm"]:
                obsm[k] = f["obsm"][k][:]

        varm: Dict[str, np.ndarray] = {}
        if "varm" in f:
            for k in f["varm"]:
                varm[k] = f["varm"][k][:]

        uns: Dict = {}
        if "uns" in f:
            uns_group = f["uns"]
            for k in uns_group.attrs:
                uns[k] = uns_group.attrs[k]
            for k in uns_group.keys():
                item = uns_group[k]
                if isinstance(item, h5py.Group):
                    d: Dict = {}
                    for sk in item.keys():
                        sub_item = item[sk]
                        d[sk] = sub_item[()] if sub_item.shape == () else sub_item[:]
                    for sk in item.attrs:
                        d[sk] = item.attrs[sk]
                    uns[k] = d
                else:
                    uns[k] = item[()] if item.shape == () else item[:]

    print(f"IO: Loaded {X.shape[0]:,} cells × {X.shape[1]:,} genes.")
    return SingleCellDataset(X, obs, var, uns=uns, obsm=obsm, varm=varm)


def _make_unique(names: np.ndarray) -> np.ndarray:
    names = np.asarray(names, dtype=str)
    new_names: list[str] = []
    seen: Dict[str, int] = {}
    for name in names:
        if name in seen:
            seen[name] += 1
            new_names.append(f"{name}-{seen[name]}")
        else:
            seen[name] = 0
            new_names.append(name)
    return np.array(new_names, dtype=str)
