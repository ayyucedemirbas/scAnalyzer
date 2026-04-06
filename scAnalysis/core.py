from __future__ import annotations

import copy
import warnings
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


class SingleCellDataset:
    def __init__(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Dict] = None,
        obsm: Optional[Dict[str, np.ndarray]] = None,
        varm: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if sp.issparse(X) and not sp.isspmatrix_csr(X):
            X = X.tocsr()
        self._X: Union[np.ndarray, sp.spmatrix] = X

        n_obs, n_vars = X.shape
        self._n_obs: int = n_obs
        self._n_vars: int = n_vars

        if obs is not None:
            if len(obs) != n_obs:
                raise ValueError(f"obs has {len(obs)} rows but X has {n_obs} rows.")
            self._obs = obs.copy()
        else:
            self._obs = pd.DataFrame(index=range(n_obs))

        if var is not None:
            if len(var) != n_vars:
                raise ValueError(f"var has {len(var)} rows but X has {n_vars} columns.")
            self._var = var.copy()
        else:
            self._var = pd.DataFrame(index=range(n_vars))

        self._uns: Dict = uns if uns is not None else {}
        self._obsm: Dict[str, np.ndarray] = obsm if obsm is not None else {}
        self._varm: Dict[str, np.ndarray] = varm if varm is not None else {}
        self._raw: Optional[Union[np.ndarray, sp.spmatrix, SingleCellDataset]] = None

    @property
    def X(self) -> Union[np.ndarray, sp.spmatrix]:
        return self._X

    @X.setter
    def X(self, value: Union[np.ndarray, sp.spmatrix]) -> None:
        if value.shape != (self._n_obs, self._n_vars):
            raise ValueError(
                f"Shape mismatch: expected {(self._n_obs, self._n_vars)}, "
                f"got {value.shape}."
            )
        if sp.issparse(value) and not sp.isspmatrix_csr(value):
            value = value.tocsr()
        self._X = value

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @obs.setter
    def obs(self, value: pd.DataFrame) -> None:
        if len(value) != self._n_obs:
            raise ValueError(f"obs must have {self._n_obs} rows, got {len(value)}.")
        self._obs = value

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @var.setter
    def var(self, value: pd.DataFrame) -> None:
        if len(value) != self._n_vars:
            raise ValueError(f"var must have {self._n_vars} rows, got {len(value)}.")
        self._var = value

    @property
    def uns(self) -> Dict:
        return self._uns

    @property
    def obsm(self) -> Dict[str, np.ndarray]:
        return self._obsm

    @property
    def varm(self) -> Dict[str, np.ndarray]:
        return self._varm

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, value) -> None:
        self._raw = value

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_vars(self) -> int:
        return self._n_vars

    @property
    def shape(self) -> Tuple[int, int]:
        return self._n_obs, self._n_vars

    def __len__(self) -> int:
        """Number of observations (cells)."""
        return self._n_obs

    def __contains__(self, key: str) -> bool:
        """
        ``key in data`` is True when *key* is a gene name, obs column, or
        obsm/varm/uns key.
        """
        return (
            key in self._var.index
            or key in self._obs.columns
            or key in self._obsm
            or key in self._varm
            or key in self._uns
        )

    def __iter__(self) -> Iterator[str]:
        """Iterate over cell names (obs index)."""
        return iter(self._obs.index)

    def __getitem__(
        self,
        index: Union[int, slice, np.ndarray, pd.Series, Tuple],
    ) -> "SingleCellDataset":
        if isinstance(index, tuple):
            obs_idx, var_idx = index
        else:
            obs_idx = index
            var_idx = slice(None)

        if isinstance(obs_idx, (pd.Series, pd.Index)):
            obs_idx = obs_idx.values
        if isinstance(var_idx, (pd.Series, pd.Index)):
            var_idx = var_idx.values

        new_X = (
            self._X[obs_idx, :][:, var_idx]
            if not isinstance(var_idx, slice)
            else self._X[obs_idx, var_idx]
        )

        if hasattr(new_X, "ndim") and new_X.ndim == 1:
            if isinstance(obs_idx, (int, np.integer)):
                new_X = new_X.reshape(1, -1)
            elif isinstance(var_idx, (int, np.integer)):
                new_X = new_X.reshape(-1, 1)

        if isinstance(obs_idx, (int, np.integer)):
            new_obs = self._obs.iloc[obs_idx : obs_idx + 1].copy()
        else:
            new_obs = self._obs.iloc[obs_idx].copy()

        if isinstance(var_idx, slice) and var_idx == slice(None):
            new_var = self._var.copy()
        elif isinstance(var_idx, (int, np.integer)):
            new_var = self._var.iloc[var_idx : var_idx + 1].copy()
        else:
            new_var = self._var.iloc[var_idx].copy()

        new_obsm: Dict[str, np.ndarray] = {}
        for key, mat in self._obsm.items():
            if isinstance(obs_idx, (int, np.integer)):
                new_obsm[key] = mat[obs_idx : obs_idx + 1]
            else:
                new_obsm[key] = mat[obs_idx]

        new_varm: Dict[str, np.ndarray] = {}
        for key, mat in self._varm.items():
            if isinstance(var_idx, slice) and var_idx == slice(None):
                new_varm[key] = mat.copy()
            elif isinstance(var_idx, (int, np.integer)):
                new_varm[key] = mat[var_idx : var_idx + 1]
            else:
                new_varm[key] = mat[var_idx]

        return SingleCellDataset(
            X=new_X,
            obs=new_obs,
            var=new_var,
            uns=self._uns.copy(),
            obsm=new_obsm,
            varm=new_varm,
        )

    def copy(self) -> "SingleCellDataset":
        """Return a full independent copy."""
        new = SingleCellDataset(
            X=self._X.copy(),
            obs=self._obs.copy(),
            var=self._var.copy(),
            uns=copy.deepcopy(self._uns),
            obsm=copy.deepcopy(self._obsm),
            varm=copy.deepcopy(self._varm),
        )
        if self._raw is not None:
            new.raw = (
                self._raw.copy()
                if hasattr(self._raw, "copy")
                else copy.deepcopy(self._raw)
            )
        return new

    def obs_names(self) -> pd.Index:
        """Cell names / barcodes."""
        return self._obs.index

    def var_names(self) -> pd.Index:
        """Gene names."""
        return self._var.index

    def to_df(self, layer: Optional[str] = None) -> pd.DataFrame:
        if layer == "raw" and self._raw is not None:
            mat = self._raw.X if hasattr(self._raw, "X") else self._raw
        elif layer is not None and layer in self._obsm:
            mat = self._obsm[layer]
            return pd.DataFrame(mat, index=self._obs.index)
        else:
            mat = self._X

        if sp.issparse(mat):
            mat = mat.toarray()
        return pd.DataFrame(mat, index=self._obs.index, columns=self._var.index)

    def summary(self) -> str:
        lines = [
            "─" * 56,
            f"SingleCellDataset  {self._n_obs:,} cells × {self._n_vars:,} genes",
            "─" * 56,
        ]

        # Matrix type & memory
        if sp.issparse(self._X):
            nnz = self._X.nnz
            density = 100 * nnz / (self._n_obs * self._n_vars)
            mem_mb = self._X.data.nbytes / 1024**2
            lines.append(
                f"  X : sparse CSR  nnz={nnz:,}  density={density:.2f}%  {mem_mb:.2f} MB"
            )
        else:
            mem_mb = self._X.nbytes / 1024**2
            lines.append(f"  X : dense ndarray  dtype={self._X.dtype}  {mem_mb:.2f} MB")

        # obs / var columns
        if not self._obs.empty and len(self._obs.columns):
            lines.append(f"  obs: {', '.join(self._obs.columns.tolist())}")
        if not self._var.empty and len(self._var.columns):
            lines.append(f"  var: {', '.join(self._var.columns.tolist())}")

        # Embeddings & graphs
        if self._obsm:
            details = ", ".join(f"{k} {v.shape}" for k, v in self._obsm.items())
            lines.append(f"  obsm: {details}")
        if self._varm:
            details = ", ".join(f"{k} {v.shape}" for k, v in self._varm.items())
            lines.append(f"  varm: {details}")
        if self._uns:
            lines.append(f"  uns : {', '.join(self._uns.keys())}")

        # Raw
        if self._raw is not None:
            lines.append("  raw : stored")

        lines.append("─" * 56)
        s = "\n".join(lines)
        print(s)
        return s

    def __repr__(self) -> str:
        parts = [
            f"SingleCellDataset object with n_obs × n_vars = "
            f"{self._n_obs} × {self._n_vars}"
        ]
        if not self._obs.empty and len(self._obs.columns):
            parts.append(f"    obs: {', '.join(self._obs.columns.tolist())}")
        if not self._var.empty and len(self._var.columns):
            parts.append(f"    var: {', '.join(self._var.columns.tolist())}")
        if self._uns:
            parts.append(f"    uns: {', '.join(self._uns.keys())}")
        if self._obsm:
            parts.append(f"    obsm: {', '.join(self._obsm.keys())}")
        if self._varm:
            parts.append(f"    varm: {', '.join(self._varm.keys())}")
        if sp.issparse(self._X):
            mb = self._X.data.nbytes / 1024**2
        else:
            mb = self._X.nbytes / 1024**2
        parts.append(f"    Memory (X): {mb:.2f} MB")
        return "\n".join(parts)
