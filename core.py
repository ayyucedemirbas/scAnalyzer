from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


class SingleCellDataset:
    """
    A comprehensive container for single-cell expression data.

    Structure mimics AnnData:
        X (n_obs x n_vars): The main expression matrix.
        obs (n_obs x k): Dataframe for cell metadata (e.g., barcodes, batches).
        var (n_vars x j): Dataframe for gene metadata (e.g., gene_ids, symbols).
        uns (dict): Unstructured data (e.g., global parameters, logs).
        obsm (dict): Multi-dimensional observation annotation (e.g., PCA, UMAP coordinates).
        varm (dict): Multi-dimensional variable annotation (e.g., PC loadings).
    """

    def __init__(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Dict] = None,
        obsm: Optional[Dict[str, np.ndarray]] = None,
        varm: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize the SingleCellDataset.

        Args:
            X: The expression matrix (Cells x Genes). Can be dense or sparse.
            obs: Pandas DataFrame containing cell annotations. Index should match X rows.
            var: Pandas DataFrame containing gene annotations. Index should match X columns.
            uns: Dictionary for unstructured data.
            obsm: Dictionary for storing cell embeddings (e.g., 'X_pca').
            varm: Dictionary for storing gene embeddings (e.g., 'PCs').
        """
        # Ensure X is in a consistent format (CSR is usually best for arithmetic)
        if sp.issparse(X) and not sp.isspmatrix_csr(X):
            self._X = X.tocsr()
        else:
            self._X = X

        self._n_obs, self._n_vars = self._X.shape

        # Initialize obs (Cell metadata)
        if obs is not None:
            if len(obs) != self._n_obs:
                raise ValueError(
                    f"Shape mismatch: obs has {len(obs)} rows, X has {self._n_obs} rows."
                )
            self._obs = obs.copy()
        else:
            self._obs = pd.DataFrame(index=pd.RangeIndex(self._n_obs).astype(str))

        # Initialize var (Gene metadata)
        if var is not None:
            if len(var) != self._n_vars:
                raise ValueError(
                    f"Shape mismatch: var has {len(var)} rows, X has {self._n_vars} columns."
                )
            self._var = var.copy()
        else:
            self._var = pd.DataFrame(index=pd.RangeIndex(self._n_vars).astype(str))

        # Initialize containers
        self._uns = uns if uns is not None else {}
        self._obsm = obsm if obsm is not None else {}
        self._varm = varm if varm is not None else {}

        # Raw storage for counts if normalization overwrites X
        self._raw = None

    @property
    def X(self):
        """The expression matrix."""
        return self._X

    @X.setter
    def X(self, value):
        if value.shape != (self._n_obs, self._n_vars):
            raise ValueError(
                f"Shape mismatch. Expected {(self._n_obs, self._n_vars)}, got {value.shape}"
            )
        self._X = value

    @property
    def obs(self):
        """Cell annotations (observations)."""
        return self._obs

    @obs.setter
    def obs(self, value):
        if len(value) != self._n_obs:
            raise ValueError("obs DataFrame length must match number of cells.")
        self._obs = value

    @property
    def var(self):
        """Gene annotations (variables)."""
        return self._var

    @var.setter
    def var(self, value):
        if len(value) != self._n_vars:
            raise ValueError("var DataFrame length must match number of genes.")
        self._var = value

    @property
    def uns(self):
        """Unstructured data."""
        return self._uns

    @property
    def obsm(self):
        """Observation matrices (e.g., PCA, UMAP embeddings)."""
        return self._obsm

    @property
    def varm(self):
        """Variable matrices (e.g., loadings)."""
        return self._varm

    @property
    def raw(self):
        """Raw data storage."""
        return self._raw

    @raw.setter
    def raw(self, value):
        self._raw = value

    @property
    def n_obs(self):
        return self._n_obs

    @property
    def n_vars(self):
        return self._n_vars

    @property
    def shape(self):
        return self._n_obs, self._n_vars

    def __repr__(self):
        """String representation of the dataset."""
        descr = f"SingleCellDataset object with n_obs × n_vars = {self.n_obs} × {self.n_vars}"

        if not self.obs.empty:
            descr += f"\n    obs: {', '.join(self.obs.columns)}"
        if not self.var.empty:
            descr += f"\n    var: {', '.join(self.var.columns)}"
        if self.uns:
            descr += f"\n    uns: {', '.join(self.uns.keys())}"
        if self.obsm:
            descr += f"\n    obsm: {', '.join(self.obsm.keys())}"
        if self.varm:
            descr += f"\n    varm: {', '.join(self.varm.keys())}"

        memory_usage = self._X.data.nbytes if sp.issparse(self._X) else self._X.nbytes
        descr += f"\n    Memory (X): {memory_usage / 1024**2:.2f} MB"

        return descr

    def copy(self):
        """Performs a deep copy of the dataset."""
        import copy

        new_X = self._X.copy()
        new_obs = self._obs.copy()
        new_var = self._var.copy()
        new_uns = copy.deepcopy(self._uns)
        new_obsm = copy.deepcopy(self._obsm)
        new_varm = copy.deepcopy(self._varm)

        new_scd = SingleCellDataset(
            new_X, new_obs, new_var, new_uns, new_obsm, new_varm
        )
        if self._raw is not None:
            new_scd.raw = self._raw.copy()

        return new_scd

    def __getitem__(self, index):
        """
        Slicing support.
        Usage: adata[cell_indices, gene_indices]
        """
        if isinstance(index, tuple):
            # Handle cases like [1:10, :] or [:, 5:20]
            obs_idx, var_idx = index
        else:
            # Handle cases like [1:10] (slices only rows/cells)
            obs_idx = index
            var_idx = slice(None)

        # Handle Pandas objects (convert to numpy arrays for safety)
        if isinstance(obs_idx, (pd.Series, pd.Index)):
            obs_idx = obs_idx.values
        if isinstance(var_idx, (pd.Series, pd.Index)):
            var_idx = var_idx.values

        # Slice X
        new_X = self._X[obs_idx, var_idx]

        # --- FIX 1: Ensure X stays 2D (Numpy drops dims on integer slices) ---
        if hasattr(new_X, "ndim") and new_X.ndim == 1:
            if isinstance(obs_idx, (int, np.integer)):
                # Sliced a single row -> reshape to (1, n_vars)
                new_X = new_X.reshape(1, -1)
            elif isinstance(var_idx, (int, np.integer)):
                # Sliced a single col -> reshape to (n_obs, 1)
                new_X = new_X.reshape(-1, 1)
        # --------------------------------------------------------------------

        # --- FIX 2: Ensure obs/var stay DataFrames (iloc[int] returns Series) ---
        if isinstance(obs_idx, (int, np.integer)):
            new_obs = self._obs.iloc[obs_idx : obs_idx + 1].copy()
        else:
            new_obs = self._obs.iloc[obs_idx].copy()

        if isinstance(var_idx, (int, np.integer)):
            new_var = self._var.iloc[var_idx : var_idx + 1].copy()
        elif isinstance(var_idx, slice) and var_idx == slice(None):
            new_var = self._var.copy()
        else:
            new_var = self._var.iloc[var_idx].copy()
        # ----------------------------------------------------------------------

        # Create new object
        # Note: obsm slicing requires careful handling of indices
        new_obsm = {}
        for key, mat in self.obsm.items():
            # Apply obs_idx to the observation matrices
            if isinstance(obs_idx, slice) or isinstance(
                obs_idx, (list, np.ndarray, pd.Series)
            ):
                new_obsm[key] = mat[obs_idx]
            else:
                # Single integer index handling if needed
                new_obsm[key] = mat[obs_idx : obs_idx + 1]

        # Note: varm slicing matches var_idx
        new_varm = {}
        for key, mat in self.varm.items():
            if isinstance(var_idx, slice) or isinstance(
                var_idx, (list, np.ndarray, pd.Series)
            ):
                new_varm[key] = mat[var_idx]
            else:
                new_varm[key] = mat[var_idx : var_idx + 1]

        return SingleCellDataset(
            X=new_X,
            obs=new_obs,
            var=new_var,
            uns=self.uns.copy(),  # Keep uns as is
            obsm=new_obsm,
            varm=new_varm,
        )
