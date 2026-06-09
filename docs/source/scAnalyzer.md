# scAnalyzer: Comprehensive Documentation

**Version:** 0.2.1  
**Author:** Ayyuce Demirbas  
**License:** MIT

---

## Table of Contents

1. [What Is Single-Cell RNA Sequencing?](#1-what-is-single-cell-rna-sequencing)
2. [Why scAnalyzer?](#2-why-scanalyzer)
3. [Installation](#3-installation)
4. [Architecture Overview](#4-architecture-overview)
5. [Core Data Structure](#5-core-data-structure-singlecelldataset)
6. [Input / Output (`sc_io`)](#6-input--output-sc_io)
7. [Quality Control (`preprocessing` + `quality_control`)](#7-quality-control)
8. [Preprocessing (`preprocessing`)](#8-preprocessing)
9. [Dimensionality Reduction (`dimensionality`)](#9-dimensionality-reduction)
10. [Clustering (`clustering`)](#10-clustering)
11. [Differential Expression (`differential`)](#11-differential-expression)
12. [Cell Cycle Scoring (`cell_cycle`)](#12-cell-cycle-scoring)
13. [Batch Correction (`batch_correction`)](#13-batch-correction)
14. [Gene Set Enrichment (`enrichment`)](#14-gene-set-enrichment)
15. [Trajectory Analysis (`trajectory`)](#15-trajectory-analysis)
16. [Imputation (`imputation`)](#16-imputation)
17. [Gene Regulatory Network Inference (`grn_inference`)](#17-gene-regulatory-network-inference)
18. [Visualization (`visualization`)](#18-visualization-static)
19. [Interactive Visualization (`interactive_viz`)](#19-interactive-visualization)
20. [Utilities (`utils`)](#20-utilities)
21. [Complete End-to-End Walkthrough](#21-complete-end-to-end-walkthrough)
22. [Biological Interpretation Guide](#22-biological-interpretation-guide)
23. [Performance Tips](#23-performance-tips)
24. [Troubleshooting](#24-troubleshooting)
25. [API Quick Reference](#25-api-quick-reference)

---

## 1. What Is Single-Cell RNA Sequencing?

### The Central Dogma and Gene Expression

Every cell in your body contains the same genome — the same ~3 billion base pairs of DNA. Yet a neuron looks and behaves nothing like a liver cell or a T-lymphocyte. The difference lies in **gene expression**: which genes are actively transcribed into messenger RNA (mRNA) at any given moment in any given cell.

**Bulk RNA sequencing** (bulk RNA-seq), the older technology, measures the *average* expression of every gene across millions of cells at once. It tells you "in this tissue, on average, gene X is expressed at level Y." This erases all cellular heterogeneity — rare cell types, transitional states, and cell-to-cell variability are invisible.

**Single-cell RNA sequencing (scRNA-seq)** breaks this average apart. It profiles the mRNA content of each individual cell in isolation, giving you a snapshot of the transcriptional state of thousands to millions of individual cells simultaneously. This reveals:

- **Cell type identity** — which cells are T cells, B cells, monocytes, neurons, etc.
- **Cell states** — activated vs. resting T cells, stressed vs. healthy hepatocytes
- **Rare cell populations** — a cell type making up 0.1% of a tissue
- **Developmental trajectories** — the continuum from a stem cell to a terminally differentiated cell
- **Cell-to-cell variability** — stochastic noise in gene expression

### How scRNA-seq Works (10x Genomics Chromium)

The most widely used platform is the **10x Genomics Chromium system**:

1. **Cell capture**: Individual cells are encapsulated in nanodroplets together with gel beads carrying unique molecular barcodes.
2. **Lysis & reverse transcription**: Inside each droplet, the cell is lysed, releasing its mRNA. Each mRNA molecule is tagged with:
   - A **cell barcode** (10–16 nt): identifies which cell the molecule came from
   - A **unique molecular identifier (UMI)** (10–12 nt): a random sequence that tags each individual mRNA molecule, allowing PCR duplicates to be collapsed
3. **Library preparation & sequencing**: Tagged cDNA is amplified and sequenced.
4. **Alignment & counting**: Reads are aligned to the reference genome; UMIs per gene per cell barcode are counted, producing a **cell × gene count matrix**.

### The Count Matrix

The raw output is a sparse matrix of shape `(n_cells × n_genes)`. Each entry `X[i, j]` represents the number of UMIs (unique mRNA molecules) detected for gene `j` in cell `i`. This matrix is:

- **Very large**: 2,000–1,000,000 cells × 20,000–33,000 genes
- **Extremely sparse**: 90–98% of entries are zero, because at single-cell resolution, most genes are simply not detected (a phenomenon called **dropout**)
- **Noisy**: Stochastic capture means two identical cells will produce different counts

This is precisely the kind of data scAnalyzer is built to handle.

---

## 2. Why scAnalyzer?

scAnalyzer is a **pure-Python, dependency-light** single-cell analysis library that reimplements the core analytical machinery of the field. It is designed for:

- **Educational purposes**: Every algorithm is implemented explicitly, so you can see exactly what each step does
- **Customization**: No black-box wrappers; every function is directly modifiable
- **Lightweight deployment**: Minimal hard dependencies compared to AnnData/Scanpy
- **Reproducibility**: Deterministic random seeds throughout

### Comparison with Scanpy / AnnData

| Feature | scAnalyzer | Scanpy/AnnData |
|---|---|---|
| Core data structure | `SingleCellDataset` | `AnnData` |
| Sparse matrix support | ✅ `scipy.sparse` | ✅ |
| HDF5 I/O | ✅ custom `.h5ad` | ✅ |
| PCA / UMAP / t-SNE | ✅ | ✅ |
| Leiden / Louvain | ✅ | ✅ |
| Differential expression | ✅ t-test, Wilcoxon | ✅ |
| Cell cycle scoring | ✅ | ✅ |
| Trajectory (DPT) | ✅ | ✅ |
| Imputation | ✅ WNID, kNN, diffusion | ❌ (external) |
| GRN inference | ✅ Ridge regression | ❌ (external) |
| MNN batch correction | ✅ | ❌ (external) |
| Interactive plots | ✅ Plotly | ❌ (external) |

---

## 3. Installation

### Prerequisites

- **Python** ≥ 3.8
- **pip** ≥ 21.0

### Install from PyPI

```bash
pip install scAnalysis
```

### Install from Source

```bash
git clone https://github.com/ayyucedemirbas/scAnalysis.git
cd scAnalysis
pip install -e .
```

### Install All Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.21 | Numerical arrays |
| `pandas` | ≥1.3 | DataFrames for metadata |
| `scipy` | ≥1.7 | Sparse matrices, statistics |
| `scikit-learn` | ≥1.0 | PCA, k-means, neighbors |
| `statsmodels` | ≥0.13 | Multiple testing correction, GLMs |
| `matplotlib` | ≥3.4 | Static plotting |
| `seaborn` | ≥0.11 | Statistical visualization |
| `h5py` | ≥3.1 | HDF5 file I/O |
| `umap-learn` | ≥0.5 | UMAP dimensionality reduction |
| `leidenalg` | ≥0.8 | Leiden clustering |
| `igraph` | ≥0.9 | Graph operations |
| `plotly` | any | Interactive visualization |

### Optional Dependencies

```bash
pip install phate       # PHATE dimensionality reduction
pip install gseapy      # Full MSigDB gene set access
```

### Verify Installation

```python
import sys
sys.path.insert(0, '/path/to/scAnalysis')
exec(open('scAnalysis/check_setup.py').read())
```

This will print the status of all modules and dependencies.

---

## 4. Architecture Overview

scAnalyzer follows a **modular, functional design**. Each module is a Python file containing functions that operate on a central `SingleCellDataset` object. There are no complex class hierarchies — functions take a dataset in, modify it in place or return a new one, and return it.

```
scAnalysis/
├── core.py               ← SingleCellDataset data structure
├── sc_io.py              ← Read/write 10x MTX, CSV, H5AD
├── preprocessing.py      ← QC metrics, filtering, normalization, HVG, scaling
├── quality_control.py    ← Doublet detection (Scrublet), outlier removal
├── dimensionality.py     ← PCA, t-SNE, UMAP, Diffusion Map, PHATE
├── clustering.py         ← K-Means, Leiden, Louvain, DBSCAN, Spectral, Hierarchical
├── differential.py       ← t-test, Wilcoxon rank-sum, marker gene extraction
├── cell_cycle.py         ← S/G2M gene scoring, phase assignment, regression
├── batch_correction.py   ← ComBat, Harmony, MNN
├── enrichment.py         ← Gene set scoring, hypergeometric/Fisher enrichment, GSEA
├── trajectory.py         ← Diffusion pseudotime, gene trends, branching detection
├── imputation.py         ← WNID, kNN smoothing, diffusion imputation
├── grn_inference.py      ← Ridge regression GRN inference
├── visualization.py      ← Static Matplotlib/Seaborn plots
├── interactive_viz.py    ← Interactive Plotly plots
├── utils.py              ← Merge, subsample, filter, rename utilities
└── check_setup.py        ← Environment verification script
```

### Typical Analysis Flow

```
Raw Count Matrix
       │
       ▼
  Quality Control  ──────────────────────── remove low-quality cells & doublets
       │
       ▼
  Normalization    ──────────────────────── total counts, log1p, scran, sctransform
       │
       ▼
  Feature Selection ─────────────────────── highly variable genes
       │
       ▼
  Scaling          ──────────────────────── zero-center, clip
       │
       ▼
  Dimensionality Reduction ──────────────── PCA → UMAP / t-SNE / Diffusion Map
       │
       ▼
  Clustering       ──────────────────────── Leiden / Louvain / K-Means
       │
       ├──── Differential Expression ──────── marker genes per cluster
       ├──── Cell Type Annotation ─────────── manual or automated
       ├──── Cell Cycle Scoring ────────────── G1 / S / G2M phase
       ├──── Batch Correction ──────────────── ComBat / Harmony / MNN
       ├──── Gene Set Enrichment ───────────── pathway activity scores
       ├──── Trajectory Analysis ───────────── pseudotime, branching
       ├──── Imputation ────────────────────── fill dropout zeros
       └──── GRN Inference ─────────────────── TF → target gene edges
```

---

## 5. Core Data Structure: `SingleCellDataset`

The `SingleCellDataset` class (in `core.py`) is the central object that carries all information about your experiment. It mirrors the design philosophy of AnnData.

### Structure

```
SingleCellDataset
├── X          : (n_obs × n_vars) — the expression matrix, sparse or dense
├── obs        : (n_obs × p) DataFrame — cell metadata (barcodes, QC metrics, cluster labels…)
├── var        : (n_vars × q) DataFrame — gene metadata (gene IDs, HVG flags, dispersion…)
├── obsm       : dict of (n_obs × k) arrays — low-dimensional embeddings (PCA, UMAP, t-SNE…)
├── varm       : dict of (n_vars × k) arrays — gene-level matrices (PCA loadings…)
├── uns        : dict — unstructured metadata (neighbor graphs, PCA stats, DE results…)
└── raw        : optional backup of X before normalization/scaling
```

### Creating a Dataset

```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scAnalysis.core import SingleCellDataset

# Random count data: 500 cells, 2000 genes
X = sp.csr_matrix(np.random.poisson(1, (500, 2000)).astype(float))

obs = pd.DataFrame({
    'sample_id': ['sample_A'] * 250 + ['sample_B'] * 250,
    'time_point': ['0h'] * 125 + ['24h'] * 125 + ['0h'] * 125 + ['24h'] * 125,
}, index=[f'cell_{i}' for i in range(500)])

var = pd.DataFrame(index=[f'gene_{i}' for i in range(2000)])

data = SingleCellDataset(X=X, obs=obs, var=var)
print(data)
```

### Subsetting

```python
# Select first 100 cells
subset = data[:100, :]

# Select specific genes
t_cell_genes = data[:, ['CD3D', 'CD3E', 'CD8A']]

# Boolean mask on cells
activated = data[data.obs['activation_score'] > 0.5, :]

# Both axes simultaneously
roi = data[:100, ['CD3D', 'MS4A1']]
```

### Key Properties

| Property | Type | Description |
|---|---|---|
| `data.n_obs` | `int` | Number of cells |
| `data.n_vars` | `int` | Number of genes |
| `data.shape` | `(int, int)` | `(n_obs, n_vars)` |
| `data.obs_names()` | `pd.Index` | Cell barcodes |
| `data.var_names()` | `pd.Index` | Gene names |
| `'CD3E' in data` | `bool` | Test gene/obs/obsm membership |
| `data.summary()` | `str` | Print detailed summary |
| `data.to_df()` | `pd.DataFrame` | Dense expression DataFrame |
| `data.copy()` | `SingleCellDataset` | Full independent copy |

### The `raw` Backup

Before scaling and dimensionality reduction, you should save a backup of the log-normalized counts:

```python
data.raw = data.copy()  # save log-normalized, un-scaled counts
preprocessing.scale(data)  # now scale in place

# Later, differential expression uses data.raw automatically
# (use_raw=True is the default)
```

This is critical because differential expression should be run on log-normalized but **not** zero-centered/scaled data (scaling distorts fold-changes).

---

## 6. Input / Output (`sc_io`)

### Reading 10x Genomics MTX Format

The most common input format. Produced by Cell Ranger, STARsolo, Alevin, and Kallisto/bustools.

```python
from scAnalysis import sc_io as io

data = io.read_10x_mtx('/path/to/filtered_feature_bc_matrix/')
```

The directory must contain:
- `matrix.mtx` (or `.mtx.gz`) — sparse market exchange format count matrix
- `barcodes.tsv` (or `.tsv.gz`) — one cell barcode per line
- `features.tsv` or `genes.tsv` (or `.gz`) — gene IDs and symbols

The matrix is automatically **transposed** from 10x's (genes × cells) orientation to scAnalyzer's (cells × genes) convention.

#### Making Gene Names Unique

Gene symbols are not always unique (e.g., `LINC00886` may appear twice). Use `_make_unique` to disambiguate:

```python
data.var.index = io._make_unique(data.var.index.values)
# 'MALAT1', 'MALAT1' → 'MALAT1', 'MALAT1-1'
```

### Reading CSV / TSV

```python
# CSV: rows = cells, columns = genes
data = io.read_csv('counts.csv', delimiter=',')

# TSV
data = io.read_text('counts.tsv', delimiter='\t')
```

### HDF5 (H5AD) Format

H5AD is scAnalyzer's native binary format (compatible with AnnData's `.h5ad` spec):

```python
# Save
io.write_h5ad(data, 'processed.h5ad')

# Load
data = io.read_h5ad('processed.h5ad')
```

H5AD stores:
- Sparse or dense `X` matrix with full CSR/CSC structure
- `obs` and `var` DataFrames including categorical columns
- All `obsm` embeddings (PCA, UMAP coordinates…)
- All `uns` dictionaries (neighbor graphs, DE results…)

### Writing CSVs

```python
io.write_csvs(data, prefix='pbmc3k')
# Creates: pbmc3k_X.csv, pbmc3k_obs.csv, pbmc3k_var.csv
```

---

## 7. Quality Control

### Why QC Matters

After sequencing, not every "cell barcode" corresponds to a real, live, single cell. Common artifacts include:

| Artifact | Description | Signature |
|---|---|---|
| **Empty droplet** | Droplet captured ambient RNA but no cell | Very low UMI count, few genes detected |
| **Dead / dying cell** | Cell membrane ruptured before capture | High mitochondrial RNA fraction (cytoplasmic mRNA leaked out) |
| **Doublet** | Two cells captured in the same droplet | Anomalously high UMI and gene counts |
| **Multiplet** | Three or more cells | Extremely high counts |

### QC Metrics (`preprocessing.calculate_qc_metrics`)

```python
from scAnalysis import preprocessing as pp

pp.calculate_qc_metrics(data, qc_vars=['MT-'])
# Adds to data.obs:
#   n_genes_by_counts  — number of unique genes detected per cell
#   total_counts       — total UMI count per cell
#   pct_counts_MT-     — % of UMIs mapping to mitochondrial genes
```

**Mitochondrial genes** (prefixed `MT-` in human, `mt-` in mouse) are encoded on the mitochondrial genome. When a cell is dying, its nuclear membrane ruptures and nuclear mRNA diffuses away, but mitochondria remain intact, so mitochondrial mRNA is over-represented. A threshold of **>5% mitochondrial reads** is commonly used to flag dying cells (though this varies by tissue type — neurons and cardiomyocytes have naturally higher mitochondrial content).

### Doublet Detection (`quality_control.scrublet`)

**Scrublet** (Wolock et al., 2019) detects doublets by:

1. **Simulating doublets** by combining pairs of real cells' expression profiles
2. **Embedding** real and simulated cells together in PCA space
3. **Computing a doublet score** for each real cell based on what fraction of its k nearest neighbors are simulated doublets

```python
from scAnalysis import quality_control as qc

qc.scrublet(
    data,
    expected_doublet_rate=0.06,   # ~6% for a 3k cell capture (~1% per 1000 cells)
    n_prin_comps=30,
    verbose=True
)
# Adds to data.obs:
#   doublet_score      — continuous score (0=singlet, 1=doublet)
#   predicted_doublet  — boolean flag
```

The `expected_doublet_rate` scales with the number of cells captured. 10x Genomics provides estimates: ~0.8% per 1,000 cells recovered. For a 6,000-cell experiment, expect ~5% doublets.

```python
# Optionally remove doublets before downstream analysis
data = qc.filter_doublets(data, use_prediction=True)
```

### Filtering Cells and Genes

```python
# Filter cells
data = pp.filter_cells(
    data,
    min_genes=200,       # remove empty droplets
    max_genes=2500,      # remove doublets (heuristic upper bound)
    max_pct_mito=5.0     # remove dying cells
)

# Filter genes: remove genes seen in fewer than 3 cells
# (these contribute noise but no statistical power)
data = pp.filter_genes(data, min_cells=3)
```

**Why `min_genes=200`?** A real cell should express hundreds of genes. Droplets containing only ambient RNA typically have <200 detected genes.

**Why `max_genes=2500`?** Doublets have approximately twice the normal gene count. Adjust this based on your data's distribution (plot `data.obs['n_genes_by_counts']` first).

### Outlier Detection

For more principled filtering using robust statistics:

```python
qc.detect_outliers(
    data,
    metric='total_counts',
    n_mads=5.0,      # flag cells > 5 MADs from median
    method='both'    # flag both extremes
)
```

MAD (Median Absolute Deviation) is more robust than standard deviation for skewed count data.

---

## 8. Preprocessing

### 1. Normalization

#### Total Count Normalization (`normalize_total`)

```python
pp.normalize_total(data, target_sum=1e4)
```

**Why?** The sequencing machine reads some cells 5,000 times and others 20,000 times — a technical artifact called **sequencing depth variation**. If cell A has 1,000 counts of gene CD3E and cell B has 500, is cell A a T-cell and cell B a non-T-cell, or is cell A simply sequenced twice as deeply? By dividing each cell's counts by its total count and multiplying by a constant (10,000 by convention, so values are "counts per 10,000 UMIs"), we make counts comparable across cells.

This method saves the raw counts to `data.raw` automatically.

#### scran Pooling-Based Normalization (`normalize_scran_pooling`)

```python
pp.normalize_scran_pooling(data, n_pools=50, target_sum=1e4)
```

Total-count normalization assumes all cells contain the same amount of RNA — false for cells of very different sizes or types. **scran** (Lun et al., 2016) addresses this by:

1. Coarsely clustering cells into pools of similar depth using K-Means
2. Summing expression across cells in each pool to reduce noise (pools cancel out the zeros that plague single-cell data)
3. Solving a system of linear equations to deconvolve pool-level size factors into cell-level size factors
4. Normalizing each cell by its individual size factor

scAnalyzer implements a fast approximation using K-Means clustering and median-based pool size factors.

#### sctransform Normalization (`normalize_sctransform`)

```python
pp.normalize_sctransform(data)
```

**sctransform** (Hafemeister & Satija, 2019) addresses a fundamental statistical problem: in scRNA-seq, the variance of a gene increases with its mean expression (overdispersion), meaning highly expressed genes dominate downstream analyses even after log normalization. sctransform:

1. Fits a **Negative Binomial regression** for each gene with log(sequencing depth) as a covariate
2. Computes **Pearson residuals** — how much each cell's expression deviates from what you'd expect given its sequencing depth
3. Returns these residuals, which are naturally variance-stabilized and depth-corrected

After sctransform, `data.X` contains Pearson residuals (can be negative), not counts.

### 2. Log Transformation (`log1p`)

```python
pp.log1p(data)
# X[i,j] = log(X[i,j] + 1)
```

**Why?** Gene expression counts span many orders of magnitude. A gene might have 1 count in a resting cell and 10,000 in an activated cell. PCA is sensitive to variance; without log transformation, a handful of highly expressed genes dominate every principal component. `log1p` (log of x+1, to handle zeros) compresses this dynamic range, making the data approximately normally distributed and allowing PCA to capture biologically meaningful variance from all genes rather than just the most abundant ones.

The `+1` prevents `log(0)` errors and ensures zero-expression values remain zero.

### 3. Highly Variable Gene Selection (`highly_variable_genes`)

```python
pp.highly_variable_genes(data, n_top_genes=2000, n_bins=20)
# Adds to data.var:
#   means             — mean expression per gene
#   dispersions       — variance / mean (Fano factor)
#   dispersions_norm  — dispersion normalized within expression bins
#   highly_variable   — boolean flag for top 2000 HVGs
```

**Why?** Of ~20,000 human genes, most are **housekeeping genes** — GAPDH, ACTB, ribosomal proteins — that are expressed at roughly constant levels in all cell types. Their expression doesn't vary much between cells. Including them in PCA adds noise without signal.

**Highly variable genes (HVGs)** are genes whose expression fluctuates meaningfully across cells — these are the cell-type markers, signaling molecules, and regulatory genes that define different cellular identities. By selecting the top 2,000–4,000 HVGs, we focus downstream analyses on biologically informative features.

**The dispersion normalization step** is critical: a gene with mean expression 0.01 naturally has higher dispersion (variance/mean) than a gene with mean 100, purely due to count statistics (Poisson noise). By binning genes by expression level and computing z-scores of dispersion within bins, we identify genes that are more variable *than expected* for their expression level.

### 4. Scaling (`scale`)

```python
pp.scale(data, max_value=10.0, zero_center=True)
```

Converts each gene's expression across cells to a z-score:

```
z[i,j] = (X[i,j] - mean_j) / std_j
```

**Why?** Even among HVGs, absolute expression levels differ. A gene expressed at 500 counts on average would dominate PCA over a gene expressed at 5 counts, even if both have equally informative variation. After scaling, every gene has mean 0 and standard deviation 1, so PCA captures relative variance rather than absolute magnitude.

**`max_value=10`** clips extreme values (usually arising from outlier cells) to prevent them from distorting PCA. This is a standard practice; the exact value matters little.

**Note:** Scaling **densifies** a sparse matrix. Run it after HVG selection to minimize memory usage.

---

## 9. Dimensionality Reduction

### Why Dimensionality Reduction?

Even after HVG selection, you're working with ~2,000 features per cell. High-dimensional spaces are pathological for distance metrics (the "curse of dimensionality"). Additionally, much of the gene-level variation is correlated — groups of co-expressed genes rise and fall together. Dimensionality reduction finds the low-dimensional manifold underlying the data.

### PCA (`run_pca`)

```python
from scAnalysis import dimensionality as dim

dim.run_pca(data, n_components=50, use_highly_variable=True)
# Adds:
#   data.obsm['X_pca']           — (n_cells × 50) PCA coordinates
#   data.varm['PCs']             — (n_genes × 50) loadings
#   data.uns['pca']['variance_ratio']  — variance explained per PC
```

**Principal Component Analysis (PCA)** finds the orthogonal directions in gene expression space that capture the most variance. PC1 captures the most variance, PC2 captures the next most (orthogonal to PC1), and so on.

For sparse data, `TruncatedSVD` is used automatically (equivalent to PCA but memory-efficient for sparse matrices).

**How many PCs to use?** Typically 20–50 for downstream steps. Look at the elbow in the variance-explained plot. The first 10–20 PCs often capture major biological variation (cell type, activation state), while later PCs capture technical noise.

```python
var_ratio = data.uns['pca']['variance_ratio']
print(f"PC1-10 explain {var_ratio[:10].sum()*100:.1f}% of variance")
```

### Neighbor Graph (`neighbors`)

```python
dim.neighbors(data, n_neighbors=15, n_pcs=40)
# Adds to data.uns['neighbors']:
#   distances        — sparse (n_cells × n_cells) distance matrix
#   connectivities   — sparse Gaussian-weighted adjacency matrix
```

Before UMAP or graph-based clustering, a **k-nearest neighbor (kNN) graph** is built in PCA space. Each cell is connected to its k most similar cells, with edge weights computed by a Gaussian kernel:

```
w(i,j) = exp(-d(i,j)² / σᵢ²)
```

where σᵢ is the distance to cell i's k-th nearest neighbor (adaptive bandwidth). This graph is the backbone of both clustering and UMAP.

### UMAP (`run_umap`)

```python
dim.run_umap(data, min_dist=0.3, n_components=2)
# Adds: data.obsm['X_umap'] — (n_cells × 2)
```

**Uniform Manifold Approximation and Projection (UMAP)** (McInnes et al., 2018) projects the kNN graph into 2D while preserving local (and partially global) structure. UMAP optimizes an embedding that keeps neighbors close while pushing non-neighbors apart, using a cross-entropy loss between the high-dimensional fuzzy topological representation and the low-dimensional one.

**`min_dist`** controls how tightly points are packed. Lower values (0.1) produce tighter clusters; higher values (0.5) produce more diffuse, evenly-spaced layouts that better preserve global structure.

**Important**: UMAP coordinates are for **visualization only**. Do not perform clustering or statistical tests on UMAP coordinates — use PCA space for those.

### t-SNE (`run_tsne`)

```python
dim.run_tsne(data, n_pcs=30, perplexity=30.0, n_iter=1000)
# Adds: data.obsm['X_tsne'] — (n_cells × 2)
```

**t-Distributed Stochastic Neighbor Embedding (t-SNE)** (van der Maaten & Hinton, 2008) minimizes the KL divergence between probability distributions over pairs in high-dimensional and low-dimensional space. It uses a Student-t distribution in the low-dimensional space to avoid the "crowding problem."

t-SNE is excellent for revealing local cluster structure but does **not** preserve global distances — clusters that appear far apart in t-SNE may not actually be globally distant. UMAP is generally preferred for modern scRNA-seq analysis.

**Perplexity** (default 30) is a smoothing parameter roughly equal to the effective number of neighbors. For large datasets (>100,000 cells), try perplexity 50–100.

### Diffusion Map (`run_diffmap`)

```python
dim.run_diffmap(data, n_components=15)
# Adds: data.obsm['X_diffmap'] — (n_cells × 15)
#        data.uns['diffmap_evals']
```

**Diffusion maps** (Coifman & Lafon, 2006) model the data as a diffusion process on a graph: the coordinates are the eigenvectors of the normalized graph Laplacian, scaled by their eigenvalues. They are especially powerful for **continuous developmental trajectories** because diffusion distance reflects the number of paths between cells, capturing the underlying manifold geometry more faithfully than Euclidean distance.

Diffusion components are the input to diffusion pseudotime (see Section 15).

### PHATE (`run_phate`)

```python
dim.run_phate(data, n_components=2, knn=5)
# Requires: pip install phate
# Adds: data.obsm['X_phate']
```

**PHATE** (Moon et al., 2019) is designed specifically for biological trajectory data. It preserves both local and global structure by using a "potential of heat diffusion" — a double-diffusion process that amplifies the structure of the data manifold.

---

## 10. Clustering

### Why Cluster?

Clustering groups cells by transcriptional similarity, operationally defining **cell types and states**. In practice, a "cluster" is a hypothesis — a group of cells that share enough expression patterns to warrant being treated as a distinct population. Clusters are subsequently annotated as cell types using marker genes.

### Leiden Clustering (`cluster_leiden`) ⭐ Recommended

```python
from scAnalysis import clustering as cl

cl.cluster_leiden(data, resolution=0.5, key_added='leiden')
# Adds: data.obs['leiden'] — categorical cluster labels
```

**Leiden clustering** (Traag et al., 2019) is the current gold standard for scRNA-seq. It optimizes the **modularity** of the kNN graph — a measure of how well-separated the communities are. It is a refinement of Louvain clustering that guarantees well-connected communities.

**`resolution`** controls granularity:
- Lower values (0.2–0.5): fewer, larger clusters
- Higher values (1.0–2.0): more, smaller clusters

Start at 0.5 and adjust based on biological knowledge. Run the pipeline at multiple resolutions and compare.

```python
# Explore multiple resolutions
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    cl.cluster_leiden(data, resolution=res, key_added=f'leiden_r{res}')
```

Requires: `pip install leidenalg igraph`

### Louvain Clustering (`cluster_louvain`)

```python
cl.cluster_louvain(data, resolution=1.0, key_added='louvain')
```

The predecessor to Leiden. Faster on very large graphs but can produce internally disconnected communities. Use Leiden when possible.

### K-Means Clustering (`cluster_kmeans`)

```python
cl.cluster_kmeans(data, n_clusters=10, use_rep='X_pca', key_added='kmeans')
```

K-Means requires specifying the number of clusters `k`. Use it when you have a strong prior on cell type count, or with `auto_select_k=True`:

```python
cl.cluster_kmeans(
    data,
    auto_select_k=True,
    k_range=(2, 20),    # search k from 2 to 20
    use_rep='X_pca'
)
```

Auto-selection uses the **silhouette score** to pick the best k.

### DBSCAN Clustering (`cluster_dbscan`)

```python
cl.cluster_dbscan(data, eps=0.5, min_samples=5, use_rep='X_pca')
```

**DBSCAN** identifies dense regions as clusters and labels low-density cells as noise (`-1`). Useful for detecting rare populations without specifying cluster count. Sensitive to the `eps` parameter.

### Spectral Clustering (`cluster_spectral`)

```python
cl.cluster_spectral(data, n_clusters=10, use_rep='X_pca', affinity='rbf')
```

Spectral clustering uses the eigenvectors of the graph Laplacian. Works well for non-convex cluster shapes.

### Hierarchical Clustering (`cluster_hierarchical`)

```python
cl.cluster_hierarchical(data, n_clusters=10, linkage='ward', use_rep='X_pca')
```

Agglomerative hierarchical clustering. Produces a dendrogram-compatible grouping. Computationally intensive for >10,000 cells.

### Cluster Statistics (`cluster_stats`)

```python
stats = cl.cluster_stats(data, cluster_key='leiden')
# Returns DataFrame with n_cells, pct_of_total, and mean expression per cluster
print(stats[['n_cells', 'pct_of_total']].head())
```

---

## 11. Differential Expression

### Biology of Differential Expression

To annotate clusters as cell types, we ask: **which genes are significantly more expressed in cluster X than in all other cells?** These **marker genes** serve as molecular fingerprints. For example:

- `CD3D`, `CD3E`, `CD3G` → T cells (TCR complex components)
- `CD19`, `MS4A1` (CD20), `CD79A` → B cells
- `CD14`, `LYZ`, `S100A8` → Monocytes
- `GNLY`, `NKG7`, `GZMA` → NK cells
- `FCER1A`, `CST3` → Dendritic cells

### `rank_genes_groups`

```python
from scAnalysis import differential as diff

diff.rank_genes_groups(
    data,
    groupby='leiden',
    method='t-test',      # or 'wilcoxon'
    use_raw=True,         # use log-normalized (not scaled) counts
    reference='rest'      # compare each cluster vs. all others
)
# Stores results in data.uns['rank_genes_groups']
# Each cluster gets a DataFrame with columns:
#   names          — gene name
#   scores         — test statistic
#   logfoldchanges — mean expression difference (log-scale)
#   pvals          — raw p-value
#   pvals_adj      — Benjamini-Hochberg adjusted p-value
#   pct_in         — fraction of cells in group expressing this gene
#   pct_out        — fraction of cells outside group expressing this gene
```

#### t-test (`method='t-test'`)

Welch's t-test compares the mean expression of each gene between the target cluster and all other cells. It accounts for different variances in each group (Welch's correction). Fast and robust for large datasets.

#### Wilcoxon Rank-Sum (`method='wilcoxon'`)

The Wilcoxon test (Mann-Whitney U) is non-parametric — it ranks all expression values and tests whether the target cluster ranks systematically higher. It makes no distributional assumptions and is recommended when normality cannot be assumed (which is the usual case for sparse count data).

#### Multiple Testing Correction

With ~20,000 genes tested per cluster, many false positives are expected by chance. **Benjamini-Hochberg FDR correction** controls the false discovery rate: if `pvals_adj < 0.05`, we expect fewer than 5% of rejected hypotheses to be false positives.

### Extracting Marker Genes (`get_marker_genes`)

```python
markers = diff.get_marker_genes(
    data,
    group='0',          # cluster label
    pval_cutoff=0.05,
    lfc_cutoff=0.5,     # log fold-change threshold
    top_n=20
)
print(markers[['names', 'logfoldchanges', 'pvals_adj', 'pct_in', 'pct_out']])
```

A good marker gene has:
- High `logfoldchanges` (specifically expressed in the target cluster)
- Low `pvals_adj` (statistically significant)
- High `pct_in` (expressed in most cells of the target cluster)
- Low `pct_out` (not expressed in other clusters)

---

## 12. Cell Cycle Scoring

### The Cell Cycle and Its Confound

Actively dividing cells progress through four phases:
- **G1** (Gap 1): Cell grows, prepares for DNA synthesis
- **S** (Synthesis): DNA is replicated
- **G2** (Gap 2): Cell prepares for mitosis
- **M** (Mitosis): Cell divides

Cell cycle phase is a **major source of transcriptional variation** that can confound cell type clustering. A G1 T-cell and an S-phase T-cell may cluster separately even though they're the same cell type at different cell cycle stages. Cell cycle scoring allows you to identify phase and, if desired, regress it out.

### Scoring (`score_cell_cycle`)

```python
from scAnalysis import cell_cycle as cc

cc.score_cell_cycle(data, organism='human')
# Adds to data.obs:
#   S_score    — score for S-phase gene expression
#   G2M_score  — score for G2/M-phase gene expression
#   phase      — categorical: 'G1', 'S', or 'G2M'
```

Scoring uses **Seurat's gene set scoring approach**:

1. For each phase gene set (S genes or G2M genes), compute the mean expression across all genes in the set for each cell
2. Subtract the mean expression of a **control gene set** — randomly selected genes with similar overall expression levels — to correct for sequencing depth
3. The score is: `mean(target genes) - mean(control genes)`

**Human S-phase genes** (43 genes): MCM2, MCM4, MCM5, PCNA, TYMS, RRM1, RRM2, CDC45, CDC6, EXO1, GINS2, POLA1, BRIP1, RFC2, and more.

**Human G2M genes** (54 genes): CDK1, TOP2A, MKI67, BIRC5, NDC80, TPX2, AURKB, AURKA, KIF11, KIF23, CCNB2, BUB1, and more.

### Regressing Out Cell Cycle (`regress_out_cell_cycle`)

```python
cc.regress_out_cell_cycle(
    data,
    difference_only=True   # regress out S vs G2M difference only (preserves proliferation signal)
    # difference_only=False would regress out all cycling, removing proliferation signal too
)
```

`difference_only=True` (default) removes the *difference* between S and G2M scores — correcting for phase assignment while preserving the proliferation signal. Use `difference_only=False` only if you want to fully remove all cell cycle effects (e.g., to study quiescence vs. proliferation is not your question).

---

## 13. Batch Correction

### What Is a Batch Effect?

When cells from different samples, experiments, days, laboratories, or sequencing runs are analyzed together, **technical batch effects** — systematic differences in gene expression unrelated to biology — can dominate the analysis. Cells may cluster by batch rather than by cell type.

### ComBat (`combat`)

```python
from scAnalysis import batch_correction as bc

bc.combat(data, batch_key='sample_id', inplace=True)
```

**ComBat** (Johnson et al., 2007) is a parametric batch correction method. For each gene, it models the batch effect as an additive shift (`γ`) and multiplicative scaling (`δ`) of expression:

```
X_corrected = (X_batch - μ_batch) / σ_batch × σ_global + μ_global
```

This standardizes each gene's distribution across batches to match a global distribution. It assumes the batch effect is consistent across cell types — an assumption that is often violated. Use ComBat for straightforward batch differences (different sequencing runs of the same experiment).

### Harmony (`harmony_integrate`)

```python
bc.harmony_integrate(
    data,
    batch_key='batch',
    basis='X_pca',                 # correct the PCA embedding
    adjusted_basis='X_pca_harmony',
    theta=2.0,                     # diversity penalty: higher = more correction
    sigma=0.1,                     # kernel bandwidth
    max_iter_harmony=20
)
# Adds: data.obsm['X_pca_harmony']
```

**Harmony** (Korsunsky et al., 2019) operates on the PCA embedding rather than the raw count matrix. It iteratively:

1. **Clusters** cells softly using fuzzy k-means
2. **Corrects** each cell's PCA coordinates by removing the contribution of batch membership, weighted by cluster assignment

The key innovation is that Harmony corrects each cluster independently — so a cell type that genuinely differs between conditions (e.g., activated T cells only in treatment) is not over-corrected. Use `X_pca_harmony` instead of `X_pca` for neighbor graph construction:

```python
dim.neighbors(data, use_rep='X_pca_harmony', n_neighbors=15)
```

### MNN Correction (`mnn_correct`)

```python
corrected = bc.mnn_correct(
    [data_sample1, data_sample2, data_sample3],
    batch_key='batch',
    k=20,        # number of mutual nearest neighbors
    sigma=1.0    # smoothing bandwidth
)
```

**Mutual Nearest Neighbors (MNN)** (Haghverdi et al., 2018) correction:

1. For each pair of batches, finds cells that are each other's nearest neighbors across batches (**mutual nearest neighbors** = likely the same cell type)
2. Computes correction vectors from MNN pairs
3. Applies weighted correction to all cells, with weight decaying with distance from anchor pairs

MNN is well-suited for multi-sample integration where cell type composition varies between samples.

---

## 14. Gene Set Enrichment

### Scoring Gene Sets (`gene_set_score`)

```python
from scAnalysis import enrichment as enrich

hypoxia_genes = ['VEGFA', 'HIF1A', 'LDHA', 'PDK1', 'ENO1', 'PFKP']
enrich.gene_set_score(data, hypoxia_genes, score_name='hypoxia_score')
# Adds: data.obs['hypoxia_score']
```

Uses the same bin-controlled scoring as cell cycle (Seurat's `AddModuleScore`). Scores can be used to color UMAP plots, correlate with other metadata, or define high-activity subpopulations.

```python
# Score multiple pathways at once
enrich.score_multiple_gene_sets(data, {
    'T_cell':  ['CD3D', 'CD3E', 'CD3G'],
    'B_cell':  ['CD19', 'MS4A1', 'CD79A'],
    'Myeloid': ['CD14', 'LYZ', 'S100A8'],
    'MHCII':   ['HLA-DRA', 'HLA-DRB1', 'CD74']
})
```

### Hypergeometric / Fisher Enrichment (`rank_genes_groups_by_enrichment`)

Tests whether marker genes of each cluster are enriched for genes in known pathways:

```python
gene_sets = enrich.load_gene_sets('msigdb', categories=['HALLMARK'])

enrichment = enrich.rank_genes_groups_by_enrichment(
    data,
    gene_sets,
    groupby='leiden',
    method='hypergeometric'   # or 'fisher'
)

# View enrichment for cluster 0
print(enrichment['0'][['gene_set', 'overlap_size', 'fold_enrichment', 'pval_adj']].head(10))
```

**Hypergeometric test**: Given a background of M genes, a pathway of n genes, and N marker genes, the probability of seeing k or more overlapping genes by chance follows the hypergeometric distribution.

**Fisher's exact test**: Equivalent but framed as a 2×2 contingency table.

### GSEA Pre-Ranked (`gsea_preranked`)

```python
# Get ranked gene list for a cluster
ranked = data.uns['rank_genes_groups']['0'][['names', 'scores']].copy()

result = enrich.gsea_preranked(
    ranked,
    gene_set=['VEGFA', 'HIF1A', 'LDHA', 'PDK1'],
    nperm=1000
)
print(f"ES={result['ES']:.3f}, NES={result['NES']:.3f}, p={result['pval']:.4f}")
```

**Gene Set Enrichment Analysis (GSEA)** computes an enrichment score (ES) that reflects whether the genes in a set are enriched at the top or bottom of a ranked gene list. The normalized enrichment score (NES) accounts for gene set size. A permutation test assesses significance.

---

## 15. Trajectory Analysis

### What Is a Trajectory?

Not all biological processes are discrete cell types. Hematopoiesis (blood cell development), embryonic development, and T-cell differentiation involve **continuous transitions** from one cell state to another. A cell doesn't jump from a stem cell to a mature T cell; it traverses a continuous differentiation path.

**Trajectory analysis** reconstructs this continuum from a snapshot (the scRNA-seq data), ordering cells along a pseudotime axis that reflects their progress through a biological process.

### Selecting the Root Cell (`select_root_cell`)

```python
from scAnalysis import trajectory

root = trajectory.select_root_cell(
    data,
    cluster_key='leiden',
    root_cluster='0',         # the cluster corresponding to starting cell type
    strategy='extreme'        # pick cell most distant from the overall centroid
    # strategy='medoid'       # pick cell closest to cluster center
)
```

The root cell defines the start (pseudotime=0) of the trajectory. Biologically, this is typically a stem cell, progenitor, or the earliest developmental stage. Choose based on prior knowledge.

### Diffusion Pseudotime (`diffusion_pseudotime`)

```python
trajectory.diffusion_pseudotime(
    data,
    root_cell=root,
    n_dcs=10,                # number of diffusion components to use
    key_added='dpt_pseudotime',
    n_branchings=0           # set >0 to detect branch points
)
# Adds: data.obs['dpt_pseudotime'] — float in [0, 1], 0 = root
```

**Diffusion Pseudotime (DPT)** (Haghverdi et al., 2016) computes pseudotime as the diffusion distance from the root cell in diffusion map space. Diffusion distance reflects the number of paths connecting two cells in the graph — cells on the same trajectory but far in real time have high diffusion distance from the root.

DPT handles **branching trajectories**: after the main trunk, lineages diverge, and cells on different branches receive different pseudotimes. Set `n_branchings=1` or higher to detect and label branch points.

```python
# Detect branching
trajectory.diffusion_pseudotime(data, root_cell=root, n_branchings=2)
# Adds: data.obs['dpt_groups'] — which branch each cell is on
```

### Gene Expression Trends (`gene_trends`)

```python
trends = trajectory.gene_trends(
    data,
    genes=['MKI67', 'CD34', 'CD38', 'CD19', 'PAX5'],
    pseudotime_key='dpt_pseudotime',
    n_bins=50,
    use_raw=True
)
# Returns DataFrame: rows = pseudotime bins, columns = genes
```

This bins cells by pseudotime and computes mean expression per bin, revealing how gene expression changes along the trajectory. Use this to find:
- **Early markers**: Genes expressed at high pseudotime that drop off
- **Late markers**: Genes that increase with pseudotime
- **Transient markers**: Genes with a peak in the middle

---

## 16. Imputation

### The Dropout Problem

At single-cell resolution, **stochastic capture** means that any given mRNA molecule has roughly a 10–20% chance of being captured. A gene expressed at 5 molecules per cell will appear as zero in ~60% of cells purely due to capture failure — a technical zero, not a biological one. This is called **dropout**, and it can:

- Obscure expression patterns
- Inflate zero-zero correlations
- Distort trajectory analysis

Imputation attempts to fill these technical zeros with estimated values. **Use imputation cautiously** — it can also introduce spurious correlations.

### WNID (`impute_wnid`) — Weighted Nearest-neighbor Imputation of Dropouts

```python
from scAnalysis import imputation

imputation.impute_wnid(
    data,
    k=15,                  # number of neighbors
    dropout_thresh=0.5,    # only impute high-probability dropouts
    n_pcs=30,
    weight_method='gaussian',
    inplace=True
)
```

**WNID** is scAnalyzer's primary imputation method. The algorithm:

1. **Identifies likely dropout entries**: Uses a CV²-based model to estimate the dropout probability of each zero entry. Genes with high variance-to-mean ratio (bursty expression) are more likely to have technical zeros than stably expressed genes.
2. **Builds a kNN graph** in PCA space
3. **Imputes** each candidate dropout with a weighted average of the gene's expression in k nearest neighbor cells, with Gaussian weights based on distance

The `dropout_thresh` controls conservatism: higher values impute only the most likely dropouts; 0 would impute all zeros (aggressive and not recommended).

### kNN Smoothing (`impute_knn_smooth`)

```python
imputation.impute_knn_smooth(
    data,
    k=10,
    weight_method='gaussian',
    n_pcs=30
)
```

Smooths the entire expression matrix (not just zeros) by replacing each cell's expression with a weighted average of itself and its k nearest neighbors. Reduces noise broadly but may blur sharp boundaries between cell types.

### Diffusion Imputation (`impute_diffusion`)

```python
imputation.impute_diffusion(
    data,
    t=3,       # number of diffusion steps
    alpha=1.0  # anisotropy parameter
)
```

Applies the diffusion operator (the Markov transition matrix T) t times to the expression matrix. Each application smooths expression over the kNN graph, propagating information from neighbors. More steps = more smoothing. Preserves the global structure of the data better than pure kNN approaches.

### Comparing Before/After Imputation

```python
data_before = data.copy()
imputation.impute_wnid(data)
imputation.compare_imputation(data_before, data)
```

### Dropout Statistics

```python
stats = imputation.dropout_stats(data, top_n=10)
# Prints: global dropout rate, top 10 most dropout-affected genes
```

---

## 17. Gene Regulatory Network Inference

### What Is a GRN?

A **Gene Regulatory Network (GRN)** describes which transcription factors (TFs) regulate which target genes. Each edge `TF → gene` represents a regulatory relationship: the TF's expression predicts the target's expression.

### Ridge Regression GRN (`infer_grn_ridge`)

```python
from scAnalysis import grn_inference

# A list of known transcription factors
human_tfs = ['GATA1', 'SPI1', 'PAX5', 'TBX21', 'RORC', 'FOXP3', 'IRF4', 'IRF8']

grn = grn_inference.infer_grn_ridge(
    data,
    tf_list=human_tfs,
    top_n_edges=50000
)
# Returns DataFrame: source (TF), target (gene), weight (regulatory strength)
print(grn.head(20))
```

For each target gene, a **Ridge regression** is fitted with TF expression values as predictors. The absolute regression coefficients indicate the strength of the TF→gene relationship. Ridge regression's L2 penalty stabilizes the solution when TFs are correlated (which they typically are).

**Limitations**: This is a correlation-based approach. It identifies statistical associations between TF and target expression but cannot distinguish direct binding from indirect regulatory chains. For more sophisticated GRN inference, consider SCENIC (pySCENIC).

---

## 18. Visualization (Static)

All plotting functions accept a `save` parameter. When provided, the figure is saved to disk at 300 DPI. When omitted, `plt.show()` is called.

### UMAP / Embedding Plots

```python
from scAnalysis import visualization as vis

# Color by cluster
vis.plot_umap(data, color='leiden', title='Clusters', legend_loc='on data')

# Color by gene expression
vis.plot_umap(data, color='CD3E', cmap='Reds', save='umap_CD3E.png')

# Color by continuous metadata
vis.plot_umap(data, color='dpt_pseudotime', cmap='viridis')

# Overlay on custom axes
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vis.plot_umap(data, color='leiden', ax=axes[0])
vis.plot_umap(data, color='S_score', ax=axes[1])
vis.plot_umap(data, color='total_counts', ax=axes[2])
plt.tight_layout()
plt.savefig('umap_panel.png', dpi=300)
```

Supported embedding bases: `X_umap`, `X_tsne`, `X_pca`, `X_diffmap`, `X_phate`, or any key in `data.obsm`.

### Violin Plot

```python
vis.plot_violin(
    data,
    keys=['n_genes_by_counts', 'total_counts', 'pct_counts_MT-'],
    groupby='leiden',
    save='qc_violin.png'
)
```

### Dot Plot

```python
canonical_markers = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'LYZ', 'CD14', 'GNLY', 'NKG7', 'FCER1A', 'CST3']

vis.plot_dotplot(
    data,
    var_names=canonical_markers,
    groupby='leiden',
    standard_scale=True,   # scale expression per gene to [0,1]
    cmap='Reds',
    save='dotplot.png'
)
```

Each dot encodes two quantities:
- **Color intensity**: Mean expression level
- **Dot size**: Fraction of cells in the group expressing the gene (> 0)

This is the most information-dense plot for cell type annotation.

### Heatmap

```python
vis.plot_heatmap(
    data,
    var_names=top_markers,
    groupby='leiden',
    standard_scale='var',
    cmap='viridis',
    save='heatmap.png'
)
```

### Volcano Plot

```python
vis.volcano_plot(
    data,
    group='3',                 # cluster to visualize
    pval_threshold=0.05,
    lfc_threshold=0.5,
    top_n_genes=15,
    save='volcano_cluster3.png'
)
```

The volcano plot shows log fold-change (x-axis) vs. −log₁₀(adjusted p-value) (y-axis). Significant upregulated genes (red) and downregulated genes (blue) are highlighted.

### QC Violin

```python
vis.plot_qc_violin(
    data,
    metrics=['n_genes_by_counts', 'total_counts', 'pct_counts_MT-'],
    groupby=None,     # or a cluster key for per-cluster QC
    save='qc.png'
)
```

### Highest-Expressed Genes

```python
vis.plot_highest_expr_genes(data, n_top=20, save='top_genes.png')
```

Useful early in the analysis to spot rRNA genes (`RPS*`, `RPL*`) or mitochondrial genes that may need to be excluded.

---

## 19. Interactive Visualization

Requires `pip install plotly`.

### Interactive UMAP

```python
from scAnalysis import interactive_viz as iviz

fig = iviz.interactive_embedding(
    data,
    basis='X_umap',
    color='leiden',
    hover_data=['n_genes_by_counts', 'total_counts', 'doublet_score', 'phase'],
    title='PBMC 3k — Interactive UMAP',
    save_html='umap_interactive.html'
)
```

The HTML file can be opened in any browser. You can pan, zoom, hover over individual cells to see their metadata, and toggle cluster visibility by clicking the legend.

### Interactive 3D PCA

```python
iviz.interactive_3d_embedding(
    data,
    basis='X_pca',
    color='leiden',
    dimensions=[0, 1, 2],    # which PCs to show
    save_html='pca_3d.html'
)
```

Rotating the 3D plot often reveals structure invisible in 2D UMAP projections.

### Interactive Heatmap

```python
iviz.interactive_heatmap(
    data,
    var_names=canonical_markers,
    groupby='leiden',
    use_raw=True,
    standard_scale=True,
    save_html='heatmap_interactive.html'
)
```

### Interactive Violin

```python
iviz.interactive_violin(
    data,
    keys=['CD3E', 'MS4A1', 'LYZ'],
    groupby='leiden',
    save_html='violin_interactive.html'
)
```

---

## 20. Utilities

### Merging Datasets (`merge`)

```python
from scAnalysis import utils

# Merge multiple experiments
merged = utils.merge(
    [data_sample1, data_sample2, data_sample3],
    batch_keys=['ctrl', 'treat_6h', 'treat_24h'],
    batch_category='condition',
    join='inner'    # 'inner' = genes in all datasets; 'outer' = union
)
```

### Subsampling (`subsample`)

```python
# Random subsample
sub = utils.subsample(data, n=5000, random_state=42)

# Fraction-based
sub = utils.subsample(data, fraction=0.1)

# Stratified (preserve cluster proportions)
sub = utils.subsample(data, n=5000, stratify='leiden', random_state=42)
```

### Filtering (`filter_obs`, `filter_var`)

```python
# Keep only cells in cluster 3
cluster3 = utils.filter_obs(data, lambda obs: obs['leiden'] == '3')

# Boolean array
t_cells = utils.filter_obs(data, data.obs['leiden'].isin(['1', '3', '7']))

# Filter genes by mask
hvg_data = utils.filter_var(data, data.var['highly_variable'])
```

### Renaming Metadata (`rename_obs`)

```python
# After manual annotation, rename leiden clusters to cell types
utils.rename_obs(data, {'leiden': 'cell_type'})
data.obs['cell_type'] = data.obs['cell_type'].map({
    '0': 'CD4 T cell',
    '1': 'CD8 T cell',
    '2': 'B cell',
    '3': 'Monocyte',
    '4': 'NK cell',
    '5': 'Dendritic cell',
})
```

### Describing Metadata (`describe_obs`)

```python
utils.describe_obs(data, 'total_counts')   # numeric: prints pandas .describe()
utils.describe_obs(data, 'leiden')         # categorical: prints value_counts()
```

### Computing Mean and Variance (`get_mean_var`)

```python
gene_mean, gene_var = utils.get_mean_var(data, axis=0)  # per gene
cell_mean, cell_var = utils.get_mean_var(data, axis=1)  # per cell
```

---

## 21. Complete End-to-End Walkthrough

This walkthrough processes the PBMC 3k dataset (2,700 peripheral blood mononuclear cells from a healthy donor), the most widely used scRNA-seq benchmark dataset.

```python
import sys
sys.path.insert(0, '/path/to/scAnalysis')

from scAnalysis import sc_io as io
from scAnalysis import preprocessing as pp
from scAnalysis import quality_control as qc
from scAnalysis import cell_cycle as cc
from scAnalysis import dimensionality as dim
from scAnalysis import clustering as cl
from scAnalysis import differential as diff
from scAnalysis import enrichment as enrich
from scAnalysis import trajectory
from scAnalysis import visualization as vis
from scAnalysis import utils


# STEP 1: Load data

data = io.read_10x_mtx('./filtered_gene_bc_matrices/hg19/')
data.var.index = io._make_unique(data.var.index.values)
print(data)  # 2700 cells × 32738 genes

# STEP 2: QC metrics

pp.calculate_qc_metrics(data, qc_vars=['MT-'])
vis.plot_qc_violin(data, save='qc_before_filtering.png')

# STEP 3: Doublet detection

qc.scrublet(data, expected_doublet_rate=0.06, verbose=True)
print(f"Doublets: {data.obs['predicted_doublet'].sum()}")

# STEP 4: Filter cells and genes

data = pp.filter_cells(data, min_genes=200, max_genes=2500, max_pct_mito=5.0)
data = pp.filter_genes(data, min_cells=3)
print(data)  # ~2638 cells × ~13714 genes

# STEP 5: Normalize and log transform

pp.normalize_total(data, target_sum=1e4)
pp.log1p(data)

# STEP 6: Cell cycle scoring

cc.score_cell_cycle(data, organism='human')
print(data.obs['phase'].value_counts())

# STEP 7: Highly variable genes + scaling

pp.highly_variable_genes(data, n_top_genes=2000)

data.raw = data.copy()          # backup log-normalized counts

pp.scale(data, max_value=10)    # now scale for PCA

# STEP 8: Dimensionality reduction

dim.run_pca(data, n_components=50)
print(f"PC1-10: {data.uns['pca']['variance_ratio'][:10].sum()*100:.1f}% variance")

dim.neighbors(data, n_neighbors=10, n_pcs=40)
dim.run_umap(data, min_dist=0.3)

# STEP 9: Clustering

cl.cluster_leiden(data, resolution=0.5, key_added='leiden')
print(f"Found {data.obs['leiden'].nunique()} clusters")
print(data.obs['leiden'].value_counts().sort_index())

# STEP 10: Differential expression

diff.rank_genes_groups(data, groupby='leiden', method='t-test', use_raw=True)

# Print top 5 markers for each cluster
for cluster in sorted(data.obs['leiden'].unique()):
    markers = diff.get_marker_genes(data, group=cluster, top_n=5)
    if len(markers):
        print(f"\nCluster {cluster}: {', '.join(markers['names'].tolist())}")

# STEP 11: Cell type annotation (manual)

# Based on marker genes, annotate clusters:
cell_type_map = {
    '0': 'CD4 T cells',
    '1': 'CD14 Monocytes',
    '2': 'B cells',
    '3': 'CD8 T cells',
    '4': 'NK cells',
    '5': 'FCGR3A Monocytes',
    '6': 'Dendritic cells',
    '7': 'Megakaryocytes',
}
data.obs['cell_type'] = data.obs['leiden'].map(cell_type_map)

# STEP 12: Visualization

vis.plot_umap(data, color='cell_type', legend_loc='on data',
              title='PBMC 3k Cell Types', save='umap_celltypes.png')

canonical = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'LYZ', 'CD14',
             'GNLY', 'NKG7', 'FCER1A', 'CST3', 'PPBP']
vis.plot_dotplot(data, var_names=canonical, groupby='cell_type',
                 standard_scale=True, save='dotplot_celltypes.png')

# Volcano for monocytes
vis.volcano_plot(data, group='1', pval_threshold=0.05,
                 lfc_threshold=0.5, save='volcano_mono.png')

# STEP 13: Save results

io.write_h5ad(data, 'pbmc3k_final.h5ad')
print("Analysis complete")
```

---

## 22. Biological Interpretation Guide

### Reading a UMAP

- **Clusters**: Discrete groups of cells with similar transcriptional profiles. Each cluster typically corresponds to a cell type or state.
- **Distance between clusters**: Partially meaningful — clusters that are close share more transcriptional similarity. However, UMAP does not perfectly preserve global distances.
- **Trajectory structure**: Cells arranged in an elongated arc or continuum often represent a developmental or activation trajectory.
- **Isolated clusters**: Small, isolated clusters may represent rare cell types, cell cycle artifacts, or low-quality cells.

### Common PBMC Cell Types and Their Markers

| Cell Type | Key Markers | Function |
|---|---|---|
| CD4 T cells | IL7R, CD4, CCR7, CD27 | Helper T cells; orchestrate adaptive immunity |
| CD8 T cells | CD8A, CD8B, GZMK, GZMB | Cytotoxic T cells; kill infected/cancer cells |
| Naive T cells | CCR7, SELL, LEF1, TCF7 | Undifferentiated; circulate in blood |
| Memory T cells | IL7R, S100A4 | Long-lived; rapid response on re-exposure |
| Regulatory T cells | FOXP3, IL2RA, IKZF2 | Suppress immune responses |
| B cells | CD19, MS4A1, CD79A, CD79B | Produce antibodies |
| NK cells | GNLY, NKG7, GZMA, GZMB, PRF1 | Innate cytotoxic cells |
| CD14 Monocytes | CD14, LYZ, S100A8, S100A9 | Phagocytes; classical monocytes |
| FCGR3A Monocytes | FCGR3A, MS4A7 | Non-classical monocytes |
| Dendritic cells | FCER1A, CST3, CLEC10A | Antigen-presenting cells |
| Megakaryocytes | PPBP, PF4, GP1BA | Platelet precursors |
| Plasma cells | JCHAIN, IGHG1, MZB1 | Antibody-secreting B cells |

### What Log Fold-Change Means

In scRNA-seq differential expression, log fold-change is computed in log-normalized space:

```
LFC = mean_log_expr(group) − mean_log_expr(rest)
      ≈ log(mean_expr_group / mean_expr_rest)
```

A LFC of 1.0 means the gene is ~2.7× higher in the group (e**1 ≈ 2.7). A LFC of 2.0 means ~7.4× higher. Note that because scRNA-seq data uses natural log, these are not the same as log2 fold-changes common in bulk RNA-seq.

### Cell Cycle Interpretation

- Most cells should be in **G1** (quiescent or post-mitotic)
- High **S** or **G2M** fractions indicate a proliferating population (e.g., tumor cells, activated T cells, hematopoietic progenitors)
- If S/G2M cells form their own cluster, consider regressing out the cell cycle signal

### Quality Control Thresholds

These are general guidelines; always inspect your data's distribution:

| Metric | Typical Threshold | Notes |
|---|---|---|
| `n_genes_by_counts` min | 200–500 | Below = likely empty droplet |
| `n_genes_by_counts` max | 2,500–6,000 | Above = likely doublet |
| `total_counts` min | 500–1,000 | Below = low quality |
| `total_counts` max | 20,000–50,000 | Above = doublet or unusual cell |
| `pct_counts_MT-` max | 5–20% | Tissue-dependent |
| `doublet_score` | >0.3 | Flag for removal |

---

## 23. Performance Tips

### Memory Management

```python
# Use sparse matrices throughout (default for 10x data)
# Avoid .toarray() on large datasets

# Check memory usage
import sys
print(f"X matrix: {data.X.data.nbytes / 1e9:.2f} GB")

# Use HVG selection BEFORE scaling to avoid densifying the full matrix
pp.highly_variable_genes(data, n_top_genes=2000)
data.raw = data.copy()
pp.scale(data)  # only scales ~2000 HVGs; but note: still densifies

# For very large datasets (>100k cells), use zero_center=False
pp.scale(data, zero_center=False)  # preserves sparsity
```

### Speed

```python
# Wilcoxon test is slower than t-test for large datasets
# Use t-test for >50,000 cells
diff.rank_genes_groups(data, groupby='leiden', method='t-test')

# Truncate DE to top N genes per cluster
diff.rank_genes_groups(data, groupby='leiden', n_genes=200)

# Use n_jobs=-1 for parallelism where available (t-SNE, spectral clustering)
dim.run_tsne(data)  # uses n_jobs=-1 internally

# For very large datasets, subsample for visualization
sub = utils.subsample(data, n=50000, stratify='leiden', random_state=0)
vis.plot_umap(sub, color='leiden')
```

### Reproducibility

All functions that involve randomness accept a `random_state` parameter. Set this consistently:

```python
dim.run_pca(data, random_state=42)
dim.run_umap(data, random_state=42)
cl.cluster_leiden(data, random_state=42)
qc.scrublet(data, random_state=42)
```

---

## 24. Troubleshooting

### ImportError: leidenalg not found

```bash
pip install leidenalg igraph
```
If this fails on Apple Silicon: `conda install -c conda-forge leidenalg`

### UMAP not available

```bash
pip install umap-learn
```

### `ValueError: 'X_pca' not found in obsm`

Run PCA before neighbors/UMAP/clustering:
```python
dim.run_pca(data, n_components=50)
dim.neighbors(data, n_neighbors=15)
dim.run_umap(data)
```

### `ValueError: Column 'total_counts' not found`

Run QC metrics before filtering:
```python
pp.calculate_qc_metrics(data, qc_vars=['MT-'])
data = pp.filter_cells(data, ...)
```

### Scaling densifies my sparse matrix

Expected behavior. Use `zero_center=False` to avoid densification:
```python
pp.scale(data, zero_center=False, max_value=10)
```

### No marker genes found

Lower the thresholds:
```python
markers = diff.get_marker_genes(data, group='5', pval_cutoff=0.1, lfc_cutoff=0.25)
```

Or check that `rank_genes_groups` has been run:
```python
'rank_genes_groups' in data.uns  # should be True
```

### Leiden finds too many / too few clusters

Adjust resolution:
```python
cl.cluster_leiden(data, resolution=0.2)   # fewer clusters
cl.cluster_leiden(data, resolution=1.5)   # more clusters
```

### `__init__.py` ImportError for grn_inference

The current `__init__.py` imports `grn_inference`. Ensure all dependencies are installed:
```bash
pip install scikit-learn
```

---

## 25. API Quick Reference

### `sc_io`

| Function | Description |
|---|---|
| `read_10x_mtx(path)` | Read 10x CellRanger MTX directory |
| `read_csv(filename)` | Read CSV count matrix |
| `read_h5ad(filename)` | Read H5AD file |
| `write_h5ad(data, filename)` | Write H5AD file |
| `write_csvs(data, prefix)` | Write X, obs, var as CSVs |
| `_make_unique(names)` | Disambiguate duplicate gene names |

### `preprocessing`

| Function | Description |
|---|---|
| `calculate_qc_metrics(data, qc_vars)` | Compute QC columns in obs |
| `filter_cells(data, ...)` | Remove low-quality cells |
| `filter_genes(data, min_cells)` | Remove rarely detected genes |
| `normalize_total(data, target_sum)` | Normalize to fixed total counts |
| `normalize_scran_pooling(data)` | scran-like pooling normalization |
| `normalize_sctransform(data)` | Pearson residual normalization |
| `log1p(data)` | Log(x+1) transform |
| `highly_variable_genes(data, n_top_genes)` | Select HVGs |
| `scale(data, max_value)` | Z-score scale |

### `quality_control`

| Function | Description |
|---|---|
| `scrublet(data, expected_doublet_rate)` | Detect doublets |
| `filter_doublets(data)` | Remove predicted doublets |
| `detect_outliers(data, metric, n_mads)` | Flag statistical outliers |

### `dimensionality`

| Function | Description |
|---|---|
| `run_pca(data, n_components)` | PCA |
| `neighbors(data, n_neighbors, n_pcs)` | kNN graph |
| `run_umap(data, min_dist)` | UMAP |
| `run_tsne(data, perplexity)` | t-SNE |
| `run_diffmap(data, n_components)` | Diffusion Map |
| `run_phate(data, n_components)` | PHATE |

### `clustering`

| Function | Description |
|---|---|
| `cluster_leiden(data, resolution)` | Leiden (recommended) |
| `cluster_louvain(data, resolution)` | Louvain |
| `cluster_kmeans(data, n_clusters)` | K-Means |
| `cluster_dbscan(data, eps)` | DBSCAN |
| `cluster_spectral(data, n_clusters)` | Spectral |
| `cluster_hierarchical(data, n_clusters)` | Hierarchical |
| `cluster_stats(data, cluster_key)` | Cluster summary statistics |

### `differential`

| Function | Description |
|---|---|
| `rank_genes_groups(data, groupby, method)` | Run DE analysis |
| `get_marker_genes(data, group, ...)` | Extract significant markers |

### `cell_cycle`

| Function | Description |
|---|---|
| `score_cell_cycle(data, organism)` | Assign S/G2M/G1 phase |
| `score_genes(data, gene_list)` | Score arbitrary gene set |
| `regress_out_cell_cycle(data)` | Remove cell cycle signal |

### `batch_correction`

| Function | Description |
|---|---|
| `combat(data, batch_key)` | ComBat batch correction |
| `harmony_integrate(data, batch_key)` | Harmony embedding correction |
| `mnn_correct(datasets, k)` | Mutual nearest neighbors |

### `enrichment`

| Function | Description |
|---|---|
| `gene_set_score(data, gene_list, score_name)` | Score one gene set |
| `score_multiple_gene_sets(data, gene_sets)` | Score multiple gene sets |
| `rank_genes_groups_by_enrichment(data, gene_sets, groupby)` | Cluster enrichment |
| `gsea_preranked(ranked_genes, gene_set)` | GSEA |
| `load_gene_sets(source, categories)` | Load MSigDB sets (placeholder) |

### `trajectory`

| Function | Description |
|---|---|
| `select_root_cell(data, cluster_key, root_cluster)` | Pick pseudotime root |
| `diffusion_pseudotime(data, root_cell)` | Compute DPT |
| `gene_trends(data, genes, pseudotime_key)` | Expression along pseudotime |

### `imputation`

| Function | Description |
|---|---|
| `impute_wnid(data, k, dropout_thresh)` | Dropout-targeted kNN imputation |
| `impute_knn_smooth(data, k)` | Full kNN smoothing |
| `impute_diffusion(data, t)` | Diffusion-based imputation |
| `dropout_stats(data)` | Report dropout statistics |
| `compare_imputation(before, after)` | Compare pre/post imputation |

### `visualization`

| Function | Description |
|---|---|
| `plot_umap(data, color, ...)` | UMAP scatter plot |
| `plot_tsne(data, color, ...)` | t-SNE scatter plot |
| `plot_pca(data, color, ...)` | PCA scatter plot |
| `plot_embedding(data, basis, color, ...)` | Any embedding |
| `plot_violin(data, keys, groupby)` | Violin plot |
| `plot_heatmap(data, var_names, groupby)` | Heatmap of mean expression |
| `plot_dotplot(data, var_names, groupby)` | Dot plot |
| `volcano_plot(data, group)` | Volcano plot |
| `plot_qc_violin(data, metrics)` | QC violin panel |
| `plot_highest_expr_genes(data, n_top)` | Top expressed genes |

### `interactive_viz`

| Function | Description |
|---|---|
| `interactive_embedding(data, basis, color)` | Interactive 2D scatter |
| `interactive_3d_embedding(data, basis)` | Interactive 3D scatter |
| `interactive_heatmap(data, var_names, groupby)` | Interactive heatmap |
| `interactive_violin(data, keys, groupby)` | Interactive violin |

### `utils`

| Function | Description |
|---|---|
| `merge(datasets, join)` | Concatenate datasets |
| `subsample(data, n, stratify)` | Random subsampling |
| `filter_obs(data, mask)` | Filter cells |
| `filter_var(data, mask)` | Filter genes |
| `rename_obs(data, mapping)` | Rename obs columns |
| `rename_var(data, mapping)` | Rename gene names |
| `get_mean_var(data, axis)` | Compute mean and variance |
| `describe_obs(data, col)` | Summarize obs column |

### `grn_inference`

| Function | Description |
|---|---|
| `infer_grn_ridge(data, tf_list, top_n_edges)` | Ridge regression GRN |

---

*scAnalyzer v0.2.1 — © 2026 Ayyuce Demirbas*
