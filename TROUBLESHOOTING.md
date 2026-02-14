# Troubleshooting Guide

## Common Issues and Solutions

### 1. "n_components must be <= n_features" Error

**Problem**: Scrublet tries to compute more PCs than available genes.

**Cause**: After filtering for variable genes, not enough genes remain.

**Solutions**:
```python
# Option 1: Use fewer PCs
qc.scrublet(data, n_prin_comps=20)  # Default is 30

# Option 2: Be more lenient with gene selection
qc.scrublet(data, min_gene_variability_pctl=75)  # Default is 85

# Option 3: Use both
qc.scrublet(
    data, 
    n_prin_comps=20,
    min_gene_variability_pctl=75
)
```

**When to adjust**:
- Small datasets (<1000 genes): Use `n_prin_comps=10-20`
- Synthetic/simulated data: Use `min_gene_variability_pctl=70-80`
- Real datasets: Default parameters usually work fine

---

### 2. "Cannot cast ufunc 'multiply'" Error

**Problem**: Trying to multiply integer array by float in-place.

**Solution**: This is fixed in the updated code. Make sure you have the latest version of the files.

---

### 3. Scrublet Detects Too Many/Few Doublets

**Problem**: Doublet detection is too sensitive or not sensitive enough.

**Cause**: Incorrect expected doublet rate for your protocol.

**Solutions**:
```python
# 10x Chromium rates (approximate):
# - 3,000 cells: ~2.5%
# - 5,000 cells: ~4%
# - 10,000 cells: ~8%
# - 20,000 cells: ~16%

# Adjust based on your capture rate
qc.scrublet(data, expected_doublet_rate=0.05)  # For ~5000 cells
```

**Rule of thumb**: ~0.8% doublet rate per 1000 cells captured.

---

### 4. Harmony Doesn't Mix Batches Well

**Problem**: Batches still cluster separately after Harmony.

**Solutions**:
```python
# Increase diversity penalty (theta)
bc.harmony_integrate(data, batch_key='batch', theta=3.0)  # Default is 2.0

# Use more PCs
dim.run_pca(data, n_components=100)  # Default is 50
bc.harmony_integrate(data, batch_key='batch')

# Try more iterations
bc.harmony_integrate(
    data, 
    batch_key='batch',
    max_iter_harmony=20  # Default is 10
)
```

---

### 5. Cell Cycle Genes Not Found

**Problem**: "Warning: X/Y genes not found in dataset"

**Cause**: Gene names don't match (human vs mouse, or different naming conventions).

**Solutions**:
```python
# For human data with gene symbols
cc.score_cell_cycle(data, organism='human')

# For mouse data
cc.score_cell_cycle(data, organism='mouse')

# If using Ensembl IDs, convert first
# Or provide custom gene lists
s_genes = ['ENSG00000...', 'ENSG00000...']  # Your S phase genes
g2m_genes = ['ENSG00000...', 'ENSG00000...']  # Your G2M genes
cc.score_cell_cycle(data, s_genes=s_genes, g2m_genes=g2m_genes)
```

---

### 6. Interactive Plots Don't Show

**Problem**: Plotly plots don't display or save.

**Solutions**:
```bash
# Install plotly
pip install plotly

# For Jupyter notebooks
pip install jupyterlab ipywidgets

# Then restart your kernel
```

**In scripts**:
```python
# Always use save_html parameter
iviz.interactive_embedding(
    data,
    color='leiden',
    save_html='my_plot.html'  # Will save to file
)
# Then open the HTML file in your browser
```

---

### 7. Out of Memory Errors

**Problem**: Process killed or memory error.

**Solutions**:
```python
# 1. Subsample your data
from utils import subsample
data_subset = subsample(data, n=5000)  # Use 5000 cells

# 2. Use fewer PCs
dim.run_pca(data, n_components=30)  # Instead of 50

# 3. Filter genes more aggressively
data = pp.filter_genes(data, min_cells=10)  # Instead of 3

# 4. For Harmony, use fewer iterations
bc.harmony_integrate(data, max_iter_harmony=5)
```

---

### 8. Import Errors

**Problem**: "ModuleNotFoundError: No module named 'quality_control'"

**Solutions**:
```bash
# Check all files are in the same directory
ls -la *.py

# Verify you're running from the correct directory
pwd

# Run setup check
python check_setup.py
```

**Common causes**:
- Files in wrong directory
- Running from parent directory instead of scAnalyzer directory
- Typo in import statement

---

### 9. Leiden/Louvain Not Available

**Problem**: "Please install 'leidenalg' and 'python-igraph'"

**Solutions**:
```bash
# Install both packages
pip install leidenalg python-igraph

# Or use conda
conda install -c conda-forge leidenalg python-igraph

# If that fails, use alternative clustering
# The code will automatically fall back to K-Means
```

---

### 10. UMAP Not Available

**Problem**: "umap-learn is not installed"

**Solutions**:
```bash
# Install UMAP
pip install umap-learn

# Or with conda
conda install -c conda-forge umap-learn

# If installation fails, skip UMAP
# You can still use t-SNE for visualization
```

---

## Parameter Quick Reference

### Scrublet Parameters

| Parameter | Default | Typical Range | When to Adjust |
|-----------|---------|---------------|----------------|
| `expected_doublet_rate` | 0.06 | 0.03-0.20 | Based on capture rate |
| `sim_doublet_ratio` | 2.0 | 1.0-5.0 | More doublets = better null distribution |
| `n_prin_comps` | 30 | 10-50 | Small datasets = fewer PCs |
| `min_gene_variability_pctl` | 85 | 70-95 | Synthetic data = lower threshold |
| `n_neighbors` | 30 | 10-50 | Affects local density calculation |

### Harmony Parameters

| Parameter | Default | Typical Range | When to Adjust |
|-----------|---------|---------------|----------------|
| `theta` | 2.0 | 1.0-5.0 | Higher = more batch diversity |
| `max_iter_harmony` | 10 | 5-20 | If not converging |
| `sigma` | 0.1 | 0.01-1.0 | Cluster width parameter |
| `lamb` | 1.0 | 0.1-10.0 | Ridge penalty strength |

### Cell Cycle Parameters

| Parameter | Default | When to Use |
|-----------|---------|-------------|
| `organism='human'` | - | Human samples |
| `organism='mouse'` | - | Mouse samples |
| `difference_only=True` | - | Keep proliferation signal |
| `difference_only=False` | - | Remove all cell cycle |

---

## Best Practices

### 1. Order of Operations

Correct order for analysis:

```python
# 1. Load data
data = io.read_10x_mtx('./data/')

# 2. QC and doublet detection (BEFORE normalization)
pp.calculate_qc_metrics(data, qc_vars=['MT-'])
qc.scrublet(data)
data = qc.filter_doublets(data)

# 3. Filter cells and genes
data = pp.filter_cells(data, min_genes=200)
data = pp.filter_genes(data, min_cells=3)

# 4. Normalize
pp.normalize_total(data)
pp.log1p(data)

# 5. Cell cycle (AFTER normalization, BEFORE scaling)
cc.score_cell_cycle(data)

# 6. HVG selection and scaling
pp.highly_variable_genes(data)
data.raw = data.copy()
pp.scale(data)

# 7. Dimensionality reduction
dim.run_pca(data)

# 8. Batch correction (if needed)
bc.harmony_integrate(data, batch_key='batch')

# 9. Neighbors and UMAP
dim.neighbors(data)
dim.run_umap(data)

# 10. Clustering
cl.cluster_leiden(data)

# 11. Differential expression
diff.rank_genes_groups(data, groupby='leiden')

# 12. Visualization
vis.plot_umap(data, color='leiden')
```

### 2. When to Use Which Feature

**Doublet Detection**: Always recommended, especially for:
- 10x Chromium data
- High-throughput protocols
- >5000 cells captured

**Cell Cycle Scoring**: Use when:
- Cell cycle dominates your clustering
- You see cycling vs non-cycling split
- Studying proliferation is NOT your goal

**Batch Correction**: Use when:
- Multiple batches/samples
- Batch effects visible in UMAP
- Want to compare across conditions

**Gene Set Enrichment**: Use when:
- Interpreting clusters biologically
- Testing specific hypotheses
- Identifying cell states

---

## Getting More Help

1. Check `IMPLEMENTATION_SUMMARY.md` for detailed docs
2. See `extended_analysis_example.py` for complete workflow
3. Run `test_new_features.py` to test individually
4. Open an issue on GitHub with:
   - Error message
   - Minimal code to reproduce
   - Dataset size (n_cells, n_genes)
   - Python/package versions

---

## Useful Diagnostic Commands

```python
# Check dataset info
print(data)

# Check what's computed
print("PCA:", "X_pca" in data.obsm)
print("UMAP:", "X_umap" in data.obsm)
print("Clusters:", "leiden" in data.obs.columns)

# Check column names
print(data.obs.columns)
print(data.var.columns)

# Check for missing values
print("Missing in obs:", data.obs.isnull().sum())
print("Missing in var:", data.var.isnull().sum())

# Memory usage
import sys
print(f"Data size: {sys.getsizeof(data.X) / 1024**2:.1f} MB")

# Gene names sample
print("First 10 genes:", data.var.index[:10].tolist())

# Cell metadata sample
print(data.obs.head())
```
