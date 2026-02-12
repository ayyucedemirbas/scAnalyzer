# API Reference

## Core (`core.py`)
### `SingleCellDataset`
The main container for data.
* **Attributes**:
    * `X`: The expression matrix (Cells × Genes).
    * `obs`: DataFrame for cell metadata.
    * `var`: DataFrame for gene metadata.
    * `obsm`: Dictionary for multi-dimensional cell annotations (e.g., PCA coordinates).
    * `varm`: Dictionary for multi-dimensional gene annotations (e.g., PC loadings).
    * `uns`: Unstructured metadata.

---

## Preprocessing (`preprocessing.py`)
* `calculate_qc_metrics(data, qc_vars=['MT-'])`: Computes n_genes, total_counts, and pct_mito.
* `filter_cells(data, min_genes, max_genes, max_pct_mito)`: Removes low-quality cells.
* `filter_genes(data, min_cells)`: Removes genes expressed in too few cells.
* `normalize_total(data, target_sum=1e4)`: Normalizes library size.
* `log1p(data)`: Log-transforms the data matrix.
* `highly_variable_genes(data, n_top_genes=2000)`: Selects highly variable genes.
* `scale(data, max_value=10)`: Scales data to unit variance and zero mean.

---

## Dimensionality Reduction (`dimensionality.py`)
* `run_pca(data, n_components=50)`: Computes PCA.
* `neighbors(data, n_neighbors=15)`: Computes k-Nearest Neighbors graph.
* `run_umap(data)`: Computes UMAP embedding (requires `umap-learn`).
* `run_tsne(data)`: Computes t-SNE embedding.

---

## Clustering (`clustering.py`)
* `cluster_leiden(data, resolution=1.0)`: Leiden graph clustering (Standard).
* `cluster_louvain(data, resolution=1.0)`: Louvain graph clustering.
* `cluster_kmeans(data, n_clusters)`: K-Means clustering.
* `cluster_dbscan(data, eps=0.5)`: DBSCAN density clustering.

---

## Differential Expression (`differential.py`)
* `rank_genes_groups(data, groupby, method='t-test')`: Identifies differentially expressed genes.
* `get_marker_genes(data, group)`: Returns a filtered DataFrame of markers for a specific group.

---

## Visualization (`visualization.py`)
All plotting functions support a `save='filename.png'` argument.
* `plot_umap(data, color)`: Scatter plot of UMAP embeddings.
* `plot_tsne(data, color)`: Scatter plot of t-SNE embeddings.
* `plot_violin(data, keys, groupby)`: Violin plot of expression distributions.
* `plot_dotplot(data, var_names, groupby)`: Dot plot of expression frequency and intensity.
* `plot_heatmap(data, var_names, groupby)`: Heatmap of mean expression.

---

## Input/Output (`sc_io.py`)
* `read_10x_mtx(path)`: Reads 10x Genomics directory.
* `read_h5ad(filename)`: Reads H5AD files.
* `write_h5ad(data, filename)`: Writes H5AD files.
* `read_csv(filename)`: Reads dense CSV matrix.