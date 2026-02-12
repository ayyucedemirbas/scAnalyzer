# scAnalyzer: A Single-Cell Analysis Toolkit

A Python toolkit for single-cell RNA sequencing (scRNA-seq) analysis.

🚧 Warning this project is under heavy development and not ready for production. ABI changes can happen frequently until reach stable version 🚧


<p align="center">
  <img alt="GitHub" src="https://img.shields.io/github/license/ayyucedemirbas/scAnalyzer">
  <img alt="Black" src="https://img.shields.io/badge/code%20style-black-black"/>
  <img alt="isort" src="https://img.shields.io/badge/isort-checked-yellow"/>
</p>


<p align="center">
<a href="https://pypi.org/project/scAnalysis/" target="_blank">
    <img src="https://img.shields.io/pypi/v/scAnalysis?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
</p>


pip install scAnalysis

## 🚀 Features

* **Core Data Structure**: `SingleCellDataset` (AnnData-like) for efficient handling of sparse matrices and metadata.
* **Preprocessing**: QC metrics, filtering (cells/genes), normalization, log-transformation, and highly variable gene (HVG) selection.
* **Dimensionality Reduction**: PCA, t-SNE, and UMAP implementations.
* **Clustering**: Graph-based (Leiden, Louvain), geometric (K-Means, Hierarchical), and density-based (DBSCAN) clustering.
* **Differential Expression**: Statistical testing (T-test, Wilcoxon) to identify marker genes.
* **Visualization**: Publication-ready plots (UMAP, t-SNE, Violin, Dotplot, Heatmap).
* **I/O**: Support for 10x Genomics (`.mtx`), H5AD (`.h5ad`), and CSV formats.

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/demirbasayyuce/scAnalyzer.git](https://github.com/demirbasayyuce/scAnalyzer.git)
cd sc_analysis
pip install -r requirements.txt

## ⚡ Quick Start

Here is a minimal example of how to run a full analysis pipeline:

```python
import sc_io as io
import preprocessing as pp
import dimensionality as dim
import clustering as cl
import visualization as vis

# 1. Load Data
data = io.read_10x_mtx('./data/pbmc3k/')

# 2. Preprocess
pp.filter_cells(data, min_genes=200, max_pct_mito=5.0)
pp.normalize_total(data)
pp.log1p(data)
pp.highly_variable_genes(data, n_top_genes=2000)
pp.scale(data)

# 3. Embed & Cluster
dim.run_pca(data)
dim.neighbors(data)
dim.run_umap(data)
cl.cluster_leiden(data, resolution=0.5)

# 4. Visualize
vis.plot_umap(data, color='leiden', save='umap_clusters.png')

```

## 📂 Project Structure

* `core.py`: Main data structure (`SingleCellDataset`).
* `preprocessing.py`: Filtering, normalization, and scaling functions.
* `dimensionality.py`: PCA, Neighborhood Graph, t-SNE, UMAP.
* `clustering.py`: Community detection algorithms.
* `differential.py`: Marker gene identification.
* `visualization.py`: Plotting functions.
* `sc_io.py`: Input/Output handlers.
* `utils.py`: Helpers for merging and subsampling.

## 🧪 Running Tests

The project includes a comprehensive suite of unit tests. Run them using:

```bash
python -m unittest discover test

```

## 📄 License

MIT License.

```

