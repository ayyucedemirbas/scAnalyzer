# Analysis Tutorial: PBMC 3k

This tutorial guides you through the analysis of 2,700 Peripheral Blood Mononuclear Cells (PBMCs) from 10x Genomics.

## 1. Quality Control
We calculate the percentage of mitochondrial genes. High mitochondrial content often indicates dying cells.
* **Filter**: Remove cells with >5% mitochondrial counts.
* **Filter**: Remove cells with <200 detected genes (empty droplets).

## 2. Normalization
Single-cell data is sparse and library sizes vary.
* We normalize counts so every cell has 10,000 counts.
* We apply `log(x+1)` transformation to stabilize variance.

## 3. Dimensionality Reduction
* **PCA**: Reduces the data to 50 principal components to remove noise.
* **Neighbors**: We build a kNN graph in PCA space (k=10).
* **UMAP**: A non-linear embedding to visualize the global structure.

## 4. Clustering & Marker Genes
We use **Leiden clustering** to identify cell populations.
Typical Markers found in this dataset:
* **IL7R**: CD4 T-cells
* **CD8A**: CD8 T-cells
* **MS4A1**: B-cells
* **GNLY**: NK cells
* **CD14**: CD14+ Monocytes
* **FCGR3A**: FCGR3A+ Monocytes