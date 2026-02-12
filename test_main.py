import os
import shutil
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import clustering

# Import our toolkit
import core
import differential
import dimensionality
import preprocessing
import sc_io as io
import visualization

# --- Configuration ---
DATA_URL = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
DATA_DIR = "./data"
FILENAME = "pbmc3k.tar.gz"
EXTRACT_DIR = "filtered_gene_bc_matrices/hg19"


def download_and_extract_data():
    """
    Downloads the PBMC 3k dataset from 10x Genomics and extracts it.
    Includes User-Agent headers to prevent HTTP 403 Forbidden errors.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    filepath = os.path.join(DATA_DIR, FILENAME)

    # 1. Download
    if not os.path.exists(filepath):
        print(f"Downloading PBMC 3k dataset from {DATA_URL}...")
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with (
                urllib.request.urlopen(req) as response,
                open(filepath, "wb") as out_file,
            ):
                shutil.copyfileobj(response, out_file)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            return None
    else:
        print("Dataset already downloaded.")

    # 2. Extract
    extract_path = os.path.join(DATA_DIR, "pbmc3k_extracted")
    final_path = os.path.join(extract_path, EXTRACT_DIR)

    if not os.path.exists(final_path):
        print("Extracting files...")
        try:
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=extract_path)
            print("Extraction complete.")
        except Exception as e:
            print(f"Extraction failed: {e}")
            return None

    return final_path


def main():
    print("=== Single Cell Analysis Pipeline: PBMC 3k (Real Data) ===")

    # 1. Load Data
    data_path = download_and_extract_data()
    if data_path is None:
        return

    print(f"Loading 10x data from {data_path}...")
    adata = io.read_10x_mtx(data_path)
    adata.var.index = io._make_unique(adata.var.index.values)

    print(f"Data Loaded: {adata}")
    print(f"Memory usage: {adata.X.data.nbytes / 1024**2:.2f} MB")

    # 2. Preprocessing
    print("\n--- Preprocessing ---")
    preprocessing.calculate_qc_metrics(adata, qc_vars=["MT-"])
    adata = preprocessing.filter_cells(
        adata, min_genes=200, max_genes=2500, max_pct_mito=5.0
    )
    adata = preprocessing.filter_genes(adata, min_cells=3)
    preprocessing.normalize_total(adata, target_sum=1e4)
    preprocessing.log1p(adata)
    preprocessing.highly_variable_genes(adata, n_top_genes=2000)
    adata.raw = adata.copy()
    preprocessing.scale(adata, max_value=10)

    # 3. Dimensionality Reduction
    print("\n--- Dimensionality Reduction ---")
    dimensionality.run_pca(adata, n_components=50)
    dimensionality.neighbors(adata, n_neighbors=10, n_pcs=40)
    try:
        dimensionality.run_umap(adata, min_dist=0.3)
        has_umap = True
        print("UMAP computed.")
    except ImportError:
        print("UMAP not installed. Skipping.")
        has_umap = False

    # 4. Clustering
    print("\n--- Clustering ---")
    try:
        clustering.cluster_leiden(adata, resolution=0.5, key_added="leiden")
        cluster_key = "leiden"
        print("Leiden clustering complete.")
    except ImportError:
        print("Leiden not found, trying Louvain...")
        try:
            clustering.cluster_louvain(adata, resolution=0.5, key_added="louvain")
            cluster_key = "louvain"
        except ImportError:
            print("Graph clustering not found. Using K-Means...")
            clustering.cluster_kmeans(adata, n_clusters=8, key_added="kmeans")
            cluster_key = "kmeans"

    print(f"Clusters found: {adata.obs[cluster_key].unique().tolist()}")

    # 5. Differential Expression
    print("\n--- Differential Expression ---")
    differential.rank_genes_groups(adata, groupby=cluster_key, method="t-test")

    first_cluster = sorted(adata.obs[cluster_key].unique())[0]
    top_markers = differential.get_marker_genes(adata, group=first_cluster)
    print(f"Top markers for Cluster {first_cluster}:")
    print(top_markers[["names", "scores", "logfoldchanges"]].head(5))

    # 6. Visualization
    print("\n--- Visualization ---")

    if has_umap:
        # Save UMAP of Clusters
        visualization.plot_umap(
            adata,
            color=cluster_key,
            title="PBMC 3k Clustering",
            figsize=(7, 7),
            legend_loc="on data",
            save="umap_clusters.png",  # <--- Saving here
        )

        # Save UMAP of specific marker
        if "CD3E" in adata.var.index:
            visualization.plot_umap(
                adata,
                color="CD3E",
                title="CD3E Expression (T Cells)",
                cmap="Reds",
                save="umap_CD3E.png",  # <--- Saving here
            )

    # Save Dot Plot
    print("Generating Dot Plot for canonical markers...")
    marker_genes = ["MS4A1", "GNLY", "CD3E", "CD14", "FCGR3A", "FCER1A", "CD8A", "CST3"]
    try:
        visualization.plot_dotplot(
            adata,
            var_names=marker_genes,
            groupby=cluster_key,
            standard_scale=True,
            save="dotplot_markers.png",  # <--- Saving here
        )
    except Exception as e:
        print(f"Could not generate dotplot: {e}")

    # 7. Save Data
    output_file = "pbmc3k_processed.h5ad"
    print(f"\nSaving results to {output_file}...")
    io.write_h5ad(adata, output_file)
    print("Analysis Complete.")


if __name__ == "__main__":
    main()
