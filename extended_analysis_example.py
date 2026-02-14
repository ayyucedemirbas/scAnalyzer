"""
Example with PBMC 3k (Clean Single-Sample Pipeline)
"""

import numpy as np
import pandas as pd
import os
import shutil
import tarfile
import urllib.request
import sys

# Import core modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import SingleCellDataset
import preprocessing as pp
import dimensionality as dim
import clustering as cl
import differential as diff
import visualization as vis
import sc_io as io

# Import extended modules
import quality_control as qc
import cell_cycle as cc
import enrichment as enrich
import interactive_viz as iviz


# Dataset configuration
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

    if not os.path.exists(filepath):
        print(f"Downloading PBMC 3k dataset from 10x Genomics...")
        print(f"URL: {DATA_URL}")
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
    else:
        print("Dataset already extracted.")

    return final_path


def main():
    print("Extended scAnalyzer Pipeline: PBMC 3k Dataset")
    
    # 1. Load Data
    data_path = download_and_extract_data()
    if data_path is None:
        print("\nFailed to download/extract data. Exiting.")
        return None
    
    print(f"\nLoading 10x data from {data_path}...")
    data = io.read_10x_mtx(data_path)
    
    # Ensure unique gene names
    data.var.index = io._make_unique(data.var.index.values)
    
    print(f"Data Loaded: {data}")
    print(f"  Memory usage: {data.X.data.nbytes / 1024**2:.2f} MB")
    
    # 2. Quality Control & Filtering
    # Calculate basic metrics (n_genes, total_counts, percent_mito)
    pp.calculate_qc_metrics(data, qc_vars=['MT-'])
    
    print("\nQC Summary (before filtering):")
    print(f"  Cells: {data.n_obs}")
    print(f"  Genes: {data.n_vars}")
    print(f"  Mean genes per cell: {data.obs['n_genes_by_counts'].mean():.0f}")
    print(f"  Mean counts per cell: {data.obs['total_counts'].mean():.0f}")
    print(f"  Mean MT%: {data.obs['pct_counts_MT-'].mean():.2f}%")
    
    # Doublet Detection
    print("\nDetecting doublets with Scrublet...")
    qc.scrublet(
        data, 
        expected_doublet_rate=0.06,  # ~6% for 3k cells
        n_prin_comps=30,
        verbose=True
    )
    
    print("\nDoublet detection summary:")
    print(f"  Detected doublets: {data.obs['predicted_doublet'].sum()} ({data.obs['predicted_doublet'].mean()*100:.1f}%)")
    print(f"  Mean doublet score: {data.obs['doublet_score'].mean():.3f}")
    
    # Filter Cells
    # Standard PBMC cutoffs: <5% Mito, >200 Genes, <2500 Genes
    data = pp.filter_cells(data, min_genes=200, max_genes=2500, max_pct_mito=5.0)
    data = pp.filter_genes(data, min_cells=3)
    
    print(f"After filtering:")
    print(f"  Cells: {data.n_obs}")
    print(f"  Genes: {data.n_vars}")
    
    pp.normalize_total(data, target_sum=1e4)
    pp.log1p(data)
    print("Data normalized and log-transformed")
    
    cc.score_cell_cycle(data, organism='human')
    print("\nCell cycle phase distribution:")
    print(data.obs['phase'].value_counts())
    
    pp.highly_variable_genes(data, n_top_genes=2000)
    
    data.raw = data.copy()
    
    # Scale data (centers mean to 0, scales variance to 1) for PCA
    pp.scale(data, max_value=10)
    print(f"Selected {data.var['highly_variable'].sum()} highly variable genes")
    print("Data scaled")
    
    # Dimensionality Reduction (PCA)
    dim.run_pca(data, n_components=50)
    print(f"PCA computed: {data.obsm['X_pca'].shape}")
    
    # Variance explained stats
    var_ratio = data.uns['pca']['variance_ratio']
    print(f"  Variance explained by PC1-10: {var_ratio[:10].sum()*100:.1f}%")
    print(f"  Variance explained by PC1-50: {var_ratio[:50].sum()*100:.1f}%")
    
    # Neighborhood Graph & UMAP
    dim.neighbors(data, n_neighbors=10, n_pcs=40)
    print(" Neighbor graph computed")
    
    try:
        dim.run_umap(data, min_dist=0.3)
        has_umap = True
        print("UMAP computed")
    except ImportError:
        print("UMAP not installed. Skipping UMAP.")
        has_umap = False
    
    # PCA
    try:
        cl.cluster_leiden(data, resolution=0.5, key_added='leiden')
        cluster_key = 'leiden'
        print(f"Leiden clustering complete")
    except ImportError:
        print("Leiden not available, trying Louvain...")
        try:
            cl.cluster_louvain(data, resolution=0.5, key_added='louvain')
            cluster_key = 'louvain'
            print(f"Louvain clustering complete")
        except ImportError:
            print("Graph clustering not found. Using K-Means...")
            cl.cluster_kmeans(data, n_clusters=8, key_added='kmeans')
            cluster_key = 'kmeans'
            print(f"K-Means clustering complete")
    
    n_clusters = len(data.obs[cluster_key].unique())
    print(f"  Found {n_clusters} clusters")
    print(f"  Cluster sizes:\n{data.obs[cluster_key].value_counts().sort_index()}")
    
    # Note: We use 'use_raw=True' to perform stats on the unscaled (log-normalized) data
    diff.rank_genes_groups(data, groupby=cluster_key, method='t-test', use_raw=True)
    
    # Show top markers for first cluster
    first_cluster = sorted(data.obs[cluster_key].unique())[0]
    top_markers = diff.get_marker_genes(data, group=first_cluster, pval_cutoff=0.05, lfc_cutoff=0.5)
    
    print(f"\nTop 10 markers for cluster {first_cluster}:")
    if len(top_markers) > 0:
        print(top_markers[['names', 'scores', 'logfoldchanges', 'pvals_adj']].head(10))
    else:
        print("  No significant markers found with current thresholds")
    
    # Define PBMC cell type marker gene sets
    pbmc_markers = {
        'T_cell': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD4'],
        'B_cell': ['CD19', 'MS4A1', 'CD79A', 'CD79B'],
        'NK': ['GNLY', 'NKG7', 'GZMA', 'GZMB', 'PRF1'],
        'Monocyte': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'FCGR3A'],
        'DC': ['FCER1A', 'CST3', 'CLEC10A'],
    }
    
    print("Scoring cell type marker gene sets...")
    enrich.score_multiple_gene_sets(data, pbmc_markers)
    
    print("\nTesting enrichment of cell type signatures in cluster markers...")
    enrichment_results = enrich.rank_genes_groups_by_enrichment(
        data, pbmc_markers, groupby=cluster_key
    )
    
    print(f"\nEnrichment results for cluster {first_cluster}:")
    if first_cluster in enrichment_results and len(enrichment_results[first_cluster]) > 0:
        print(enrichment_results[first_cluster][['gene_set', 'overlap_size', 'fold_enrichment', 'pval_adj']].head())
    
    if has_umap:
        print("\nGenerating UMAP plots...")
        
        # Clusters
        vis.plot_umap(data, color=cluster_key, title='PBMC 3k - Clusters', 
                     save='pbmc_umap_clusters.png', legend_loc='right margin')
        
        # Doublet scores
        vis.plot_umap(data, color='doublet_score', title='Doublet Score', 
                     cmap='Reds', save='pbmc_umap_doublet.png')
        
        # Cell cycle
        vis.plot_umap(data, color='phase', title='Cell Cycle Phase', 
                     save='pbmc_umap_phase.png', legend_loc='right margin')
        
        # Gene expression examples
        marker_genes = ['CD3D', 'MS4A1', 'CD14', 'GNLY']
        available_markers = [g for g in marker_genes if g in data.var.index]
        
        for gene in available_markers[:2]:  # Plot first 2 available
            vis.plot_umap(data, color=gene, title=f'{gene} Expression', 
                         cmap='Reds', save=f'pbmc_umap_{gene}.png')
        
        print(f"  ✓ Saved {2 + len(available_markers[:2])} UMAP plots")
    
    # QC violin plots (Grouped by Cluster now, since 'batch' was removed)
    print("\nGenerating QC plots...")
    vis.plot_qc_violin(
        data,
        metrics=['n_genes_by_counts', 'total_counts', 'pct_counts_MT-'],
        groupby=cluster_key, # Changed from 'batch' to cluster_key
        save='pbmc_qc_violin.png'
    )
    print(" Saved QC violin plot")
    
    # Volcano plot for differential expression
    vis.volcano_plot(
        data,
        group=first_cluster,
        pval_threshold=0.05,
        lfc_threshold=0.5,
        top_n_genes=10,
        save=f'pbmc_volcano_cluster{first_cluster}.png'
    )
    print(f"Saved volcano plot for cluster {first_cluster}")
    
    # Top expressed genes
    vis.plot_highest_expr_genes(data, n_top=20, save='pbmc_top_genes.png')
    print("  Saved top genes plot")
    
    # Dotplot of canonical markers
    canonical_markers = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 
                         'LYZ', 'CD14', 'LGALS3', 'S100A8', 'GNLY', 
                         'NKG7', 'KLRB1', 'FCGR3A', 'MS4A7', 'FCER1A', 
                         'CST3', 'PPBP']
    available_markers = [g for g in canonical_markers if g in data.var.index]
    
    if len(available_markers) > 0:
        vis.plot_dotplot(
            data,
            var_names=available_markers,
            groupby=cluster_key,
            standard_scale=True,
            save='pbmc_dotplot_markers.png'
        )
        print(f" Saved dotplot with {len(available_markers)} markers")
    
    try:
        if has_umap:
            # Interactive UMAP
            print("\nGenerating interactive UMAP...")
            iviz.interactive_embedding(
                data,
                basis='X_umap',
                color=cluster_key,
                hover_data=['n_genes_by_counts', 'total_counts', 'doublet_score', 'phase'],
                title='PBMC 3k Interactive UMAP',
                save_html='pbmc_interactive_umap.html'
            )
            print("  Saved interactive UMAP")
            
            # 3D PCA
            iviz.interactive_3d_embedding(
                data,
                basis='X_pca', # Reverted to standard PCA
                color=cluster_key,
                dimensions=[0, 1, 2],
                save_html='pbmc_interactive_3d_pca.html'
            )
            print(" Saved 3D PCA")
        
        # Interactive heatmap
        print("Generating interactive heatmap...")
        if len(available_markers) > 0:
            iviz.interactive_heatmap(
                data,
                var_names=available_markers[:15],  # Use first 15
                groupby=cluster_key,
                use_raw=True,
                save_html='pbmc_interactive_heatmap.html'
            )
            print(" Saved interactive heatmap")
        
        print("\n Interactive plots created! Open the HTML files in a browser.")
        
    except ImportError:
        print("\nPlotly not installed. Skipping interactive visualizations.")
        print("  Install with: pip install plotly")
    except Exception as e:
        print(f"\nCould not create interactive plots: {e}")
    
    # 13. Summary & Save
    print(f"\nFinal dataset summary:")
    print(f"  Cells: {data.n_obs}")
    print(f"  Genes: {data.n_vars}")
    print(f"  Clusters: {n_clusters}")
    
    print(f"\nCell cycle phase distribution:")
    for phase in ['G1', 'S', 'G2M']:
        count = (data.obs['phase'] == phase).sum()
        pct = 100 * count / data.n_obs
        print(f"  {phase}: {count} cells ({pct:.1f}%)")
    
    print(f"\nDoublet detection:")
    print(f"  Predicted doublets: {data.obs['predicted_doublet'].sum()} ({data.obs['predicted_doublet'].mean()*100:.1f}%)")
    
    print("\nGenerated files:")
    files_created = [
        'pbmc_umap_clusters.png',
        'pbmc_umap_doublet.png',
        'pbmc_umap_phase.png',
        'pbmc_qc_violin.png',
        f'pbmc_volcano_cluster{first_cluster}.png',
        'pbmc_top_genes.png',
        'pbmc_dotplot_markers.png',
        'pbmc_interactive_umap.html',
        'pbmc_interactive_3d_pca.html',
        'pbmc_interactive_heatmap.html'
    ]
    
    for f in files_created:
        if os.path.exists(f):
            print(f"  {f}")
    
    io.write_h5ad(data, 'pbmc3k_processed.h5ad')
    print(" Saved to pbmc3k_processed.h5ad")
    
    return data


if __name__ == "__main__":
    data = main()