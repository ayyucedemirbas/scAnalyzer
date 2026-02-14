from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from core import SingleCellDataset

S_GENES_HUMAN = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2',
    'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2',
    'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7',
    'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1',
    'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B',
    'BRIP1', 'E2F8'
]

G2M_GENES_HUMAN = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80',
    'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A',
    'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E',
    'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK',
    'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8',
    'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5',
    'CENPE', 'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
]

# Mouse orthologs (capitalized first letter only)
S_GENES_MOUSE = [g.capitalize() for g in S_GENES_HUMAN]
G2M_GENES_MOUSE = [g.capitalize() for g in G2M_GENES_HUMAN]


def score_genes(
    data: SingleCellDataset,
    gene_list: List[str],
    ctrl_size: int = 50,
    n_bins: int = 25,
    random_state: int = 0,
    use_raw: bool = False,
) -> np.ndarray:
    """
    Calculate per-cell score for a gene set.
    
    Similar to Seurat's AddModuleScore. For each cell:
    1. Calculate mean expression of genes in the gene set
    2. Subtract mean expression of control genes
    
    Control genes are selected from genes with similar average expression.
    
    Parameters

    data : SingleCellDataset
        Annotated data matrix.
    gene_list : List[str]
        List of gene names to score.
    ctrl_size : int, default: 50
        Number of control genes per gene in gene_list.
    n_bins : int, default: 25
        Number of bins for matching control genes by expression.
    random_state : int, default: 0
        Random seed.
    use_raw : bool, default: False
        Use raw counts if available.
    
    Returns

    np.ndarray
        Score for each cell.
    """
    
    np.random.seed(random_state)
    
    # Get expression matrix
    if use_raw and data.raw is not None:
        if hasattr(data.raw, 'X'):
            X = data.raw.X
            var_names = data.raw.var.index
        else:
            X = data.raw
            var_names = data.var.index
    else:
        X = data.X
        var_names = data.var.index
    
    # Find genes in dataset
    gene_list_found = [g for g in gene_list if g in var_names]
    
    if len(gene_list_found) == 0:
        print(f"Warning: None of the {len(gene_list)} genes found in dataset.")
        return np.zeros(data.n_obs)
    
    if len(gene_list_found) < len(gene_list):
        missing = len(gene_list) - len(gene_list_found)
        print(f"Warning: {missing}/{len(gene_list)} genes not found in dataset.")
    
    # Get indices
    gene_indices = [var_names.get_loc(g) for g in gene_list_found]
    
    # Calculate mean expression per gene (for binning)
    if sp.issparse(X):
        gene_means = np.ravel(X.mean(axis=0))
    else:
        gene_means = X.mean(axis=0)
    
    # Bin genes by expression level
    gene_bins = pd.cut(gene_means, bins=n_bins, labels=False)
    
    # For each gene in gene_list, select control genes from same bin
    ctrl_genes = []
    
    for gene_idx in gene_indices:
        gene_bin = gene_bins[gene_idx]
        
        # Candidate control genes: same bin, not in gene_list
        candidates = np.where(
            (gene_bins == gene_bin) & (~np.isin(np.arange(len(var_names)), gene_indices))
        )[0]
        
        if len(candidates) == 0:
            # No candidates in same bin, use adjacent bins
            adjacent_bins = [gene_bin - 1, gene_bin, gene_bin + 1]
            adjacent_bins = [b for b in adjacent_bins if 0 <= b < n_bins]
            candidates = np.where(
                (np.isin(gene_bins, adjacent_bins)) & 
                (~np.isin(np.arange(len(var_names)), gene_indices))
            )[0]
        
        # Randomly select ctrl_size genes
        n_select = min(ctrl_size, len(candidates))
        selected = np.random.choice(candidates, n_select, replace=False)
        ctrl_genes.extend(selected)
    
    ctrl_genes = np.array(ctrl_genes)
    
    # Calculate scores
    if sp.issparse(X):
        # Mean expression of gene set
        gene_expr = X[:, gene_indices].mean(axis=1).A1
        # Mean expression of control genes
        ctrl_expr = X[:, ctrl_genes].mean(axis=1).A1
    else:
        gene_expr = X[:, gene_indices].mean(axis=1)
        ctrl_expr = X[:, ctrl_genes].mean(axis=1)
    
    scores = gene_expr - ctrl_expr
    
    return scores


def score_cell_cycle(
    data: SingleCellDataset,
    s_genes: Optional[List[str]] = None,
    g2m_genes: Optional[List[str]] = None,
    organism: str = 'human',
    ctrl_size: int = 50,
    random_state: int = 0,
    use_raw: bool = False,
) -> SingleCellDataset:
    """
    Score cells for cell cycle phase.
    
    Calculates S phase and G2M phase scores, then assigns cells to phases:
    - G1: Low S and G2M scores
    - S: High S score
    - G2M: High G2M score
    
    Parameters

    data : SingleCellDataset
        Annotated data matrix (should be normalized and log-transformed).
    s_genes : List[str], optional
        S phase marker genes. If None, uses default list.
    g2m_genes : List[str], optional
        G2M phase marker genes. If None, uses default list.
    organism : {'human', 'mouse'}, default: 'human'
        Organism for default gene lists.
    ctrl_size : int, default: 50
        Number of control genes per gene set.
    random_state : int, default: 0
        Random seed.
    use_raw : bool, default: False
        Use raw counts.
    
    Returns

    SingleCellDataset
        Updates data.obs with:
        - 'S_score': S phase score
        - 'G2M_score': G2M phase score
        - 'phase': Predicted cell cycle phase
    
    Examples

    >>> from cell_cycle import score_cell_cycle
    >>> score_cell_cycle(data, organism='human')
    >>> print(data.obs['phase'].value_counts())
    """
    
    print("Cell Cycle: Scoring cells for S and G2M phases...")
    
    # Use default gene lists if not provided
    if s_genes is None:
        if organism == 'human':
            s_genes = S_GENES_HUMAN
        elif organism == 'mouse':
            s_genes = S_GENES_MOUSE
        else:
            raise ValueError("organism must be 'human' or 'mouse'")
    
    if g2m_genes is None:
        if organism == 'human':
            g2m_genes = G2M_GENES_HUMAN
        elif organism == 'mouse':
            g2m_genes = G2M_GENES_MOUSE
        else:
            raise ValueError("organism must be 'human' or 'mouse'")
    
    # Calculate scores
    s_score = score_genes(
        data, s_genes, ctrl_size=ctrl_size, random_state=random_state, use_raw=use_raw
    )
    g2m_score = score_genes(
        data, g2m_genes, ctrl_size=ctrl_size, random_state=random_state, use_raw=use_raw
    )
    
    # Store scores
    data.obs['S_score'] = s_score
    data.obs['G2M_score'] = g2m_score
    
    # Assign phases
    # Simple heuristic: if S_score and G2M_score are both negative, assign G1
    # If S_score > G2M_score and positive, assign S
    # Otherwise assign G2M
    
    phases = np.array(['G1'] * data.n_obs)
    
    # S phase: high S score
    s_mask = (s_score > 0) & (s_score > g2m_score)
    phases[s_mask] = 'S'
    
    # G2M phase: high G2M score
    g2m_mask = (g2m_score > 0) & (g2m_score >= s_score)
    phases[g2m_mask] = 'G2M'
    
    data.obs['phase'] = pd.Categorical(phases, categories=['G1', 'S', 'G2M'])
    
    # Print summary
    phase_counts = data.obs['phase'].value_counts()
    print("Cell Cycle: Phase distribution:")
    for phase in ['G1', 'S', 'G2M']:
        count = phase_counts.get(phase, 0)
        pct = 100 * count / data.n_obs
        print(f"  {phase}: {count} cells ({pct:.1f}%)")
    
    return data


def regress_out_cell_cycle(
    data: SingleCellDataset,
    s_genes: Optional[List[str]] = None,
    g2m_genes: Optional[List[str]] = None,
    organism: str = 'human',
    difference_only: bool = True,
) -> SingleCellDataset:
    """
    Regress out cell cycle effects from expression data.
    
    Two modes:
    1. difference_only=True: Regress out S-G2M difference (preserves proliferating vs non-proliferating)
    2. difference_only=False: Regress out both S and G2M scores (removes all cell cycle signal)
    
    Parameters
    data : SingleCellDataset
        Dataset with cell cycle scores (run score_cell_cycle first).
    s_genes : List[str], optional
        S phase genes (for re-scoring if needed).
    g2m_genes : List[str], optional
        G2M phase genes (for re-scoring if needed).
    organism : str, default: 'human'
        Organism.
    difference_only : bool, default: True
        If True, regress S-G2M difference. If False, regress both.
    
    Returns

    SingleCellDataset
        Expression data with cell cycle effects removed.
    """
    
    # Ensure scores are present
    if 'S_score' not in data.obs.columns or 'G2M_score' not in data.obs.columns:
        print("Cell cycle scores not found. Calculating...")
        score_cell_cycle(data, s_genes, g2m_genes, organism)
    
    print(f"Cell Cycle: Regressing out cell cycle (difference_only={difference_only})...")
    
    s_score = data.obs['S_score'].values
    g2m_score = data.obs['G2M_score'].values
    
    X = data.X
    if sp.issparse(X):
        X = X.toarray()
    
    # Regress for each gene
    for gene_idx in range(data.n_vars):
        y = X[:, gene_idx]
        
        if difference_only:
            # Regress out S-G2M difference
            diff = s_score - g2m_score
            # Fit: y = a + b * diff
            # Residual: y - b * diff (keeping intercept)
            if np.std(diff) > 0:
                beta = np.cov(y, diff)[0, 1] / np.var(diff)
                residuals = y - beta * diff
                X[:, gene_idx] = residuals
        else:
            # Regress out both S and G2M
            # Design matrix: [intercept, S_score, G2M_score]
            if np.std(s_score) > 0 and np.std(g2m_score) > 0:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                predictors = np.column_stack([s_score, g2m_score])
                reg.fit(predictors, y)
                residuals = y - reg.predict(predictors) + reg.intercept_
                X[:, gene_idx] = residuals
    
    # Update data
    if sp.issparse(data.X):
        data.X = sp.csr_matrix(X)
    else:
        data.X = X
    
    print("Cell Cycle: Regression complete.")
    return data
