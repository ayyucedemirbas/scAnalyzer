from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from .core import SingleCellDataset


def gene_set_score(
    data: SingleCellDataset,
    gene_list: List[str],
    score_name: Optional[str] = None,
    ctrl_size: int = 50,
    n_bins: int = 25,
    random_state: int = 0,
    use_raw: bool = False,
) -> SingleCellDataset:

    # Examples

    # hypoxia_genes = ['VEGFA', 'HIF1A', 'LDHA', 'PDK1']
    # gene_set_score(data, hypoxia_genes, score_name='hypoxia_score')
    # Cells with high hypoxia score
    # hypoxic = data[data.obs['hypoxia_score'] > 0.5, :]

    from .cell_cycle import score_genes

    scores = score_genes(
        data,
        gene_list,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        random_state=random_state,
        use_raw=use_raw,
    )

    if score_name is None:
        score_name = "gene_set_score"

    data.obs[score_name] = scores

    print(
        f"Gene Set: Added '{score_name}' to obs (mean={scores.mean():.3f}, std={scores.std():.3f})"
    )

    return data


def score_multiple_gene_sets(
    data: SingleCellDataset,
    gene_sets: Dict[str, List[str]],
    ctrl_size: int = 50,
    n_bins: int = 25,
    random_state: int = 0,
    use_raw: bool = False,
) -> SingleCellDataset:
    """
    Examples

     gene_sets = {
    ...     'T_cell': ['CD3D', 'CD3E', 'CD3G'],
    ...     'B_cell': ['CD19', 'MS4A1', 'CD79A'],
    ...     'Myeloid': ['CD14', 'LYZ', 'S100A8']
    ... }
     score_multiple_gene_sets(data, gene_sets)
    """

    print(f"Gene Sets: Scoring {len(gene_sets)} gene sets...")

    for name, genes in gene_sets.items():
        gene_set_score(
            data,
            genes,
            score_name=f"{name}_score",
            ctrl_size=ctrl_size,
            n_bins=n_bins,
            random_state=random_state,
            use_raw=use_raw,
        )

    return data


def rank_genes_groups_by_enrichment(
    data: SingleCellDataset,
    gene_sets: Dict[str, List[str]],
    groupby: str,
    key: str = "rank_genes_groups",
    method: str = "hypergeometric",
    background: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Examples

    >>> from differential import rank_genes_groups
    >>> rank_genes_groups(data, groupby='leiden')
    >>>
    >>> gene_sets = load_msigdb_sets(['HALLMARK_HYPOXIA', 'HALLMARK_GLYCOLYSIS'])
    >>> enrichment = rank_genes_groups_by_enrichment(data, gene_sets, groupby='leiden')
    >>> print(enrichment['0'])  # Enrichment for cluster 0
    """

    if key not in data.uns:
        raise ValueError(f"{key} not found. Run rank_genes_groups() first.")

    if background is None:
        background = data.var.index.tolist()

    results = {}

    for group, df in data.uns[key].items():
        markers = df[df["pvals_adj"] < 0.05]["names"].tolist()

        if len(markers) == 0:
            continue

        enrichment_results = []

        for set_name, gene_set in gene_sets.items():
            overlap = set(markers) & set(gene_set) & set(background)

            # Hypergeometric test
            # k = overlap, M = background size, n = gene set size, N = markers size
            M = len(background)
            n = len(set(gene_set) & set(background))
            N = len(markers)
            k = len(overlap)

            if method == "hypergeometric":
                # P(X >= k) where X ~ Hypergeometric(M, n, N)
                pval = stats.hypergeom.sf(k - 1, M, n, N)
            elif method == "fisher":
                # Fisher's exact test
                # Contingency table:
                #                  In gene set | Not in gene set
                # Markers:         k           | N - k
                # Non-markers:     n - k       | M - N - n + k
                from scipy.stats import fisher_exact

                table = [[k, N - k], [n - k, M - N - n + k]]
                _, pval = fisher_exact(table, alternative="greater")
            else:
                raise ValueError("method must be 'hypergeometric' or 'fisher'")

            # Enrichment ratio
            expected = (N * n) / M
            fold_enrichment = k / expected if expected > 0 else 0

            enrichment_results.append(
                {
                    "gene_set": set_name,
                    "overlap_size": k,
                    "gene_set_size": n,
                    "markers_size": N,
                    "expected": expected,
                    "fold_enrichment": fold_enrichment,
                    "pval": pval,
                    "overlap_genes": ",".join(sorted(overlap)),
                }
            )

        results_df = pd.DataFrame(enrichment_results)

        # Multiple testing correction
        if len(results_df) > 0:
            from statsmodels.stats.multitest import multipletests

            _, padj, _, _ = multipletests(results_df["pval"], method="fdr_bh")
            results_df["pval_adj"] = padj
            results_df = results_df.sort_values("pval")

        results[group] = results_df

    return results


def gsea_preranked(
    ranked_genes: pd.DataFrame,
    gene_set: List[str],
    weight: float = 1.0,
    nperm: int = 1000,
    random_state: int = 0,
) -> Dict[str, float]:

    np.random.seed(random_state)

    gene_names = ranked_genes["names"].values
    gene_scores = ranked_genes["scores"].values

    gene_set_present = [g for g in gene_set if g in gene_names]

    if len(gene_set_present) == 0:
        return {"ES": 0, "NES": 0, "pval": 1.0, "n_genes": 0}

    in_set = np.array([g in gene_set_present for g in gene_names])

    N = len(gene_names)
    N_hit = in_set.sum()
    N_miss = N - N_hit

    hit_scores = np.abs(gene_scores) ** weight * in_set
    miss_scores = 1 - in_set

    hit_sum = hit_scores.sum()
    if hit_sum == 0:
        return {"ES": 0, "NES": 0, "pval": 1.0, "n_genes": N_hit}

    cum_hit = np.cumsum(hit_scores / hit_sum)
    cum_miss = np.cumsum(miss_scores / N_miss)

    running_es = cum_hit - cum_miss

    ES = running_es[np.abs(running_es).argmax()]

    peak_idx = np.abs(running_es).argmax()
    leading_edge = gene_names[in_set][: peak_idx + 1]

    es_null = []
    for _ in range(nperm):
        shuffled = np.random.permutation(in_set)

        hit_scores_perm = np.abs(gene_scores) ** weight * shuffled
        miss_scores_perm = 1 - shuffled

        hit_sum_perm = hit_scores_perm.sum()
        if hit_sum_perm == 0:
            continue

        cum_hit_perm = np.cumsum(hit_scores_perm / hit_sum_perm)
        cum_miss_perm = np.cumsum(miss_scores_perm / N_miss)

        running_es_perm = cum_hit_perm - cum_miss_perm
        es_perm = running_es_perm[np.abs(running_es_perm).argmax()]
        es_null.append(es_perm)

    es_null = np.array(es_null)

    if ES >= 0:
        NES = ES / (es_null[es_null >= 0].mean() + 1e-10)
        pval = (es_null >= ES).sum() / len(es_null)
    else:
        NES = -ES / ((-es_null[es_null < 0]).mean() + 1e-10)
        pval = (es_null <= ES).sum() / len(es_null)

    return {
        "ES": ES,
        "NES": NES,
        "pval": pval,
        "n_genes": N_hit,
        "leading_edge": ",".join(leading_edge[:10]),  # Top 10
    }


def load_gene_sets(
    source: str = "msigdb",
    categories: Optional[List[str]] = None,
    organism: str = "human",
) -> Dict[str, List[str]]:
    """

    Examples

    >>> gene_sets = load_gene_sets('msigdb', categories=['HALLMARK'])
    >>> # Returns Hallmark gene sets

    Note

    This is a placeholder.
    - gseapy.get_library_name()
    - Or download GMT files from MSigDB
    """

    print(f"Loading gene sets from {source}...")
    print("Note: This is a placeholder. Install 'gseapy' for full functionality.")

    # Placeholder: return some example gene sets
    example_sets = {
        "HALLMARK_HYPOXIA": ["VEGFA", "HIF1A", "LDHA", "PDK1", "ENO1", "PFKP"],
        "HALLMARK_GLYCOLYSIS": ["HK2", "LDHA", "PFKL", "PFKP", "ENO1", "PKM"],
        "HALLMARK_APOPTOSIS": ["BAX", "BCL2", "CASP3", "CASP9", "CYCS", "BID"],
        "GO_T_CELL_ACTIVATION": ["CD3D", "CD3E", "CD3G", "CD28", "ICOS"],
    }

    return example_sets
