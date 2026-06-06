import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from .core import SingleCellDataset

def infer_grn_ridge(
    data: SingleCellDataset, 
    tf_list: list, 
    top_n_edges: int = 50000
) -> pd.DataFrame:

    valid_tfs = [tf for tf in tf_list if tf in data.var.index]
    if not valid_tfs:
        raise ValueError("Not found.")
        
    tf_indices = [data.var.index.get_loc(tf) for tf in valid_tfs]
    
    X = data.X.toarray() if sp.issparse(data.X) else data.X
    
    X_tfs = X[:, tf_indices]
    
    edges = []
    
    
    model = Ridge(alpha=1.0)
    
    for gene_idx, gene_name in enumerate(data.var.index):
        y_target = X[:, gene_idx]
        

        model.fit(X_tfs, y_target)
        
        weights = np.abs(model.coef_)
        
        top_tf_local_indices = np.argsort(weights)[::-1][:100]
        
        for local_idx in top_tf_local_indices:
            tf_name = valid_tfs[local_idx]
            weight = weights[local_idx]
            if weight > 0:
                edges.append((tf_name, gene_name, weight))


    df_grn = pd.DataFrame(edges, columns=["source", "target", "weight"])
    df_grn = df_grn.sort_values(by="weight", ascending=False).head(top_n_edges).reset_index(drop=True)
    
    df_grn['weight'] = df_grn['weight'].astype(str)
    
    return df_grn