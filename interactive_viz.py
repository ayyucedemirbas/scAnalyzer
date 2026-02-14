"""
Interactive Visualizations using Plotly
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from core import SingleCellDataset


def interactive_embedding(
    data: SingleCellDataset,
    basis: str = 'X_umap',
    color: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600,
    save_html: Optional[str] = None,
):
    """
    Interactive scatter plot of embeddings using Plotly.
    
    Features:
    - Zoom, pan, and select cells
    - Hover to see cell information
    - Interactive legend
    - Export to HTML for sharing
    
    Parameters

    data : SingleCellDataset
        Dataset with embeddings.
    basis : str, default: 'X_umap'
        Embedding to plot.
    color : str, optional
        Column in obs or gene name to color by.
    hover_data : List[str], optional
        Additional columns to show on hover.
    title : str, optional
        Plot title.
    width : int, default: 800
        Plot width in pixels.
    height : int, default: 600
        Plot height in pixels.
    save_html : str, optional
        Save interactive plot to HTML file.
    
    Examples

    >>> from interactive_viz import interactive_embedding
    >>> interactive_embedding(
    ...     data,
    ...     color='leiden',
    ...     hover_data=['n_genes', 'total_counts'],
    ...     save_html='umap_interactive.html'
    ... )
    """
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")
    
    if basis not in data.obsm:
        raise ValueError(f"{basis} not found. Run dimensionality reduction first.")
    
    # Get coordinates
    coords = data.obsm[basis]
    
    # Prepare data frame
    plot_df = pd.DataFrame({
        f'{basis}_1': coords[:, 0],
        f'{basis}_2': coords[:, 1],
        'cell_id': data.obs.index
    })
    
    # Add color data
    if color:
        from visualization import _get_color_data
        values, is_categorical, label = _get_color_data(data, color)
        plot_df[color] = values
    
    # Add hover data
    if hover_data:
        for col in hover_data:
            if col in data.obs.columns:
                plot_df[col] = data.obs[col].values
    
    # Create plot
    if color and pd.api.types.is_categorical_dtype(plot_df[color]) or \
       color and pd.api.types.is_object_dtype(plot_df[color]):
        # Categorical coloring
        fig = px.scatter(
            plot_df,
            x=f'{basis}_1',
            y=f'{basis}_2',
            color=color,
            hover_data=hover_data if hover_data else ['cell_id'],
            title=title if title else f'{basis} colored by {color}',
            width=width,
            height=height,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    elif color:
        # Continuous coloring
        fig = px.scatter(
            plot_df,
            x=f'{basis}_1',
            y=f'{basis}_2',
            color=color,
            hover_data=hover_data if hover_data else ['cell_id'],
            title=title if title else f'{basis} colored by {color}',
            width=width,
            height=height,
            color_continuous_scale='Viridis'
        )
    else:
        # No coloring
        fig = px.scatter(
            plot_df,
            x=f'{basis}_1',
            y=f'{basis}_2',
            hover_data=hover_data if hover_data else ['cell_id'],
            title=title if title else basis,
            width=width,
            height=height
        )
    
    # Update layout
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    # Save or show
    if save_html:
        fig.write_html(save_html)
        print(f"Interactive plot saved to {save_html}")
    else:
        fig.show()
    
    return fig


def interactive_violin(
    data: SingleCellDataset,
    keys: List[str],
    groupby: str,
    width: int = 1000,
    height: int = 600,
    save_html: Optional[str] = None,
):
    """
    Interactive violin plot using Plotly.
    
    Parameters

    data : SingleCellDataset
        Annotated data matrix.
    keys : List[str]
        Genes or obs columns to plot.
    groupby : str
        Grouping variable.
    width : int, default: 1000
        Plot width.
    height : int, default: 600
        Plot height.
    save_html : str, optional
        Save to HTML.
    """
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")
    
    from visualization import _get_color_data
    
    fig = go.Figure()
    
    for key in keys:
        values, _, _ = _get_color_data(data, key)
        groups = data.obs[groupby].values
        
        for group in np.unique(groups):
            mask = groups == group
            fig.add_trace(go.Violin(
                y=values[mask],
                name=f'{key}_{group}',
                legendgroup=key,
                scalegroup=key,
                x=[group] * mask.sum()
            ))
    
    fig.update_layout(
        title=f'Expression by {groupby}',
        xaxis_title=groupby,
        yaxis_title='Expression',
        width=width,
        height=height
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"Interactive violin plot saved to {save_html}")
    else:
        fig.show()
    
    return fig


def interactive_heatmap(
    data: SingleCellDataset,
    var_names: List[str],
    groupby: str,
    use_raw: bool = False,
    standard_scale: bool = True,
    width: int = 800,
    height: int = 600,
    save_html: Optional[str] = None,
):
    """
    Interactive heatmap using Plotly.
    
    Parameters

    data : SingleCellDataset
        Annotated data matrix.
    var_names : List[str]
        Genes to include.
    groupby : str
        Grouping variable.
    use_raw : bool, default: False
        Use raw counts.
    standard_scale : bool, default: True
        Z-score normalize.
    width : int, default: 800
        Plot width.
    height : int, default: 600
        Plot height.
    save_html : str, optional
        Save to HTML.
    """
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")
    
    import scipy.sparse as sp
    
    # Get data
    valid_vars = [v for v in var_names if v in data.var.index]
    var_indices = [data.var.index.get_loc(v) for v in valid_vars]
    
    if use_raw and data.raw is not None:
        raw_X = data.raw.X if hasattr(data.raw, 'X') else data.raw
        X_subset = raw_X[:, var_indices]
    else:
        X_subset = data.X[:, var_indices]
    
    if sp.issparse(X_subset):
        X_subset = X_subset.toarray()
    
    # Compute mean per group
    groups = data.obs[groupby]
    unique_groups = np.sort(groups.unique())
    
    mean_expr = []
    for g in unique_groups:
        mask = (groups == g).values
        mean_expr.append(np.mean(X_subset[mask, :], axis=0))
    
    heatmap_data = np.array(mean_expr)
    
    # Standardize
    if standard_scale:
        heatmap_data = (heatmap_data - heatmap_data.mean(axis=0)) / (heatmap_data.std(axis=0) + 1e-10)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=valid_vars,
        y=unique_groups,
        colorscale='RdBu_r',
        zmid=0 if standard_scale else None
    ))
    
    fig.update_layout(
        title=f'Mean expression by {groupby}',
        xaxis_title='Genes',
        yaxis_title=groupby,
        width=width,
        height=height
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"Interactive heatmap saved to {save_html}")
    else:
        fig.show()
    
    return fig


def interactive_3d_embedding(
    data: SingleCellDataset,
    basis: str = 'X_pca',
    color: Optional[str] = None,
    dimensions: List[int] = [0, 1, 2],
    width: int = 900,
    height: int = 700,
    save_html: Optional[str] = None,
):
    """
    3D interactive plot of embeddings.
    
    Parameters
    data : SingleCellDataset
        Dataset with embeddings.
    basis : str, default: 'X_pca'
        Embedding key in obsm.
    color : str, optional
        Variable to color by.
    dimensions : List[int], default: [0, 1, 2]
        Which dimensions to plot (for PCA, use first 3 PCs).
    width : int, default: 900
        Plot width.
    height : int, default: 700
        Plot height.
    save_html : str, optional
        Save to HTML.
    
    Examples

    >>> interactive_3d_embedding(data, basis='X_pca', color='leiden', dimensions=[0,1,2])
    """
    
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")
    
    if basis not in data.obsm:
        raise ValueError(f"{basis} not found.")
    
    coords = data.obsm[basis]
    
    # Check dimensions
    if coords.shape[1] < 3:
        raise ValueError(f"{basis} has only {coords.shape[1]} dimensions. Need at least 3 for 3D plot.")
    
    # Prepare data
    plot_df = pd.DataFrame({
        'dim1': coords[:, dimensions[0]],
        'dim2': coords[:, dimensions[1]],
        'dim3': coords[:, dimensions[2]],
        'cell_id': data.obs.index
    })
    
    if color:
        from visualization import _get_color_data
        values, is_categorical, label = _get_color_data(data, color)
        plot_df[color] = values
    
    # Create 3D plot
    fig = px.scatter_3d(
        plot_df,
        x='dim1',
        y='dim2',
        z='dim3',
        color=color if color else None,
        hover_data=['cell_id'],
        title=f'3D {basis}',
        width=width,
        height=height
    )
    
    fig.update_traces(marker=dict(size=2, opacity=0.7))
    
    if save_html:
        fig.write_html(save_html)
        print(f"3D plot saved to {save_html}")
    else:
        fig.show()
    
    return fig
