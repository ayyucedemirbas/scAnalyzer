from typing import List, Optional

import numpy as np
import pandas as pd

from .core import SingleCellDataset


def interactive_embedding(
    data: SingleCellDataset,
    basis: str = "X_umap",
    color: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600,
    save_html: Optional[str] = None,
):

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")

    if basis not in data.obsm:
        raise ValueError(f"{basis} not found. Run dimensionality reduction first.")

    coords = data.obsm[basis]

    plot_df = pd.DataFrame(
        {
            f"{basis}_1": coords[:, 0],
            f"{basis}_2": coords[:, 1],
            "cell_id": data.obs.index,
        }
    )

    if color:
        from visualization import _get_color_data

        values, is_categorical, label = _get_color_data(data, color)
        plot_df[color] = values

    if hover_data:
        for col in hover_data:
            if col in data.obs.columns:
                plot_df[col] = data.obs[col].values

    if (
        color
        and pd.api.types.is_categorical_dtype(plot_df[color])
        or color
        and pd.api.types.is_object_dtype(plot_df[color])
    ):
        fig = px.scatter(
            plot_df,
            x=f"{basis}_1",
            y=f"{basis}_2",
            color=color,
            hover_data=hover_data if hover_data else ["cell_id"],
            title=title if title else f"{basis} colored by {color}",
            width=width,
            height=height,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
    elif color:
        fig = px.scatter(
            plot_df,
            x=f"{basis}_1",
            y=f"{basis}_2",
            color=color,
            hover_data=hover_data if hover_data else ["cell_id"],
            title=title if title else f"{basis} colored by {color}",
            width=width,
            height=height,
            color_continuous_scale="Viridis",
        )
    else:
        fig = px.scatter(
            plot_df,
            x=f"{basis}_1",
            y=f"{basis}_2",
            hover_data=hover_data if hover_data else ["cell_id"],
            title=title if title else basis,
            width=width,
            height=height,
        )

    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
    )

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
            fig.add_trace(
                go.Violin(
                    y=values[mask],
                    name=f"{key}_{group}",
                    legendgroup=key,
                    scalegroup=key,
                    x=[group] * mask.sum(),
                )
            )

    fig.update_layout(
        title=f"Expression by {groupby}",
        xaxis_title=groupby,
        yaxis_title="Expression",
        width=width,
        height=height,
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

    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")

    import scipy.sparse as sp

    valid_vars = [v for v in var_names if v in data.var.index]
    var_indices = [data.var.index.get_loc(v) for v in valid_vars]

    if use_raw and data.raw is not None:
        raw_X = data.raw.X if hasattr(data.raw, "X") else data.raw
        X_subset = raw_X[:, var_indices]
    else:
        X_subset = data.X[:, var_indices]

    if sp.issparse(X_subset):
        X_subset = X_subset.toarray()

    groups = data.obs[groupby]
    unique_groups = np.sort(groups.unique())

    mean_expr = []
    for g in unique_groups:
        mask = (groups == g).values
        mean_expr.append(np.mean(X_subset[mask, :], axis=0))

    heatmap_data = np.array(mean_expr)

    if standard_scale:
        heatmap_data = (heatmap_data - heatmap_data.mean(axis=0)) / (
            heatmap_data.std(axis=0) + 1e-10
        )

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=valid_vars,
            y=unique_groups,
            colorscale="RdBu_r",
            zmid=0 if standard_scale else None,
        )
    )

    fig.update_layout(
        title=f"Mean expression by {groupby}",
        xaxis_title="Genes",
        yaxis_title=groupby,
        width=width,
        height=height,
    )

    if save_html:
        fig.write_html(save_html)
        print(f"Interactive heatmap saved to {save_html}")
    else:
        fig.show()

    return fig


def interactive_3d_embedding(
    data: SingleCellDataset,
    basis: str = "X_pca",
    color: Optional[str] = None,
    dimensions: List[int] = [0, 1, 2],
    width: int = 900,
    height: int = 700,
    save_html: Optional[str] = None,
):

    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("Please install plotly: pip install plotly")

    if basis not in data.obsm:
        raise ValueError(f"{basis} not found.")

    coords = data.obsm[basis]

    if coords.shape[1] < 3:
        raise ValueError(
            f"{basis} has only {coords.shape[1]} dimensions. Need at least 3 for 3D plot."
        )

    plot_df = pd.DataFrame(
        {
            "dim1": coords[:, dimensions[0]],
            "dim2": coords[:, dimensions[1]],
            "dim3": coords[:, dimensions[2]],
            "cell_id": data.obs.index,
        }
    )

    if color:
        from visualization import _get_color_data

        values, is_categorical, label = _get_color_data(data, color)
        plot_df[color] = values

    fig = px.scatter_3d(
        plot_df,
        x="dim1",
        y="dim2",
        z="dim3",
        color=color if color else None,
        hover_data=["cell_id"],
        title=f"3D {basis}",
        width=width,
        height=height,
    )

    fig.update_traces(marker=dict(size=2, opacity=0.7))

    if save_html:
        fig.write_html(save_html)
        print(f"3D plot saved to {save_html}")
    else:
        fig.show()

    return fig
