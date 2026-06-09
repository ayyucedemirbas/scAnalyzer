import os
import sys
sys.path.insert(0, os.path.abspath('..'))

html_theme = 'sphinx_rtd_theme'

autodoc_mock_imports = [
    "numpy", 
    "pandas", 
    "scipy", 
    "sklearn", 
    "statsmodels", 
    "matplotlib", 
    "seaborn", 
    "h5py", 
    "umap", 
    "leidenalg", 
    "louvain", 
    "igraph", 
    "plotly"
]
