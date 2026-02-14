import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Checking scAnalyzer setup...")
print("=" * 60)

# Check core modules
modules_to_check = [
    ('core', 'Core data structure'),
    ('preprocessing', 'Preprocessing functions'),
    ('dimensionality', 'Dimensionality reduction'),
    ('clustering', 'Clustering algorithms'),
    ('differential', 'Differential expression'),
    ('visualization', 'Visualization functions'),
    ('sc_io', 'Input/Output'),
    ('utils', 'Utility functions'),
    # New modules
    ('quality_control', 'Quality control & doublet detection'),
    ('cell_cycle', 'Cell cycle scoring'),
    ('batch_correction', 'Batch correction'),
    ('enrichment', 'Gene set enrichment'),
    ('interactive_viz', 'Interactive visualizations'),
]

all_ok = True

for module_name, description in modules_to_check:
    try:
        __import__(module_name)
        print(f"✓ {module_name:20s} - {description}")
    except ImportError as e:
        print(f"✗ {module_name:20s} - MISSING: {e}")
        all_ok = False
    except Exception as e:
        print(f"✗ {module_name:20s} - ERROR: {e}")
        all_ok = False

print("=" * 60)

dependencies = [
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('scipy', 'SciPy'),
    ('sklearn', 'scikit-learn'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('h5py', 'HDF5'),
]

optional_deps = [
    ('umap', 'UMAP'),
    ('leidenalg', 'Leiden clustering'),
    ('igraph', 'igraph'),
    ('plotly', 'Plotly (for interactive viz)'),
]

for dep_name, description in dependencies:
    try:
        __import__(dep_name)
        print(f"{dep_name:20s} - {description}")
    except ImportError:
        print(f"{dep_name:20s} - MISSING (required)")
        all_ok = False

for dep_name, description in optional_deps:
    try:
        __import__(dep_name)
        print(f"{dep_name:20s} - {description}")
    except ImportError:
        print(f"{dep_name:20s} - Not installed (optional)")

print("=" * 60)

if all_ok:
    pass
else:
    print("\nSetup incomplete. Please install missing modules.")
    print("\nTo install missing modules:")
    print("  pip install -r requirements.txt")
    
print("\nCurrent directory:", os.getcwd())
print("Script directory:", os.path.dirname(os.path.abspath(__file__)))