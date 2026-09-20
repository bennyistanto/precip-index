"""
Shared test utilities and configuration for precip-index test suite.

This module provides common functions, constants, and fixtures used across
all test modules. Import from here to maintain consistency.

Author: Benny Istanto
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Configure for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

import numpy as np
import xarray as xr

# =============================================================================
# CONSTANTS
# =============================================================================

# Input data files
INPUT_DIR = REPO_ROOT / 'input'
INPUT_FILES = {
    'precip': INPUT_DIR / 'terraclimate_bali_ppt_1950_2025.nc',
    'tmean': INPUT_DIR / 'terraclimate_bali_tmean_1950_2025.nc',
    'tmin': INPUT_DIR / 'terraclimate_bali_tmin_1950_2025.nc',
    'tmax': INPUT_DIR / 'terraclimate_bali_tmax_1950_2025.nc',
    'pet': INPUT_DIR / 'terraclimate_bali_pet_1950_2025.nc',
}

# Variable names in NetCDF files
VAR_NAMES = {
    'precip': 'ppt',
    'tmean': 'tmean',
    'tmin': 'tmin',
    'tmax': 'tmax',
    'pet': 'pet',
}

# Output directories
OUTPUT_DIR = REPO_ROOT / 'tests' / 'output'
OUTPUT_NETCDF = OUTPUT_DIR / 'netcdf'
OUTPUT_PLOTS = OUTPUT_DIR / 'plots'
OUTPUT_REPORTS = OUTPUT_DIR / 'reports'

# Test parameters
CALIBRATION_START = 1991
CALIBRATION_END = 2020
DATA_START_YEAR = 1950

# Distributions to test
TEST_DISTRIBUTIONS = ['gamma', 'pearson3', 'log_logistic']

# Scales to test
TEST_SCALES = [3, 12]

# Sample location for time series (center of Bali grid)
SAMPLE_LAT_IDX = 12  # Approximate center
SAMPLE_LON_IDX = 17  # Approximate center


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_header(title: str, width: int = 70):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 70):
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)


def print_ok(message: str):
    """Print success message."""
    print(f"  [OK] {message}")


def print_fail(message: str):
    """Print failure message."""
    print(f"  [FAIL] {message}")


def print_info(message: str):
    """Print info message."""
    print(f"  [INFO] {message}")


def print_stat(label: str, value, unit: str = ""):
    """Print a statistic with consistent formatting."""
    if isinstance(value, float):
        print(f"    {label}: {value:.3f} {unit}".rstrip())
    else:
        print(f"    {label}: {value} {unit}".rstrip())


# =============================================================================
# DATA LOADING
# =============================================================================

def setup_output_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NETCDF.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)


def check_input_files() -> dict:
    """
    Check which input files exist.

    Returns:
        dict: {name: {'exists': bool, 'path': Path, 'size_mb': float}}
    """
    status = {}
    for name, path in INPUT_FILES.items():
        exists = path.exists()
        size_mb = path.stat().st_size / (1024 * 1024) if exists else 0
        status[name] = {
            'exists': exists,
            'path': path,
            'size_mb': size_mb
        }
    return status


def load_data(name: str) -> xr.DataArray:
    """
    Load a dataset by name.

    Args:
        name: One of 'precip', 'tmean', 'tmin', 'tmax', 'pet'

    Returns:
        xr.DataArray: The loaded data variable
    """
    if name not in INPUT_FILES:
        raise ValueError(f"Unknown data name: {name}. Use one of {list(INPUT_FILES.keys())}")

    path = INPUT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ds = xr.open_dataset(path)
    var_name = VAR_NAMES[name]

    # Filter out 'crs' variable if present
    if var_name not in ds.data_vars:
        # Try to find the variable
        available = [v for v in ds.data_vars if v != 'crs']
        if len(available) == 1:
            var_name = available[0]
        else:
            raise ValueError(f"Cannot find variable '{var_name}' in {path}")

    return ds[var_name]


def load_all_data() -> dict:
    """
    Load all available input data.

    Returns:
        dict: {name: xr.DataArray} for all existing input files
    """
    data = {}
    status = check_input_files()

    for name, info in status.items():
        if info['exists']:
            try:
                data[name] = load_data(name)
            except Exception as e:
                print_fail(f"Could not load {name}: {e}")

    return data


# =============================================================================
# DATA QUALITY UTILITIES
# =============================================================================

def get_data_summary(da: xr.DataArray) -> dict:
    """
    Get summary statistics for a DataArray.

    Returns:
        dict with keys: shape, dims, time_range, lat_range, lon_range,
                       mean, std, min, max, nan_pct, zero_pct
    """
    summary = {
        'shape': da.shape,
        'dims': da.dims,
        'mean': float(da.mean()),
        'std': float(da.std()),
        'min': float(da.min()),
        'max': float(da.max()),
        'nan_pct': float(da.isnull().mean() * 100),
    }

    # Time range
    if 'time' in da.coords:
        summary['time_start'] = str(da.time.values[0])[:10]
        summary['time_end'] = str(da.time.values[-1])[:10]
        summary['n_timesteps'] = len(da.time)

    # Spatial extent
    lat_coord = 'lat' if 'lat' in da.coords else 'latitude'
    lon_coord = 'lon' if 'lon' in da.coords else 'longitude'

    if lat_coord in da.coords:
        summary['lat_range'] = (float(da[lat_coord].min()), float(da[lat_coord].max()))
        summary['lon_range'] = (float(da[lon_coord].min()), float(da[lon_coord].max()))
        summary['n_lat'] = len(da[lat_coord])
        summary['n_lon'] = len(da[lon_coord])

    # Zero percentage (for precipitation)
    valid_data = da.values[~np.isnan(da.values)]
    if len(valid_data) > 0:
        summary['zero_pct'] = float((valid_data == 0).sum() / len(valid_data) * 100)

    return summary


def print_data_summary(name: str, summary: dict):
    """Print a formatted data summary."""
    print(f"\n  {name}:")
    print(f"    Shape: {summary['shape']}")
    print(f"    Dimensions: {summary['dims']}")

    if 'time_start' in summary:
        print(f"    Time: {summary['time_start']} to {summary['time_end']} ({summary['n_timesteps']} steps)")

    if 'lat_range' in summary:
        print(f"    Lat: [{summary['lat_range'][0]:.2f}, {summary['lat_range'][1]:.2f}] ({summary['n_lat']} cells)")
        print(f"    Lon: [{summary['lon_range'][0]:.2f}, {summary['lon_range'][1]:.2f}] ({summary['n_lon']} cells)")

    print(f"    Mean: {summary['mean']:.2f}, Std: {summary['std']:.2f}")
    print(f"    Range: [{summary['min']:.2f}, {summary['max']:.2f}]")
    print(f"    Missing: {summary['nan_pct']:.1f}%", end="")

    if 'zero_pct' in summary:
        print(f", Zeros: {summary['zero_pct']:.1f}%")
    else:
        print()


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays, ignoring NaNs."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def compute_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSE between two arrays, ignoring NaNs."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 10:
        return np.nan
    return float(np.sqrt(np.mean((a[valid] - b[valid])**2)))


def compute_bias(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean bias (a - b), ignoring NaNs."""
    valid = ~(np.isnan(a) | np.isnan(b))
    if valid.sum() < 10:
        return np.nan
    return float(np.mean(a[valid] - b[valid]))


def compare_arrays(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> dict:
    """
    Compare two arrays and return statistics.

    Returns:
        dict with correlation, rmse, bias, and summary stats
    """
    a_flat = a.flatten()
    b_flat = b.flatten()

    return {
        'name_a': name_a,
        'name_b': name_b,
        'correlation': compute_correlation(a_flat, b_flat),
        'rmse': compute_rmse(a_flat, b_flat),
        'bias': compute_bias(a_flat, b_flat),
        'mean_a': float(np.nanmean(a_flat)),
        'mean_b': float(np.nanmean(b_flat)),
        'std_a': float(np.nanstd(a_flat)),
        'std_b': float(np.nanstd(b_flat)),
    }


def print_comparison(comp: dict):
    """Print comparison results."""
    print(f"\n  {comp['name_a']} vs {comp['name_b']}:")
    print(f"    Correlation: r = {comp['correlation']:.4f}")
    print(f"    RMSE: {comp['rmse']:.4f}")
    print(f"    Bias: {comp['bias']:.4f}")
    print(f"    Mean {comp['name_a']}: {comp['mean_a']:.3f}, Mean {comp['name_b']}: {comp['mean_b']:.3f}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def write_report(filename: str, title: str, content: str):
    """Write a text report to the reports directory."""
    filepath = OUTPUT_REPORTS / filename

    header = f"""{'=' * 70}
{title}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}

"""

    with open(filepath, 'w') as f:
        f.write(header + content)

    print_ok(f"Report saved: {filepath}")


# =============================================================================
# MATPLOTLIB SETUP
# =============================================================================

def setup_matplotlib():
    """Configure matplotlib for non-interactive plotting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 150

    return plt
