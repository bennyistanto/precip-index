"""
Plot Global SPI-12 Map

Standalone script to visualize global SPI-12 output from CHIRPS v3 data.
Produces a publication-ready map with WMO 11-category classification,
equal-area projection, and country boundaries.

Usage:
    python tests/plot_global_spi.py

Input:
    global/output/netcdf/wld_cli_chirps3_d3_spi_gamma_12_month.nc

Output:
    docs/images/global-spi12-202512.png
"""

import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# ============================================================================
# Configuration
# ============================================================================

# Input / output paths
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = Path(r'E:/temp/chirps/spi/output3/wld_cli_chirps3_spi_gamma_12_month.nc')
OUTPUT_FILE = BASE_DIR / 'docs' / 'images' / 'global-spi12-202512.png'

# Time slice to plot
TARGET_DATE = '2025-12-21'

# Variable name in the NetCDF
VAR_NAME = 'spi_gamma_12_month'

# Accumulation length, read off the variable name so that pointing this script
# at another timescale needs no other edit.
TIMESCALE_LABEL = re.search(r'(\d+)_month', VAR_NAME).group(1) + '-month'

# Title: the long-format index name on its own.
INDEX_NAME = 'Standardized Precipitation Index'

# Subtitle parts. The baseline and the date are appended at plot time, both
# read from the file rather than typed here.
DATASET_LABEL = 'CHIRPS v3'
DISTRIBUTION_LABEL = 'Gamma'

# Colorbar keeps the fully qualified label.
INDEX_LABEL = 'Standardized Precipitation Index (Gamma), 12-month'

# Map settings
PROJECTION = ccrs.EqualEarth()        # Equal-area projection
DATA_CRS = ccrs.PlateCarree()         # Data is in lat/lon
FIGSIZE = (12, 8)
DPI = 200

# ============================================================================
# WMO 11-Category SPI Classification
# ============================================================================

# Boundaries for 11 categories
# -inf, -2.0, -1.5, -1.2, -0.7, -0.5, 0.5, 0.7, 1.2, 1.5, 2.0, +inf
SPI_BOUNDS = [-3.5, -2.0, -1.5, -1.2, -0.7, -0.5, 0.5, 0.7, 1.2, 1.5, 2.0, 3.5]

SPI_COLORS = [
    '#760005',   # Exceptionally dry    (< -2.0)
    '#ec0013',   # Extremely dry         (-2.0 to -1.5)
    '#ffa938',   # Severely dry          (-1.5 to -1.2)
    '#fdd28a',   # Moderately dry        (-1.2 to -0.7)
    '#fefe53',   # Abnormally dry        (-0.7 to -0.5)
    '#ffffff',   # Near normal           (-0.5 to +0.5)
    '#a2fd6e',   # Abnormally moist      (+0.5 to +0.7)
    '#00b44a',   # Moderately moist      (+0.7 to +1.2)
    '#008180',   # Very moist            (+1.2 to +1.5)
    '#2a23eb',   # Extremely moist       (+1.5 to +2.0)
    '#a21fec',   # Exceptionally moist   (> +2.0)
]

SPI_LABELS = [
    'Exceptionally\nDry',
    'Extremely\nDry',
    'Severely\nDry',
    'Moderately\nDry',
    'Abnormally\nDry',
    'Near\nNormal',
    'Abnormally\nMoist',
    'Moderately\nMoist',
    'Very\nMoist',
    'Extremely\nMoist',
    'Exceptionally\nMoist',
]


def build_colormap():
    """Build a discrete colormap and norm for the WMO 11-category classification."""
    cmap = mcolors.ListedColormap(SPI_COLORS)
    norm = mcolors.BoundaryNorm(SPI_BOUNDS, cmap.N)
    return cmap, norm


def load_data(filepath: Path, target_date: str, var_name: str) -> xr.DataArray:
    """Load a single time slice from the global SPI NetCDF."""
    print(f"Opening: {filepath.name}")
    print(f"  (file size: {filepath.stat().st_size / 1e9:.1f} GB)")

    # Open dataset — only decode the time slice we need
    ds = xr.open_dataset(filepath)

    # Select the target date
    da = ds[var_name].sel(time=target_date, method='nearest')
    actual_time = str(da.time.values)[:10]
    print(f"  Selected time: {actual_time}")

    # Load into memory (single time slice: ~65 MB for float32 2400x7200)
    print(f"  Loading data slice ({da.shape[0]} x {da.shape[1]}) ...")
    da = da.load()

    # Carry the calibration period out of the dataset before closing it, so the
    # subtitle states the baseline the index is standardised against. Read from
    # the file rather than typed in, so a rerun on another baseline labels
    # itself instead of silently keeping the old text.
    cal_start = ds.attrs.get('calibration_start_year')
    cal_end = ds.attrs.get('calibration_end_year')
    da.attrs['baseline'] = (f'{cal_start}-{cal_end} baseline'
                            if cal_start and cal_end else None)
    print(f"  Baseline: {da.attrs['baseline'] or 'NOT RECORDED IN FILE'}")
    ds.close()

    valid = int(np.isfinite(da.values).sum())
    total = da.size
    print(f"  Valid cells: {valid:,} / {total:,} ({100*valid/total:.1f}%)")

    return da


def plot_global_spi(da: xr.DataArray, output_path: Path):
    """Create publication-ready global SPI map."""
    print("\nCreating map ...")

    cmap, norm = build_colormap()

    # Create figure — map fills most of the space, extra room at bottom for legend labels
    fig = plt.figure(figsize=FIGSIZE, facecolor='white')
    ax = fig.add_axes([0.02, 0.17, 0.96, 0.74], projection=PROJECTION)

    # Background
    ax.set_global()
    ax.set_facecolor('#d9d9d9')  # Light gray for ocean/background

    # Plot SPI data
    im = ax.pcolormesh(
        da.lon.values, da.lat.values, da.values,
        transform=DATA_CRS,
        cmap=cmap,
        norm=norm,
        shading='auto',
        rasterized=True,      # Keeps file size reasonable
    )

    # Country boundaries (50m resolution for better detail)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', '50m',
            edgecolor='#404040', facecolor='none',
        ),
        linewidth=0.3,
        alpha=0.7,
    )

    # Coastlines (50m resolution)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m',
            edgecolor='#202020', facecolor='none',
        ),
        linewidth=0.4,
    )

    # Gridlines
    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.3,
        color='gray',
        alpha=0.4,
        linestyle='--',
    )
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(range(-60, 81, 30))

    # Title and subtitle with clear spacing
    date_label = f"{da.time.dt.strftime('%b %Y').values}"
    fig.text(
        0.5, 0.97, INDEX_NAME,
        ha='center', va='top',
        fontsize=14, fontweight='bold',
    )
    fig.text(
        0.5, 0.93,
        ', '.join(part for part in [
            DATASET_LABEL, DISTRIBUTION_LABEL,
            f'{TIMESCALE_LABEL} accumulation', da.attrs.get('baseline'),
        ] if part) + f'  ·  as of {date_label}',
        ha='center', va='top',
        fontsize=11, color='#404040',
    )

    # Colorbar — horizontal below the map, ticks at class boundaries
    cbar_left = 0.12
    cbar_width = 0.76
    cbar_ax = fig.add_axes([cbar_left, 0.08, cbar_width, 0.018])
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation='horizontal',
        extend='neither',
        ticks=[-2.0, -1.5, -1.2, -0.7, -0.5, 0.5, 0.7, 1.2, 1.5, 2.0],
    )
    cbar.ax.set_xticklabels(
        ['-2.0', '-1.5', '-1.2', '-0.7', '-0.5', '0.5', '0.7', '1.2', '1.5', '2.0'],
        fontsize=7,
    )
    cbar.ax.set_xlabel(
        INDEX_LABEL,
        fontsize=9, labelpad=6,
    )

    # Category labels above each color segment — use data coordinates on cbar axis
    # BoundaryNorm maps N colors to N+1 boundaries, colorbar x-axis matches data values
    label_fontsize = 5.5

    for i, label in enumerate(SPI_LABELS):
        # Midpoint in SPI data space — the colorbar x-axis IS in data space
        mid = (SPI_BOUNDS[i] + SPI_BOUNDS[i + 1]) / 2.0
        cbar.ax.text(mid, 1.3, label, transform=cbar.ax.get_xaxis_transform(),
                     ha='center', va='bottom',
                     fontsize=label_fontsize, color='#404040')

    # Attribution text
    fig.text(0.02, 0.02, 'Data: CHIRPS v3', fontsize=8, color='#606060',
             ha='left', va='bottom')
    fig.text(0.98, 0.02, 'GOST/DEC Data Group/WBG', fontsize=8, color='#606060',
             ha='right', va='bottom')

    # Save — use pad_inches to control whitespace
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    """Main entry point."""
    print("=" * 60)
    print(f" GLOBAL SPI-12 MAP — {TARGET_DATE}")
    print("=" * 60)

    # Check input exists
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    da = load_data(INPUT_FILE, TARGET_DATE, VAR_NAME)

    # Plot
    plot_global_spi(da, OUTPUT_FILE)

    print("\nDone!")


if __name__ == '__main__':
    main()
