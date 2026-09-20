"""
Precipitation Index Package - SPI and SPEI for Climate Extremes Monitoring

Monitor both drought (dry) and wet (flood/excess) conditions using Standardized
Precipitation Index (SPI) and Standardized Precipitation Evapotranspiration
Index (SPEI) with Gamma distribution fitting.

The indices work for both climate extremes:
- Negative values indicate dry conditions (drought)
- Positive values indicate wet conditions (flooding/excess precipitation)

Optimized for global-scale gridded data following CF Convention (time, lat, lon).

---
Author: Benny Istanto, GOST/DEC Data Group/The World Bank

Built upon the foundation of climate-indices by James Adams, 
with substantial modifications for multi-distribution support, 
bidirectional event analysis, and scalable processing.
---

References:
    McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought
    frequency and duration to time scales. 8th Conference on Applied Climatology.

    Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010). A Multiscalar
    Drought Index Sensitive to Global Warming: The Standardized Precipitation
    Evapotranspiration Index. Journal of Climate, 23(7), 1696-1718.

Both import styles work:

    # Flat modules - what the notebooks, tests and docs use
    >>> import sys
    >>> sys.path.insert(0, 'src')
    >>> from indices import spi, spei, save_fitting_params, load_fitting_params

    # Package namespace - put the repository root on sys.path instead
    >>> import sys
    >>> sys.path.insert(0, '.')
    >>> import src
    >>> src.spi, src.compute_spei_dask, src.ChunkedProcessor

Example:
    >>> # Calculate SPI-12 for both dry and wet extremes
    >>> spi_12, params = spi(precip_da, scale=12, return_params=True)
    >>>
    >>> # Save parameters for reuse
    >>> save_fitting_params(params, 'spi_params.nc', scale=12, periodicity='monthly')
    >>>
    >>> # Calculate SPEI with PET (monitors both extremes)
    >>> spei_12 = spei(precip_da, pet=pet_da, scale=12)
"""

import os as _os
import sys as _sys

# ---------------------------------------------------------------------------
# Import bootstrap
# ---------------------------------------------------------------------------
# The submodules in this directory import each other by flat, absolute name
# (``from config import ...``, ``from utils import ...``), and every notebook,
# test and doc example does ``sys.path.insert(0, '.../src')`` before importing
# them that way. There is no packaging metadata, so this directory is not an
# installed package.
#
# Previously this file used relative imports (``from .config import ...``),
# which meant ``import src`` always failed with
# ``ModuleNotFoundError: No module named 'config'`` - the submodules could not
# resolve their own flat imports. Everything below was therefore unreachable.
#
# Putting this directory on sys.path first makes ``import src`` work while
# leaving the flat-import style used everywhere else untouched. Converting the
# submodules to relative imports instead would break every existing caller.
_MODULE_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _MODULE_DIR not in _sys.path:
    _sys.path.insert(0, _MODULE_DIR)

from config import __version__

__author__ = "Benny Istanto"
__email__ = "bistanto@worldbank.org"

# Core index functions
from indices import (
    spi,
    spi_multi_scale,
    spei,
    spei_multi_scale,
)

# Parameter I/O
from indices import (
    save_fitting_params,
    load_fitting_params,
)

# Output utilities
from indices import (
    save_index_to_netcdf,
    classify_drought,
    get_drought_area_percentage,
)

# Whole-grid convenience wrappers and memory estimation
from indices import (
    spi_global,
    spei_global,
    estimate_memory_requirements,
)

# Configuration
from config import (
    Periodicity,
    FITTED_INDEX_VALID_MIN,
    FITTED_INDEX_VALID_MAX,
    DEFAULT_CALIBRATION_START_YEAR,
    DEFAULT_CALIBRATION_END_YEAR,
    DEFAULT_METADATA,
    METADATA_PRESETS,
    CREATOR_PRESETS,
    build_metadata,
)

# Utility functions
from utils import (
    calculate_pet,
    eto_thornthwaite,
    ensure_cf_compliant,
    get_data_year_range,
)

# Climate extremes analysis (run theory - works for both dry and wet events)
from runtheory import (
    identify_runs,
    identify_events,  # Works for both dry (negative threshold) and wet (positive threshold)
    calculate_timeseries,
    calculate_events_spatial,
    calculate_interarrival_times,
    summarize_events,
    get_event_state,
    # Temporal aggregation for decision makers
    calculate_period_statistics,
    calculate_annual_statistics,
    compare_periods,
)

# Visualization functions
from visualization import (
    generate_location_filename,
    plot_index,
    plot_events,
    plot_event_characteristics,
    plot_event_timeline,
    plot_spatial_stats,
)

# Low-level compute functions (for advanced users)
from compute import (
    sum_to_scale,
    gamma_parameters,
    transform_fitted_gamma,
    compute_index_parallel,
    compute_index_dask,
    compute_index_dask_to_zarr,
    compute_spi_1d,
    compute_spei_1d,
)

# Chunked processing for grids larger than RAM (serial tiles)
# Only numpy/xarray are imported at module level here, so this does not pull
# in dask; dask is required when the tiling functions actually run.
from chunked import (
    ChunkedProcessor,
    ChunkInfo,
    MemoryEstimate,
    estimate_memory,
    estimate_memory_from_data,
    iter_chunks,
    compute_spi_global,
    compute_spei_global,
)

# Dask processing (parallel tiles, keeps distribution fitting parameters)
from dask_processor import (
    DaskLayout,
    plan_layout,
    compute_spi_dask,
    compute_spei_dask,
    zarr_to_netcdf,
)

__all__ = [
    # Version
    "__version__",
    # Core functions
    "spi",
    "spi_multi_scale", 
    "spei",
    "spei_multi_scale",
    # Parameter I/O
    "save_fitting_params",
    "load_fitting_params",
    # Output utilities
    "save_index_to_netcdf",
    "classify_drought",
    "get_drought_area_percentage",
    # Whole-grid wrappers and memory estimation
    "spi_global",
    "spei_global",
    "estimate_memory_requirements",
    # Configuration
    "Periodicity",
    "FITTED_INDEX_VALID_MIN",
    "FITTED_INDEX_VALID_MAX",
    "DEFAULT_CALIBRATION_START_YEAR",
    "DEFAULT_CALIBRATION_END_YEAR",
    "DEFAULT_METADATA",
    "METADATA_PRESETS",
    "CREATOR_PRESETS",
    "build_metadata",
    # Utilities
    "calculate_pet",
    "eto_thornthwaite",
    "ensure_cf_compliant",
    "get_data_year_range",
    # Climate extremes analysis (run theory - works for both dry and wet events)
    "identify_runs",
    "identify_events",  # Works for both dry and wet with threshold direction
    "calculate_timeseries",
    "calculate_events_spatial",
    "calculate_interarrival_times",
    "summarize_events",
    "get_event_state",
    "calculate_period_statistics",
    "calculate_annual_statistics",
    "compare_periods",
    # Visualization
    "generate_location_filename",
    "plot_index",
    "plot_events",
    "plot_event_characteristics",
    "plot_event_timeline",
    "plot_spatial_stats",
    # Low-level compute
    "sum_to_scale",
    "gamma_parameters",
    "transform_fitted_gamma",
    "compute_index_parallel",
    "compute_index_dask",
    "compute_index_dask_to_zarr",
    "compute_spi_1d",
    "compute_spei_1d",
    # Chunked processing (serial tiles)
    "ChunkedProcessor",
    "ChunkInfo",
    "MemoryEstimate",
    "estimate_memory",
    "estimate_memory_from_data",
    "iter_chunks",
    "compute_spi_global",
    "compute_spei_global",
    # Dask processing (parallel tiles, keeps fitting parameters)
    "DaskLayout",
    "plan_layout",
    "compute_spi_dask",
    "compute_spei_dask",
    "zarr_to_netcdf",
]
