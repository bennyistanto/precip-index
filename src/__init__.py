"""
Precipitation Index Package - SPI and SPEI for Climate Extremes Monitoring

Monitor both drought (dry) and wet (flood/excess) conditions using Standardized
Precipitation Index (SPI) and Standardized Precipitation Evapotranspiration
Index (SPEI) with Gamma distribution fitting.

The indices work for both climate extremes:
- Negative values indicate dry conditions (drought)
- Positive values indicate wet conditions (flooding/excess precipitation)

Optimized for global-scale gridded data following CF Convention (time, lat, lon).

Modified/adapted from James Adams' climate-indices package
https://github.com/monocongo/climate_indices

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank

References:
    McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought
    frequency and duration to time scales. 8th Conference on Applied Climatology.

    Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010). A Multiscalar
    Drought Index Sensitive to Global Warming: The Standardized Precipitation
    Evapotranspiration Index. Journal of Climate, 23(7), 1696-1718.

Example:
    >>> import sys
    >>> sys.path.insert(0, 'src')
    >>> from indices import spi, spei, save_fitting_params, load_fitting_params
    >>>
    >>> # Calculate SPI-12 for both dry and wet extremes
    >>> spi_12, params = spi(precip_da, scale=12, return_params=True)
    >>>
    >>> # Save parameters for reuse
    >>> save_fitting_params(params, 'spi_params.nc', scale=12, periodicity='monthly')
    >>>
    >>> # Calculate SPEI with PET (monitors both extremes)
    >>> spei_12 = spei(precip_da, pet=pet_da, scale=12)
"""

__version__ = "2026.1"
__author__ = "Benny Istanto"
__email__ = "bistanto@worldbank.org"

# Core index functions
from .indices import (
    spi,
    spi_multi_scale,
    spei,
    spei_multi_scale,
)

# Parameter I/O
from .indices import (
    save_fitting_params,
    load_fitting_params,
)

# Output utilities
from .indices import (
    save_index_to_netcdf,
    classify_drought,
    get_drought_area_percentage,
)

# Configuration
from .config import (
    Periodicity,
    FITTED_INDEX_VALID_MIN,
    FITTED_INDEX_VALID_MAX,
    DEFAULT_CALIBRATION_START_YEAR,
    DEFAULT_CALIBRATION_END_YEAR,
)

# Utility functions
from .utils import (
    calculate_pet,
    eto_thornthwaite,
    ensure_cf_compliant,
    get_data_year_range,
)

# Climate extremes analysis (run theory - works for both dry and wet events)
from .runtheory import (
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
from .visualization import (
    generate_location_filename,
    plot_index,
    plot_events,
    plot_event_characteristics,
    plot_event_timeline,
    plot_spatial_stats,
)

# Low-level compute functions (for advanced users)
from .compute import (
    sum_to_scale,
    gamma_parameters,
    transform_fitted_gamma,
    compute_index_parallel,
    compute_index_dask,
    compute_spi_1d,
    compute_spei_1d,
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
    # Configuration
    "Periodicity",
    "FITTED_INDEX_VALID_MIN",
    "FITTED_INDEX_VALID_MAX",
    "DEFAULT_CALIBRATION_START_YEAR",
    "DEFAULT_CALIBRATION_END_YEAR",
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
    "compute_spi_1d",
    "compute_spei_1d",
]
