"""
Configuration module for SPI/SPEI climate indices calculation.

Central configuration hub: contains all enums, constants, and user-configurable
settings. All helper/utility functions have been moved to utils.py.

---
Author: Benny Istanto, GOST/DEC Data Group/The World Bank

Built upon the foundation of climate-indices by James Adams, 
with substantial modifications for multi-distribution support, 
bidirectional event analysis, and scalable processing.
---
"""

from enum import Enum


# =============================================================================
# VERSION
# =============================================================================

__version__ = "2026.1"


# =============================================================================
# ENUMS
# =============================================================================

class Periodicity(Enum):
    """
    Enumeration type for specifying dataset periodicity.

    'monthly': array of monthly values, assumed to span full years,
        i.e. the first value corresponds to January of the initial year
        and any missing final months of the final year filled with NaN values,
        with size == # of years * 12

    'daily': array of full years of daily values with 366 days per year,
        as if each year were a leap year and any missing final months of the
        final year filled with NaN values, with array size == (# years * 366)
    """

    monthly = 12
    daily = 366

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str) -> 'Periodicity':
        """
        Convert string to Periodicity enum.

        :param s: string value ('monthly' or 'daily')
        :return: Periodicity enum value
        :raises ValueError: if string doesn't match any periodicity
        """
        try:
            return Periodicity[s.lower()]
        except KeyError:
            raise ValueError(
                f"Invalid periodicity: '{s}'. Must be 'monthly' or 'daily'."
            )

    def unit(self) -> str:
        """
        Return the unit name for this periodicity.

        :return: 'month' for monthly, 'day' for daily
        """
        if self == Periodicity.monthly:
            return "month"
        elif self == Periodicity.daily:
            return "day"
        else:
            raise ValueError(f"No unit defined for periodicity: {self.name}")


# =============================================================================
# CONSTANTS
# =============================================================================

# Valid range for fitted SPI/SPEI values
# Values outside this range are clipped
FITTED_INDEX_VALID_MIN = -3.09
FITTED_INDEX_VALID_MAX = 3.09

# Fill value for missing data in NetCDF files
NC_FILL_VALUE = -9999.0

# Minimum number of non-NaN values required for the vectorized gamma fast path
# in compute.py. Deliberately permissive; gamma has only two parameters.
MIN_VALUES_FOR_GAMMA_FIT = 4

# Minimum number of valid values required to fit a distribution in
# distributions.py (Pearson III, Log-Logistic, GEV, Generalized Logistic).
#
# This MUST stay below the calibration window length. The WMO standard
# calibration 1991-2020 yields exactly 30 values per calendar month, so the
# previous value of 30 left zero tolerance for gaps: a single missing year in
# any calendar month rejected the fit and turned that month's entire
# multi-decade series into NaN, while the gamma path (threshold 4) kept it.
# Measured on a 68-year series, one missing calibration year took Pearson III
# from 68 valid January values to 0.
#
# 20 keeps a three-parameter fit statistically defensible while tolerating up
# to a third of a 30-year window being absent. Raise it if you calibrate over
# a longer period and want stricter completeness.
MIN_VALUES_FOR_FIT = 20

# Minimum number of non-zero values for reliable fitting
MIN_NONZERO_VALUES = 10

# Maximum proportion of zeros allowed (beyond this, fitting is unreliable)
MAX_ZERO_PROPORTION = 0.95

# Default calibration period (WMO standard)
DEFAULT_CALIBRATION_START_YEAR = 1991
DEFAULT_CALIBRATION_END_YEAR = 2020

# Variable naming pattern: {index}_{distribution}_{scale}_{periodicity}
# e.g., spi_gamma_12_month, spei_pearson3_3_month
VAR_NAME_PATTERN = "{index}_{distribution}_{scale}_{periodicity}"

# Default distribution (for backward compatibility)
DEFAULT_DISTRIBUTION = "gamma"

# Supported distributions and their fitting parameter names
DISTRIBUTION_PARAM_NAMES = {
    "gamma": ("alpha", "beta", "prob_zero"),
    "pearson3": ("skew", "loc", "scale", "prob_zero"),
    "log_logistic": ("alpha", "beta", "prob_zero"),
    "gev": ("shape", "loc", "scale", "prob_zero"),
    "gen_logistic": ("shape", "loc", "scale", "prob_zero"),
}

# Human-readable display names for distributions
DISTRIBUTION_DISPLAY_NAMES = {
    'gamma': 'Gamma',
    'pearson3': 'Pearson III',
    'log_logistic': 'Log-Logistic',
    'gev': 'GEV',
    'gen_logistic': 'Generalized Logistic',
}

# Fitting parameter variable names (backward-compatible alias for gamma)
FITTING_PARAM_NAMES = ("alpha", "beta", "prob_zero")

# Offset added to SPEI water balance (P - PET) to ensure positive values
# for probability distribution fitting. Standard value from literature.
SPEI_WATER_BALANCE_OFFSET = 1000.0


# =============================================================================
# VARIABLE DETECTION PATTERNS
# =============================================================================

# Variable name patterns for auto-detection in NetCDF datasets.
# Add patterns here if your data uses different variable names.
PRECIP_VAR_PATTERNS = ['precip', 'prcp', 'precipitation', 'pr', 'ppt', 'rainfall']
PET_VAR_PATTERNS = ['pet', 'eto', 'et', 'evap']
TEMP_VAR_PATTERNS = ['temp', 'tas', 'tasmin', 'tasmax', 't2m', 'tmean', 'tmin', 'tmax']


# =============================================================================
# MEMORY AND PERFORMANCE CONSTANTS
# =============================================================================

# --- Serial chunked processing (chunked.py, utils.get_optimal_chunk_size) ---

# Memory multiplier: peak memory as multiple of input array size.
# Accounts for scaled_data, parameters and intermediate arrays during
# computation, assuming float64 input.
MEMORY_MULTIPLIER = 12.0

# SPEI additionally holds precipitation, PET and the water balance, so its
# peak is higher than SPI's for the same grid.
MEMORY_MULTIPLIER_SPEI = 18.0

# Safety factor: fraction of available memory to use (conservative)
MEMORY_SAFETY_FACTOR = 0.7

# Default chunk sizes for global processing
DEFAULT_CHUNK_LAT = 500
DEFAULT_CHUNK_LON = 500

# Minimum chunk size (too small = inefficient)
MIN_CHUNK_SIZE = 100

# Arrays at or below this size are processed in one shot rather than tiled
MAX_SINGLE_CHUNK_GB = 2.0

# --- Dask tiled processing (dask_processor.py) ---

# Peak working set per tile as a multiple of one float32 tile array. Lower
# than MEMORY_MULTIPLIER because the Dask path works in float32 throughout.
# These are multiplied by the worker count, since Dask holds n_workers tiles
# at once - unlike serial chunking, which holds one.
#
# Raised by 1.0 on 2026-09-19: compute._rolling_sum_3d now accumulates its
# cumulative sum in float64 to avoid catastrophic cancellation, which costs one
# extra tile-sized array at peak. See that function for the measurement.
DASK_TILE_MULTIPLIER_SPI = 9.0
DASK_TILE_MULTIPLIER_SPEI = 12.0

# Upper bound on a Dask tile edge, regardless of available memory
MAX_DASK_TILE = 2048


# =============================================================================
# DEFAULT METADATA ATTRIBUTES
# =============================================================================

# =============================================================================
# METADATA CONFIGURATION
# =============================================================================

# Methodological references for SPI and SPEI.
# These are always included regardless of data source.
_REFERENCES_SPI = (
    'McKee, T.B., Doesken, N.J., Kleist, J. (1993). '
    'The relationship of drought frequency and duration to time scales. '
    '8th Conference on Applied Climatology'
)
_REFERENCES_SPEI = (
    'Vicente-Serrano, S.M., Begueria, S., Lopez-Moreno, J.I. (2010). '
    'A Multiscalar Drought Index Sensitive to Global Warming: '
    'The Standardized Precipitation Evapotranspiration Index. '
    'Journal of Climate, 23(7), 1696-1718'
)
_REFERENCES_DEFAULT = f'{_REFERENCES_SPI}; {_REFERENCES_SPEI}'

# ---------------------------------------------------------------------------
# Data source presets
# ---------------------------------------------------------------------------
# Each preset provides 'source' and 'source_references' for a specific dataset.
# Use with build_metadata(): build_metadata(source='terraclimate', ...)

METADATA_PRESETS = {
    'terraclimate': {
        'source': 'TerraClimate monthly gridded data',
        'source_references': (
            'Abatzoglou, J.T., Dobrowski, S.Z., Parks, S.A., Hegewisch, K.C. (2018). '
            'TerraClimate, a high-resolution global dataset of monthly climate and '
            'climatic water balance from 1958-2015. Scientific Data, 5, 170191'
        ),
    },
    'chirps': {
        'source': 'CHIRPS v3.0 monthly precipitation',
        'source_references': (
            'Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., '
            'Shukla, S., Husak, G., Rowland, J., Harrison, L., Hoell, A., '
            'Michaelsen, J. (2015). The climate hazards infrared precipitation '
            'with stations - a new environmental record for monitoring extremes. '
            'Scientific Data, 2, 150066'
        ),
    },
    'era5_land': {
        'source': 'ERA5-Land monthly averaged reanalysis',
        'source_references': (
            'Munoz-Sabater, J., Dutra, E., Agust-Panareda, A., Albergel, C., '
            'Arduini, G., Balsamo, G., Boussetta, S., Choulga, M., Harrigan, S., '
            'Hersbach, H., Martens, B., Miralles, D.G., Piles, M., '
            'Rodriguez-Fernandez, N.J., Zsoter, E., Buontempo, C., Thepaut, J.N. '
            '(2021). ERA5-Land: a state-of-the-art global reanalysis dataset for '
            'land applications. Earth System Science Data, 13(9), 4349-4383'
        ),
    },
    'imerg': {
        'source': 'GPM IMERG Final Precipitation L3 monthly',
        'source_references': (
            'Huffman, G.J., Stocker, E.F., Bolvin, D.T., Nelkin, E.J., '
            'Tan, J. (2023). GPM IMERG Final Precipitation L3 1 month 0.1 '
            'degree x 0.1 degree V07. Greenbelt, MD, Goddard Earth Sciences '
            'Data and Information Services Center (GES DISC)'
        ),
    },
    'cmorph': {
        'source': 'CMORPH Climate Data Record monthly precipitation',
        'source_references': (
            'Xie, P., Joyce, R., Wu, S., Yoo, S.H., Yarosh, Y., Sun, F., '
            'Lin, R. (2017). Reprocessed, Bias-Corrected CMORPH Global '
            'High-Resolution Precipitation Estimates from 1998. '
            'Journal of Hydrometeorology, 18(6), 1617-1641'
        ),
    },
    'mswep': {
        'source': 'MSWEP v2.8 multi-source weighted-ensemble precipitation',
        'source_references': (
            'Beck, H.E., Wood, E.F., Pan, M., Fisher, C.K., Miralles, D.G., '
            'van Dijk, A.I.J.M., McVicar, T.R., Adler, R.F. (2019). '
            'MSWEP V2 Global 3-Hourly 0.1 Precipitation: Methodology and '
            'Quantitative Assessment. Bulletin of the American Meteorological '
            'Society, 100(3), 473-500'
        ),
    },
    'gpcc': {
        'source': 'GPCC Full Data Monthly Product',
        'source_references': (
            'Schneider, U., Becker, A., Finger, P., Rustemeier, E., Ziese, M. '
            '(2020). GPCC Full Data Monthly Product Version 2020 at 0.25: '
            'Monthly Land-Surface Precipitation from Rain-Gauges built on '
            'GTS-based and Historical Data. '
            'Global Precipitation Climatology Centre (GPCC)'
        ),
    },
}

# ---------------------------------------------------------------------------
# Creator presets
# ---------------------------------------------------------------------------
# Predefined creator profiles. Use with build_metadata(creator='worldbank').

CREATOR_PRESETS = {
    'worldbank': {
        'institution': 'GOST/DEC Data Group, The World Bank',
        'creator_name': 'Benny Istanto',
        'creator_role': 'Climate Geographer',
        'creator_email': 'bistanto@worldbank.org',
    },
}

# ---------------------------------------------------------------------------
# Default metadata
# ---------------------------------------------------------------------------
# Default global attributes for NetCDF output files.
# Uses TerraClimate + World Bank as defaults. Override via:
#   1. build_metadata(source='chirps', creator='worldbank')
#   2. Passing global_attrs={...} to any compute function
#   3. Modifying DEFAULT_METADATA dict directly at runtime
DEFAULT_METADATA = {
    'Conventions': 'CF-1.8',
    'cdm_data_type': 'GRID',
    'institution': CREATOR_PRESETS['worldbank']['institution'],
    'source': METADATA_PRESETS['terraclimate']['source'],
    'references': (
        f"{_REFERENCES_DEFAULT}; "
        f"{METADATA_PRESETS['terraclimate']['source_references']}"
    ),
    'comment': (
        'Computed using precip-index package. '
        'Calibration period defines the baseline for distribution fitting. '
        'The data is developed to support hydrometeorological monitoring '
        'and assessment of extreme dry and wet periods.'
    ),
    'creator_name': CREATOR_PRESETS['worldbank']['creator_name'],
    'creator_role': CREATOR_PRESETS['worldbank']['creator_role'],
    'creator_email': CREATOR_PRESETS['worldbank']['creator_email'],
}


def build_metadata(
    source: str = 'terraclimate',
    creator: str = 'worldbank',
    **kwargs
) -> dict:
    """
    Build a metadata dict from presets and custom overrides.

    Combines data source preset, creator preset, and any custom kwargs
    into a ready-to-use metadata dict for the ``global_attrs`` parameter.

    :param source: data source key from METADATA_PRESETS (e.g., 'terraclimate',
        'chirps', 'era5_land', 'imerg', 'cmorph', 'mswep', 'gpcc'),
        or a custom source string (used as-is if not a known preset key)
    :param creator: creator key from CREATOR_PRESETS (e.g., 'worldbank'),
        or 'custom' to provide creator fields via kwargs
    :param kwargs: additional overrides (highest priority). Common keys:
        institution, creator_name, creator_email, creator_role, comment

    :return: metadata dict ready for ``global_attrs`` parameter

    Examples::

        # Use CHIRPS with default creator
        meta = build_metadata(source='chirps')

        # Use ERA5-Land with custom creator
        meta = build_metadata(
            source='era5_land',
            creator='custom',
            institution='Bogor Agricultural University',
            creator_name='Benny Istanto',
            creator_email='bennyistanto@ipb.ac.id',
        )

        # Fully custom
        meta = build_metadata(
            source='My custom dataset v1.0',
            creator='custom',
            institution='My Institute',
            creator_name='Hello World',
            creator_email='hello@example.com',
        )

        # Pass to compute function
        spi_12 = spi(precip, scale=12, global_attrs=meta)
    """
    meta = {
        'Conventions': 'CF-1.8',
        'cdm_data_type': 'GRID',
    }

    # --- Data source ---
    if source in METADATA_PRESETS:
        preset = METADATA_PRESETS[source]
        meta['source'] = preset['source']
        meta['references'] = (
            f"{_REFERENCES_DEFAULT}; {preset['source_references']}"
        )
    else:
        # Custom source string
        meta['source'] = source
        meta['references'] = _REFERENCES_DEFAULT

    # --- Creator ---
    if creator in CREATOR_PRESETS:
        meta.update(CREATOR_PRESETS[creator])
    # If creator='custom', user provides via kwargs

    # --- Standard comment ---
    meta['comment'] = (
        'Computed using precip-index package. '
        'Calibration period defines the baseline for distribution fitting. '
        'The data is developed to support hydrometeorological monitoring '
        'and assessment of extreme dry and wet periods.'
    )

    # --- User overrides (highest priority) ---
    meta.update(kwargs)

    return meta
