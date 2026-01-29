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

# Minimum number of non-NaN values required for gamma fitting
MIN_VALUES_FOR_GAMMA_FIT = 4

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

# Memory multiplier: peak memory as multiple of input array size
# Accounts for scaled_data, parameters, intermediate arrays during computation
MEMORY_MULTIPLIER = 12.0

# Safety factor: fraction of available memory to use (conservative)
MEMORY_SAFETY_FACTOR = 0.7

# Default chunk sizes for global processing
DEFAULT_CHUNK_LAT = 500
DEFAULT_CHUNK_LON = 500

# Minimum chunk size (too small = inefficient)
MIN_CHUNK_SIZE = 100

# Maximum recommended array size in GB for single-chunk processing
MAX_SINGLE_CHUNK_GB = 2.0


# =============================================================================
# DEFAULT METADATA ATTRIBUTES
# =============================================================================

# Default global attributes for NetCDF output files.
# Users can override these by passing global_attrs parameter or
# by modifying these defaults before calling computation functions.
#
# Example — set your institution once at the start of your script:
#   from config import DEFAULT_METADATA
#   DEFAULT_METADATA['institution'] = 'My University'
#   DEFAULT_METADATA['references'] = 'McKee et al. (1993)'
DEFAULT_METADATA = {
    'institution': '',
    'source': 'precip-index package',
    'Conventions': 'CF-1.8',
    'references': '',
    'comment': '',
}
