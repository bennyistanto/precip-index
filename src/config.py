"""
Configuration module for SPI/SPEI climate indices calculation.

Contains enums, constants, and logging setup.

Modified/adapted from James Adams' climate-indices package
https://github.com/monocongo/climate_indices

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank
"""

import logging
from enum import Enum


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

# Variable naming pattern: {index}_gamma_{scale}_{periodicity}
# e.g., spi_gamma_12_month, spei_gamma_3_month
VAR_NAME_PATTERN = "{index}_gamma_{scale}_{periodicity}"

# Fitting parameter variable names
FITTING_PARAM_NAMES = ("alpha", "beta", "prob_zero")


# =============================================================================
# LOGGING
# =============================================================================

def get_logger(
    name: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up and return a logger with consistent formatting.

    :param name: logger name (typically __name__ of calling module)
    :param level: logging level (default: logging.INFO)
    :return: configured logger instance
    """
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_variable_name(
    index: str,
    scale: int,
    periodicity: Periodicity
) -> str:
    """
    Generate standardized variable name for SPI/SPEI output.

    :param index: index type ('spi' or 'spei')
    :param scale: time scale (e.g., 1, 3, 6, 12)
    :param periodicity: Periodicity enum value
    :return: formatted variable name (e.g., 'spi_gamma_12_month')
    """
    return VAR_NAME_PATTERN.format(
        index=index.lower(),
        scale=scale,
        periodicity=periodicity.unit()
    )


def get_fitting_param_name(
    param: str,
    scale: int,
    periodicity: Periodicity
) -> str:
    """
    Generate standardized variable name for fitting parameters.

    :param param: parameter name ('alpha', 'beta', or 'prob_zero')
    :param scale: time scale (e.g., 1, 3, 6, 12)
    :param periodicity: Periodicity enum value
    :return: formatted parameter name (e.g., 'alpha_12_month')
    """
    if param not in FITTING_PARAM_NAMES:
        raise ValueError(
            f"Invalid parameter name: '{param}'. "
            f"Must be one of: {FITTING_PARAM_NAMES}"
        )
    return f"{param}_{scale}_{periodicity.unit()}"


def get_long_name(
    index: str,
    scale: int,
    periodicity: Periodicity
) -> str:
    """
    Generate long descriptive name for NetCDF attributes.

    :param index: index type ('spi' or 'spei')
    :param scale: time scale (e.g., 1, 3, 6, 12)
    :param periodicity: Periodicity enum value
    :return: formatted long name
    """
    index_names = {
        'spi': 'Standardized Precipitation Index',
        'spei': 'Standardized Precipitation Evapotranspiration Index'
    }
    
    index_full = index_names.get(index.lower(), index.upper())
    return f"{index_full} (Gamma), {scale}-{periodicity.unit()}"


def get_variable_attributes(
    index: str,
    scale: int,
    periodicity: Periodicity
) -> dict:
    """
    Generate standard NetCDF variable attributes for SPI/SPEI.

    :param index: index type ('spi' or 'spei')
    :param scale: time scale (e.g., 1, 3, 6, 12)
    :param periodicity: Periodicity enum value
    :return: dictionary of attributes
    """
    return {
        'long_name': get_long_name(index, scale, periodicity),
        'standard_name': f'{index.lower()}_gamma_{scale}_{periodicity.unit()}',
        'units': '1',  # dimensionless
        'valid_min': FITTED_INDEX_VALID_MIN,
        'valid_max': FITTED_INDEX_VALID_MAX,
        'distribution': 'gamma',
        'scale': scale,
        'periodicity': periodicity.name,
    }


def get_fitting_param_attributes(
    param: str,
    scale: int,
    periodicity: Periodicity
) -> dict:
    """
    Generate NetCDF attributes for fitting parameter variables.

    :param param: parameter name ('alpha', 'beta', or 'prob_zero')
    :param scale: time scale
    :param periodicity: Periodicity enum value
    :return: dictionary of attributes
    """
    descriptions = {
        'alpha': (
            f"Shape parameter (alpha) of the gamma distribution computed from "
            f"{scale}-{periodicity.unit()} scaled precipitation values"
        ),
        'beta': (
            f"Scale parameter (beta) of the gamma distribution computed from "
            f"{scale}-{periodicity.unit()} scaled precipitation values"
        ),
        'prob_zero': (
            f"Probability of zero precipitation within calibration period for "
            f"{scale}-{periodicity.unit()} scale"
        ),
    }
    
    return {
        'long_name': f"Gamma {param} parameter ({scale}-{periodicity.unit()})",
        'description': descriptions.get(param, f"{param} parameter"),
        'units': '1',
    }
