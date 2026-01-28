"""
Utility functions for SPI/SPEI climate indices calculation.

Includes data transformations, array reshaping, PET calculation,
and helper functions for variable naming and metadata generation.
All functions follow CF Convention with dimension order: (time, lat, lon)

---
Author: Benny Istanto, GOST/DEC Data Group/The World Bank

Built upon the foundation of climate-indices by James Adams, 
with substantial modifications for multi-distribution support, 
bidirectional event analysis, and scalable processing.
---
"""

import calendar
import logging
import math
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

from config import (
    DEFAULT_DISTRIBUTION,
    DEFAULT_METADATA,
    DISTRIBUTION_DISPLAY_NAMES,
    DISTRIBUTION_PARAM_NAMES,
    FITTED_INDEX_VALID_MAX,
    FITTED_INDEX_VALID_MIN,
    FITTING_PARAM_NAMES,
    Periodicity,
    VAR_NAME_PATTERN,
)


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


# Module logger
_logger = get_logger(__name__)


# =============================================================================
# ARRAY VALIDATION AND RESHAPING
# =============================================================================

def reshape_to_2d(
    values: np.ndarray,
    periods_per_year: int
) -> np.ndarray:
    """
    Reshape a 1-D array of values to 2-D array with shape (years, periods).
    
    For monthly data: (total_months,) -> (years, 12)
    For daily data: (total_days,) -> (years, 366)

    :param values: 1-D numpy array of values
    :param periods_per_year: 12 for monthly, 366 for daily
    :return: 2-D numpy array with shape (years, periods_per_year)
    :raises ValueError: if input array has invalid shape
    """
    shape = values.shape
    
    # If already 2-D with correct shape, return as-is
    if len(shape) == 2:
        if shape[1] == periods_per_year:
            return values
        else:
            raise ValueError(
                f"2-D array has incorrect second dimension: {shape[1]}. "
                f"Expected: {periods_per_year}"
            )
    
    # Must be 1-D
    if len(shape) != 1:
        raise ValueError(
            f"Invalid array shape: {shape}. Expected 1-D or 2-D array."
        )
    
    # Pad array if necessary to make it evenly divisible
    total_values = shape[0]
    remainder = total_values % periods_per_year
    
    if remainder != 0:
        padding_size = periods_per_year - remainder
        values = np.pad(
            values,
            (0, padding_size),
            mode='constant',
            constant_values=np.nan
        )
        _logger.debug(
            f"Padded array with {padding_size} NaN values to complete "
            f"final year ({total_values} -> {values.size})"
        )
    
    # Reshape to (years, periods_per_year)
    num_years = values.size // periods_per_year
    return values.reshape(num_years, periods_per_year)


def validate_array(
    values: np.ndarray,
    periodicity: Periodicity
) -> np.ndarray:
    """
    Validate and reshape input array for index calculation.
    
    Converts 1-D array to 2-D array with shape (years, periods).

    :param values: input array (1-D or 2-D)
    :param periodicity: data periodicity (monthly or daily)
    :return: validated 2-D array with shape (years, periods)
    :raises ValueError: if array shape is invalid
    """
    periods_per_year = periodicity.value  # 12 or 366
    
    if len(values.shape) == 1:
        return reshape_to_2d(values, periods_per_year)
    
    elif len(values.shape) == 2:
        if values.shape[1] not in (12, 366):
            raise ValueError(
                f"Invalid 2-D array shape: {values.shape}. "
                f"Second dimension must be 12 (monthly) or 366 (daily)."
            )
        return values
    
    else:
        raise ValueError(
            f"Invalid array dimensions: {len(values.shape)}. "
            f"Expected 1-D or 2-D array."
        )


def is_data_valid(data: np.ndarray) -> bool:
    """
    Check if data array contains at least one non-NaN value.

    :param data: numpy array or masked array
    :return: True if array has at least one valid (non-NaN) value
    """
    if np.ma.isMaskedArray(data):
        return bool(data.count() > 0)
    elif isinstance(data, np.ndarray):
        return not np.all(np.isnan(data))
    else:
        _logger.warning(f"Unexpected data type: {type(data)}")
        return False


# =============================================================================
# DAILY DATA CALENDAR TRANSFORMS (366-day <-> Gregorian)
# =============================================================================

def transform_to_366day(
    original: np.ndarray,
    year_start: int,
    total_years: int
) -> np.ndarray:
    """
    Convert daily values from Gregorian calendar to 366-day calendar.
    
    Non-leap years get a synthetic Feb 29th value (average of Feb 28 and Mar 1).
    This is required for consistent array shapes in daily calculations.

    :param original: 1-D array of daily values in Gregorian calendar
    :param year_start: starting year of the data
    :param total_years: total number of years in the data
    :return: 1-D array with shape (total_years * 366,)
    """
    if len(original.shape) != 1:
        raise ValueError("Input array must be 1-D")
    
    # Allocate output array
    all_leap = np.full((total_years * 366,), np.nan)
    
    original_index = 0
    all_leap_index = 0
    
    for year in range(year_start, year_start + total_years):
        if calendar.isleap(year):
            # Copy all 366 days directly
            days_to_copy = min(366, len(original) - original_index)
            all_leap[all_leap_index:all_leap_index + days_to_copy] = \
                original[original_index:original_index + days_to_copy]
            original_index += 366
        else:
            # Copy Jan 1 through Feb 28 (59 days)
            days_available = len(original) - original_index
            jan_feb = min(59, days_available)
            all_leap[all_leap_index:all_leap_index + jan_feb] = \
                original[original_index:original_index + jan_feb]
            
            # Create synthetic Feb 29th (average of Feb 28 and Mar 1)
            if days_available > 59:
                all_leap[all_leap_index + 59] = (
                    original[original_index + 58] + 
                    original[original_index + 59]
                ) / 2.0
            
            # Copy Mar 1 through Dec 31 (306 days)
            if days_available > 59:
                remaining = min(306, days_available - 59)
                all_leap[all_leap_index + 60:all_leap_index + 60 + remaining] = \
                    original[original_index + 59:original_index + 59 + remaining]
            
            original_index += 365
        
        all_leap_index += 366
    
    return all_leap


def transform_to_gregorian(
    original: np.ndarray,
    year_start: int
) -> np.ndarray:
    """
    Convert daily values from 366-day calendar back to Gregorian calendar.
    
    Removes synthetic Feb 29th from non-leap years.

    :param original: 1-D array with 366 days per year
    :param year_start: starting year of the data
    :return: 1-D array in Gregorian calendar
    """
    if len(original.shape) != 1:
        raise ValueError("Input array must be 1-D")
    
    if original.size % 366 != 0:
        raise ValueError(
            f"Array size ({original.size}) must be a multiple of 366"
        )
    
    total_years = original.size // 366
    year_end = year_start + total_years - 1
    
    # Calculate actual number of days
    days_actual = (
        datetime(year_end, 12, 31) - datetime(year_start, 1, 1)
    ).days + 1
    
    gregorian = np.full((days_actual,), np.nan)
    
    original_index = 0
    gregorian_index = 0
    
    for year in range(year_start, year_start + total_years):
        if calendar.isleap(year):
            # Copy all 366 days
            gregorian[gregorian_index:gregorian_index + 366] = \
                original[original_index:original_index + 366]
            gregorian_index += 366
        else:
            # Copy Jan 1 through Feb 28 (59 days)
            gregorian[gregorian_index:gregorian_index + 59] = \
                original[original_index:original_index + 59]
            # Skip Feb 29, copy Mar 1 through Dec 31 (306 days)
            gregorian[gregorian_index + 59:gregorian_index + 365] = \
                original[original_index + 60:original_index + 366]
            gregorian_index += 365
        
        original_index += 366
    
    return gregorian


def gregorian_length_as_366day(
    length_gregorian: int,
    year_start: int
) -> int:
    """
    Calculate equivalent 366-day calendar length for a Gregorian length.

    :param length_gregorian: number of days in Gregorian calendar
    :param year_start: starting year
    :return: equivalent length in 366-day calendar
    """
    year = year_start
    remaining = length_gregorian
    length_366day = 0
    
    while remaining > 0:
        days_in_year = 366 if calendar.isleap(year) else 365
        
        if remaining >= days_in_year:
            length_366day += 366
        else:
            length_366day += remaining
        
        remaining -= days_in_year
        year += 1
    
    return length_366day


# =============================================================================
# TIME COORDINATE UTILITIES
# =============================================================================

def compute_time_values(
    initial_year: int,
    total_periods: int,
    periodicity: Periodicity,
    initial_month: int = 1,
    units_start_year: int = 1800
) -> np.ndarray:
    """
    Compute time coordinate values in "days since" units.
    
    Useful for creating CF-compliant time coordinates.

    :param initial_year: starting year of the data
    :param total_periods: total number of time steps
    :param periodicity: monthly or daily
    :param initial_month: starting month (1=January, default)
    :param units_start_year: reference year for "days since" (default: 1800)
    :return: array of time values in days since reference date
    """
    start_date = datetime(units_start_year, 1, 1)
    days = np.empty(total_periods, dtype=int)
    
    if periodicity == Periodicity.monthly:
        for i in range(total_periods):
            years = (i + initial_month - 1) // 12
            months = (i + initial_month - 1) % 12
            current_date = datetime(initial_year + years, 1 + months, 1)
            days[i] = (current_date - start_date).days
    
    elif periodicity == Periodicity.daily:
        current_date = datetime(initial_year, initial_month, 1)
        for i in range(total_periods):
            days[i] = (current_date - start_date).days
            # Advance one day
            try:
                current_date = datetime.fromordinal(current_date.toordinal() + 1)
            except ValueError:
                break
    
    return days


# =============================================================================
# POTENTIAL EVAPOTRANSPIRATION (PET) - THORNTHWAITE METHOD
# =============================================================================

# Days in each month
_MONTH_DAYS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_DAYS_LEAP = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# Solar constant [MJ m-2 min-1]
_SOLAR_CONSTANT = 0.0820

# Valid latitude range in radians
_LAT_RAD_MIN = np.deg2rad(-90.0)
_LAT_RAD_MAX = np.deg2rad(90.0)

# Valid solar declination range in radians (±23.45°)
_SOLAR_DEC_MIN = np.deg2rad(-23.45)
_SOLAR_DEC_MAX = np.deg2rad(23.45)


def _solar_declination(day_of_year: int) -> float:
    """
    Calculate solar declination angle for a given day of year.
    
    Based on FAO equation 24 in Allen et al. (1998).

    :param day_of_year: day of year (1-366)
    :return: solar declination in radians
    """
    if not 1 <= day_of_year <= 366:
        raise ValueError(f"Day of year must be 1-366, got: {day_of_year}")
    
    return 0.409 * math.sin((2.0 * math.pi / 365.0) * day_of_year - 1.39)


def _sunset_hour_angle(
    latitude_rad: float,
    solar_dec_rad: float
) -> float:
    """
    Calculate sunset hour angle from latitude and solar declination.
    
    Based on FAO equation 25 in Allen et al. (1998).

    :param latitude_rad: latitude in radians
    :param solar_dec_rad: solar declination in radians
    :return: sunset hour angle in radians
    """
    # Validate inputs
    if not _LAT_RAD_MIN <= latitude_rad <= _LAT_RAD_MAX:
        raise ValueError(
            f"Latitude must be between {_LAT_RAD_MIN:.4f} and "
            f"{_LAT_RAD_MAX:.4f} radians, got: {latitude_rad:.4f}"
        )
    
    # Calculate cosine of sunset hour angle
    cos_sha = -math.tan(latitude_rad) * math.tan(solar_dec_rad)
    
    # Clamp to valid range for acos [-1, 1]
    cos_sha = max(-1.0, min(1.0, cos_sha))
    
    return math.acos(cos_sha)


def _daylight_hours(sunset_hour_angle_rad: float) -> float:
    """
    Calculate daylight hours from sunset hour angle.
    
    Based on FAO equation 34 in Allen et al. (1998).

    :param sunset_hour_angle_rad: sunset hour angle in radians
    :return: daylight hours
    """
    return (24.0 / math.pi) * sunset_hour_angle_rad


def _monthly_mean_daylight_hours(
    latitude_rad: float,
    leap: bool = False
) -> np.ndarray:
    """
    Calculate mean daylight hours for each month at given latitude.

    :param latitude_rad: latitude in radians
    :param leap: whether to calculate for leap year
    :return: array of 12 monthly mean daylight hours
    """
    month_days = _MONTH_DAYS_LEAP if leap else _MONTH_DAYS_NONLEAP
    monthly_dlh = np.zeros(12)
    
    day_of_year = 1
    for month_idx, days_in_month in enumerate(month_days):
        cumulative_hours = 0.0
        for _ in range(days_in_month):
            solar_dec = _solar_declination(day_of_year)
            sunset_angle = _sunset_hour_angle(latitude_rad, solar_dec)
            cumulative_hours += _daylight_hours(sunset_angle)
            day_of_year += 1
        
        monthly_dlh[month_idx] = cumulative_hours / days_in_month
    
    return monthly_dlh


def eto_thornthwaite(
    temperature_celsius: np.ndarray,
    latitude_degrees: float,
    data_start_year: int
) -> np.ndarray:
    """
    Calculate monthly potential evapotranspiration (PET) using Thornthwaite method.
    
    Reference:
        Thornthwaite, C.W. (1948) An approach toward a rational classification
        of climate. Geographical Review, Vol. 38, 55-94.
    
    Thornthwaite equation:
        PET = 1.6 * (L/12) * (N/30) * (10*Ta / I)^a
    
    where:
        - Ta: mean daily air temperature (°C, clipped to ≥0)
        - N: number of days in month
        - L: mean day length (hours)
        - I: annual heat index
        - a: coefficient based on heat index

    :param temperature_celsius: array of monthly mean temperatures in °C
        Can be 1-D (months,) or 2-D (years, 12)
    :param latitude_degrees: latitude in degrees north (-90 to 90)
    :param data_start_year: starting year of the data
    :return: array of monthly PET values in mm/month, same shape as input
    """
    original_length = temperature_celsius.size
    
    # Reshape to (years, 12)
    temps = reshape_to_2d(temperature_celsius.copy(), 12)
    
    # Convert latitude to radians
    latitude_rad = math.radians(float(latitude_degrees))
    
    # Clip negative temperatures to zero (no evaporation below freezing)
    temps = np.where(temps < 0, 0.0, temps)
    
    # Calculate monthly means across all years
    mean_monthly_temps = np.nanmean(temps, axis=0)
    
    # Calculate heat index (I)
    heat_index = np.sum(np.power(mean_monthly_temps / 5.0, 1.514))
    
    if heat_index == 0:
        _logger.warning("Heat index is zero, returning zero PET")
        return np.zeros(original_length)
    
    # Calculate exponent coefficient (a)
    a = (
        6.75e-07 * heat_index**3 -
        7.71e-05 * heat_index**2 +
        1.792e-02 * heat_index +
        0.49239
    )
    
    # Get mean daylight hours for leap and non-leap years
    dlh_nonleap = _monthly_mean_daylight_hours(latitude_rad, leap=False)
    dlh_leap = _monthly_mean_daylight_hours(latitude_rad, leap=True)
    
    # Calculate PET for each year
    pet = np.full(temps.shape, np.nan)
    
    for year_idx in range(temps.shape[0]):
        year = data_start_year + year_idx
        
        if calendar.isleap(year):
            month_days = _MONTH_DAYS_LEAP
            dlh = dlh_leap
        else:
            month_days = _MONTH_DAYS_NONLEAP
            dlh = dlh_nonleap
        
        # Thornthwaite equation
        pet[year_idx, :] = (
            16.0 *
            (dlh / 12.0) *
            (month_days / 30.0) *
            np.power(10.0 * temps[year_idx, :] / heat_index, a)
        )
    
    # Reshape back to 1-D and truncate to original length
    return pet.reshape(-1)[:original_length]


def calculate_pet(
    temperature: Union[np.ndarray, xr.DataArray],
    latitude: Union[float, np.ndarray, xr.DataArray],
    data_start_year: int
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate PET from temperature data, handling both arrays and DataArrays.
    
    Wrapper around eto_thornthwaite that handles xarray inputs.

    :param temperature: monthly temperature in °C
        - numpy array: shape (time,) or (time, lat, lon) following CF Convention
        - xarray DataArray: with 'time' dimension
    :param latitude: latitude in degrees
        - float: single value for all data
        - array: latitude values matching spatial dimensions
    :param data_start_year: starting year of the data
    :return: PET array in mm/month, same type and shape as input temperature
    """
    # Handle xarray DataArray
    if isinstance(temperature, xr.DataArray):
        _logger.info("Calculating PET from xarray DataArray")

        # Ensure CF Convention dimension order (time, lat, lon)
        if temperature.ndim == 3:
            expected_order = ('time', 'lat', 'lon')
            if temperature.dims != expected_order:
                _logger.info(f"Transposing temperature dimensions from {temperature.dims} to {expected_order}")
                temperature = temperature.transpose(*expected_order)

        # Get latitude values
        if isinstance(latitude, xr.DataArray):
            lat_values = latitude.values
        elif isinstance(latitude, np.ndarray):
            lat_values = latitude
        else:
            lat_values = float(latitude)

        # Check if we have spatial dimensions
        if 'lat' in temperature.dims and 'lon' in temperature.dims:
            # 3D data: (time, lat, lon) - CF Convention
            pet_data = np.full(temperature.shape, np.nan)
            
            # Get latitude array
            if isinstance(lat_values, (int, float)):
                lat_array = np.full(temperature.shape[1], lat_values)
            else:
                lat_array = lat_values
            
            # Process each grid point
            for lat_idx in range(temperature.shape[1]):
                for lon_idx in range(temperature.shape[2]):
                    temp_series = temperature[:, lat_idx, lon_idx].values
                    lat_val = lat_array[lat_idx] if lat_array.ndim >= 1 else lat_array
                    
                    if not np.all(np.isnan(temp_series)) and -90 < lat_val < 90:
                        pet_data[:, lat_idx, lon_idx] = eto_thornthwaite(
                            temp_series, lat_val, data_start_year
                        )
            
            # Return as DataArray with same coordinates
            return xr.DataArray(
                data=pet_data,
                dims=temperature.dims,
                coords=temperature.coords,
                attrs={
                    'long_name': 'Potential Evapotranspiration (Thornthwaite)',
                    'units': 'mm/month',
                    'method': 'Thornthwaite (1948)',
                }
            )
        else:
            # 1D data: just time series
            pet_values = eto_thornthwaite(
                temperature.values,
                float(lat_values) if np.ndim(lat_values) == 0 else float(lat_values[0]),
                data_start_year
            )
            return xr.DataArray(
                data=pet_values,
                dims=temperature.dims,
                coords=temperature.coords,
                attrs={
                    'long_name': 'Potential Evapotranspiration (Thornthwaite)',
                    'units': 'mm/month',
                    'method': 'Thornthwaite (1948)',
                }
            )
    
    # Handle numpy array
    else:
        if isinstance(latitude, (int, float)):
            return eto_thornthwaite(temperature, latitude, data_start_year)
        else:
            raise ValueError(
                "For numpy array input with multiple latitudes, "
                "use xarray DataArray instead"
            )


# =============================================================================
# XARRAY UTILITIES
# =============================================================================

def ensure_cf_compliant(
    ds: xr.Dataset,
    var_name: str
) -> xr.Dataset:
    """
    Ensure dataset follows CF Convention dimension order: (time, lat, lon).
    
    Transposes dimensions if necessary.

    :param ds: xarray Dataset
    :param var_name: name of the main data variable
    :return: Dataset with CF-compliant dimension order
    """
    da = ds[var_name]
    dims = da.dims
    
    # Expected CF Convention order
    cf_order_3d = ('time', 'lat', 'lon')
    cf_order_2d = ('lat', 'lon')
    cf_order_1d = ('time',)
    
    if len(dims) == 3:
        if dims != cf_order_3d:
            _logger.info(f"Transposing dimensions from {dims} to {cf_order_3d}")
            ds[var_name] = da.transpose(*cf_order_3d)
    elif len(dims) == 2:
        if set(dims) == {'lat', 'lon'} and dims != cf_order_2d:
            _logger.info(f"Transposing dimensions from {dims} to {cf_order_2d}")
            ds[var_name] = da.transpose(*cf_order_2d)
    
    return ds


def get_data_year_range(
    ds: xr.Dataset
) -> Tuple[int, int]:
    """
    Extract start and end year from dataset time coordinate.

    :param ds: xarray Dataset with 'time' coordinate
    :return: tuple of (start_year, end_year)
    """
    time_coord = ds['time']
    
    # Handle different time coordinate types
    if np.issubdtype(time_coord.dtype, np.datetime64):
        start_year = int(time_coord[0].dt.year)
        end_year = int(time_coord[-1].dt.year)
    else:
        # Assume CF time units, try to decode
        start_year = int(str(time_coord[0].values)[:4])
        end_year = int(str(time_coord[-1].values)[:4])
    
    return start_year, end_year


def count_zeros_and_non_missing(values: np.ndarray) -> Tuple[int, int]:
    """
    Count zeros and non-missing values in an array.

    :param values: numpy array
    :return: tuple of (zero_count, non_missing_count)
    """
    values = np.asarray(values)
    zeros = np.sum(values == 0)
    non_missing = np.sum(~np.isnan(values))
    return int(zeros), int(non_missing)


# =============================================================================
# MEMORY AND PERFORMANCE UTILITIES
# =============================================================================

def get_optimal_chunk_size(
    n_time: int,
    n_lat: int,
    n_lon: int,
    available_memory_gb: float = None,
    memory_multiplier: float = 12.0,
    safety_factor: float = 0.7
) -> Tuple[int, int]:
    """
    Calculate optimal spatial chunk size based on available memory.

    :param n_time: Number of time steps
    :param n_lat: Number of latitude points
    :param n_lon: Number of longitude points
    :param available_memory_gb: Available RAM in GB (auto-detected if None)
    :param memory_multiplier: Peak memory as multiple of input
    :param safety_factor: Fraction of available memory to use
    :return: Tuple of (chunk_lat, chunk_lon)
    """
    # Auto-detect available memory
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            _logger.warning("psutil not installed, assuming 16GB available")
            available_memory_gb = 16.0

    # Calculate target chunk size
    usable_memory_gb = available_memory_gb * safety_factor
    bytes_per_element = 8  # float64

    # Target cells that fit in memory with multiplier
    target_cells = int(
        (usable_memory_gb / memory_multiplier) * (1024**3) / (n_time * bytes_per_element)
    )

    # Make roughly square chunks
    chunk_size = int(np.sqrt(target_cells))

    # Apply bounds
    chunk_lat = min(max(chunk_size, 100), n_lat)
    chunk_lon = min(max(chunk_size, 100), n_lon)

    return chunk_lat, chunk_lon


def format_bytes(n_bytes: int) -> str:
    """
    Format byte count as human-readable string.

    :param n_bytes: Number of bytes
    :return: Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.2f} PB"


def get_array_memory_size(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> int:
    """
    Calculate memory size of array with given shape and dtype.

    :param shape: Array shape tuple
    :param dtype: Data type (default: float64)
    :return: Size in bytes
    """
    return int(np.prod(shape) * np.dtype(dtype).itemsize)


def print_memory_info():
    """Print current memory usage information."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        _logger.info(
            f"Memory: {format_bytes(mem.used)} used / "
            f"{format_bytes(mem.total)} total "
            f"({mem.percent}% used)"
        )
    except ImportError:
        _logger.warning("psutil not installed, cannot report memory info")


# =============================================================================
# VARIABLE NAMING AND METADATA HELPERS
# =============================================================================

def get_variable_name(
    index: str,
    scale: int,
    periodicity: Periodicity,
    distribution: str = DEFAULT_DISTRIBUTION
) -> str:
    """
    Generate standardized variable name for SPI/SPEI output.

    :param index: index type ('spi' or 'spei')
    :param scale: time scale (e.g., 1, 3, 6, 12)
    :param periodicity: Periodicity enum value
    :param distribution: distribution name (e.g., 'gamma', 'pearson3')
    :return: formatted variable name (e.g., 'spi_gamma_12_month')
    """
    return VAR_NAME_PATTERN.format(
        index=index.lower(),
        distribution=distribution.lower(),
        scale=scale,
        periodicity=periodicity.unit()
    )


def get_fitting_param_name(
    param: str,
    scale: int,
    periodicity: Periodicity,
    distribution: str = DEFAULT_DISTRIBUTION
) -> str:
    """
    Generate standardized variable name for fitting parameters.

    :param param: parameter name (e.g., 'alpha', 'beta', 'skew', 'loc', 'scale', 'prob_zero')
    :param scale: time scale (e.g., 1, 3, 6, 12)
    :param periodicity: Periodicity enum value
    :param distribution: distribution name (e.g., 'gamma', 'pearson3')
    :return: formatted parameter name (e.g., 'alpha_12_month')
    """
    dist_key = distribution.lower()
    valid_params = DISTRIBUTION_PARAM_NAMES.get(dist_key, FITTING_PARAM_NAMES)
    if param not in valid_params:
        raise ValueError(
            f"Invalid parameter name '{param}' for distribution '{dist_key}'. "
            f"Must be one of: {valid_params}"
        )
    return f"{param}_{scale}_{periodicity.unit()}"


def get_long_name(
    index: str,
    scale: int,
    periodicity: Periodicity,
    distribution: str = DEFAULT_DISTRIBUTION
) -> str:
    """
    Generate long descriptive name for NetCDF attributes.

    :param index: index type ('spi' or 'spei')
    :param scale: time scale (e.g., 1, 3, 6, 12)
    :param periodicity: Periodicity enum value
    :param distribution: distribution name (e.g., 'gamma', 'pearson3')
    :return: formatted long name
    """
    index_names = {
        'spi': 'Standardized Precipitation Index',
        'spei': 'Standardized Precipitation Evapotranspiration Index'
    }

    index_full = index_names.get(index.lower(), index.upper())
    dist_name = DISTRIBUTION_DISPLAY_NAMES.get(distribution.lower(), distribution)
    return f"{index_full} ({dist_name}), {scale}-{periodicity.unit()}"


def get_variable_attributes(
    index: str,
    scale: int,
    periodicity: Periodicity,
    distribution: str = DEFAULT_DISTRIBUTION
) -> dict:
    """
    Generate standard NetCDF variable attributes for SPI/SPEI.

    :param index: index type ('spi' or 'spei')
    :param scale: time scale (e.g., 1, 3, 6, 12)
    :param periodicity: Periodicity enum value
    :param distribution: distribution name (e.g., 'gamma', 'pearson3')
    :return: dictionary of attributes
    """
    dist = distribution.lower()
    return {
        'long_name': get_long_name(index, scale, periodicity, dist),
        'standard_name': f'{index.lower()}_{dist}_{scale}_{periodicity.unit()}',
        'units': '1',  # dimensionless
        'valid_min': FITTED_INDEX_VALID_MIN,
        'valid_max': FITTED_INDEX_VALID_MAX,
        'distribution': dist,
        'scale': scale,
        'periodicity': periodicity.name,
    }


def get_fitting_param_attributes(
    param: str,
    scale: int,
    periodicity: Periodicity,
    distribution: str = DEFAULT_DISTRIBUTION
) -> dict:
    """
    Generate NetCDF attributes for fitting parameter variables.

    :param param: parameter name (e.g., 'alpha', 'beta', 'skew', 'loc', 'scale', 'prob_zero')
    :param scale: time scale
    :param periodicity: Periodicity enum value
    :param distribution: distribution name (e.g., 'gamma', 'pearson3')
    :return: dictionary of attributes
    """
    dist = distribution.lower()
    dist_name = DISTRIBUTION_DISPLAY_NAMES.get(dist, dist)

    descriptions = {
        'alpha': f"Shape parameter (alpha) of the {dist_name} distribution computed from "
                 f"{scale}-{periodicity.unit()} scaled values",
        'beta': f"Scale parameter (beta) of the {dist_name} distribution computed from "
                f"{scale}-{periodicity.unit()} scaled values",
        'skew': f"Skewness parameter of the {dist_name} distribution computed from "
                f"{scale}-{periodicity.unit()} scaled values",
        'loc': f"Location parameter of the {dist_name} distribution computed from "
               f"{scale}-{periodicity.unit()} scaled values",
        'scale': f"Scale parameter of the {dist_name} distribution computed from "
                 f"{scale}-{periodicity.unit()} scaled values",
        'shape': f"Shape parameter of the {dist_name} distribution computed from "
                 f"{scale}-{periodicity.unit()} scaled values",
        'prob_zero': f"Probability of zero values within calibration period for "
                     f"{scale}-{periodicity.unit()} scale",
    }

    return {
        'long_name': f"{dist_name} {param} parameter ({scale}-{periodicity.unit()})",
        'description': descriptions.get(param, f"{param} parameter"),
        'units': '1',
    }


def get_global_attributes(
    title: str,
    distribution: str = DEFAULT_DISTRIBUTION,
    calibration_start_year=None,
    calibration_end_year=None,
    extra_attrs: dict = None,
    global_attrs: dict = None,
) -> dict:
    """
    Build global attributes dict for NetCDF output.

    Merges DEFAULT_METADATA with computed attributes and user overrides.
    Empty default strings are excluded from output.

    Priority order (lowest to highest):
        1. DEFAULT_METADATA (module-level defaults)
        2. Computed attributes (title, history, distribution, calibration)
        3. extra_attrs (additional computed attributes like scales list)
        4. global_attrs (user overrides — highest priority)

    :param title: dataset title
    :param distribution: distribution type used
    :param calibration_start_year: calibration start year
    :param calibration_end_year: calibration end year
    :param extra_attrs: additional computed attributes (e.g., scales list)
    :param global_attrs: user overrides (highest priority)
    :return: merged attributes dict
    """
    attrs = {}
    # Start with defaults (skip empty strings)
    for k, v in DEFAULT_METADATA.items():
        if v:  # Only include non-empty defaults
            attrs[k] = v

    # Add computed attributes
    attrs['title'] = title
    attrs['history'] = f'Created {datetime.now().isoformat()}'
    attrs['distribution'] = distribution.lower()

    if calibration_start_year is not None:
        attrs['calibration_start_year'] = calibration_start_year
    if calibration_end_year is not None:
        attrs['calibration_end_year'] = calibration_end_year

    # Add any extra computed attributes
    if extra_attrs:
        attrs.update(extra_attrs)

    # User overrides (highest priority)
    if global_attrs:
        attrs.update(global_attrs)

    return attrs


# =============================================================================
# DATA COMPLETENESS REPORTING
# =============================================================================

def summarize_data_completeness(
    data: Union[np.ndarray, 'xr.DataArray'],
    time_dim: str = 'time'
) -> dict:
    """
    Report land-aware data completeness for gridded datasets.

    Separates ocean/no-data cells (all-NaN across time) from land cells
    (at least one valid timestep), then reports temporal completeness
    only for land cells. This avoids misleading NaN percentages for
    island or coastal datasets where ocean cells are expected to be NaN.

    :param data: 3-D array (time, lat, lon) — xarray DataArray or numpy array
    :param time_dim: name of the time dimension (for xarray inputs)
    :return: dictionary with completeness statistics
    """
    # Convert to numpy if xarray
    if hasattr(data, 'values'):
        values = data.values
        # Get spatial dimension sizes
        dims = data.dims
        time_axis = dims.index(time_dim) if time_dim in dims else 0
        spatial_shape = tuple(s for i, s in enumerate(data.shape) if i != time_axis)
    else:
        values = np.asarray(data)
        time_axis = 0
        spatial_shape = values.shape[1:]

    n_timesteps = values.shape[time_axis]
    total_cells = int(np.prod(spatial_shape))

    # Count valid (non-NaN) timesteps per spatial cell
    valid_counts = np.sum(~np.isnan(values), axis=time_axis)
    flat_counts = valid_counts.flatten()

    # Land mask: cells with at least one valid timestep
    land_mask = flat_counts > 0
    n_land = int(np.sum(land_mask))
    n_ocean = total_cells - n_land

    result = {
        'total_cells': total_cells,
        'spatial_shape': spatial_shape,
        'land_cells': n_land,
        'ocean_cells': n_ocean,
        'land_fraction': n_land / total_cells * 100 if total_cells > 0 else 0.0,
        'total_timesteps': n_timesteps,
    }

    if n_land > 0:
        land_counts = flat_counts[land_mask]
        completeness = land_counts / n_timesteps * 100
        n_fully_complete = int(np.sum(land_counts == n_timesteps))

        result.update({
            'mean_temporal_completeness': float(np.mean(completeness)),
            'min_temporal_completeness': float(np.min(completeness)),
            'max_temporal_completeness': float(np.max(completeness)),
            'fully_complete_land_cells': n_fully_complete,
        })
    else:
        result.update({
            'mean_temporal_completeness': 0.0,
            'min_temporal_completeness': 0.0,
            'max_temporal_completeness': 0.0,
            'fully_complete_land_cells': 0,
        })

    return result


def print_data_completeness(report: dict, indent: str = "   ") -> None:
    """
    Print a formatted data completeness report.

    :param report: dictionary from summarize_data_completeness()
    :param indent: prefix string for each line
    """
    sp = report['spatial_shape']
    sp_str = ' x '.join(str(s) for s in sp)

    print(f"{indent}Spatial coverage:")
    print(f"{indent}  Total grid cells: {report['total_cells']:,} ({sp_str})")
    print(f"{indent}  Land cells: {report['land_cells']:,} ({report['land_fraction']:.1f}%)")
    print(f"{indent}  Ocean/NoData cells: {report['ocean_cells']:,} ({100 - report['land_fraction']:.1f}%)")

    print(f"{indent}Temporal completeness (land cells only):")
    print(f"{indent}  Time steps: {report['total_timesteps']:,}")

    if report['land_cells'] > 0:
        print(f"{indent}  Mean completeness: {report['mean_temporal_completeness']:.1f}%")
        print(f"{indent}  Min completeness: {report['min_temporal_completeness']:.1f}%")
        fc = report['fully_complete_land_cells']
        lc = report['land_cells']
        print(f"{indent}  Fully complete cells: {fc} / {lc} ({100*fc/lc:.1f}%)")
    else:
        print(f"{indent}  No land cells found — dataset is entirely NaN")
