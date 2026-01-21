"""
Utility functions for SPI/SPEI climate indices calculation.

Includes data transformations, array reshaping, and PET calculation.
All functions follow CF Convention with dimension order: (time, lat, lon)

Modified/adapted from James Adams' climate-indices package
https://github.com/monocongo/climate_indices

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank
"""

import calendar
import math
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

from config import Periodicity, get_logger

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
