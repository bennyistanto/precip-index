"""
SPI and SPEI climate indices calculation module.

High-level API for computing Standardized Precipitation Index (SPI) and
Standardized Precipitation Evapotranspiration Index (SPEI) with support
for saving/loading fitting parameters.

Optimized for global-scale gridded data following CF Convention (time, lat, lon).

---
Author: Benny Istanto, GOST/DEC Data Group/The World Bank

Built upon the foundation of climate-indices by James Adams, 
with substantial modifications for multi-distribution support, 
bidirectional event analysis, and scalable processing.
---

References:
    - McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought
      frequency and duration to time scales. 8th Conference on Applied Climatology.
    - Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010). A Multiscalar
      Drought Index Sensitive to Global Warming: The Standardized Precipitation
      Evapotranspiration Index. Journal of Climate, 23(7), 1696-1718.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from config import (
    DEFAULT_CALIBRATION_END_YEAR,
    DEFAULT_CALIBRATION_START_YEAR,
    DEFAULT_DISTRIBUTION,
    DISTRIBUTION_DISPLAY_NAMES,
    DISTRIBUTION_PARAM_NAMES,
    FITTED_INDEX_VALID_MAX,
    FITTED_INDEX_VALID_MIN,
    FITTING_PARAM_NAMES,
    NC_FILL_VALUE,
    PET_VAR_PATTERNS,
    Periodicity,
    PRECIP_VAR_PATTERNS,
    SPEI_WATER_BALANCE_OFFSET,
    TEMP_VAR_PATTERNS,
)
from utils import (
    get_fitting_param_attributes,
    get_fitting_param_name,
    get_global_attributes,
    get_logger,
    get_variable_attributes,
    get_variable_name,
)
from compute import (
    compute_index_dask,
    compute_index_dask_to_zarr,
    compute_index_parallel,
    compute_spi_1d,
    compute_spei_1d,
    sum_to_scale,
)
from utils import (
    calculate_pet,
    ensure_cf_compliant,
    get_data_year_range,
    is_data_valid,
)

# Module logger
_logger = get_logger(__name__)


# =============================================================================
# FITTING PARAMETERS I/O
# =============================================================================

def save_fitting_params(
    params: Dict[str, Union[np.ndarray, xr.DataArray]],
    filepath: str,
    scale: int,
    periodicity: Periodicity,
    index_type: str = 'spi',
    calibration_start_year: Optional[int] = None,
    calibration_end_year: Optional[int] = None,
    coords: Optional[Dict] = None,
    global_attrs: Optional[Dict] = None,
    distribution: str = DEFAULT_DISTRIBUTION
) -> str:
    """
    Save distribution fitting parameters to NetCDF file for later reuse.

    Parameters can be loaded later to speed up recalculation or
    to apply the same calibration to new data. Supports any distribution
    type — parameter names are determined from DISTRIBUTION_PARAM_NAMES.

    :param params: dictionary containing distribution-specific parameter arrays.
        For gamma: 'alpha', 'beta', 'prob_zero'.
        For pearson3: 'skew', 'loc', 'scale', 'prob_zero'.
        Arrays should have shape (periods,) for 1D or (periods, lat, lon) for 3D.
    :param filepath: output NetCDF file path
    :param scale: accumulation scale (e.g., 1, 3, 6, 12)
    :param periodicity: monthly or daily
    :param index_type: 'spi' or 'spei'
    :param calibration_start_year: start year of calibration period
    :param calibration_end_year: end year of calibration period
    :param coords: optional coordinate dict with 'lat', 'lon' for gridded data
    :param global_attrs: optional additional global attributes
    :param distribution: distribution type used for fitting (default: 'gamma')
    :return: filepath of saved file
    """
    dist = distribution.lower()
    _logger.info(f"Saving {dist} fitting parameters to: {filepath}")

    # Convert periodicity if string
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    # Determine which parameter names to save
    # Use distribution from params dict if available, else use argument
    dist_key = params.get('distribution', dist)
    if isinstance(dist_key, str):
        dist_key = dist_key.lower()
    param_names = DISTRIBUTION_PARAM_NAMES.get(dist_key, FITTING_PARAM_NAMES)

    # Create dataset
    ds = xr.Dataset()

    # Find first array param to determine dimensionality
    first_array = None
    for pname in param_names:
        if pname in params:
            val = params[pname]
            if isinstance(val, xr.DataArray):
                first_array = val.values
            elif isinstance(val, np.ndarray):
                first_array = val
            if first_array is not None:
                break

    if first_array is None:
        raise ValueError(f"No parameter arrays found in params dict for distribution '{dist_key}'")

    ndim = first_array.ndim
    periods = periodicity.value
    period_dim = periodicity.unit()  # 'month' or 'day'

    # Create period coordinate
    period_coord = np.arange(periods)

    if ndim == 1:
        # 1D: shape (periods,)
        dims = (period_dim,)
        coords_dict = {period_dim: period_coord}
    elif ndim == 3:
        # 3D: shape (periods, lat, lon)
        dims = (period_dim, 'lat', 'lon')
        if coords is not None:
            coords_dict = {
                period_dim: period_coord,
                'lat': coords.get('lat', np.arange(first_array.shape[1])),
                'lon': coords.get('lon', np.arange(first_array.shape[2]))
            }
        else:
            coords_dict = {
                period_dim: period_coord,
                'lat': np.arange(first_array.shape[1]),
                'lon': np.arange(first_array.shape[2])
            }
    else:
        raise ValueError(f"Unsupported parameter array dimensions: {ndim}")

    # Add each parameter as a variable
    for param_name in param_names:
        if param_name not in params:
            _logger.warning(f"Parameter '{param_name}' not found in params dict")
            continue

        param_data = params[param_name]
        if isinstance(param_data, xr.DataArray):
            param_data = param_data.values
        if not isinstance(param_data, np.ndarray):
            continue  # Skip non-array entries like 'distribution'

        var_name = get_fitting_param_name(param_name, scale, periodicity, distribution=dist_key)
        var_attrs = get_fitting_param_attributes(param_name, scale, periodicity, distribution=dist_key)

        ds[var_name] = xr.DataArray(
            data=param_data,
            dims=dims,
            coords=coords_dict,
            attrs=var_attrs
        )

    # Global attributes (centralized via config.get_global_attributes)
    ds.attrs = get_global_attributes(
        title=f'{DISTRIBUTION_DISPLAY_NAMES.get(dist_key, dist_key)} distribution fitting parameters for {index_type.upper()}',
        distribution=dist_key,
        calibration_start_year=calibration_start_year or 'not specified',
        calibration_end_year=calibration_end_year or 'not specified',
        extra_attrs={
            'index_type': index_type.upper(),
            'scale': scale,
            'periodicity': periodicity.name,
        },
        global_attrs=global_attrs,
    )

    # Ensure directory exists
    dir_path = os.path.dirname(os.path.abspath(filepath))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Set encoding
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            'dtype': 'float32',
            '_FillValue': NC_FILL_VALUE,
            'zlib': True,
            'complevel': 4
        }

    # Save
    ds.to_netcdf(filepath, encoding=encoding)
    _logger.info(f"Fitting parameters saved: {filepath}")

    return filepath


def load_fitting_params(
    filepath: str,
    scale: int,
    periodicity: Union[str, Periodicity],
    distribution: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load distribution fitting parameters from NetCDF file.

    Automatically detects the distribution type from file attributes
    if not specified, and loads the corresponding parameter set.

    :param filepath: path to NetCDF file with fitting parameters
    :param scale: accumulation scale to load (e.g., 12)
    :param periodicity: monthly or daily
    :param distribution: distribution type to load (auto-detected from file if None)
    :return: dictionary with distribution-specific parameter arrays and 'distribution' key
    :raises FileNotFoundError: if file doesn't exist
    :raises KeyError: if required variables not found
    """
    _logger.info(f"Loading fitting parameters from: {filepath}")

    # Convert periodicity if string
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fitting parameters file not found: {filepath}")

    ds = xr.open_dataset(filepath)

    # Determine distribution from file or argument
    if distribution is not None:
        dist = distribution.lower()
    else:
        dist = ds.attrs.get('distribution', DEFAULT_DISTRIBUTION)
        if isinstance(dist, str):
            dist = dist.lower()

    # Get parameter names for this distribution
    param_names = DISTRIBUTION_PARAM_NAMES.get(dist, FITTING_PARAM_NAMES)

    params = {}
    for param_name in param_names:
        var_name = get_fitting_param_name(param_name, scale, periodicity, distribution=dist)

        if var_name not in ds:
            raise KeyError(
                f"Variable '{var_name}' not found in {filepath}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        params[param_name] = ds[var_name].values

    params['distribution'] = dist

    ds.close()

    # Find first array param for shape logging
    first_key = next((k for k in param_names if k in params and isinstance(params[k], np.ndarray)), None)
    shape_info = params[first_key].shape if first_key else 'unknown'

    _logger.info(
        f"Loaded {dist} parameters for scale={scale}, periodicity={periodicity.name}, "
        f"shape={shape_info}"
    )

    return params


# =============================================================================
# SPI CALCULATION
# =============================================================================

def spi(
    precip: Union[np.ndarray, xr.DataArray, xr.Dataset],
    scale: int,
    periodicity: Union[str, Periodicity] = Periodicity.monthly,
    data_start_year: Optional[int] = None,
    calibration_start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    calibration_end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
    return_params: bool = False,
    var_name: Optional[str] = None,
    distribution: str = DEFAULT_DISTRIBUTION
) -> Union[xr.DataArray, Tuple[xr.DataArray, Dict[str, np.ndarray]]]:
    """
    Calculate Standardized Precipitation Index (SPI).

    SPI is computed by fitting precipitation data to a probability distribution
    and transforming to standard normal distribution.

    :param precip: precipitation data in mm
        - numpy array: 1D (time,) or 3D (time, lat, lon)
        - xarray DataArray: with 'time' dimension
        - xarray Dataset: specify var_name parameter
    :param scale: accumulation period in time steps (e.g., 1, 3, 6, 12 months)
    :param periodicity: 'monthly' or 'daily' (or Periodicity enum)
    :param data_start_year: first year of data (auto-detected for xarray)
    :param calibration_start_year: first year of calibration period (default: 1991)
    :param calibration_end_year: last year of calibration period (default: 2020)
    :param fitting_params: optional pre-computed parameters from save_fitting_params()
    :param return_params: if True, return (result, params) tuple
    :param var_name: variable name if precip is a Dataset
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :return: SPI values as xarray DataArray, or tuple (SPI, params) if return_params=True

    Example:
        >>> # Basic usage (Gamma distribution, default)
        >>> spi_12 = spi(precip_da, scale=12)

        >>> # Using Pearson III distribution
        >>> spi_12 = spi(precip_da, scale=12, distribution='pearson3')

        >>> # With parameter saving
        >>> spi_12, params = spi(precip_da, scale=12, return_params=True)
        >>> save_fitting_params(params, 'spi_params.nc', scale=12, periodicity='monthly')

        >>> # Using pre-computed parameters
        >>> params = load_fitting_params('spi_params.nc', scale=12, periodicity='monthly')
        >>> spi_12 = spi(new_precip_da, scale=12, fitting_params=params)
    """
    dist = distribution.lower()
    _logger.info(f"Computing SPI-{scale} (distribution={dist})")

    # Convert periodicity string to enum
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    # Handle different input types
    if isinstance(precip, xr.Dataset):
        if var_name is None:
            # Try to find precipitation variable
            precip_vars = [v for v in precip.data_vars
                          if any(p in v.lower() for p in PRECIP_VAR_PATTERNS)]
            if len(precip_vars) == 1:
                var_name = precip_vars[0]
            else:
                raise ValueError(
                    "Multiple/no precipitation variables found. Specify var_name parameter. "
                    f"Available: {list(precip.data_vars)}"
                )
        precip_da = precip[var_name]
    elif isinstance(precip, xr.DataArray):
        precip_da = precip
    else:
        # Numpy array
        precip_da = None
        precip_array = np.asarray(precip)

    # Extract data and metadata from xarray
    if precip_da is not None:
        # Ensure CF Convention dimension order (time, lat, lon)
        if precip_da.ndim == 3:
            expected_order = ('time', 'lat', 'lon')
            if precip_da.dims != expected_order:
                _logger.info(f"Transposing dimensions from {precip_da.dims} to {expected_order}")
                precip_da = precip_da.transpose(*expected_order)

        # Auto-detect start year
        if data_start_year is None:
            data_start_year, _ = get_data_year_range(
                xr.Dataset({'var': precip_da})
            )

        # Get coordinates for output
        coords = dict(precip_da.coords)
        dims = precip_da.dims

        # Convert to numpy
        precip_array = precip_da.values

        _logger.info(
            f"Input shape: {precip_array.shape}, dims: {dims}, "
            f"data_start_year: {data_start_year}"
        )
    else:
        if data_start_year is None:
            raise ValueError("data_start_year required for numpy array input")
        coords = None
        dims = None

    # Clip negative values
    precip_array = np.clip(precip_array, 0, None)

    # Compute SPI based on array dimensions
    if precip_array.ndim == 1:
        # 1D time series
        result, params = compute_spi_1d(
            precip_array,
            scale=scale,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            periodicity=periodicity,
            fitting_params=fitting_params,
            distribution=dist
        )
    elif precip_array.ndim == 3:
        # 3D gridded data (time, lat, lon) - CF Convention
        result, params = compute_index_parallel(
            precip_array,
            scale=scale,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            periodicity=periodicity,
            fitting_params=fitting_params,
            distribution=dist
        )
    else:
        raise ValueError(
            f"Unsupported array dimensions: {precip_array.ndim}. "
            f"Expected 1D (time,) or 3D (time, lat, lon)"
        )

    # Create output DataArray
    output_var_name = get_variable_name('spi', scale, periodicity, distribution=dist)
    output_attrs = get_variable_attributes('spi', scale, periodicity, distribution=dist)
    output_attrs.update({
        'calibration_start_year': calibration_start_year,
        'calibration_end_year': calibration_end_year,
    })

    if coords is not None:
        result_da = xr.DataArray(
            data=result,
            dims=dims,
            coords=coords,
            name=output_var_name,
            attrs=output_attrs
        )
    else:
        result_da = xr.DataArray(
            data=result,
            name=output_var_name,
            attrs=output_attrs
        )

    _logger.info(f"SPI-{scale} computation complete. Output shape: {result.shape}")

    if return_params:
        return result_da, params
    else:
        return result_da


def spi_multi_scale(
    precip: Union[np.ndarray, xr.DataArray, xr.Dataset],
    scales: List[int],
    periodicity: Union[str, Periodicity] = Periodicity.monthly,
    data_start_year: Optional[int] = None,
    calibration_start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    calibration_end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    return_params: bool = False,
    var_name: Optional[str] = None,
    distribution: str = DEFAULT_DISTRIBUTION,
    global_attrs: Optional[Dict] = None
) -> Union[xr.Dataset, Tuple[xr.Dataset, Dict[int, Dict[str, np.ndarray]]]]:
    """
    Calculate SPI for multiple time scales.

    :param precip: precipitation data
    :param scales: list of accumulation scales (e.g., [1, 3, 6, 12])
    :param periodicity: 'monthly' or 'daily'
    :param data_start_year: first year of data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param return_params: if True, return (result, params_dict) tuple
    :param var_name: variable name if precip is a Dataset
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :param global_attrs: optional dict of global attributes to override defaults
        (e.g., {'institution': 'My Org', 'source': 'My Project'})
    :return: Dataset with SPI for all scales, or tuple (Dataset, params_dict)

    Example:
        >>> spi_ds = spi_multi_scale(precip_da, scales=[1, 3, 6, 12])
        >>> spi_12 = spi_ds['spi_gamma_12_month']
        >>> # With Pearson III
        >>> spi_ds = spi_multi_scale(precip_da, scales=[3, 12], distribution='pearson3')
        >>> # With custom metadata
        >>> spi_ds = spi_multi_scale(precip_da, scales=[3, 12],
        ...     global_attrs={'institution': 'My University'})
    """
    dist = distribution.lower()
    _logger.info(f"Computing SPI for scales: {scales} (distribution={dist})")

    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    results = {}
    all_params = {}

    for s in scales:
        _logger.info(f"Processing scale {s}...")

        result_da, params = spi(
            precip,
            scale=s,
            periodicity=periodicity,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            return_params=True,
            var_name=var_name,
            distribution=dist
        )

        var_name_out = get_variable_name('spi', s, periodicity, distribution=dist)
        results[var_name_out] = result_da
        all_params[s] = params

    # Create output Dataset
    ds = xr.Dataset(results)
    ds.attrs = get_global_attributes(
        title=f'Standardized Precipitation Index (SPI) - {DISTRIBUTION_DISPLAY_NAMES.get(dist, dist)}',
        distribution=dist,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        extra_attrs={'scales': scales},
        global_attrs=global_attrs,
    )

    _logger.info(f"Multi-scale SPI complete. Variables: {list(ds.data_vars)}")

    if return_params:
        return ds, all_params
    else:
        return ds


# =============================================================================
# SPEI CALCULATION
# =============================================================================

def spei(
    precip: Union[np.ndarray, xr.DataArray, xr.Dataset],
    pet: Optional[Union[np.ndarray, xr.DataArray]] = None,
    temperature: Optional[Union[np.ndarray, xr.DataArray]] = None,
    latitude: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
    scale: int = 12,
    periodicity: Union[str, Periodicity] = Periodicity.monthly,
    data_start_year: Optional[int] = None,
    calibration_start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    calibration_end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
    return_params: bool = False,
    precip_var_name: Optional[str] = None,
    pet_var_name: Optional[str] = None,
    temp_var_name: Optional[str] = None,
    distribution: str = DEFAULT_DISTRIBUTION,
    pet_method: str = 'thornthwaite',
    temp_min: Optional[Union[np.ndarray, xr.DataArray]] = None,
    temp_max: Optional[Union[np.ndarray, xr.DataArray]] = None
) -> Union[xr.DataArray, Tuple[xr.DataArray, Dict[str, np.ndarray]]]:
    """
    Calculate Standardized Precipitation Evapotranspiration Index (SPEI).

    SPEI uses the water balance (P - PET) instead of just precipitation.
    PET can be provided directly or calculated from temperature using
    Thornthwaite or Hargreaves-Samani method.

    :param precip: precipitation data in mm
    :param pet: potential evapotranspiration in mm (optional if temperature provided)
    :param temperature: mean temperature in C for PET calculation (optional if PET provided)
    :param latitude: latitude for PET calculation (required if using temperature)
    :param scale: accumulation period in time steps
    :param periodicity: 'monthly' or 'daily'
    :param data_start_year: first year of data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param fitting_params: optional pre-computed distribution parameters
    :param return_params: if True, return (result, params) tuple
    :param precip_var_name: variable name for precipitation in Dataset
    :param pet_var_name: variable name for PET in Dataset
    :param temp_var_name: variable name for temperature in Dataset
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'.
        Note: Pearson III or Log-Logistic are recommended for SPEI.
    :param pet_method: PET calculation method ('thornthwaite' or 'hargreaves').
        - 'thornthwaite': Uses only mean temperature (default)
        - 'hargreaves': Uses mean, min, max temperature (better for arid regions)
    :param temp_min: minimum temperature in C (required for Hargreaves method)
    :param temp_max: maximum temperature in C (required for Hargreaves method)
    :return: SPEI values as xarray DataArray, or tuple (SPEI, params)

    Example:
        >>> # With pre-computed PET (Gamma, default)
        >>> spei_12 = spei(precip_da, pet=pet_da, scale=12)

        >>> # With Pearson III distribution (recommended for SPEI)
        >>> spei_12 = spei(precip_da, pet=pet_da, scale=12, distribution='pearson3')

        >>> # With temperature - Thornthwaite method (default)
        >>> spei_12 = spei(precip_da, temperature=temp_da, latitude=lat_da, scale=12)

        >>> # With temperature - Hargreaves method (better for arid regions)
        >>> spei_12 = spei(precip_da, temperature=temp_mean, latitude=lat_da, scale=12,
        ...               pet_method='hargreaves', temp_min=tmin, temp_max=tmax)

        >>> # Save and reuse parameters
        >>> spei_12, params = spei(precip_da, pet=pet_da, scale=12, return_params=True)
        >>> save_fitting_params(params, 'spei_params.nc', scale=12,
        ...                     periodicity='monthly', index_type='spei')
    """
    dist = distribution.lower()
    _logger.info(f"Computing SPEI-{scale} (distribution={dist})")

    # Convert periodicity
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    # Handle Dataset input for precip
    if isinstance(precip, xr.Dataset):
        if precip_var_name is None:
            precip_vars = [v for v in precip.data_vars
                          if any(p in v.lower() for p in PRECIP_VAR_PATTERNS)]
            if len(precip_vars) == 1:
                precip_var_name = precip_vars[0]
            else:
                raise ValueError(f"Specify precip_var_name. Available: {list(precip.data_vars)}")
        precip_da = precip[precip_var_name]
    elif isinstance(precip, xr.DataArray):
        precip_da = precip
    else:
        precip_da = None
        precip_array = np.asarray(precip)

    # Get/compute PET
    if pet is not None:
        # PET provided directly
        if isinstance(pet, xr.Dataset):
            if pet_var_name is None:
                pet_vars = [v for v in pet.data_vars if any(p in v.lower() for p in PET_VAR_PATTERNS)]
                if len(pet_vars) == 1:
                    pet_var_name = pet_vars[0]
                else:
                    raise ValueError(f"Specify pet_var_name. Available: {list(pet.data_vars)}")
            pet_da = pet[pet_var_name]
        elif isinstance(pet, xr.DataArray):
            pet_da = pet
        else:
            pet_da = None
            pet_array = np.asarray(pet)
    elif temperature is not None:
        # Compute PET from temperature
        pet_method = pet_method.lower()
        _logger.info(f"Computing PET from temperature using {pet_method.capitalize()} method")

        if latitude is None:
            raise ValueError("latitude required for PET calculation from temperature")

        if pet_method == 'hargreaves' and (temp_min is None or temp_max is None):
            raise ValueError(
                "Hargreaves method requires temp_min and temp_max parameters. "
                "Use pet_method='thornthwaite' if only mean temperature is available."
            )

        # Handle temperature input
        if isinstance(temperature, xr.Dataset):
            if temp_var_name is None:
                temp_vars = [v for v in temperature.data_vars
                            if any(p in v.lower() for p in TEMP_VAR_PATTERNS)]
                if len(temp_vars) == 1:
                    temp_var_name = temp_vars[0]
                else:
                    raise ValueError(f"Specify temp_var_name. Available: {list(temperature.data_vars)}")
            temp_da = temperature[temp_var_name]
        elif isinstance(temperature, xr.DataArray):
            temp_da = temperature
        else:
            temp_da = xr.DataArray(np.asarray(temperature))

        # Auto-detect start year
        if data_start_year is None and precip_da is not None:
            data_start_year, _ = get_data_year_range(xr.Dataset({'var': precip_da}))

        if data_start_year is None:
            raise ValueError("data_start_year required for PET calculation")

        # Calculate PET using specified method
        pet_da = calculate_pet(
            temp_da, latitude, data_start_year,
            method=pet_method,
            temp_min=temp_min,
            temp_max=temp_max
        )
        pet_array = pet_da.values if isinstance(pet_da, xr.DataArray) else pet_da
    else:
        raise ValueError("Either 'pet' or 'temperature' (with 'latitude') must be provided")

    # Extract arrays and metadata
    if precip_da is not None:
        # Ensure CF Convention dimension order (time, lat, lon)
        if precip_da.ndim == 3:
            expected_order = ('time', 'lat', 'lon')
            if precip_da.dims != expected_order:
                _logger.info(f"Transposing precipitation dimensions from {precip_da.dims} to {expected_order}")
                precip_da = precip_da.transpose(*expected_order)

        if data_start_year is None:
            data_start_year, _ = get_data_year_range(xr.Dataset({'var': precip_da}))

        coords = dict(precip_da.coords)
        dims = precip_da.dims
        precip_array = precip_da.values

        if pet_da is not None:
            # Ensure PET also has correct dimension order
            if isinstance(pet_da, xr.DataArray) and pet_da.ndim == 3:
                if pet_da.dims != expected_order:
                    _logger.info(f"Transposing PET dimensions from {pet_da.dims} to {expected_order}")
                    pet_da = pet_da.transpose(*expected_order)
            pet_array = pet_da.values
    else:
        if data_start_year is None:
            raise ValueError("data_start_year required for numpy array input")
        coords = None
        dims = None
        if pet_da is not None:
            pet_array = pet_da.values

    # Validate shapes match
    if precip_array.shape != pet_array.shape:
        raise ValueError(
            f"Precipitation and PET shapes must match: "
            f"{precip_array.shape} vs {pet_array.shape}"
        )

    _logger.info(
        f"Input shape: {precip_array.shape}, "
        f"data_start_year: {data_start_year}"
    )

    # Compute water balance: P - PET
    # Add offset to ensure positive values for gamma fitting
    water_balance = (precip_array - pet_array) + SPEI_WATER_BALANCE_OFFSET

    # Compute SPEI (same algorithm as SPI, but on water balance)
    if water_balance.ndim == 1:
        result, params = compute_spei_1d(
            precip_array,
            pet_array,
            scale=scale,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            periodicity=periodicity,
            fitting_params=fitting_params,
            distribution=dist
        )
    elif water_balance.ndim == 3:
        result, params = compute_index_parallel(
            water_balance,
            scale=scale,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            periodicity=periodicity,
            fitting_params=fitting_params,
            distribution=dist
        )
    else:
        raise ValueError(
            f"Unsupported array dimensions: {water_balance.ndim}. "
            f"Expected 1D (time,) or 3D (time, lat, lon)"
        )

    # Create output DataArray
    output_var_name = get_variable_name('spei', scale, periodicity, distribution=dist)
    output_attrs = get_variable_attributes('spei', scale, periodicity, distribution=dist)
    output_attrs.update({
        'calibration_start_year': calibration_start_year,
        'calibration_end_year': calibration_end_year,
    })

    if coords is not None:
        result_da = xr.DataArray(
            data=result,
            dims=dims,
            coords=coords,
            name=output_var_name,
            attrs=output_attrs
        )
    else:
        result_da = xr.DataArray(
            data=result,
            name=output_var_name,
            attrs=output_attrs
        )

    _logger.info(f"SPEI-{scale} computation complete. Output shape: {result.shape}")

    if return_params:
        return result_da, params
    else:
        return result_da


def spei_multi_scale(
    precip: Union[np.ndarray, xr.DataArray, xr.Dataset],
    pet: Optional[Union[np.ndarray, xr.DataArray]] = None,
    temperature: Optional[Union[np.ndarray, xr.DataArray]] = None,
    latitude: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
    scales: List[int] = [1, 3, 6, 12],
    periodicity: Union[str, Periodicity] = Periodicity.monthly,
    data_start_year: Optional[int] = None,
    calibration_start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    calibration_end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    return_params: bool = False,
    precip_var_name: Optional[str] = None,
    pet_var_name: Optional[str] = None,
    temp_var_name: Optional[str] = None,
    distribution: str = DEFAULT_DISTRIBUTION,
    global_attrs: Optional[Dict] = None,
    pet_method: str = 'thornthwaite',
    temp_min: Optional[Union[np.ndarray, xr.DataArray]] = None,
    temp_max: Optional[Union[np.ndarray, xr.DataArray]] = None
) -> Union[xr.Dataset, Tuple[xr.Dataset, Dict[int, Dict[str, np.ndarray]]]]:
    """
    Calculate SPEI for multiple time scales.

    :param precip: precipitation data
    :param pet: potential evapotranspiration (optional if temperature provided)
    :param temperature: mean temperature for PET calculation
    :param latitude: latitude for PET calculation
    :param scales: list of accumulation scales (e.g., [1, 3, 6, 12])
    :param periodicity: 'monthly' or 'daily'
    :param data_start_year: first year of data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param return_params: if True, return (result, params_dict) tuple
    :param precip_var_name: variable name for precipitation
    :param pet_var_name: variable name for PET
    :param temp_var_name: variable name for temperature
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :param global_attrs: optional dict of global attributes to override defaults
        (e.g., {'institution': 'My Org', 'source': 'My Project'})
    :param pet_method: PET calculation method ('thornthwaite' or 'hargreaves')
    :param temp_min: minimum temperature (required for Hargreaves)
    :param temp_max: maximum temperature (required for Hargreaves)
    :return: Dataset with SPEI for all scales

    Example:
        >>> spei_ds = spei_multi_scale(precip_da, pet=pet_da, scales=[1, 3, 6, 12])
        >>> spei_12 = spei_ds['spei_gamma_12_month']
        >>> # With Pearson III (recommended for SPEI)
        >>> spei_ds = spei_multi_scale(precip_da, pet=pet_da, scales=[3, 12],
        ...                            distribution='pearson3')
        >>> # With Hargreaves PET
        >>> spei_ds = spei_multi_scale(precip_da, temperature=tmean, latitude=lat,
        ...                            scales=[3, 12], pet_method='hargreaves',
        ...                            temp_min=tmin, temp_max=tmax)
    """
    dist = distribution.lower()
    _logger.info(f"Computing SPEI for scales: {scales} (distribution={dist})")

    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    results = {}
    all_params = {}

    for s in scales:
        _logger.info(f"Processing scale {s}...")

        result_da, params = spei(
            precip,
            pet=pet,
            temperature=temperature,
            latitude=latitude,
            scale=s,
            periodicity=periodicity,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            return_params=True,
            precip_var_name=precip_var_name,
            pet_var_name=pet_var_name,
            temp_var_name=temp_var_name,
            distribution=dist,
            pet_method=pet_method,
            temp_min=temp_min,
            temp_max=temp_max
        )

        var_name_out = get_variable_name('spei', s, periodicity, distribution=dist)
        results[var_name_out] = result_da
        all_params[s] = params

    # Create output Dataset
    ds = xr.Dataset(results)
    ds.attrs = get_global_attributes(
        title=f'Standardized Precipitation Evapotranspiration Index (SPEI) - {DISTRIBUTION_DISPLAY_NAMES.get(dist, dist)}',
        distribution=dist,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        extra_attrs={'scales': scales},
        global_attrs=global_attrs,
    )

    _logger.info(f"Multi-scale SPEI complete. Variables: {list(ds.data_vars)}")

    if return_params:
        return ds, all_params
    else:
        return ds


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_index_to_netcdf(
    data: Union[xr.DataArray, xr.Dataset],
    filepath: str,
    compress: bool = True,
    complevel: int = 5,
    chunksizes: Optional[Tuple[int, ...]] = None
) -> str:
    """
    Save SPI/SPEI results to NetCDF file with proper encoding.

    :param data: DataArray or Dataset to save
    :param filepath: output file path
    :param compress: whether to use compression
    :param complevel: compression level (1-9)
    :param chunksizes: optional chunk sizes for NetCDF, e.g., (12, 300, 300)
    :return: filepath of saved file
    """
    _logger.info(f"Saving to: {filepath}")
    
    # Ensure directory exists
    dir_path = os.path.dirname(os.path.abspath(filepath))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Convert to Dataset if needed
    if isinstance(data, xr.DataArray):
        data = data.to_dataset()
    
    # Set encoding
    encoding = {}
    for var in data.data_vars:
        encoding[var] = {
            'dtype': 'float32',
            '_FillValue': NC_FILL_VALUE,
        }
        if compress:
            encoding[var]['zlib'] = True
            encoding[var]['complevel'] = complevel
        if chunksizes is not None:
            encoding[var]['chunksizes'] = chunksizes
    
    # Add coordinate encoding
    for coord in data.coords:
        if coord in ['lat', 'lon']:
            encoding[coord] = {'dtype': 'float32', '_FillValue': None}
        elif coord == 'time':
            encoding[coord] = {'dtype': 'float64', '_FillValue': None}
    
    data.to_netcdf(filepath, encoding=encoding)
    _logger.info(f"Saved: {filepath}")
    
    return filepath


def classify_drought(
    index_values: Union[np.ndarray, xr.DataArray],
    classification: str = 'mckee'
) -> Union[np.ndarray, xr.DataArray]:
    """
    Classify SPI/SPEI values into drought categories.

    :param index_values: SPI or SPEI values
    :param classification: classification scheme ('mckee' or 'custom')
    :return: array of drought categories (integers)
    
    McKee et al. (1993) classification:
        >= 2.0:  Extremely wet (4)
        1.5 to 2.0: Very wet (3)
        1.0 to 1.5: Moderately wet (2)
        -1.0 to 1.0: Near normal (1)
        -1.5 to -1.0: Moderately dry (0)
        -2.0 to -1.5: Severely dry (-1)
        <= -2.0: Extremely dry (-2)
    """
    values = index_values.values if isinstance(index_values, xr.DataArray) else index_values
    
    # Initialize with NaN
    categories = np.full(values.shape, np.nan)
    
    # Apply classification (order matters - most extreme first)
    if classification == 'mckee':
        # Wet categories
        categories = np.where(values >= 2.0, 4, categories)
        categories = np.where((values >= 1.5) & (values < 2.0), 3, categories)
        categories = np.where((values >= 1.0) & (values < 1.5), 2, categories)
        # Near normal
        categories = np.where((values > -1.0) & (values < 1.0), 1, categories)
        # Dry categories
        categories = np.where((values <= -1.0) & (values > -1.5), 0, categories)
        categories = np.where((values <= -1.5) & (values > -2.0), -1, categories)
        categories = np.where(values <= -2.0, -2, categories)
    
    if isinstance(index_values, xr.DataArray):
        return xr.DataArray(
            data=categories,
            dims=index_values.dims,
            coords=index_values.coords,
            name='drought_category',
            attrs={
                'long_name': 'Drought classification (McKee et al., 1993)',
                'classification': classification,
                'flag_values': [-2, -1, 0, 1, 2, 3, 4],
                'flag_meanings': 'extremely_dry severely_dry moderately_dry '
                                'near_normal moderately_wet very_wet extremely_wet'
            }
        )
    else:
        return categories


def get_drought_area_percentage(
    index_values: Union[np.ndarray, xr.DataArray],
    threshold: float = -1.0
) -> Union[float, xr.DataArray]:
    """
    Calculate percentage of area under drought conditions.

    :param index_values: SPI or SPEI values (2D or 3D array)
    :param threshold: drought threshold (default: -1.0 for moderate drought)
    :return: percentage of area under drought (0-100)

    Example:
        >>> # Get time series of drought area percentage
        >>> drought_pct = get_drought_area_percentage(spi_12, threshold=-1.5)
    """
    values = index_values.values if isinstance(index_values, xr.DataArray) else index_values

    if values.ndim == 2:
        # Single time slice (lat, lon)
        valid_count = np.sum(~np.isnan(values))
        drought_count = np.sum(values <= threshold)
        return 100.0 * drought_count / valid_count if valid_count > 0 else np.nan

    elif values.ndim == 3:
        # Time series (time, lat, lon)
        n_time = values.shape[0]
        percentages = np.full(n_time, np.nan)

        for t in range(n_time):
            slice_vals = values[t, :, :]
            valid_count = np.sum(~np.isnan(slice_vals))
            drought_count = np.sum(slice_vals <= threshold)
            if valid_count > 0:
                percentages[t] = 100.0 * drought_count / valid_count

        if isinstance(index_values, xr.DataArray):
            return xr.DataArray(
                data=percentages,
                dims=['time'],
                coords={'time': index_values.coords['time']},
                name='drought_area_percentage',
                attrs={
                    'long_name': f'Percentage of area with index <= {threshold}',
                    'units': '%',
                    'threshold': threshold
                }
            )
        return percentages

    else:
        raise ValueError(f"Unsupported array dimensions: {values.ndim}")


# =============================================================================
# GLOBAL-SCALE PROCESSING (MEMORY-EFFICIENT)
# =============================================================================

def spi_global(
    precip_path: str,
    output_path: str,
    scale: int = 12,
    periodicity: Union[str, Periodicity] = Periodicity.monthly,
    calibration_start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    calibration_end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    chunk_size: int = 500,
    var_name: Optional[str] = None,
    save_params: bool = True,
    distribution: str = DEFAULT_DISTRIBUTION,
    global_attrs: Optional[Dict] = None
) -> xr.Dataset:
    """
    Calculate SPI for global-scale datasets with automatic memory management.

    This function handles datasets that exceed available RAM by processing
    data in spatial chunks and streaming results to disk.

    :param precip_path: Path to precipitation NetCDF file
    :param output_path: Path for output SPI NetCDF file
    :param scale: Accumulation scale (default: 12)
    :param periodicity: 'monthly' or 'daily'
    :param calibration_start_year: Start of calibration period
    :param calibration_end_year: End of calibration period
    :param chunk_size: Spatial chunk size (default: 500)
    :param var_name: Precipitation variable name (auto-detected if None)
    :param save_params: Whether to save fitting parameters
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :param global_attrs: optional dict of global attributes to override defaults
        (e.g., {'institution': 'My Org', 'source': 'My Project'})
    :return: Dataset with computed SPI

    Example:
        >>> # Process Global CHIRPS data
        >>> result = spi_global(
        ...     'chirps_global_monthly_1981_2024.nc',
        ...     'spi_12_global.nc',
        ...     scale=12,
        ...     chunk_size=500  # Adjust based on available RAM
        ... )
        >>> # With Pearson III distribution
        >>> result = spi_global(
        ...     'chirps_global_monthly_1981_2024.nc',
        ...     'spi_pearson3_12_global.nc',
        ...     scale=12,
        ...     distribution='pearson3'
        ... )
    """
    from chunked import ChunkedProcessor

    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    processor = ChunkedProcessor(
        chunk_lat=chunk_size,
        chunk_lon=chunk_size
    )

    return processor.compute_spi_chunked(
        precip=precip_path,
        output_path=output_path,
        scale=scale,
        periodicity=periodicity,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        var_name=var_name,
        save_params=save_params,
        distribution=distribution,
        global_attrs=global_attrs
    )


def spei_global(
    precip_path: str,
    pet_path: str,
    output_path: str,
    scale: int = 12,
    periodicity: Union[str, Periodicity] = Periodicity.monthly,
    calibration_start_year: int = DEFAULT_CALIBRATION_START_YEAR,
    calibration_end_year: int = DEFAULT_CALIBRATION_END_YEAR,
    chunk_size: int = 500,
    precip_var_name: Optional[str] = None,
    pet_var_name: Optional[str] = None,
    save_params: bool = True,
    distribution: str = DEFAULT_DISTRIBUTION,
    global_attrs: Optional[Dict] = None
) -> xr.Dataset:
    """
    Calculate SPEI for global-scale datasets with automatic memory management.

    :param precip_path: Path to precipitation NetCDF file
    :param pet_path: Path to PET NetCDF file
    :param output_path: Path for output SPEI NetCDF file
    :param scale: Accumulation scale
    :param periodicity: 'monthly' or 'daily'
    :param calibration_start_year: Start of calibration period
    :param calibration_end_year: End of calibration period
    :param chunk_size: Spatial chunk size
    :param precip_var_name: Precipitation variable name
    :param pet_var_name: PET variable name
    :param save_params: Whether to save fitting parameters
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'.
        Note: Pearson III or Log-Logistic are recommended for SPEI.
    :param global_attrs: optional dict of global attributes to override defaults
        (e.g., {'institution': 'My Org', 'source': 'My Project'})
    :return: Dataset with computed SPEI

    Example:
        >>> result = spei_global(
        ...     'chirps_global_monthly.nc',
        ...     'pet_global_monthly.nc',
        ...     'spei_12_global.nc',
        ...     scale=12
        ... )
        >>> # With Pearson III (recommended for SPEI)
        >>> result = spei_global(
        ...     'chirps_global_monthly.nc',
        ...     'pet_global_monthly.nc',
        ...     'spei_pearson3_12_global.nc',
        ...     scale=12,
        ...     distribution='pearson3'
        ... )
    """
    from chunked import ChunkedProcessor

    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    processor = ChunkedProcessor(
        chunk_lat=chunk_size,
        chunk_lon=chunk_size
    )

    return processor.compute_spei_chunked(
        precip=precip_path,
        pet=pet_path,
        output_path=output_path,
        scale=scale,
        periodicity=periodicity,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        precip_var_name=precip_var_name,
        pet_var_name=pet_var_name,
        save_params=save_params,
        distribution=distribution,
        global_attrs=global_attrs
    )


def estimate_memory_requirements(
    precip: Union[str, xr.DataArray, xr.Dataset],
    var_name: Optional[str] = None,
    available_memory_gb: Optional[float] = None
):
    """
    Estimate memory requirements before running SPI/SPEI computation.

    Use this function to check if your data will fit in memory and
    get recommended chunk sizes if chunking is needed.

    :param precip: Precipitation data path or xarray object
    :param var_name: Variable name if Dataset
    :param available_memory_gb: Available RAM in GB (auto-detected if None)
    :return: MemoryEstimate object with recommendations

    Example:
        >>> mem = estimate_memory_requirements('chirps_global.nc')
        >>> print(mem)
        MemoryEstimate(
          Input size: 35.80 GB
          Peak memory needed: 429.60 GB
          Available memory: 150.00 GB
          Status: ✗ Requires chunking
          Recommended chunk size: (500, 500) (lat, lon)
          Number of chunks: 36
        )
    """
    from chunked import estimate_memory, estimate_memory_from_data

    if isinstance(precip, str):
        ds = xr.open_dataset(precip)
        if var_name is None:
            precip_vars = [v for v in ds.data_vars
                          if any(x in v.lower() for x in PRECIP_VAR_PATTERNS)]
            var_name = precip_vars[0] if precip_vars else list(ds.data_vars)[0]
        return estimate_memory_from_data(ds, var_name, available_memory_gb)
    else:
        return estimate_memory_from_data(precip, var_name, available_memory_gb)
