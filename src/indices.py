"""
SPI and SPEI climate indices calculation module.

High-level API for computing Standardized Precipitation Index (SPI) and
Standardized Precipitation Evapotranspiration Index (SPEI) with support
for saving/loading fitting parameters.

Optimized for global-scale gridded data following CF Convention (time, lat, lon).

Modified/adapted from James Adams' climate-indices package
https://github.com/monocongo/climate_indices

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank

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
    FITTED_INDEX_VALID_MAX,
    FITTED_INDEX_VALID_MIN,
    FITTING_PARAM_NAMES,
    NC_FILL_VALUE,
    Periodicity,
    get_fitting_param_attributes,
    get_fitting_param_name,
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
    global_attrs: Optional[Dict] = None
) -> str:
    """
    Save gamma fitting parameters to NetCDF file for later reuse.
    
    Parameters can be loaded later to speed up recalculation or
    to apply the same calibration to new data.

    :param params: dictionary containing 'alpha', 'beta', 'prob_zero' arrays
        Arrays should have shape (periods,) for 1D or (periods, lat, lon) for 3D
    :param filepath: output NetCDF file path
    :param scale: accumulation scale (e.g., 1, 3, 6, 12)
    :param periodicity: monthly or daily
    :param index_type: 'spi' or 'spei'
    :param calibration_start_year: start year of calibration period
    :param calibration_end_year: end year of calibration period
    :param coords: optional coordinate dict with 'lat', 'lon' for gridded data
    :param global_attrs: optional additional global attributes
    :return: filepath of saved file
    """
    _logger.info(f"Saving fitting parameters to: {filepath}")
    
    # Convert periodicity if string
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)
    
    # Create dataset
    ds = xr.Dataset()
    
    # Determine dimensionality from alpha shape
    alpha = params['alpha']
    if isinstance(alpha, xr.DataArray):
        alpha = alpha.values
    
    ndim = alpha.ndim
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
                'lat': coords.get('lat', np.arange(alpha.shape[1])),
                'lon': coords.get('lon', np.arange(alpha.shape[2]))
            }
        else:
            coords_dict = {
                period_dim: period_coord,
                'lat': np.arange(alpha.shape[1]),
                'lon': np.arange(alpha.shape[2])
            }
    else:
        raise ValueError(f"Unsupported parameter array dimensions: {ndim}")
    
    # Add each parameter as a variable
    for param_name in FITTING_PARAM_NAMES:
        if param_name not in params:
            _logger.warning(f"Parameter '{param_name}' not found in params dict")
            continue
        
        param_data = params[param_name]
        if isinstance(param_data, xr.DataArray):
            param_data = param_data.values
        
        var_name = get_fitting_param_name(param_name, scale, periodicity)
        var_attrs = get_fitting_param_attributes(param_name, scale, periodicity)
        
        ds[var_name] = xr.DataArray(
            data=param_data,
            dims=dims,
            coords=coords_dict,
            attrs=var_attrs
        )
    
    # Global attributes
    ds.attrs = {
        'title': f'Gamma distribution fitting parameters for {index_type.upper()}',
        'institution': 'GFDRR/GOST, The World Bank',
        'source': 'climate_indices package',
        'history': f'Created {datetime.now().isoformat()}',
        'Conventions': 'CF-1.8',
        'index_type': index_type.upper(),
        'distribution': 'gamma',
        'scale': scale,
        'periodicity': periodicity.name,
        'calibration_start_year': calibration_start_year or 'not specified',
        'calibration_end_year': calibration_end_year or 'not specified',
    }
    
    if global_attrs:
        ds.attrs.update(global_attrs)
    
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
    periodicity: Union[str, Periodicity]
) -> Dict[str, np.ndarray]:
    """
    Load gamma fitting parameters from NetCDF file.

    :param filepath: path to NetCDF file with fitting parameters
    :param scale: accumulation scale to load (e.g., 12)
    :param periodicity: monthly or daily
    :return: dictionary with 'alpha', 'beta', 'prob_zero' arrays
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
    
    params = {}
    for param_name in FITTING_PARAM_NAMES:
        var_name = get_fitting_param_name(param_name, scale, periodicity)
        
        if var_name not in ds:
            raise KeyError(
                f"Variable '{var_name}' not found in {filepath}. "
                f"Available variables: {list(ds.data_vars)}"
            )
        
        params[param_name] = ds[var_name].values
    
    ds.close()
    
    _logger.info(
        f"Loaded parameters for scale={scale}, periodicity={periodicity.name}, "
        f"shape={params['alpha'].shape}"
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
    var_name: Optional[str] = None
) -> Union[xr.DataArray, Tuple[xr.DataArray, Dict[str, np.ndarray]]]:
    """
    Calculate Standardized Precipitation Index (SPI).
    
    SPI is computed by fitting precipitation data to a gamma distribution
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
    :param fitting_params: optional pre-computed gamma parameters from save_fitting_params()
    :param return_params: if True, return (result, params) tuple
    :param var_name: variable name if precip is a Dataset
    :return: SPI values as xarray DataArray, or tuple (SPI, params) if return_params=True
    
    Example:
        >>> # Basic usage
        >>> spi_12 = spi(precip_da, scale=12)
        
        >>> # With parameter saving
        >>> spi_12, params = spi(precip_da, scale=12, return_params=True)
        >>> save_fitting_params(params, 'spi_params.nc', scale=12, periodicity='monthly')
        
        >>> # Using pre-computed parameters
        >>> params = load_fitting_params('spi_params.nc', scale=12, periodicity='monthly')
        >>> spi_12 = spi(new_precip_da, scale=12, fitting_params=params)
    """
    _logger.info(f"Computing SPI-{scale}")
    
    # Convert periodicity string to enum
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)
    
    # Handle different input types
    if isinstance(precip, xr.Dataset):
        if var_name is None:
            # Try to find precipitation variable
            precip_vars = [v for v in precip.data_vars 
                          if 'precip' in v.lower() or 'prcp' in v.lower() or v.lower() == 'pr']
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
            fitting_params=fitting_params
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
            fitting_params=fitting_params
        )
    else:
        raise ValueError(
            f"Unsupported array dimensions: {precip_array.ndim}. "
            f"Expected 1D (time,) or 3D (time, lat, lon)"
        )
    
    # Create output DataArray
    output_var_name = get_variable_name('spi', scale, periodicity)
    output_attrs = get_variable_attributes('spi', scale, periodicity)
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
    var_name: Optional[str] = None
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
    :return: Dataset with SPI for all scales, or tuple (Dataset, params_dict)
    
    Example:
        >>> spi_ds = spi_multi_scale(precip_da, scales=[1, 3, 6, 12])
        >>> spi_12 = spi_ds['spi_gamma_12_month']
    """
    _logger.info(f"Computing SPI for scales: {scales}")
    
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)
    
    results = {}
    all_params = {}
    
    for scale in scales:
        _logger.info(f"Processing scale {scale}...")
        
        result_da, params = spi(
            precip,
            scale=scale,
            periodicity=periodicity,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            return_params=True,
            var_name=var_name
        )
        
        var_name_out = get_variable_name('spi', scale, periodicity)
        results[var_name_out] = result_da
        all_params[scale] = params
    
    # Create output Dataset
    ds = xr.Dataset(results)
    ds.attrs = {
        'title': 'Standardized Precipitation Index (SPI)',
        'institution': 'GFDRR/GOST, The World Bank',
        'source': 'climate_indices package',
        'history': f'Created {datetime.now().isoformat()}',
        'Conventions': 'CF-1.8',
        'scales': scales,
        'distribution': 'gamma',
        'calibration_start_year': calibration_start_year,
        'calibration_end_year': calibration_end_year,
    }
    
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
    temp_var_name: Optional[str] = None
) -> Union[xr.DataArray, Tuple[xr.DataArray, Dict[str, np.ndarray]]]:
    """
    Calculate Standardized Precipitation Evapotranspiration Index (SPEI).
    
    SPEI uses the water balance (P - PET) instead of just precipitation.
    PET can be provided directly or calculated from temperature using
    the Thornthwaite method.

    :param precip: precipitation data in mm
    :param pet: potential evapotranspiration in mm (optional if temperature provided)
    :param temperature: temperature in °C for PET calculation (optional if PET provided)
    :param latitude: latitude for PET calculation (required if using temperature)
    :param scale: accumulation period in time steps
    :param periodicity: 'monthly' or 'daily'
    :param data_start_year: first year of data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param fitting_params: optional pre-computed gamma parameters
    :param return_params: if True, return (result, params) tuple
    :param precip_var_name: variable name for precipitation in Dataset
    :param pet_var_name: variable name for PET in Dataset
    :param temp_var_name: variable name for temperature in Dataset
    :return: SPEI values as xarray DataArray, or tuple (SPEI, params)
    
    Example:
        >>> # With pre-computed PET
        >>> spei_12 = spei(precip_da, pet=pet_da, scale=12)
        
        >>> # With temperature (auto-compute PET)
        >>> spei_12 = spei(precip_da, temperature=temp_da, latitude=lat_da, scale=12)
        
        >>> # Save and reuse parameters
        >>> spei_12, params = spei(precip_da, pet=pet_da, scale=12, return_params=True)
        >>> save_fitting_params(params, 'spei_params.nc', scale=12, 
        ...                     periodicity='monthly', index_type='spei')
    """
    _logger.info(f"Computing SPEI-{scale}")
    
    # Convert periodicity
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)
    
    # Handle Dataset input for precip
    if isinstance(precip, xr.Dataset):
        if precip_var_name is None:
            precip_vars = [v for v in precip.data_vars 
                          if 'precip' in v.lower() or 'prcp' in v.lower() or v.lower() == 'pr']
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
                pet_vars = [v for v in pet.data_vars if 'pet' in v.lower() or 'et' in v.lower()]
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
        _logger.info("Computing PET from temperature using Thornthwaite method")
        
        if latitude is None:
            raise ValueError("latitude required for PET calculation from temperature")
        
        # Handle temperature input
        if isinstance(temperature, xr.Dataset):
            if temp_var_name is None:
                temp_vars = [v for v in temperature.data_vars 
                            if 'temp' in v.lower() or 'tas' in v.lower() or 't2m' in v.lower()]
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
        
        # Calculate PET
        pet_da = calculate_pet(temp_da, latitude, data_start_year)
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
    water_balance = (precip_array - pet_array) + 1000.0
    
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
            fitting_params=fitting_params
        )
    elif water_balance.ndim == 3:
        result, params = compute_index_parallel(
            water_balance,
            scale=scale,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            periodicity=periodicity,
            fitting_params=fitting_params
        )
    else:
        raise ValueError(
            f"Unsupported array dimensions: {water_balance.ndim}. "
            f"Expected 1D (time,) or 3D (time, lat, lon)"
        )
    
    # Create output DataArray
    output_var_name = get_variable_name('spei', scale, periodicity)
    output_attrs = get_variable_attributes('spei', scale, periodicity)
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
    temp_var_name: Optional[str] = None
) -> Union[xr.Dataset, Tuple[xr.Dataset, Dict[int, Dict[str, np.ndarray]]]]:
    """
    Calculate SPEI for multiple time scales.

    :param precip: precipitation data
    :param pet: potential evapotranspiration (optional if temperature provided)
    :param temperature: temperature for PET calculation
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
    :return: Dataset with SPEI for all scales
    
    Example:
        >>> spei_ds = spei_multi_scale(precip_da, pet=pet_da, scales=[1, 3, 6, 12])
        >>> spei_12 = spei_ds['spei_gamma_12_month']
    """
    _logger.info(f"Computing SPEI for scales: {scales}")
    
    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)
    
    results = {}
    all_params = {}
    
    for scale in scales:
        _logger.info(f"Processing scale {scale}...")
        
        result_da, params = spei(
            precip,
            pet=pet,
            temperature=temperature,
            latitude=latitude,
            scale=scale,
            periodicity=periodicity,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            return_params=True,
            precip_var_name=precip_var_name,
            pet_var_name=pet_var_name,
            temp_var_name=temp_var_name
        )
        
        var_name_out = get_variable_name('spei', scale, periodicity)
        results[var_name_out] = result_da
        all_params[scale] = params
    
    # Create output Dataset
    ds = xr.Dataset(results)
    ds.attrs = {
        'title': 'Standardized Precipitation Evapotranspiration Index (SPEI)',
        'institution': 'GFDRR/GOST, The World Bank',
        'source': 'climate_indices package',
        'history': f'Created {datetime.now().isoformat()}',
        'Conventions': 'CF-1.8',
        'scales': scales,
        'distribution': 'gamma',
        'calibration_start_year': calibration_start_year,
        'calibration_end_year': calibration_end_year,
    }
    
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
    params_path: Optional[str] = None
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
    :param params_path: Path for fitting parameters file (default: output_path with '_params.nc' suffix)
    :return: Dataset with computed SPI

    Example:
        >>> # Process Global CHIRPS data
        >>> result = spi_global(
        ...     'chirps_global_monthly_1981_2024.nc',
        ...     'spi_12_global.nc',
        ...     scale=12,
        ...     chunk_size=500  # Adjust based on available RAM
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
        params_path=params_path
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
    params_path: Optional[str] = None
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
    :param params_path: Path for fitting parameters file (default: output_path with '_params.nc' suffix)
    :return: Dataset with computed SPEI

    Example:
        >>> result = spei_global(
        ...     'chirps_global_monthly.nc',
        ...     'pet_global_monthly.nc',
        ...     'spei_12_global.nc',
        ...     scale=12
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
        params_path=params_path
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
                          if any(x in v.lower() for x in ['precip', 'prcp', 'pr', 'ppt'])]
            var_name = precip_vars[0] if precip_vars else list(ds.data_vars)[0]
        return estimate_memory_from_data(ds, var_name, available_memory_gb)
    else:
        return estimate_memory_from_data(precip, var_name, available_memory_gb)
