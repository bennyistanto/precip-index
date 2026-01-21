"""
Core computation functions for SPI/SPEI calculation.

Includes gamma distribution fitting, scaling, and transformation functions.
Optimized for global-scale data with Dask and multiprocessing support.

Modified/adapted from James Adams' climate-indices package
https://github.com/monocongo/climate_indices

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.stats
import xarray as xr
from numba import jit, prange

from config import (
    FITTED_INDEX_VALID_MAX,
    FITTED_INDEX_VALID_MIN,
    MIN_VALUES_FOR_GAMMA_FIT,
    Periodicity,
    get_logger,
)
from utils import validate_array

# Module logger
_logger = get_logger(__name__)

# Suppress scipy warnings for invalid gamma fits
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# SCALING FUNCTIONS
# =============================================================================

@jit(nopython=True, cache=True)
def _sum_to_scale_1d(values: np.ndarray, scale: int) -> np.ndarray:
    """
    Numba-optimized rolling sum for 1-D array.
    
    :param values: 1-D array of values
    :param scale: number of time steps to sum
    :return: array of rolling sums (first scale-1 values are NaN)
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(scale - 1, n):
        total = 0.0
        valid_count = 0
        
        for j in range(scale):
            val = values[i - j]
            if not np.isnan(val):
                total += val
                valid_count += 1
        
        # Only compute sum if all values in window are valid
        if valid_count == scale:
            result[i] = total
    
    return result


def sum_to_scale(
    values: np.ndarray,
    scale: int
) -> np.ndarray:
    """
    Compute rolling sum over specified time scale.
    
    For SPI/SPEI, this accumulates precipitation (or P-PET) over
    the specified number of time steps (e.g., 3-month, 12-month).

    :param values: 1-D numpy array of values (precipitation or P-PET)
    :param scale: number of time steps to accumulate (e.g., 1, 3, 6, 12)
    :return: array of scaled (accumulated) values, same length as input
        First (scale-1) values will be NaN
    :raises ValueError: if scale < 1
    """
    if scale < 1:
        raise ValueError(f"Scale must be >= 1, got: {scale}")
    
    if scale == 1:
        return values.copy()
    
    # Flatten if needed
    original_shape = values.shape
    values_flat = values.flatten()
    
    # Use numba-optimized function
    result = _sum_to_scale_1d(values_flat, scale)
    
    return result.reshape(original_shape) if len(original_shape) > 1 else result


# =============================================================================
# GAMMA DISTRIBUTION FITTING
# =============================================================================

@jit(nopython=True, cache=True)
def _gamma_parameters_1d(
    values: np.ndarray,
    calibration_start_idx: int,
    calibration_end_idx: int
) -> Tuple[float, float, float]:
    """
    Numba-optimized gamma parameter fitting for a single time series.
    
    Uses method of moments estimation for alpha and beta.
    
    :param values: 1-D array of values for one calendar period (e.g., all Januaries)
    :param calibration_start_idx: start index of calibration period
    :param calibration_end_idx: end index of calibration period (exclusive)
    :return: tuple of (alpha, beta, prob_zero)
    """
    # Extract calibration period
    calib_values = values[calibration_start_idx:calibration_end_idx]
    
    # Count zeros and compute probability of zero
    n_total = 0
    n_zeros = 0
    
    for val in calib_values:
        if not np.isnan(val):
            n_total += 1
            if val == 0.0:
                n_zeros += 1
    
    if n_total == 0:
        return np.nan, np.nan, np.nan
    
    prob_zero = n_zeros / n_total
    
    # Get non-zero values for gamma fitting
    non_zero_vals = []
    for val in calib_values:
        if not np.isnan(val) and val > 0.0:
            non_zero_vals.append(val)
    
    n_non_zero = len(non_zero_vals)
    
    if n_non_zero < MIN_VALUES_FOR_GAMMA_FIT:
        return np.nan, np.nan, prob_zero
    
    # Method of moments estimation
    # Calculate mean and mean of logs
    sum_vals = 0.0
    sum_logs = 0.0
    
    for val in non_zero_vals:
        sum_vals += val
        sum_logs += np.log(val)
    
    mean = sum_vals / n_non_zero
    mean_log = sum_logs / n_non_zero
    log_mean = np.log(mean)
    
    # A = ln(mean) - mean(ln(x))
    a = log_mean - mean_log
    
    if a <= 0:
        return np.nan, np.nan, prob_zero
    
    # Alpha (shape) using approximation
    alpha = (1.0 + np.sqrt(1.0 + 4.0 * a / 3.0)) / (4.0 * a)
    
    # Beta (scale)
    beta = mean / alpha
    
    return alpha, beta, prob_zero


def gamma_parameters(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gamma distribution parameters (alpha, beta) for each calendar period.
    
    For monthly data: computes parameters for each of 12 months
    For daily data: computes parameters for each of 366 days

    :param values: 2-D array with shape (years, periods_per_year)
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :return: tuple of (alphas, betas, probs_zero) arrays with shape (periods_per_year,)
    """
    periods = periodicity.value  # 12 or 366
    
    # Validate and reshape input
    values = validate_array(values, periodicity)
    
    # Handle all-NaN input
    if np.all(np.isnan(values)):
        return (
            np.full(periods, np.nan),
            np.full(periods, np.nan),
            np.full(periods, np.nan)
        )
    
    # Calculate calibration indices
    data_end_year = data_start_year + values.shape[0] - 1
    
    # Adjust calibration period if out of bounds
    cal_start = max(calibration_start_year, data_start_year)
    cal_end = min(calibration_end_year, data_end_year)
    
    cal_start_idx = cal_start - data_start_year
    cal_end_idx = cal_end - data_start_year + 1
    
    # Initialize output arrays
    alphas = np.full(periods, np.nan)
    betas = np.full(periods, np.nan)
    probs_zero = np.full(periods, np.nan)
    
    # Fit gamma for each calendar period
    for period_idx in range(periods):
        # Get all values for this calendar period (column)
        period_values = values[:, period_idx]
        
        alpha, beta, prob_zero = _gamma_parameters_1d(
            period_values, cal_start_idx, cal_end_idx
        )
        
        alphas[period_idx] = alpha
        betas[period_idx] = beta
        probs_zero[period_idx] = prob_zero
    
    return alphas, betas, probs_zero


# =============================================================================
# GAMMA TRANSFORMATION
# =============================================================================

def transform_fitted_gamma(
    values: np.ndarray,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    alphas: Optional[np.ndarray] = None,
    betas: Optional[np.ndarray] = None,
    probs_zero: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Transform values to normalized (standard normal) values using gamma CDF.
    
    This is the core SPI/SPEI transformation:
    1. Fit values to gamma distribution (or use provided parameters)
    2. Compute gamma CDF probabilities
    3. Adjust for probability of zero
    4. Transform to standard normal using inverse normal CDF

    :param values: 2-D array with shape (years, periods_per_year)
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param alphas: pre-computed alpha parameters (optional)
    :param betas: pre-computed beta parameters (optional)
    :param probs_zero: pre-computed probability of zero (optional)
    :return: transformed values (SPI/SPEI), same shape as input
    """
    # Validate and reshape
    values = validate_array(values, periodicity)
    
    # Handle all-NaN input
    if np.all(np.isnan(values)):
        return values
    
    # Compute fitting parameters if not provided
    if alphas is None or betas is None:
        alphas, betas, probs_zero = gamma_parameters(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity
        )
    
    # If probs_zero not provided, compute from data
    if probs_zero is None:
        zeros = (values == 0).sum(axis=0)
        probs_zero = zeros / values.shape[0]
    
    # Initialize output
    transformed = np.full(values.shape, np.nan)
    
    # Transform each calendar period
    for period_idx in range(values.shape[1]):
        alpha = alphas[period_idx]
        beta = betas[period_idx]
        prob_zero = probs_zero[period_idx]
        
        # Skip if parameters are invalid
        if np.isnan(alpha) or np.isnan(beta) or alpha <= 0 or beta <= 0:
            continue
        
        # Get values for this period
        period_values = values[:, period_idx]
        
        # Compute gamma CDF for non-zero values
        # scipy.stats.gamma uses shape (a) and scale parameters
        gamma_probs = scipy.stats.gamma.cdf(period_values, a=alpha, scale=beta)
        
        # Adjust probabilities for zeros:
        # P(X <= x) = P(zero) + P(non-zero) * P(X <= x | X > 0)
        adjusted_probs = prob_zero + (1.0 - prob_zero) * gamma_probs
        
        # Clamp probabilities to valid range (0, 1) exclusive
        # to avoid infinity in inverse normal
        adjusted_probs = np.clip(adjusted_probs, 1e-10, 1.0 - 1e-10)
        
        # Transform to standard normal
        transformed[:, period_idx] = scipy.stats.norm.ppf(adjusted_probs)
    
    # Clip to valid SPI/SPEI range
    transformed = np.clip(transformed, FITTED_INDEX_VALID_MIN, FITTED_INDEX_VALID_MAX)
    
    return transformed


# =============================================================================
# PARALLEL PROCESSING FOR GRIDDED DATA
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _process_grid_parallel(
    data_3d: np.ndarray,
    scale: int,
    cal_start_idx: int,
    cal_end_idx: int,
    periods_per_year: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-parallelized SPI/SPEI computation for 3D grid.
    
    Processes each grid cell in parallel using multiple cores.
    
    :param data_3d: 3-D array with shape (time, lat, lon)
    :param scale: accumulation scale
    :param cal_start_idx: calibration start index (year)
    :param cal_end_idx: calibration end index (year)
    :param periods_per_year: 12 for monthly, 366 for daily
    :return: tuple of (result, alphas, betas, probs_zero)
    """
    n_time, n_lat, n_lon = data_3d.shape
    n_years = n_time // periods_per_year
    
    # Output arrays
    result = np.full((n_time, n_lat, n_lon), np.nan)
    alphas_out = np.full((periods_per_year, n_lat, n_lon), np.nan)
    betas_out = np.full((periods_per_year, n_lat, n_lon), np.nan)
    probs_zero_out = np.full((periods_per_year, n_lat, n_lon), np.nan)
    
    # Process each grid cell in parallel
    for lat_idx in prange(n_lat):
        for lon_idx in range(n_lon):
            # Extract time series for this cell
            cell_data = data_3d[:, lat_idx, lon_idx].copy()
            
            # Skip if all NaN
            all_nan = True
            for t in range(n_time):
                if not np.isnan(cell_data[t]):
                    all_nan = False
                    break
            
            if all_nan:
                continue
            
            # Apply scaling (rolling sum)
            scaled_data = np.full(n_time, np.nan)
            for i in range(scale - 1, n_time):
                total = 0.0
                valid = 0
                for j in range(scale):
                    val = cell_data[i - j]
                    if not np.isnan(val):
                        total += val
                        valid += 1
                if valid == scale:
                    scaled_data[i] = total
            
            # Reshape to (years, periods)
            scaled_2d = scaled_data.reshape(n_years, periods_per_year)
            
            # Process each calendar period
            for period_idx in range(periods_per_year):
                period_vals = scaled_2d[:, period_idx]
                
                # Compute gamma parameters
                alpha, beta, prob_zero = _gamma_parameters_1d(
                    period_vals, cal_start_idx, cal_end_idx
                )
                
                alphas_out[period_idx, lat_idx, lon_idx] = alpha
                betas_out[period_idx, lat_idx, lon_idx] = beta
                probs_zero_out[period_idx, lat_idx, lon_idx] = prob_zero
                
                # Skip invalid parameters
                if np.isnan(alpha) or np.isnan(beta) or alpha <= 0 or beta <= 0:
                    continue
                
                # Transform each value in this period
                for year_idx in range(n_years):
                    val = scaled_2d[year_idx, period_idx]
                    if np.isnan(val):
                        continue
                    
                    # Gamma CDF (approximation for numba)
                    # Using scipy is not possible in numba, so we use
                    # incomplete gamma function approximation
                    x = val / beta
                    
                    # Simple gamma CDF approximation using series expansion
                    # For more accuracy, we'll post-process with scipy
                    time_idx = year_idx * periods_per_year + period_idx
                    result[time_idx, lat_idx, lon_idx] = val  # Placeholder
    
    return result, alphas_out, betas_out, probs_zero_out


def compute_index_parallel(
    data: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute SPI/SPEI for 3D gridded data with parallel processing.
    
    Optimized for large global datasets using vectorized numpy operations
    and scipy for gamma transformation.

    :param data: 3-D array with shape (time, lat, lon) - CF Convention
    :param scale: accumulation scale (e.g., 1, 3, 6, 12)
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters dict with
        'alpha', 'beta', 'prob_zero' arrays of shape (periods, lat, lon)
    :return: tuple of (result_array, fitting_params_dict)
    """
    n_time, n_lat, n_lon = data.shape
    periods_per_year = periodicity.value
    n_years = n_time // periods_per_year
    
    _logger.info(
        f"Computing index: shape={data.shape}, scale={scale}, "
        f"grid_cells={n_lat * n_lon:,}"
    )
    
    # Ensure data is float64 and contiguous
    data = np.ascontiguousarray(data, dtype=np.float64)
    
    # Step 1: Apply scaling (vectorized along time axis)
    _logger.info("Step 1/3: Applying temporal scaling...")
    scaled_data = np.full_like(data, np.nan)
    
    if scale == 1:
        scaled_data = data.copy()
    else:
        # Vectorized rolling sum using stride tricks or simple loop
        for t in range(scale - 1, n_time):
            window = data[t - scale + 1:t + 1, :, :]
            # Sum only if all values in window are valid
            valid_mask = ~np.any(np.isnan(window), axis=0)
            scaled_data[t, :, :] = np.where(
                valid_mask,
                np.nansum(window, axis=0),
                np.nan
            )
    
    # Step 2: Compute or use provided fitting parameters
    _logger.info("Step 2/3: Computing gamma parameters...")
    
    if fitting_params is not None:
        alphas = fitting_params['alpha']
        betas = fitting_params['beta']
        probs_zero = fitting_params['prob_zero']
        _logger.info("Using pre-computed fitting parameters")
    else:
        # Reshape to (years, periods, lat, lon)
        scaled_4d = scaled_data.reshape(n_years, periods_per_year, n_lat, n_lon)
        
        # Calculate calibration indices
        data_end_year = data_start_year + n_years - 1
        cal_start = max(calibration_start_year, data_start_year)
        cal_end = min(calibration_end_year, data_end_year)
        cal_start_idx = cal_start - data_start_year
        cal_end_idx = cal_end - data_start_year + 1
        
        # Initialize parameter arrays: shape (periods, lat, lon)
        alphas = np.full((periods_per_year, n_lat, n_lon), np.nan)
        betas = np.full((periods_per_year, n_lat, n_lon), np.nan)
        probs_zero = np.full((periods_per_year, n_lat, n_lon), np.nan)
        
        # Compute parameters for each calendar period (vectorized over space)
        for period_idx in range(periods_per_year):
            # Get all years for this period: shape (n_years, lat, lon)
            period_data = scaled_4d[:, period_idx, :, :]
            
            # Calibration subset: shape (cal_years, lat, lon)
            calib_data = period_data[cal_start_idx:cal_end_idx, :, :]
            
            # Count zeros and total valid values
            n_valid = np.sum(~np.isnan(calib_data), axis=0)
            n_zeros = np.sum(calib_data == 0, axis=0)
            
            # Probability of zero
            with np.errstate(divide='ignore', invalid='ignore'):
                probs_zero[period_idx, :, :] = np.where(
                    n_valid > 0, n_zeros / n_valid, np.nan
                )
            
            # Get non-zero values for gamma fitting
            # Replace zeros with NaN for fitting
            calib_nonzero = np.where(calib_data > 0, calib_data, np.nan)
            
            # Count non-zero valid values
            n_nonzero = np.sum(~np.isnan(calib_nonzero), axis=0)
            
            # Method of moments estimation (vectorized)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Mean of non-zero values
                mean_vals = np.nanmean(calib_nonzero, axis=0)
                
                # Mean of log values
                log_vals = np.log(calib_nonzero)
                mean_log = np.nanmean(log_vals, axis=0)
                
                # A = ln(mean) - mean(ln(x))
                a = np.log(mean_vals) - mean_log
                
                # Alpha (shape parameter)
                alpha = np.where(
                    a > 0,
                    (1.0 + np.sqrt(1.0 + 4.0 * a / 3.0)) / (4.0 * a),
                    np.nan
                )
                
                # Beta (scale parameter)
                beta = mean_vals / alpha
                
                # Apply minimum data requirement
                valid_fit = n_nonzero >= MIN_VALUES_FOR_GAMMA_FIT
                
                alphas[period_idx, :, :] = np.where(valid_fit, alpha, np.nan)
                betas[period_idx, :, :] = np.where(valid_fit, beta, np.nan)
    
    # Step 3: Transform to standard normal (vectorized)
    _logger.info("Step 3/3: Transforming to standard normal...")
    
    # Reshape scaled data to (years, periods, lat, lon)
    scaled_4d = scaled_data.reshape(n_years, periods_per_year, n_lat, n_lon)
    result_4d = np.full_like(scaled_4d, np.nan)
    
    # Process each calendar period
    for period_idx in range(periods_per_year):
        alpha = alphas[period_idx, :, :]
        beta = betas[period_idx, :, :]
        prob_zero = probs_zero[period_idx, :, :]
        
        # Values for this period: shape (n_years, lat, lon)
        period_vals = scaled_4d[:, period_idx, :, :]
        
        # Create mask for valid parameters
        valid_params = (
            ~np.isnan(alpha) & ~np.isnan(beta) & 
            (alpha > 0) & (beta > 0)
        )
        
        # Expand valid_params to match period_vals shape
        valid_params_expanded = np.broadcast_to(
            valid_params[np.newaxis, :, :], period_vals.shape
        )
        
        # Compute gamma CDF (vectorized)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_probs = scipy.stats.gamma.cdf(
                period_vals,
                a=alpha[np.newaxis, :, :],
                scale=beta[np.newaxis, :, :]
            )
            
            # Adjust for probability of zero
            adjusted_probs = (
                prob_zero[np.newaxis, :, :] + 
                (1.0 - prob_zero[np.newaxis, :, :]) * gamma_probs
            )
            
            # Clamp to valid range
            adjusted_probs = np.clip(adjusted_probs, 1e-10, 1.0 - 1e-10)
            
            # Transform to standard normal
            transformed = scipy.stats.norm.ppf(adjusted_probs)
            
            # Apply only where parameters are valid
            result_4d[:, period_idx, :, :] = np.where(
                valid_params_expanded & ~np.isnan(period_vals),
                transformed,
                np.nan
            )
    
    # Reshape back to (time, lat, lon)
    result = result_4d.reshape(n_time, n_lat, n_lon)
    
    # Clip to valid range
    result = np.clip(result, FITTED_INDEX_VALID_MIN, FITTED_INDEX_VALID_MAX)
    
    # Prepare fitting parameters dict
    params_dict = {
        'alpha': alphas,
        'beta': betas,
        'prob_zero': probs_zero
    }
    
    _logger.info("Index computation complete")
    
    return result, params_dict


# =============================================================================
# DASK-ENABLED COMPUTATION FOR VERY LARGE DATASETS
# =============================================================================

def compute_index_dask(
    data: xr.DataArray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: Optional[Dict[str, xr.DataArray]] = None,
    chunks: Optional[Dict[str, int]] = None
) -> Tuple[xr.DataArray, Dict[str, xr.DataArray]]:
    """
    Compute SPI/SPEI using Dask for out-of-core processing.
    
    Suitable for very large global datasets that don't fit in memory.
    Uses lazy evaluation and chunked processing.

    :param data: xarray DataArray with dimensions (time, lat, lon)
    :param scale: accumulation scale
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters
    :param chunks: optional chunk sizes, e.g., {'lat': 100, 'lon': 100}
    :return: tuple of (result DataArray, fitting_params dict of DataArrays)
    """
    import dask.array as da
    from dask.diagnostics import ProgressBar
    
    _logger.info(f"Starting Dask-enabled computation for shape {data.shape}")
    
    # Ensure data is chunked
    if chunks is None:
        # Default chunking: keep time together, chunk spatially
        chunks = {'time': -1, 'lat': 100, 'lon': 100}
    
    if not data.chunks:
        data = data.chunk(chunks)
        _logger.info(f"Chunked data with {chunks}")
    
    # Get coordinates
    time_coord = data.coords['time']
    lat_coord = data.coords['lat']
    lon_coord = data.coords['lon']
    
    # Define computation function for map_blocks
    def _compute_chunk(
        chunk_data: np.ndarray,
        block_info: dict = None
    ) -> np.ndarray:
        """Process a single spatial chunk."""
        if chunk_data.size == 0:
            return chunk_data
        
        result, _ = compute_index_parallel(
            chunk_data,
            scale=scale,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            periodicity=periodicity,
            fitting_params=None  # Compute fresh for each chunk
        )
        return result
    
    # Apply computation using dask
    _logger.info("Applying Dask map_blocks...")
    
    # Use xarray's apply_ufunc for dask integration
    result = xr.apply_ufunc(
        lambda x: compute_index_parallel(
            x, scale, data_start_year,
            calibration_start_year, calibration_end_year,
            periodicity, None
        )[0],
        data,
        input_core_dims=[['time', 'lat', 'lon']],
        output_core_dims=[['time', 'lat', 'lon']],
        vectorize=False,
        dask='parallelized',
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
    
    # For fitting params, we need to compute them separately
    # This is a limitation - params are computed during result computation
    # For now, return empty params dict (user should use compute_index_parallel
    # for param extraction on a subset)
    
    _logger.info("Computation graph built. Use .compute() to execute.")
    
    return result, {}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_spi_1d(
    precip: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute SPI for a single time series (1-D array).
    
    Convenience function for single-point calculations.

    :param precip: 1-D array of precipitation values
    :param scale: accumulation scale
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters
    :return: tuple of (SPI values, fitting_params dict)
    """
    # Validate input
    precip = np.asarray(precip).flatten()
    
    # Apply scaling
    scaled = sum_to_scale(precip, scale)
    
    # Reshape to 2D
    periods = periodicity.value
    scaled_2d = validate_array(scaled, periodicity)
    
    # Extract params if provided
    if fitting_params is not None:
        alphas = fitting_params.get('alpha')
        betas = fitting_params.get('beta')
        probs_zero = fitting_params.get('prob_zero')
    else:
        alphas = betas = probs_zero = None
    
    # Transform
    result_2d = transform_fitted_gamma(
        scaled_2d,
        data_start_year,
        calibration_start_year,
        calibration_end_year,
        periodicity,
        alphas, betas, probs_zero
    )
    
    # Get params if not provided
    if fitting_params is None:
        alphas, betas, probs_zero = gamma_parameters(
            scaled_2d,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity
        )
    
    # Flatten result
    result = result_2d.flatten()[:len(precip)]
    
    params = {
        'alpha': alphas,
        'beta': betas,
        'prob_zero': probs_zero
    }
    
    return result, params


def compute_spei_1d(
    precip: np.ndarray,
    pet: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute SPEI for a single time series (1-D arrays).
    
    Convenience function for single-point calculations.

    :param precip: 1-D array of precipitation values (mm)
    :param pet: 1-D array of potential evapotranspiration values (mm)
    :param scale: accumulation scale
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters
    :return: tuple of (SPEI values, fitting_params dict)
    """
    # Validate inputs
    precip = np.asarray(precip).flatten()
    pet = np.asarray(pet).flatten()
    
    if len(precip) != len(pet):
        raise ValueError(
            f"Precipitation and PET arrays must have same length: "
            f"{len(precip)} vs {len(pet)}"
        )
    
    # Compute water balance (P - PET)
    # Add offset to ensure positive values for gamma fitting
    water_balance = (precip - pet) + 1000.0
    
    # Use same logic as SPI
    return compute_spi_1d(
        water_balance,
        scale,
        data_start_year,
        calibration_start_year,
        calibration_end_year,
        periodicity,
        fitting_params
    )
