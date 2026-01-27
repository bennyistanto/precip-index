"""
Core computation functions for SPI/SPEI calculation.

Includes gamma distribution fitting, scaling, and transformation functions.
Optimized for global-scale data with memory-efficient chunked processing,
Dask integration, and Numba JIT compilation for performance.

Modified/adapted from James Adams' climate-indices package
https://github.com/monocongo/climate_indices

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank
"""

import gc
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
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
    memory_efficient: bool = True
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute SPI/SPEI for 3D gridded data with parallel processing.

    Optimized for large global datasets using vectorized numpy operations
    and scipy for gamma transformation. Memory-efficient mode reduces
    peak memory usage by processing in-place where possible.

    :param data: 3-D array with shape (time, lat, lon) - CF Convention
    :param scale: accumulation scale (e.g., 1, 3, 6, 12)
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters dict with
        'alpha', 'beta', 'prob_zero' arrays of shape (periods, lat, lon)
    :param memory_efficient: if True, optimize for lower memory usage
    :return: tuple of (result_array, fitting_params_dict)
    """
    n_time, n_lat, n_lon = data.shape
    periods_per_year = periodicity.value
    n_years = n_time // periods_per_year

    _logger.info(
        f"Computing index: shape={data.shape}, scale={scale}, "
        f"grid_cells={n_lat * n_lon:,}"
    )

    # Memory-efficient: work with float32 internally, convert at end
    dtype = np.float32 if memory_efficient else np.float64

    # Ensure data is contiguous (avoid copy if already correct dtype)
    if data.dtype != dtype:
        data = data.astype(dtype, copy=False)
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    # Step 1: Apply scaling (vectorized along time axis)
    _logger.info("Step 1/3: Applying temporal scaling...")

    if scale == 1:
        # No scaling needed - use data directly (no copy)
        scaled_data = data
    else:
        # Memory-efficient rolling sum
        scaled_data = _rolling_sum_3d(data, scale, dtype)

    # Step 2: Compute or use provided fitting parameters
    _logger.info("Step 2/3: Computing gamma parameters...")

    if fitting_params is not None:
        alphas = fitting_params['alpha'].astype(dtype, copy=False)
        betas = fitting_params['beta'].astype(dtype, copy=False)
        probs_zero = fitting_params['prob_zero'].astype(dtype, copy=False)
        _logger.info("Using pre-computed fitting parameters")
    else:
        # Calculate calibration indices
        data_end_year = data_start_year + n_years - 1
        cal_start = max(calibration_start_year, data_start_year)
        cal_end = min(calibration_end_year, data_end_year)
        cal_start_idx = cal_start - data_start_year
        cal_end_idx = cal_end - data_start_year + 1

        # Compute parameters efficiently
        alphas, betas, probs_zero = _compute_gamma_params_vectorized(
            scaled_data, n_years, periods_per_year, n_lat, n_lon,
            cal_start_idx, cal_end_idx, dtype
        )

    # Step 3: Transform to standard normal (vectorized)
    _logger.info("Step 3/3: Transforming to standard normal...")

    # Transform in-place to save memory
    result = _transform_to_normal_vectorized(
        scaled_data, alphas, betas, probs_zero,
        n_years, periods_per_year, n_lat, n_lon, dtype
    )

    # Clip to valid range
    np.clip(result, FITTED_INDEX_VALID_MIN, FITTED_INDEX_VALID_MAX, out=result)

    # Prepare fitting parameters dict (convert back to float64 for storage)
    params_dict = {
        'alpha': alphas.astype(np.float64) if memory_efficient else alphas,
        'beta': betas.astype(np.float64) if memory_efficient else betas,
        'prob_zero': probs_zero.astype(np.float64) if memory_efficient else probs_zero
    }

    # Convert result to float64 for output consistency
    if memory_efficient:
        result = result.astype(np.float64)

    _logger.info("Index computation complete")

    # Explicit garbage collection to free intermediate arrays
    if memory_efficient:
        gc.collect()

    return result, params_dict


def _rolling_sum_3d(data: np.ndarray, scale: int, dtype: np.dtype) -> np.ndarray:
    """
    Memory-efficient rolling sum for 3D array.

    Uses cumulative sum approach for O(n) complexity instead of O(n*scale).
    """
    n_time, n_lat, n_lon = data.shape

    # Initialize output with NaN
    result = np.full((n_time, n_lat, n_lon), np.nan, dtype=dtype)

    # Use cumulative sum for efficiency (handles NaN correctly)
    # Replace NaN with 0 for cumsum, track valid counts
    data_filled = np.where(np.isnan(data), 0, data)
    valid_mask = (~np.isnan(data)).astype(np.int8)

    # Cumulative sums
    cumsum_data = np.cumsum(data_filled, axis=0, dtype=dtype)
    cumsum_valid = np.cumsum(valid_mask, axis=0, dtype=np.int16)

    # Calculate rolling sums using cumsum difference
    for t in range(scale - 1, n_time):
        if t == scale - 1:
            window_sum = cumsum_data[t]
            window_valid = cumsum_valid[t]
        else:
            window_sum = cumsum_data[t] - cumsum_data[t - scale]
            window_valid = cumsum_valid[t] - cumsum_valid[t - scale]

        # Only valid if all values in window are valid
        all_valid = window_valid == scale
        result[t] = np.where(all_valid, window_sum, np.nan)

    # Clean up intermediate arrays
    del data_filled, valid_mask, cumsum_data, cumsum_valid

    return result


def _compute_gamma_params_vectorized(
    scaled_data: np.ndarray,
    n_years: int,
    periods_per_year: int,
    n_lat: int,
    n_lon: int,
    cal_start_idx: int,
    cal_end_idx: int,
    dtype: np.dtype
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gamma parameters with vectorized operations.

    Memory-optimized to avoid creating many large intermediate arrays.
    """
    # Initialize parameter arrays: shape (periods, lat, lon)
    alphas = np.full((periods_per_year, n_lat, n_lon), np.nan, dtype=dtype)
    betas = np.full((periods_per_year, n_lat, n_lon), np.nan, dtype=dtype)
    probs_zero = np.full((periods_per_year, n_lat, n_lon), np.nan, dtype=dtype)

    # Reshape to (years, periods, lat, lon) - this is a view, no copy
    scaled_4d = scaled_data.reshape(n_years, periods_per_year, n_lat, n_lon)

    # Process each calendar period
    for period_idx in range(periods_per_year):
        # Get calibration data for this period: shape (cal_years, lat, lon)
        calib_data = scaled_4d[cal_start_idx:cal_end_idx, period_idx, :, :]

        # Count valid and zero values
        valid_mask = ~np.isnan(calib_data)
        n_valid = np.sum(valid_mask, axis=0)
        n_zeros = np.sum((calib_data == 0) & valid_mask, axis=0)

        # Probability of zero
        with np.errstate(divide='ignore', invalid='ignore'):
            probs_zero[period_idx] = np.where(n_valid > 0, n_zeros / n_valid, np.nan)

        # Non-zero values for gamma fitting
        nonzero_mask = (calib_data > 0) & valid_mask
        n_nonzero = np.sum(nonzero_mask, axis=0)

        # Replace non-positive with NaN for stats
        calib_positive = np.where(nonzero_mask, calib_data, np.nan)

        # Method of moments estimation
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_vals = np.nanmean(calib_positive, axis=0)
            log_vals = np.log(calib_positive)
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

            alphas[period_idx] = np.where(valid_fit, alpha, np.nan)
            betas[period_idx] = np.where(valid_fit, beta, np.nan)

        # Clean up period-specific arrays
        del calib_data, valid_mask, nonzero_mask, calib_positive, log_vals

    return alphas, betas, probs_zero


def _transform_to_normal_vectorized(
    scaled_data: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    probs_zero: np.ndarray,
    n_years: int,
    periods_per_year: int,
    n_lat: int,
    n_lon: int,
    dtype: np.dtype
) -> np.ndarray:
    """
    Transform scaled data to standard normal using gamma CDF.

    Memory-optimized to process one period at a time.
    """
    n_time = n_years * periods_per_year
    result = np.full((n_time, n_lat, n_lon), np.nan, dtype=dtype)

    # Reshape for period-wise access (view, no copy)
    scaled_4d = scaled_data.reshape(n_years, periods_per_year, n_lat, n_lon)
    result_4d = result.reshape(n_years, periods_per_year, n_lat, n_lon)

    # Process each calendar period
    for period_idx in range(periods_per_year):
        alpha = alphas[period_idx]
        beta = betas[period_idx]
        prob_zero = probs_zero[period_idx]

        # Values for this period: shape (n_years, lat, lon)
        period_vals = scaled_4d[:, period_idx, :, :]

        # Create mask for valid parameters
        valid_params = (~np.isnan(alpha) & ~np.isnan(beta) &
                       (alpha > 0) & (beta > 0))

        # Skip if no valid parameters
        if not np.any(valid_params):
            continue

        # Compute gamma CDF (vectorized)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_probs = scipy.stats.gamma.cdf(
                period_vals,
                a=alpha[np.newaxis, :, :],
                scale=beta[np.newaxis, :, :]
            )

            # Adjust for probability of zero
            adjusted_probs = (prob_zero[np.newaxis, :, :] +
                            (1.0 - prob_zero[np.newaxis, :, :]) * gamma_probs)

            # Clamp to valid range
            np.clip(adjusted_probs, 1e-10, 1.0 - 1e-10, out=adjusted_probs)

            # Transform to standard normal
            transformed = scipy.stats.norm.ppf(adjusted_probs)

            # Expand valid_params mask
            valid_expanded = np.broadcast_to(valid_params[np.newaxis, :, :], period_vals.shape)

            # Apply only where parameters and values are valid
            result_4d[:, period_idx, :, :] = np.where(
                valid_expanded & ~np.isnan(period_vals),
                transformed,
                np.nan
            )

        # Clean up period-specific arrays
        del gamma_probs, adjusted_probs, transformed

    return result


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
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
    chunks: Optional[Dict[str, int]] = None
) -> Tuple[xr.DataArray, Dict[str, xr.DataArray]]:
    """
    Compute SPI/SPEI using Dask for out-of-core processing.

    Suitable for very large global datasets that don't fit in memory.
    Uses lazy evaluation and chunked processing.

    Note: For fitting parameters extraction, use ChunkedProcessor from
    the chunked module instead, which properly handles parameter collection.

    :param data: xarray DataArray with dimensions (time, lat, lon)
    :param scale: accumulation scale
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters
    :param chunks: optional chunk sizes, e.g., {'lat': 500, 'lon': 500}
    :return: tuple of (result DataArray, empty params dict)
    """
    import dask.array as da

    _logger.info(f"Starting Dask-enabled computation for shape {data.shape}")

    # Ensure data is chunked - keep full time dimension, chunk spatially
    if chunks is None:
        chunks = {'time': -1, 'lat': 500, 'lon': 500}

    if not data.chunks:
        data = data.chunk(chunks)
        _logger.info(f"Chunked data with {chunks}")

    # Wrapper function for map_blocks
    def _process_chunk(chunk: np.ndarray) -> np.ndarray:
        """Process a single chunk."""
        if chunk.size == 0 or np.all(np.isnan(chunk)):
            return np.full(chunk.shape, np.nan, dtype=np.float64)

        result, _ = compute_index_parallel(
            chunk,
            scale=scale,
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            periodicity=periodicity,
            fitting_params=fitting_params,
            memory_efficient=True
        )
        return result

    # Apply using dask map_blocks
    _logger.info("Building Dask computation graph...")

    result_dask = da.map_blocks(
        _process_chunk,
        data.data,
        dtype=np.float64,
        meta=np.array((), dtype=np.float64)
    )

    # Wrap back in xarray
    result = xr.DataArray(
        result_dask,
        dims=data.dims,
        coords=data.coords,
        attrs={
            'long_name': f'Standardized Precipitation Index (scale={scale})',
            'units': '1',
            'scale': scale,
            'calibration_start_year': calibration_start_year,
            'calibration_end_year': calibration_end_year,
        }
    )

    _logger.info(
        "Computation graph built. Use .compute() to execute, "
        "or save directly with to_netcdf() for streaming output."
    )

    # Note: Fitting parameters not collected in Dask mode
    # Use chunked.ChunkedProcessor for param extraction
    return result, {}


def compute_index_dask_to_zarr(
    data: xr.DataArray,
    output_path: str,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
    chunks: Optional[Dict[str, int]] = None,
    n_workers: int = None
) -> str:
    """
    Compute SPI/SPEI using Dask and stream output to Zarr format.

    This is the most memory-efficient method for very large datasets
    as it streams results directly to disk without loading all data.

    :param data: xarray DataArray with dimensions (time, lat, lon)
    :param output_path: Path for output Zarr store
    :param scale: accumulation scale
    :param data_start_year: first year of data
    :param calibration_start_year: calibration start year
    :param calibration_end_year: calibration end year
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters
    :param chunks: chunk sizes for processing
    :param n_workers: number of Dask workers
    :return: path to output Zarr store

    Example:
        >>> result_path = compute_index_dask_to_zarr(
        ...     precip_da,
        ...     'output/spi_global.zarr',
        ...     scale=12,
        ...     data_start_year=1981,
        ...     calibration_start_year=1991,
        ...     calibration_end_year=2020,
        ...     periodicity=Periodicity.monthly
        ... )
    """
    import dask
    from dask.distributed import Client, LocalCluster

    # Set up Dask client
    if n_workers is None:
        import os
        n_workers = max(1, os.cpu_count() - 1)

    _logger.info(f"Setting up Dask cluster with {n_workers} workers")

    # Use context manager for cluster
    with LocalCluster(n_workers=n_workers, threads_per_worker=1) as cluster:
        with Client(cluster) as client:
            _logger.info(f"Dask dashboard: {client.dashboard_link}")

            # Compute lazy result
            result, _ = compute_index_dask(
                data, scale, data_start_year,
                calibration_start_year, calibration_end_year,
                periodicity, fitting_params, chunks
            )

            # Convert to dataset for saving
            from config import get_variable_name
            var_name = get_variable_name('spi', scale, periodicity)
            result = result.rename(var_name)
            ds = result.to_dataset()

            # Stream to Zarr (memory-efficient)
            _logger.info(f"Streaming output to Zarr: {output_path}")
            ds.to_zarr(output_path, mode='w', consolidated=True)

            _logger.info(f"Computation complete: {output_path}")

    return output_path


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
