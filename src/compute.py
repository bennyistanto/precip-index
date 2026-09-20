"""
Core computation functions for SPI/SPEI calculation.

Includes gamma distribution fitting, scaling, and transformation functions.
Optimized for global-scale data with memory-efficient chunked processing,
Dask integration, and Numba JIT compilation for performance.

---
Author: Benny Istanto, GOST/DEC Data Group/The World Bank

Built upon the foundation of climate-indices by James Adams, 
with substantial modifications for multi-distribution support, 
bidirectional event analysis, and scalable processing.
---
"""

import gc
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.stats
import xarray as xr
from numba import jit

from config import (
    DEFAULT_DISTRIBUTION,
    DISTRIBUTION_DISPLAY_NAMES,
    DISTRIBUTION_PARAM_NAMES,
    FITTED_INDEX_VALID_MAX,
    FITTED_INDEX_VALID_MIN,
    MIN_VALUES_FOR_GAMMA_FIT,
    Periodicity,
    SPEI_WATER_BALANCE_OFFSET,
)
from utils import get_logger, get_variable_name, validate_array

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



def compute_index_parallel(
    data: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity: Periodicity,
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
    memory_efficient: bool = True,
    distribution: str = DEFAULT_DISTRIBUTION
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute SPI/SPEI for 3D gridded data with parallel processing.

    Optimized for large global datasets using vectorized numpy operations
    and scipy for gamma transformation. Memory-efficient mode reduces
    peak memory usage by processing in-place where possible.

    For Gamma distribution, uses the optimized Numba/NumPy fast path.
    For other distributions (Pearson III, Log-Logistic, GEV, etc.), uses
    the generic scipy-based path via distributions.py.

    :param data: 3-D array with shape (time, lat, lon) - CF Convention
    :param scale: accumulation scale (e.g., 1, 3, 6, 12)
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters dict with
        distribution-specific parameter arrays of shape (periods, lat, lon)
    :param memory_efficient: if True, optimize for lower memory usage
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :return: tuple of (result_array, fitting_params_dict)
    """
    n_time, n_lat, n_lon = data.shape
    periods_per_year = periodicity.value
    remainder = n_time % periods_per_year
    n_years = n_time // periods_per_year + (1 if remainder else 0)
    n_time_original = n_time  # remember original length before padding
    dist = distribution.lower()

    _logger.info(
        f"Computing index: shape={data.shape}, scale={scale}, "
        f"distribution={dist}, grid_cells={n_lat * n_lon:,}"
    )

    if remainder:
        _logger.info(
            f"Data has {n_time} timesteps (not divisible by {periods_per_year}). "
            f"Padding {periods_per_year - remainder} NaN timesteps to complete "
            f"final year ({n_years} years total)."
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

    # Pad to complete years if needed (after scaling, so rolling sum uses real data)
    if remainder:
        pad_count = periods_per_year - remainder
        pad_shape = (pad_count, n_lat, n_lon)
        pad_array = np.full(pad_shape, np.nan, dtype=dtype)
        scaled_data = np.concatenate([scaled_data, pad_array], axis=0)
        del pad_array

    # Calculate calibration indices (needed by both paths)
    data_end_year = data_start_year + n_years - 1
    cal_start = max(calibration_start_year, data_start_year)
    cal_end = min(calibration_end_year, data_end_year)
    cal_start_idx = cal_start - data_start_year
    cal_end_idx = cal_end - data_start_year + 1

    # Route based on distribution type
    if dist == 'gamma':
        # === GAMMA FAST PATH (existing optimized code) ===
        _logger.info("Step 2/3: Computing gamma parameters...")

        if fitting_params is not None:
            alphas = fitting_params['alpha'].astype(dtype, copy=False)
            betas = fitting_params['beta'].astype(dtype, copy=False)
            probs_zero = fitting_params['prob_zero'].astype(dtype, copy=False)
            _logger.info("Using pre-computed fitting parameters")
        else:
            alphas, betas, probs_zero = _compute_gamma_params_vectorized(
                scaled_data, n_years, periods_per_year, n_lat, n_lon,
                cal_start_idx, cal_end_idx, dtype
            )

        _logger.info("Step 3/3: Transforming to standard normal...")
        result = _transform_to_normal_vectorized(
            scaled_data, alphas, betas, probs_zero,
            n_years, periods_per_year, n_lat, n_lon, dtype
        )

        # Prepare fitting parameters dict
        params_dict = {
            'alpha': alphas.astype(np.float64) if memory_efficient else alphas,
            'beta': betas.astype(np.float64) if memory_efficient else betas,
            'prob_zero': probs_zero.astype(np.float64) if memory_efficient else probs_zero,
            'distribution': dist
        }
    else:
        # === GENERIC PATH (distributions.py for non-Gamma) ===
        _logger.info(f"Step 2/3: Computing {dist} parameters...")

        if fitting_params is not None:
            params_dict = {k: v.astype(dtype, copy=False) if isinstance(v, np.ndarray) else v
                          for k, v in fitting_params.items()}
            _logger.info("Using pre-computed fitting parameters")
        else:
            params_dict = _compute_params_generic(
                scaled_data, dist, n_years, periods_per_year,
                n_lat, n_lon, cal_start_idx, cal_end_idx, dtype
            )

        _logger.info("Step 3/3: Transforming to standard normal...")
        result = _transform_to_normal_generic(
            scaled_data, params_dict, dist,
            n_years, periods_per_year, n_lat, n_lon, dtype
        )

        # Convert params to float64 for storage
        if memory_efficient:
            params_dict = {
                k: v.astype(np.float64) if isinstance(v, np.ndarray) else v
                for k, v in params_dict.items()
            }
        params_dict['distribution'] = dist

    # Free scaled_data (can be as large as input array)
    del scaled_data

    # Trim result back to original time length if we padded
    if remainder:
        result = result[:n_time_original]

    # Clip to valid range
    np.clip(result, FITTED_INDEX_VALID_MIN, FITTED_INDEX_VALID_MAX, out=result)

    # Convert result to float64 for output consistency
    if memory_efficient:
        result = result.astype(np.float64)

    _logger.info("Index computation complete")

    # Explicit garbage collection to free intermediate arrays
    gc.collect()

    return result, params_dict


def _rolling_sum_3d(data: np.ndarray, scale: int, dtype: np.dtype) -> np.ndarray:
    """
    Memory-efficient rolling sum for 3D array.

    Uses a cumulative sum for O(n) complexity instead of O(n*scale). The window
    sum is recovered as ``cumsum[t] - cumsum[t-scale]``, which is a subtraction
    of two large, nearly equal numbers.

    The accumulator is float64 regardless of the storage dtype, because in
    float32 that subtraction loses badly. SPEI adds a +1000 offset to the water
    balance, so over a 76-year monthly record the cumulative sum reaches ~8e5,
    where the float32 spacing is 0.0625. Each 12-month window sum then carried
    roughly 0.06 of absolute error, and - worse - the error depended on how much
    data preceded the window, so the same calendar month produced different SPEI
    depending on whether the record started in 1950 or 1958. Measured on a 30x30
    land window: float32 gave mean|diff| 9.7e-05 and max 1.3e-02 between the two
    record lengths; float64 gives exactly 0.

    Cost is one extra float64 copy of the input in place of a float32 one, about
    +1x the tile size at peak. The result is still stored in ``dtype``.

    The valid-count accumulator is int32 rather than int16: int16 caps at 32,767,
    which a daily record longer than about 89 years (366 * 90) would overflow,
    silently wrapping negative and voiding every window.

    :param data: 3-D array (time, lat, lon)
    :param scale: number of time steps to accumulate
    :param dtype: storage dtype for the returned array
    :return: rolling sums, NaN where the window is not fully valid
    """
    n_time, n_lat, n_lon = data.shape

    # Initialize output with NaN
    result = np.full((n_time, n_lat, n_lon), np.nan, dtype=dtype)

    # Use cumulative sum for efficiency (handles NaN correctly)
    # Replace NaN with 0 for cumsum, track valid counts
    data_filled = np.where(np.isnan(data), 0, data)
    valid_mask = (~np.isnan(data)).astype(np.int8)

    # Cumulative sums. float64 accumulator: see the note above.
    cumsum_data = np.cumsum(data_filled, axis=0, dtype=np.float64)
    cumsum_valid = np.cumsum(valid_mask, axis=0, dtype=np.int32)

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
# GENERIC DISTRIBUTION FITTING AND TRANSFORMATION
# =============================================================================

def _compute_params_pearson3_vectorized(
    scaled_data: np.ndarray,
    n_years: int,
    periods_per_year: int,
    n_lat: int,
    n_lon: int,
    cal_start_idx: int,
    cal_end_idx: int,
    dtype: np.dtype
) -> Dict[str, np.ndarray]:
    """
    Vectorized Pearson III parameter fitting (method of moments).

    Numerically equivalent to calling fit_pearson3(values, FittingMethod.MOMENTS)
    on every grid cell, but reduces along the year axis with array operations
    instead of one scalar scipy call per cell per period. For a 1440x1440 chunk
    with monthly data that replaces ~24.9 million Python-level calls with 12
    array reductions.

    Replicates the decision order in distributions.fit_pearson3:
      1. n_total < MIN_VALUES_FOR_FIT        -> NaN params, prob_zero = 0
      2. zero_proportion > MAX_ZERO_PROPORTION -> NaN params, prob_zero = zero_prop
      3. n_nonzero < MIN_NONZERO_VALUES      -> NaN params, prob_zero = zero_prop
      4. near-constant data                  -> skew 0, loc mean, scale sqrt(var)
      5. otherwise                           -> moments fit, prob_zero = P(x <= 0)

    :param scaled_data: 3-D array (time, lat, lon) of scaled values
    :param n_years: number of years in data
    :param periods_per_year: 12 for monthly, 366 for daily
    :param n_lat: number of latitude points
    :param n_lon: number of longitude points
    :param cal_start_idx: calibration start index (year)
    :param cal_end_idx: calibration end index (year, exclusive)
    :param dtype: numpy data type for output arrays
    :return: dictionary with 'skew', 'loc', 'scale', 'prob_zero' arrays
    """
    from distributions import (
        EPSILON,
        MAX_SCALE_PARAM,
        MAX_ZERO_PROPORTION,
        MIN_NONZERO_VALUES,
        MIN_SCALE_PARAM,
        MIN_VALUES_FOR_FIT,
        MIN_VARIANCE,
    )

    shape_out = (periods_per_year, n_lat, n_lon)
    skew_out = np.full(shape_out, np.nan, dtype=dtype)
    loc_out = np.full(shape_out, np.nan, dtype=dtype)
    scale_out = np.full(shape_out, np.nan, dtype=dtype)
    pzero_out = np.full(shape_out, np.nan, dtype=dtype)

    scaled_4d = scaled_data.reshape(n_years, periods_per_year, n_lat, n_lon)

    for period_idx in range(periods_per_year):
        # Calibration values for this calendar period: (cal_years, lat, lon)
        v = scaled_4d[cal_start_idx:cal_end_idx, period_idx, :, :].astype(np.float64)

        finite = np.isfinite(v)
        n_total = finite.sum(axis=0)
        vv = np.where(finite, v, np.nan)

        # Guard against divide-by-zero on empty cells; masked out later anyway
        n_safe = np.maximum(n_total, 1)

        n_zeros = (np.abs(vv) < EPSILON).sum(axis=0)
        zero_prop = n_zeros / n_safe
        n_nonzero = n_total - n_zeros

        mean = np.nanmean(vv, axis=0)
        var = np.nanvar(vv, axis=0, ddof=1)
        var = np.where(np.isfinite(var), var, 0.0)
        std = np.sqrt(np.maximum(var, 0.0))
        data_range = np.nanmax(vv, axis=0) - np.nanmin(vv, axis=0)

        # prob_zero for the mixed distribution is P(x <= 0), not P(x == 0)
        n_non_positive = (vv <= 0).sum(axis=0)
        prob_zero = n_non_positive / n_safe

        # --- method of moments (_fit_pearson3_moments_robust, vectorized) ---
        centered = vv - mean
        m3 = np.nanmean(centered ** 3, axis=0)
        del centered

        std_cubed = std ** 3
        skew = np.where(std_cubed < EPSILON, 0.0, m3 / np.where(std_cubed < EPSILON, 1.0, std_cubed))

        # Fisher bias correction for n > 3
        bias = np.where(
            n_total > 3,
            np.sqrt(n_safe * (n_safe - 1)) / np.maximum(n_safe - 2, 1),
            1.0
        )
        skew = skew * bias
        skew = np.clip(skew, -10.0, 10.0)
        skew = np.where(np.abs(skew) < EPSILON, 0.0, skew)

        loc = mean
        scale = std

        # Inner near-constant guard inside _fit_pearson3_moments_robust
        inner_const = (std <= EPSILON) | (var < MIN_VARIANCE)
        skew = np.where(inner_const, 0.0, skew)
        scale = np.where(inner_const, np.maximum(std, EPSILON), scale)

        # Non-finite fit -> normal approximation fallback
        bad_fit = ~(np.isfinite(skew) & np.isfinite(loc) & np.isfinite(scale))
        skew = np.where(bad_fit, 0.0, skew)
        loc = np.where(bad_fit, mean, loc)
        scale = np.where(bad_fit, std, scale)

        # Outer near-constant branch in fit_pearson3
        outer_const = (var < MIN_VARIANCE) | (data_range < EPSILON)
        skew = np.where(outer_const, 0.0, skew)
        loc = np.where(outer_const, mean, loc)
        scale = np.where(outer_const, np.maximum(std, EPSILON), scale)
        pz = np.where(outer_const, zero_prop, prob_zero)

        # Final bounds
        skew = np.clip(skew, -10.0, 10.0)
        scale = np.clip(np.abs(scale), MIN_SCALE_PARAM, MAX_SCALE_PARAM)

        # --- invalid cases, applied in fit_pearson3's return order ---
        too_few = n_total < MIN_VALUES_FOR_FIT
        too_many_zeros = (~too_few) & (zero_prop > MAX_ZERO_PROPORTION)
        too_few_nonzero = (~too_few) & (~too_many_zeros) & (n_nonzero < MIN_NONZERO_VALUES)
        invalid = too_few | too_many_zeros | too_few_nonzero

        skew = np.where(invalid, np.nan, skew)
        loc = np.where(invalid, np.nan, loc)
        scale = np.where(invalid, np.nan, scale)
        pz = np.where(too_few, 0.0, np.where(too_many_zeros | too_few_nonzero, zero_prop, pz))

        # Cells with no valid data at all are skipped entirely by the scalar
        # path, leaving every parameter NaN - including prob_zero.
        empty = n_total == 0
        skew = np.where(empty, np.nan, skew)
        loc = np.where(empty, np.nan, loc)
        scale = np.where(empty, np.nan, scale)
        pz = np.where(empty, np.nan, pz)

        skew_out[period_idx] = skew
        loc_out[period_idx] = loc
        scale_out[period_idx] = scale
        pzero_out[period_idx] = pz

        del v, vv, finite, mean, var, std, data_range, m3, skew, loc, scale, pz

    return {
        'skew': skew_out,
        'loc': loc_out,
        'scale': scale_out,
        'prob_zero': pzero_out,
    }


def _transform_pearson3_vectorized(
    scaled_data: np.ndarray,
    params_dict: Dict[str, np.ndarray],
    n_years: int,
    periods_per_year: int,
    n_lat: int,
    n_lon: int,
    dtype: np.dtype
) -> np.ndarray:
    """
    Vectorized Pearson III CDF transform to standard normal.

    Equivalent to looping distributions.pearson3_cdf + cdf_to_standard_normal
    over every grid cell. scipy broadcasts the per-cell skew/loc/scale arrays
    against the (years, lat, lon) value block, so the whole period is one call.

    Matches pearson3_cdf behaviour: cells with |skew| < 0.01 use the normal
    approximation, and any non-finite Pearson III CDF falls back to normal.

    Note: as in the scalar path, prob_zero is not applied in the Pearson III
    transform (unlike the gamma path, which does adjust for it).

    :param scaled_data: 3-D array (time, lat, lon) of scaled values
    :param params_dict: parameter arrays from _compute_params_pearson3_vectorized
    :param n_years: number of years
    :param periods_per_year: 12 for monthly, 366 for daily
    :param n_lat: number of latitude points
    :param n_lon: number of longitude points
    :param dtype: numpy data type
    :return: transformed array (time, lat, lon)
    """
    n_time = n_years * periods_per_year
    result = np.full((n_time, n_lat, n_lon), np.nan, dtype=dtype)

    scaled_4d = scaled_data.reshape(n_years, periods_per_year, n_lat, n_lon)
    result_4d = result.reshape(n_years, periods_per_year, n_lat, n_lon)

    skew_all = params_dict['skew']
    loc_all = params_dict['loc']
    scale_all = params_dict['scale']

    for period_idx in range(periods_per_year):
        skew = skew_all[period_idx].astype(np.float64)
        loc = loc_all[period_idx].astype(np.float64)
        scale = scale_all[period_idx].astype(np.float64)

        valid_params = (np.isfinite(skew) & np.isfinite(loc) &
                        np.isfinite(scale) & (scale > 0))
        if not np.any(valid_params):
            continue

        vals = scaled_4d[:, period_idx, :, :].astype(np.float64)

        # Substitute safe values where params are invalid so scipy never sees
        # NaN or a non-positive scale; those cells are masked out at the end.
        skew_b = np.where(valid_params, skew, 0.0)[np.newaxis, :, :]
        loc_b = np.where(valid_params, loc, 0.0)[np.newaxis, :, :]
        scale_b = np.where(valid_params, scale, 1.0)[np.newaxis, :, :]

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            norm_cdf = scipy.stats.norm.cdf(vals, loc=loc_b, scale=scale_b)

            near_sym = np.broadcast_to(np.abs(skew_b) < 0.01, vals.shape)
            if np.all(near_sym):
                cdf = norm_cdf
            else:
                p3_cdf = scipy.stats.pearson3.cdf(vals, skew_b, loc=loc_b, scale=scale_b)
                cdf = np.where(near_sym, norm_cdf, p3_cdf)
                del p3_cdf
                # Fall back to normal where Pearson III produced nothing usable
                bad = ~np.isfinite(cdf)
                if np.any(bad):
                    cdf = np.where(bad, norm_cdf, cdf)
                del bad

            del norm_cdf, near_sym

            np.clip(cdf, 1e-10, 1.0 - 1e-10, out=cdf)
            transformed = scipy.stats.norm.ppf(cdf)
            del cdf

            np.clip(transformed, FITTED_INDEX_VALID_MIN, FITTED_INDEX_VALID_MAX,
                    out=transformed)

            keep = (np.broadcast_to(valid_params[np.newaxis, :, :], vals.shape) &
                    np.isfinite(vals))
            result_4d[:, period_idx, :, :] = np.where(keep, transformed, np.nan).astype(dtype)

        del vals, transformed, keep

    return result


def _compute_params_generic(
    scaled_data: np.ndarray,
    distribution: str,
    n_years: int,
    periods_per_year: int,
    n_lat: int,
    n_lon: int,
    cal_start_idx: int,
    cal_end_idx: int,
    dtype: np.dtype
) -> Dict[str, np.ndarray]:
    """
    Compute distribution parameters using the generic distributions.py module.

    Works for any distribution type (Pearson III, Log-Logistic, GEV, etc.).
    Processes each calendar period and grid cell using the unified
    fit_distribution() interface.

    :param scaled_data: 3-D array (time, lat, lon) of scaled values
    :param distribution: distribution name (e.g., 'pearson3', 'gev')
    :param n_years: number of years in data
    :param periods_per_year: 12 for monthly, 366 for daily
    :param n_lat: number of latitude points
    :param n_lon: number of longitude points
    :param cal_start_idx: calibration start index (year)
    :param cal_end_idx: calibration end index (year, exclusive)
    :param dtype: numpy data type for arrays
    :return: dictionary of parameter arrays keyed by parameter name
    """
    from distributions import fit_distribution

    # Fast path: Pearson III with method of moments is fully vectorizable.
    # Same numbers as the per-cell loop below, orders of magnitude faster.
    if distribution == 'pearson3':
        _logger.info("  Using vectorized Pearson III fitting")
        return _compute_params_pearson3_vectorized(
            scaled_data, n_years, periods_per_year,
            n_lat, n_lon, cal_start_idx, cal_end_idx, dtype
        )

    # Get parameter names for this distribution
    param_names = DISTRIBUTION_PARAM_NAMES.get(distribution, ('prob_zero',))

    # Initialize parameter arrays: shape (periods, lat, lon)
    params_dict = {}
    for pname in param_names:
        params_dict[pname] = np.full((periods_per_year, n_lat, n_lon), np.nan, dtype=dtype)

    # Reshape to (years, periods, lat, lon)
    scaled_4d = scaled_data.reshape(n_years, periods_per_year, n_lat, n_lon)

    # Process each calendar period
    for period_idx in range(periods_per_year):
        if period_idx % 3 == 0:
            _logger.debug(f"  Fitting period {period_idx + 1}/{periods_per_year}")

        # Calibration data for this period: shape (cal_years, lat, lon)
        calib_data = scaled_4d[cal_start_idx:cal_end_idx, period_idx, :, :]

        # Process each grid cell
        for lat_idx in range(n_lat):
            for lon_idx in range(n_lon):
                cell_values = calib_data[:, lat_idx, lon_idx].astype(np.float64)

                # Skip all-NaN cells
                if np.all(np.isnan(cell_values)):
                    continue

                # Fit distribution using unified interface
                dist_params = fit_distribution(cell_values, distribution)

                # Extract parameters into arrays
                for pname in param_names:
                    if pname == 'prob_zero':
                        params_dict[pname][period_idx, lat_idx, lon_idx] = dist_params.prob_zero
                    elif pname in dist_params.params:
                        val = dist_params.params[pname]
                        if val is not None and not np.isnan(val):
                            params_dict[pname][period_idx, lat_idx, lon_idx] = val

    return params_dict


def _transform_to_normal_generic(
    scaled_data: np.ndarray,
    params_dict: Dict[str, np.ndarray],
    distribution: str,
    n_years: int,
    periods_per_year: int,
    n_lat: int,
    n_lon: int,
    dtype: np.dtype
) -> np.ndarray:
    """
    Transform scaled data to standard normal using a generic distribution CDF.

    Uses distributions.py for CDF computation and inverse normal transformation.
    Processes one period at a time for memory efficiency.

    :param scaled_data: 3-D array (time, lat, lon) of scaled values
    :param params_dict: dictionary of parameter arrays from _compute_params_generic()
    :param distribution: distribution name
    :param n_years: number of years
    :param periods_per_year: 12 for monthly, 366 for daily
    :param n_lat: number of latitude points
    :param n_lon: number of longitude points
    :param dtype: numpy data type
    :return: transformed array (same shape as scaled_data)
    """
    from distributions import (
        DistributionParams, DistributionType, FittingMethod,
        compute_cdf, cdf_to_standard_normal
    )

    # Fast path: vectorized Pearson III transform (see fitting counterpart)
    if distribution == 'pearson3':
        _logger.info("  Using vectorized Pearson III transform")
        return _transform_pearson3_vectorized(
            scaled_data, params_dict, n_years, periods_per_year,
            n_lat, n_lon, dtype
        )

    n_time = n_years * periods_per_year
    result = np.full((n_time, n_lat, n_lon), np.nan, dtype=dtype)

    # Reshape for period-wise access
    scaled_4d = scaled_data.reshape(n_years, periods_per_year, n_lat, n_lon)
    result_4d = result.reshape(n_years, periods_per_year, n_lat, n_lon)

    # Parse distribution type
    dist_type = DistributionType(distribution)
    param_names = DISTRIBUTION_PARAM_NAMES.get(distribution, ('prob_zero',))

    # Process each calendar period
    for period_idx in range(periods_per_year):
        # Values for this period: shape (n_years, lat, lon)
        period_vals = scaled_4d[:, period_idx, :, :]

        # Process each grid cell
        for lat_idx in range(n_lat):
            for lon_idx in range(n_lon):
                # Build DistributionParams for this cell
                cell_params = {}
                prob_zero = 0.0
                has_valid = True

                for pname in param_names:
                    val = float(params_dict[pname][period_idx, lat_idx, lon_idx])
                    if pname == 'prob_zero':
                        prob_zero = val if not np.isnan(val) else 0.0
                    else:
                        if np.isnan(val):
                            has_valid = False
                            break
                        cell_params[pname] = val

                if not has_valid:
                    continue

                dp = DistributionParams(
                    distribution=dist_type,
                    params=cell_params,
                    prob_zero=prob_zero,
                    n_samples=0,
                    fitting_method=FittingMethod.LMOMENTS
                )

                # Get cell time series
                cell_values = period_vals[:, lat_idx, lon_idx].astype(np.float64)

                # Compute CDF
                cdf_vals = compute_cdf(cell_values, dp)

                # Transform to standard normal
                normal_vals = cdf_to_standard_normal(cdf_vals)

                result_4d[:, period_idx, lat_idx, lon_idx] = normal_vals.astype(dtype)

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
    chunks: Optional[Dict[str, int]] = None,
    distribution: str = DEFAULT_DISTRIBUTION
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
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :return: tuple of (result DataArray, empty params dict)
    """
    import dask.array as da

    dist = distribution.lower()
    _logger.info(
        f"Starting Dask-enabled computation for shape {data.shape}, "
        f"distribution={dist}"
    )

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
            memory_efficient=True,
            distribution=dist
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
            'long_name': f'Standardized Index ({DISTRIBUTION_DISPLAY_NAMES.get(dist, dist)}, scale={scale})',
            'units': '1',
            'scale': scale,
            'distribution': dist,
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
    n_workers: int = None,
    distribution: str = DEFAULT_DISTRIBUTION
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
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
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

    dist = distribution.lower()

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
                periodicity, fitting_params, chunks,
                distribution=dist
            )

            # Convert to dataset for saving
            var_name = get_variable_name('spi', scale, periodicity, distribution=dist)
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
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
    distribution: str = DEFAULT_DISTRIBUTION
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute SPI for a single time series (1-D array).

    Convenience function for single-point calculations.
    For Gamma distribution, uses the optimized fast path.
    For other distributions, uses the generic distributions.py module.

    :param precip: 1-D array of precipitation values
    :param scale: accumulation scale
    :param data_start_year: first year of the data
    :param calibration_start_year: first year of calibration period
    :param calibration_end_year: last year of calibration period
    :param periodicity: monthly or daily
    :param fitting_params: optional pre-computed parameters
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :return: tuple of (SPI values, fitting_params dict)
    """
    dist = distribution.lower()

    # Validate input
    precip = np.asarray(precip).flatten()

    # Apply scaling
    scaled = sum_to_scale(precip, scale)

    # Reshape to 2D
    periods = periodicity.value
    scaled_2d = validate_array(scaled, periodicity)

    if dist == 'gamma':
        # === GAMMA FAST PATH ===
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
            'prob_zero': probs_zero,
            'distribution': dist
        }
    else:
        # === GENERIC PATH (distributions.py) ===
        # Reshape to 3D (time, 1, 1) for compute_index_parallel compatibility
        n_time = scaled_2d.shape[0] * scaled_2d.shape[1]
        scaled_3d = scaled_2d.reshape(n_time, 1, 1)

        result_3d, params = compute_index_parallel(
            scaled_3d,
            scale=1,  # Already scaled
            data_start_year=data_start_year,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            periodicity=periodicity,
            fitting_params=fitting_params,
            memory_efficient=False,
            distribution=dist
        )

        # Flatten result
        result = result_3d.flatten()[:len(precip)]

        # Squeeze spatial dims from params
        params = {
            k: v.squeeze() if isinstance(v, np.ndarray) else v
            for k, v in params.items()
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
    fitting_params: Optional[Dict[str, np.ndarray]] = None,
    distribution: str = DEFAULT_DISTRIBUTION
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
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
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
    water_balance = (precip - pet) + SPEI_WATER_BALANCE_OFFSET

    # Use same logic as SPI
    return compute_spi_1d(
        water_balance,
        scale,
        data_start_year,
        calibration_start_year,
        calibration_end_year,
        periodicity,
        fitting_params,
        distribution=distribution
    )
