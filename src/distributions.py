"""
Probability distribution fitting for SPI and SPEI calculation.

This module provides implementations of various probability distributions
for standardizing precipitation and water balance data:

1. Gamma - Standard for SPI (McKee et al. 1993)
2. Pearson Type III - Recommended for SPEI (Vicente-Serrano et al. 2010)
3. Log-Logistic - Alternative for SPEI with better tail behavior
4. Generalized Extreme Value (GEV) - For extreme value analysis
5. Generalized Logistic - Used in some European drought indices

Each distribution supports:
- Parameter estimation via Method of Moments (fast) or MLE (accurate)
- L-moments estimation for improved robustness
- Mixed distribution handling (zero-inflated data)
- Goodness-of-fit testing

References:
    - McKee, T.B., Doesken, N.J., Kleist, J. (1993). SPI methodology.
    - Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010).
      A Multiscalar Drought Index Sensitive to Global Warming: SPEI.
    - Stagge, J.H., et al. (2015). Candidate Distributions for SPI and SPEI.
    - Hosking, J.R.M. (1990). L-moments: Analysis and estimation of
      distributions using linear combinations of order statistics.

---
Author: Benny Istanto, GOST/DEC Data Group/The World Bank

Built upon the foundation of climate-indices by James Adams, 
with substantial modifications for multi-distribution support, 
bidirectional event analysis, and scalable processing.
---
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func
from scipy.special import gammaln

# Suppress runtime warnings for edge cases
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Minimum number of valid values required for distribution fitting
MIN_VALUES_FOR_FIT = 30

# Minimum number of non-zero values for reliable fitting
MIN_NONZERO_VALUES = 10

# Maximum proportion of zeros allowed (beyond this, fitting is unreliable)
MAX_ZERO_PROPORTION = 0.95

# Valid range for fitted index values (±3.09 = 99.9th percentile of standard normal)
try:
    from config import FITTED_INDEX_VALID_MIN, FITTED_INDEX_VALID_MAX
except ImportError:
    FITTED_INDEX_VALID_MIN = -3.09
    FITTED_INDEX_VALID_MAX = 3.09

# Small value to avoid division by zero
EPSILON = 1e-10

# Minimum variance for reliable moment estimation
MIN_VARIANCE = 1e-8

# Bounds for parameter estimation (prevents numerical overflow)
MAX_SHAPE_PARAM = 1000.0
MIN_SCALE_PARAM = 1e-10
MAX_SCALE_PARAM = 1e10


class DistributionType(Enum):
    """Supported probability distributions."""
    GAMMA = "gamma"
    PEARSON3 = "pearson3"
    LOG_LOGISTIC = "log_logistic"
    GEV = "gev"
    GEN_LOGISTIC = "gen_logistic"


class FittingMethod(Enum):
    """Parameter estimation methods."""
    MOMENTS = "moments"      # Method of moments (fast)
    LMOMENTS = "lmoments"    # L-moments (robust)
    MLE = "mle"              # Maximum likelihood (accurate)


@dataclass
class DistributionParams:
    """Container for distribution parameters."""
    distribution: DistributionType
    params: Dict[str, float]
    prob_zero: float
    n_samples: int
    fitting_method: FittingMethod

    def is_valid(self) -> bool:
        """Check if parameters are valid for computation."""
        return all(not np.isnan(v) for v in self.params.values())

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'distribution': self.distribution.value,
            'params': self.params,
            'prob_zero': self.prob_zero,
            'n_samples': self.n_samples,
            'fitting_method': self.fitting_method.value
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'DistributionParams':
        """Create from dictionary."""
        return cls(
            distribution=DistributionType(d['distribution']),
            params=d['params'],
            prob_zero=d['prob_zero'],
            n_samples=d['n_samples'],
            fitting_method=FittingMethod(d['fitting_method'])
        )


class GoodnessOfFit(NamedTuple):
    """Goodness-of-fit test results."""
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float  # Anderson-Darling
    aic: float           # Akaike Information Criterion
    bic: float           # Bayesian Information Criterion


# =============================================================================
# L-MOMENTS COMPUTATION
# =============================================================================

def compute_lmoments(data: np.ndarray, nmom: int = 4) -> np.ndarray:
    """
    Compute L-moments from sample data.

    L-moments are more robust to outliers than conventional moments
    and provide better parameter estimates for small samples.

    :param data: 1-D array of sample values (should be sorted)
    :param nmom: number of L-moments to compute (default: 4)
    :return: array of L-moments [l1, l2, l3, l4]

    Reference: Hosking (1990)
    """
    n = len(data)
    if n < nmom:
        return np.full(nmom, np.nan)

    # Sort data
    x = np.sort(data)

    # Compute probability weighted moments (PWMs)
    b = np.zeros(nmom)
    for i in range(n):
        for r in range(nmom):
            b[r] += x[i] * _comb(i, r) / _comb(n - 1, r)
    b /= n

    # Convert PWMs to L-moments
    lmom = np.zeros(nmom)
    lmom[0] = b[0]  # L1 = mean

    if nmom >= 2:
        lmom[1] = 2 * b[1] - b[0]  # L2

    if nmom >= 3:
        lmom[2] = 6 * b[2] - 6 * b[1] + b[0]  # L3

    if nmom >= 4:
        lmom[3] = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]  # L4

    return lmom


def _comb(n: int, k: int) -> float:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0

    # Use logarithms for numerical stability
    return np.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def compute_lmoment_ratios(lmom: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute L-moment ratios (L-CV, L-skewness, L-kurtosis).

    :param lmom: array of L-moments [l1, l2, l3, l4]
    :return: (t2, t3, t4) = (L-CV, L-skewness, L-kurtosis)
    """
    if len(lmom) < 4 or lmom[1] <= 0:
        return np.nan, np.nan, np.nan

    t2 = lmom[1] / lmom[0] if lmom[0] != 0 else np.nan  # L-CV
    t3 = lmom[2] / lmom[1]  # L-skewness
    t4 = lmom[3] / lmom[1]  # L-kurtosis

    return t2, t3, t4


# =============================================================================
# GAMMA DISTRIBUTION
# =============================================================================

def fit_gamma(
    values: np.ndarray,
    method: FittingMethod = FittingMethod.MOMENTS
) -> DistributionParams:
    """
    Fit Gamma distribution to data.

    The Gamma distribution is the standard choice for SPI calculation
    as precipitation is bounded at zero and positively skewed.

    Includes robust handling for:
    - High proportion of zeros (arid/semi-arid regions)
    - Near-zero variance in positive values
    - Numerical issues in parameter estimation

    :param values: array of values (zeros will be handled separately)
    :param method: parameter estimation method
    :return: DistributionParams object
    """
    # Separate zeros and positive values
    valid_mask = ~np.isnan(values) & np.isfinite(values)
    valid_values = values[valid_mask]
    n_total = len(valid_values)

    if n_total == 0:
        return _invalid_params(DistributionType.GAMMA, method)

    # Calculate zero statistics
    n_zeros = np.sum(valid_values <= 0)  # Treat negative as zero for precip
    prob_zero = n_zeros / n_total

    # Work with positive values only
    positive_values = valid_values[valid_values > 0]
    n_positive = len(positive_values)

    # Check for excessive zeros
    if prob_zero > MAX_ZERO_PROPORTION:
        return DistributionParams(
            distribution=DistributionType.GAMMA,
            params={'alpha': np.nan, 'beta': np.nan},
            prob_zero=prob_zero,
            n_samples=n_total,
            fitting_method=method
        )

    if n_positive < MIN_NONZERO_VALUES:
        return _invalid_params(DistributionType.GAMMA, method, prob_zero, n_total)

    # Check for near-constant positive values
    if n_positive > 1:
        variance = np.var(positive_values, ddof=1)
        if variance < MIN_VARIANCE:
            # Near-constant - use approximation
            mean = np.mean(positive_values)
            alpha = mean ** 2 / max(variance, EPSILON)
            beta = max(variance, EPSILON) / mean if mean > 0 else EPSILON
            alpha = np.clip(alpha, EPSILON, MAX_SHAPE_PARAM)
            beta = np.clip(beta, MIN_SCALE_PARAM, MAX_SCALE_PARAM)
            return DistributionParams(
                distribution=DistributionType.GAMMA,
                params={'alpha': alpha, 'beta': beta},
                prob_zero=prob_zero,
                n_samples=n_total,
                fitting_method=method
            )

    # Fit parameters based on method with fallback chain
    alpha, beta = np.nan, np.nan

    if method == FittingMethod.MLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alpha, loc, beta = stats.gamma.fit(positive_values, floc=0)
        except Exception:
            # Fallback to L-moments
            alpha, beta = _fit_gamma_lmoments_robust(positive_values)
    elif method == FittingMethod.LMOMENTS:
        alpha, beta = _fit_gamma_lmoments_robust(positive_values)
    else:  # Method of moments
        alpha, beta = _fit_gamma_moments_robust(positive_values)

    # Final validation
    if np.isnan(alpha) or np.isnan(beta) or alpha <= 0 or beta <= 0:
        # Try alternative methods
        alpha, beta = _fit_gamma_moments_robust(positive_values)

    # Bound parameters
    if not np.isnan(alpha) and not np.isnan(beta):
        alpha = np.clip(alpha, EPSILON, MAX_SHAPE_PARAM)
        beta = np.clip(beta, MIN_SCALE_PARAM, MAX_SCALE_PARAM)

    return DistributionParams(
        distribution=DistributionType.GAMMA,
        params={'alpha': alpha, 'beta': beta},
        prob_zero=prob_zero,
        n_samples=n_total,
        fitting_method=method
    )


def _fit_gamma_moments(values: np.ndarray) -> Tuple[float, float]:
    """
    Fit Gamma parameters using method of moments.

    For Gamma(alpha, beta):
        mean = alpha * beta
        variance = alpha * beta^2

    Therefore:
        alpha = mean^2 / variance
        beta = variance / mean
    """
    return _fit_gamma_moments_robust(values)


def _fit_gamma_moments_robust(values: np.ndarray) -> Tuple[float, float]:
    """
    Robust Gamma fitting using method of moments.

    Handles edge cases:
    - Zero or negative mean
    - Zero or near-zero variance
    - Division by zero
    """
    n = len(values)
    if n < 2:
        return np.nan, np.nan

    mean = np.mean(values)
    variance = np.var(values, ddof=1)

    # Check for invalid statistics
    if mean <= EPSILON:
        return np.nan, np.nan

    if variance < MIN_VARIANCE:
        # Near-constant data - use high alpha (shape approaches normal)
        alpha = MAX_SHAPE_PARAM
        beta = mean / alpha
        return alpha, beta

    # Standard moment estimators
    alpha = mean ** 2 / variance
    beta = variance / mean

    # Bound parameters
    alpha = np.clip(alpha, EPSILON, MAX_SHAPE_PARAM)
    beta = np.clip(beta, MIN_SCALE_PARAM, MAX_SCALE_PARAM)

    return alpha, beta


def _fit_gamma_lmoments(values: np.ndarray) -> Tuple[float, float]:
    """
    Fit Gamma parameters using L-moments.

    Reference: Hosking (1990), Table 2
    """
    return _fit_gamma_lmoments_robust(values)


def _fit_gamma_lmoments_robust(values: np.ndarray) -> Tuple[float, float]:
    """
    Robust Gamma fitting using L-moments.

    Handles edge cases:
    - Invalid L-moments
    - Extreme L-CV values
    - Numerical overflow

    Reference: Hosking (1990), Table 2
    """
    n = len(values)
    if n < 3:
        return _fit_gamma_moments_robust(values)

    try:
        lmom = compute_lmoments(values, nmom=2)
    except Exception:
        return _fit_gamma_moments_robust(values)

    l1, l2 = lmom[0], lmom[1]

    # Check for invalid L-moments
    if np.isnan(l1) or np.isnan(l2):
        return _fit_gamma_moments_robust(values)

    if l1 <= EPSILON:
        return np.nan, np.nan

    if l2 <= EPSILON:
        # Near-constant - high alpha
        return MAX_SHAPE_PARAM, l1 / MAX_SHAPE_PARAM

    # L-CV (coefficient of L-variation)
    t = l2 / l1

    # Bound t to valid range for approximation
    t = np.clip(t, EPSILON, 0.99)

    # Approximation for alpha from L-CV (Hosking 1990)
    if t < 0.5:
        # Polynomial approximation
        z = np.pi * t ** 2
        denom = z + z ** 2 * 0.0552
        if denom > EPSILON:
            alpha = (1 + 0.2906 * z) / denom
        else:
            alpha = 1.0
    else:
        # High L-CV - lower alpha
        # Use rational approximation for t > 0.5
        alpha = (1 - t) / t if t < 1.0 else 0.1

    # Ensure alpha is positive and bounded
    alpha = np.clip(alpha, EPSILON, MAX_SHAPE_PARAM)

    # Scale parameter
    beta = l1 / alpha
    beta = np.clip(beta, MIN_SCALE_PARAM, MAX_SCALE_PARAM)

    return alpha, beta


def gamma_cdf(
    values: np.ndarray,
    params: DistributionParams
) -> np.ndarray:
    """
    Compute CDF using Gamma distribution with zero-inflation.

    Includes robust handling for:
    - Invalid parameters
    - Extreme values causing overflow
    - Mixed distribution (zeros + positive)

    :param values: array of values to transform
    :param params: fitted distribution parameters
    :return: array of CDF values in [0, 1]
    """
    alpha = params.params.get('alpha', np.nan)
    beta = params.params.get('beta', np.nan)
    prob_zero = params.prob_zero

    result = np.full(values.shape, np.nan)

    # Check for invalid parameters
    if not params.is_valid():
        return result

    if alpha <= 0 or beta <= 0 or np.isnan(alpha) or np.isnan(beta):
        return result

    # Handle zeros - assign probability of zero
    zero_mask = (values == 0) | (np.abs(values) < EPSILON)
    result[zero_mask] = prob_zero

    # Handle positive values
    positive_mask = (values > EPSILON) & np.isfinite(values)
    if np.any(positive_mask):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cdf_vals = stats.gamma.cdf(values[positive_mask], alpha, scale=beta)

                # Handle any NaN or invalid results
                invalid_mask = ~np.isfinite(cdf_vals)
                if np.any(invalid_mask):
                    # For invalid points, use empirical estimate
                    cdf_vals[invalid_mask] = 0.5

                # Mixed distribution: H(x) = q + (1-q) * G(x)
                mixed_cdf = prob_zero + (1 - prob_zero) * cdf_vals

                # Clip to valid range
                result[positive_mask] = np.clip(mixed_cdf, 1e-10, 1.0 - 1e-10)

        except Exception:
            # Fallback: use empirical CDF approximation
            pos_vals = values[positive_mask]
            ranks = np.argsort(np.argsort(pos_vals)) + 1
            empirical_cdf = ranks / (len(pos_vals) + 1)
            result[positive_mask] = prob_zero + (1 - prob_zero) * empirical_cdf

    # Handle negative values (shouldn't happen for precip, but for SPEI water balance)
    negative_mask = (values < -EPSILON) & np.isfinite(values)
    if np.any(negative_mask):
        # For negative values in Gamma context, use small probability
        result[negative_mask] = prob_zero * 0.5  # Half of zero probability

    return result


# =============================================================================
# PEARSON TYPE III DISTRIBUTION
# =============================================================================

def fit_pearson3(
    values: np.ndarray,
    method: FittingMethod = FittingMethod.MOMENTS
) -> DistributionParams:
    """
    Fit Pearson Type III distribution to data.

    Pearson III (3-parameter Gamma) is recommended for SPEI as it can
    handle negative values (water deficit) and asymmetric distributions.

    This implementation includes robust handling for:
    - High proportion of zeros (common in arid regions)
    - Near-zero variance (constant or near-constant data)
    - Division by zero in moment calculations
    - Numerical overflow in parameter estimation

    :param values: array of values (can include negatives for SPEI)
    :param method: parameter estimation method
    :return: DistributionParams object
    """
    valid_mask = ~np.isnan(values) & np.isfinite(values)
    valid_values = values[valid_mask]
    n_total = len(valid_values)

    if n_total < MIN_VALUES_FOR_FIT:
        return _invalid_params(DistributionType.PEARSON3, method)

    # Calculate zero statistics
    n_zeros = np.sum(np.abs(valid_values) < EPSILON)
    zero_proportion = n_zeros / n_total
    n_nonzero = n_total - n_zeros

    # Check for excessive zeros - this causes fitting failures
    if zero_proportion > MAX_ZERO_PROPORTION:
        # Too many zeros - return invalid params with metadata
        return DistributionParams(
            distribution=DistributionType.PEARSON3,
            params={'skew': np.nan, 'loc': np.nan, 'scale': np.nan},
            prob_zero=zero_proportion,
            n_samples=n_total,
            fitting_method=method
        )

    # Check for minimum non-zero values
    if n_nonzero < MIN_NONZERO_VALUES:
        return _invalid_params(DistributionType.PEARSON3, method, zero_proportion, n_total)

    # Check for near-constant data (variance too small)
    data_range = np.ptp(valid_values)  # peak-to-peak (max - min)
    variance = np.var(valid_values, ddof=1) if n_total > 1 else 0.0

    if variance < MIN_VARIANCE or data_range < EPSILON:
        # Near-constant data - use normal approximation
        mean = np.mean(valid_values)
        return DistributionParams(
            distribution=DistributionType.PEARSON3,
            params={'skew': 0.0, 'loc': mean, 'scale': max(np.sqrt(variance), EPSILON)},
            prob_zero=zero_proportion,
            n_samples=n_total,
            fitting_method=method
        )

    # For Pearson III with SPEI, calculate prob_zero for mixed distribution
    n_non_positive = np.sum(valid_values <= 0)
    prob_zero = n_non_positive / n_total if n_total > 0 else 0.0

    # Fit parameters based on method with fallback chain
    skew, loc, scale = np.nan, np.nan, np.nan

    if method == FittingMethod.MLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skew, loc, scale = stats.pearson3.fit(valid_values)
        except Exception:
            # Fallback to L-moments if MLE fails
            skew, loc, scale = _fit_pearson3_lmoments_robust(valid_values)
    elif method == FittingMethod.LMOMENTS:
        skew, loc, scale = _fit_pearson3_lmoments_robust(valid_values)
    else:  # Method of moments
        skew, loc, scale = _fit_pearson3_moments_robust(valid_values)

    # Final validation and bounds checking
    if np.isnan(skew) or np.isnan(loc) or np.isnan(scale):
        # All methods failed - try normal approximation as last resort
        skew, loc, scale = 0.0, np.mean(valid_values), np.std(valid_values, ddof=1)

    # Bound parameters to prevent numerical issues
    skew = np.clip(skew, -10.0, 10.0)
    scale = np.clip(abs(scale), MIN_SCALE_PARAM, MAX_SCALE_PARAM)

    return DistributionParams(
        distribution=DistributionType.PEARSON3,
        params={'skew': skew, 'loc': loc, 'scale': scale},
        prob_zero=prob_zero,
        n_samples=n_total,
        fitting_method=method
    )


def _fit_pearson3_moments(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit Pearson III parameters using method of moments.

    Parameters:
        skew: skewness parameter (shape)
        loc: location parameter
        scale: scale parameter
    """
    return _fit_pearson3_moments_robust(values)


def _fit_pearson3_moments_robust(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Robust Pearson III fitting using method of moments.

    Handles edge cases:
    - Zero or near-zero variance
    - Division by zero in skewness calculation
    - Extreme skewness values
    """
    n = len(values)
    if n < 3:
        return np.nan, np.nan, np.nan

    mean = np.mean(values)
    variance = np.var(values, ddof=1)
    std = np.sqrt(variance) if variance > 0 else 0.0

    # Check for near-zero variance
    if std <= EPSILON or variance < MIN_VARIANCE:
        # Return normal approximation with minimal scale
        return 0.0, mean, max(std, EPSILON)

    # Calculate sample skewness with overflow protection
    try:
        centered = values - mean
        m3 = np.mean(centered ** 3)

        # Prevent division by zero
        std_cubed = std ** 3
        if std_cubed < EPSILON:
            skewness = 0.0
        else:
            skewness = m3 / std_cubed

        # Apply bias correction for small samples (Fisher's adjustment)
        if n > 3:
            skewness = skewness * np.sqrt(n * (n - 1)) / (n - 2)

    except (FloatingPointError, OverflowError, ZeroDivisionError):
        skewness = 0.0

    # Bound skewness to prevent numerical issues
    skewness = np.clip(skewness, -10.0, 10.0)

    # Nearly symmetric - use normal approximation
    if abs(skewness) < EPSILON:
        return 0.0, mean, std

    return skewness, mean, std


def _fit_pearson3_lmoments(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit Pearson III parameters using L-moments.

    Reference: Hosking (1990), Appendix
    """
    return _fit_pearson3_lmoments_robust(values)


def _fit_pearson3_lmoments_robust(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Robust Pearson III fitting using L-moments.

    Handles edge cases:
    - Zero or near-zero L2 (scale)
    - Extreme L-skewness values
    - Numerical overflow in gamma function
    - Invalid parameter combinations

    Reference: Hosking (1990), Appendix
    """
    n = len(values)
    if n < 4:
        # Fall back to moments for very small samples
        return _fit_pearson3_moments_robust(values)

    try:
        lmom = compute_lmoments(values, nmom=3)
    except Exception:
        return _fit_pearson3_moments_robust(values)

    l1, l2, l3 = lmom[0], lmom[1], lmom[2]

    # Check for invalid L-moments
    if np.isnan(l1) or np.isnan(l2) or np.isnan(l3):
        return _fit_pearson3_moments_robust(values)

    # L2 must be positive (it's a measure of scale/dispersion)
    if l2 <= EPSILON:
        # Near-constant data - use normal approximation
        return 0.0, l1, max(abs(l2), EPSILON)

    # L-skewness (bounded to [-1, 1] for valid distributions)
    t3 = np.clip(l3 / l2, -0.99, 0.99)

    # Approximate skewness from L-skewness
    # Using polynomial approximation (Hosking 1990)
    if abs(t3) < 0.99:
        # Rational approximation for better accuracy
        c0, c1, c2 = 0.0, 1.0, 0.2906
        skew = 2.0 * t3 * (c0 + c1 + c2 * t3 ** 2)
    else:
        # Extreme L-skewness - cap the conventional skewness
        skew = np.sign(t3) * 2.0

    # Bound skewness
    skew = np.clip(skew, -10.0, 10.0)

    # Calculate scale and location with overflow protection
    try:
        if abs(skew) > EPSILON:
            # Shape parameter (alpha = 4/skew^2 for Pearson III)
            alpha = 4.0 / (skew ** 2)
            alpha = min(alpha, MAX_SHAPE_PARAM)  # Prevent overflow

            # Scale parameter with gamma function protection
            if alpha > 0 and alpha < 170:  # gamma function limit
                g_ratio = gamma_func(0.5 + alpha)
                if g_ratio > EPSILON:
                    scale = l2 * np.sqrt(np.pi) * np.sqrt(alpha) / g_ratio
                else:
                    scale = l2 * np.sqrt(np.pi)
            else:
                # For very large alpha, use asymptotic approximation
                scale = l2 * np.sqrt(np.pi)

            # Location parameter
            loc = l1 - alpha * scale * np.sign(skew)
        else:
            # Normal approximation (skew ≈ 0)
            skew = 0.0
            scale = l2 * np.sqrt(np.pi)
            loc = l1

    except (FloatingPointError, OverflowError, ValueError):
        # Fallback to moments
        return _fit_pearson3_moments_robust(values)

    # Final bounds check
    scale = np.clip(abs(scale), MIN_SCALE_PARAM, MAX_SCALE_PARAM)

    # Validate results
    if np.isnan(skew) or np.isnan(loc) or np.isnan(scale):
        return _fit_pearson3_moments_robust(values)

    return skew, loc, scale


def pearson3_cdf(
    values: np.ndarray,
    params: DistributionParams
) -> np.ndarray:
    """
    Compute CDF using Pearson Type III distribution.

    Includes robust handling for:
    - Invalid parameters (returns NaN)
    - Near-zero skewness (uses normal approximation)
    - Extreme values (clipped CDF)
    - Numerical overflow in scipy

    :param values: array of values to transform
    :param params: fitted distribution parameters
    :return: array of CDF values in [0, 1]
    """
    skew = params.params.get('skew', np.nan)
    loc = params.params.get('loc', np.nan)
    scale = params.params.get('scale', np.nan)

    result = np.full(values.shape, np.nan)

    # Check for invalid parameters
    if not params.is_valid():
        return result

    # Ensure scale is positive
    if scale <= 0 or np.isnan(scale):
        return result

    valid_mask = ~np.isnan(values) & np.isfinite(values)
    if not np.any(valid_mask):
        return result

    valid_values = values[valid_mask]

    # Use normal approximation for near-zero skewness
    # This avoids numerical issues in scipy.stats.pearson3
    if abs(skew) < 0.01:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cdf_vals = stats.norm.cdf(valid_values, loc=loc, scale=scale)
                result[valid_mask] = np.clip(cdf_vals, 0.0, 1.0)
        except Exception:
            pass
        return result

    # For non-zero skewness, use Pearson III with error handling
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # scipy.stats.pearson3 can fail for extreme parameters
            # Compute CDF in chunks to isolate failures
            cdf_vals = np.full(len(valid_values), np.nan)

            # Try vectorized computation first
            try:
                cdf_vals = stats.pearson3.cdf(valid_values, skew, loc=loc, scale=scale)
            except Exception:
                # Fallback: compute element-wise
                for i, v in enumerate(valid_values):
                    try:
                        cdf_vals[i] = stats.pearson3.cdf(v, skew, loc=loc, scale=scale)
                    except Exception:
                        # Use normal approximation for failed points
                        cdf_vals[i] = stats.norm.cdf(v, loc=loc, scale=scale)

            # Handle any remaining NaN or invalid values
            invalid_cdf = ~np.isfinite(cdf_vals)
            if np.any(invalid_cdf):
                # Use normal approximation for invalid points
                cdf_vals[invalid_cdf] = stats.norm.cdf(
                    valid_values[invalid_cdf], loc=loc, scale=scale
                )

            # Clip to valid CDF range
            result[valid_mask] = np.clip(cdf_vals, 1e-10, 1.0 - 1e-10)

    except Exception:
        # Complete failure - try normal approximation for all
        try:
            cdf_vals = stats.norm.cdf(valid_values, loc=loc, scale=scale)
            result[valid_mask] = np.clip(cdf_vals, 1e-10, 1.0 - 1e-10)
        except Exception:
            pass

    return result


# =============================================================================
# LOG-LOGISTIC DISTRIBUTION
# =============================================================================

def fit_log_logistic(
    values: np.ndarray,
    method: FittingMethod = FittingMethod.MLE
) -> DistributionParams:
    """
    Fit Log-Logistic (Fisk) distribution to data.

    Log-Logistic has heavier tails than Gamma, making it suitable
    for capturing extreme events in SPEI.

    :param values: array of positive values
    :param method: parameter estimation method
    :return: DistributionParams object
    """
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    n_total = len(valid_values)

    if n_total == 0:
        return _invalid_params(DistributionType.LOG_LOGISTIC, method)

    # Handle zeros
    n_zeros = np.sum(valid_values <= 0)
    prob_zero = n_zeros / n_total

    # Work with positive values only
    positive_values = valid_values[valid_values > 0]
    n_positive = len(positive_values)

    if n_positive < MIN_VALUES_FOR_FIT:
        return _invalid_params(DistributionType.LOG_LOGISTIC, method, prob_zero, n_total)

    # Fit parameters
    if method == FittingMethod.MLE:
        try:
            c, loc, scale = stats.fisk.fit(positive_values, floc=0)
            alpha = c  # shape
            beta = scale  # scale
        except Exception:
            alpha, beta = np.nan, np.nan
    elif method == FittingMethod.LMOMENTS:
        alpha, beta = _fit_loglogistic_lmoments(positive_values)
    else:
        alpha, beta = _fit_loglogistic_moments(positive_values)

    return DistributionParams(
        distribution=DistributionType.LOG_LOGISTIC,
        params={'alpha': alpha, 'beta': beta},
        prob_zero=prob_zero,
        n_samples=n_total,
        fitting_method=method
    )


def _fit_loglogistic_moments(values: np.ndarray) -> Tuple[float, float]:
    """
    Fit Log-Logistic parameters using method of moments.

    For Log-Logistic with shape alpha > 2:
        mean = beta * pi/alpha / sin(pi/alpha)
        variance = beta^2 * (2*pi/alpha / sin(2*pi/alpha) - (pi/alpha / sin(pi/alpha))^2)
    """
    mean = np.mean(values)
    variance = np.var(values, ddof=1)

    if mean <= EPSILON or variance <= EPSILON:
        return np.nan, np.nan

    cv = np.sqrt(variance) / mean  # Coefficient of variation

    # Solve for alpha using CV relationship
    # For alpha > 2: CV^2 = (pi/alpha)^2 * (csc^2(pi/alpha) - 1/2 * csc(2*pi/alpha))
    # Use approximation: alpha ≈ pi / (CV * sqrt(3))
    if cv > 0:
        alpha = np.pi / (cv * np.sqrt(3))
        alpha = max(alpha, 2.1)  # Ensure alpha > 2 for finite moments

        # Compute beta from mean
        beta = mean * np.sin(np.pi / alpha) / (np.pi / alpha)
    else:
        alpha, beta = np.nan, np.nan

    return alpha, beta


def _fit_loglogistic_lmoments(values: np.ndarray) -> Tuple[float, float]:
    """
    Fit Log-Logistic parameters using L-moments.

    Reference: Hosking (1990)
    For Log-Logistic:
        l1 = beta * Gamma(1 + 1/alpha) * Gamma(1 - 1/alpha)
        t3 = 0 (symmetric on log scale)
    """
    lmom = compute_lmoments(values, nmom=2)
    l1, l2 = lmom[0], lmom[1]

    if l1 <= EPSILON or l2 <= EPSILON:
        return np.nan, np.nan

    # For Log-Logistic, L-CV (t) relates to alpha
    t = l2 / l1

    if t <= 0 or t >= 1:
        return np.nan, np.nan

    # alpha from L-CV: t = sin(pi/alpha) * (pi/alpha)
    # Iterative solution or approximation
    alpha = np.pi / np.arcsin(t) if t < 1 else 2.0
    alpha = max(alpha, 2.1)

    # beta from l1
    beta = l1 * np.sin(np.pi / alpha) / (np.pi / alpha)

    return alpha, beta


def log_logistic_cdf(
    values: np.ndarray,
    params: DistributionParams
) -> np.ndarray:
    """
    Compute CDF using Log-Logistic distribution.

    CDF(x) = 1 / (1 + (x/beta)^(-alpha))

    :param values: array of values to transform
    :param params: fitted distribution parameters
    :return: array of CDF values in [0, 1]
    """
    alpha = params.params['alpha']
    beta = params.params['beta']
    prob_zero = params.prob_zero

    result = np.full(values.shape, np.nan)

    if not params.is_valid():
        return result

    # Handle zeros
    zero_mask = (values <= 0)
    result[zero_mask] = prob_zero

    # Handle positive values
    positive_mask = (values > 0) & ~np.isnan(values)
    if np.any(positive_mask):
        x = values[positive_mask]
        cdf_vals = 1.0 / (1.0 + (x / beta) ** (-alpha))
        result[positive_mask] = prob_zero + (1 - prob_zero) * cdf_vals

    return result


# =============================================================================
# GENERALIZED EXTREME VALUE (GEV) DISTRIBUTION
# =============================================================================

def fit_gev(
    values: np.ndarray,
    method: FittingMethod = FittingMethod.LMOMENTS
) -> DistributionParams:
    """
    Fit Generalized Extreme Value (GEV) distribution.

    GEV is appropriate for modeling extremes (block maxima/minima).
    The shape parameter determines tail behavior:
        - xi < 0: Weibull (bounded upper tail)
        - xi = 0: Gumbel (light tails)
        - xi > 0: Fréchet (heavy upper tail)

    :param values: array of values
    :param method: parameter estimation method
    :return: DistributionParams object
    """
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    n_total = len(valid_values)

    if n_total < MIN_VALUES_FOR_FIT:
        return _invalid_params(DistributionType.GEV, method)

    n_zeros = np.sum(valid_values <= 0)
    prob_zero = n_zeros / n_total if n_total > 0 else 0.0

    # Fit parameters
    if method == FittingMethod.MLE:
        try:
            shape, loc, scale = stats.genextreme.fit(valid_values)
        except Exception:
            shape, loc, scale = np.nan, np.nan, np.nan
    elif method == FittingMethod.LMOMENTS:
        shape, loc, scale = _fit_gev_lmoments(valid_values)
    else:
        # Method of moments is less reliable for GEV, use L-moments instead
        shape, loc, scale = _fit_gev_lmoments(valid_values)

    return DistributionParams(
        distribution=DistributionType.GEV,
        params={'shape': shape, 'loc': loc, 'scale': scale},
        prob_zero=prob_zero,
        n_samples=n_total,
        fitting_method=method
    )


def _fit_gev_lmoments(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit GEV parameters using L-moments.

    Reference: Hosking (1990), Table 1
    """
    lmom = compute_lmoments(values, nmom=3)
    l1, l2, l3 = lmom[0], lmom[1], lmom[2]

    if l2 <= EPSILON:
        return np.nan, np.nan, np.nan

    # L-skewness
    t3 = l3 / l2

    # Approximate shape parameter from L-skewness
    # Using polynomial approximation (Hosking et al. 1985)
    c = 2 / (3 + t3) - np.log(2) / np.log(3)
    shape = 7.8590 * c + 2.9554 * c ** 2

    # Bound shape parameter for stability
    shape = np.clip(shape, -0.5, 0.5)

    # Scale and location from L-moments
    if abs(shape) > EPSILON:
        g1 = gamma_func(1 + shape)
        g2 = gamma_func(1 + 2 * shape) if abs(shape) < 0.5 else 1.0

        scale = l2 * shape / (g1 * (1 - 2 ** (-shape)))
        loc = l1 - scale * (g1 - 1) / shape
    else:
        # Gumbel case (shape = 0)
        scale = l2 / np.log(2)
        loc = l1 - 0.5772 * scale  # Euler-Mascheroni constant

    return shape, loc, scale


def gev_cdf(
    values: np.ndarray,
    params: DistributionParams
) -> np.ndarray:
    """
    Compute CDF using GEV distribution.

    :param values: array of values to transform
    :param params: fitted distribution parameters
    :return: array of CDF values in [0, 1]
    """
    shape = params.params['shape']
    loc = params.params['loc']
    scale = params.params['scale']

    result = np.full(values.shape, np.nan)

    if not params.is_valid():
        return result

    valid_mask = ~np.isnan(values)
    if np.any(valid_mask):
        try:
            result[valid_mask] = stats.genextreme.cdf(
                values[valid_mask], shape, loc=loc, scale=scale
            )
        except Exception:
            pass

    return result


# =============================================================================
# GENERALIZED LOGISTIC DISTRIBUTION
# =============================================================================

def fit_gen_logistic(
    values: np.ndarray,
    method: FittingMethod = FittingMethod.LMOMENTS
) -> DistributionParams:
    """
    Fit Generalized Logistic distribution.

    Similar to GEV but with logistic instead of exponential base.
    Used in some European drought indices.

    :param values: array of values
    :param method: parameter estimation method
    :return: DistributionParams object
    """
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    n_total = len(valid_values)

    if n_total < MIN_VALUES_FOR_FIT:
        return _invalid_params(DistributionType.GEN_LOGISTIC, method)

    n_zeros = np.sum(valid_values <= 0)
    prob_zero = n_zeros / n_total if n_total > 0 else 0.0

    # Fit parameters using L-moments
    shape, loc, scale = _fit_genlogistic_lmoments(valid_values)

    return DistributionParams(
        distribution=DistributionType.GEN_LOGISTIC,
        params={'shape': shape, 'loc': loc, 'scale': scale},
        prob_zero=prob_zero,
        n_samples=n_total,
        fitting_method=method
    )


def _fit_genlogistic_lmoments(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit Generalized Logistic parameters using L-moments.

    Reference: Hosking (1990)
    """
    lmom = compute_lmoments(values, nmom=3)
    l1, l2, l3 = lmom[0], lmom[1], lmom[2]

    if l2 <= EPSILON:
        return np.nan, np.nan, np.nan

    # L-skewness directly gives shape parameter
    t3 = l3 / l2
    shape = -t3  # For generalized logistic

    # Bound for stability
    shape = np.clip(shape, -0.5, 0.5)

    # Scale and location
    if abs(shape) > EPSILON:
        g = gamma_func(1 + shape) * gamma_func(1 - shape)
        scale = l2 / g
        loc = l1 - scale * (1 / shape - np.pi / np.sin(np.pi * shape))
    else:
        # Standard logistic
        scale = l2
        loc = l1

    return shape, loc, scale


def gen_logistic_cdf(
    values: np.ndarray,
    params: DistributionParams
) -> np.ndarray:
    """
    Compute CDF using Generalized Logistic distribution.

    :param values: array of values to transform
    :param params: fitted distribution parameters
    :return: array of CDF values in [0, 1]
    """
    shape = params.params['shape']
    loc = params.params['loc']
    scale = params.params['scale']

    result = np.full(values.shape, np.nan)

    if not params.is_valid():
        return result

    valid_mask = ~np.isnan(values)
    if np.any(valid_mask):
        try:
            # scipy.stats.genlogistic uses different parameterization
            result[valid_mask] = stats.genlogistic.cdf(
                values[valid_mask], shape, loc=loc, scale=scale
            )
        except Exception:
            pass

    return result


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

# Mapping of distribution types to fitting and CDF functions
_DISTRIBUTION_FUNCTIONS = {
    DistributionType.GAMMA: (fit_gamma, gamma_cdf),
    DistributionType.PEARSON3: (fit_pearson3, pearson3_cdf),
    DistributionType.LOG_LOGISTIC: (fit_log_logistic, log_logistic_cdf),
    DistributionType.GEV: (fit_gev, gev_cdf),
    DistributionType.GEN_LOGISTIC: (fit_gen_logistic, gen_logistic_cdf),
}

# Recommended distributions for different indices
RECOMMENDED_DISTRIBUTIONS = {
    'spi': DistributionType.GAMMA,
    'spei': DistributionType.PEARSON3,
    'extreme': DistributionType.GEV,
}


def fit_distribution(
    values: np.ndarray,
    distribution: Union[str, DistributionType] = DistributionType.GAMMA,
    method: Union[str, FittingMethod, None] = None,
    calibration_indices: Optional[Tuple[int, int]] = None
) -> DistributionParams:
    """
    Unified interface for distribution fitting.

    :param values: 1-D array of values
    :param distribution: distribution type (string or DistributionType)
    :param method: fitting method (string or FittingMethod). If None, uses
        the recommended default for each distribution.
    :param calibration_indices: optional (start, end) indices for calibration
    :return: DistributionParams object

    Example:
        >>> params = fit_distribution(precip_data, 'gamma', 'lmoments')
        >>> cdf_values = compute_cdf(precip_data, params)
    """
    # Convert string inputs to enums
    if isinstance(distribution, str):
        distribution = DistributionType(distribution.lower().replace('-', '_'))

    if isinstance(method, str):
        method = FittingMethod(method.lower())

    # Extract calibration period if specified
    if calibration_indices is not None:
        start, end = calibration_indices
        values = values[start:end]

    # Get fitting function and fit
    fit_func, _ = _DISTRIBUTION_FUNCTIONS[distribution]
    if method is not None:
        return fit_func(values, method)
    else:
        # Use each distribution's recommended default method
        return fit_func(values)


def compute_cdf(
    values: np.ndarray,
    params: DistributionParams
) -> np.ndarray:
    """
    Compute CDF using fitted distribution parameters.

    :param values: array of values to transform
    :param params: DistributionParams from fit_distribution()
    :return: array of CDF values in [0, 1]
    """
    _, cdf_func = _DISTRIBUTION_FUNCTIONS[params.distribution]
    return cdf_func(values, params)


def cdf_to_standard_normal(cdf_values: np.ndarray) -> np.ndarray:
    """
    Transform CDF values to standard normal distribution (SPI/SPEI values).

    Uses inverse standard normal (probit) transformation.

    :param cdf_values: array of CDF values in [0, 1]
    :return: array of standard normal values (SPI/SPEI)
    """
    # Clip to avoid infinite values at extremes
    cdf_clipped = np.clip(cdf_values, 1e-10, 1 - 1e-10)

    # Inverse standard normal transformation
    result = stats.norm.ppf(cdf_clipped)

    # Clip extreme values for numerical stability
    result = np.clip(result, FITTED_INDEX_VALID_MIN, FITTED_INDEX_VALID_MAX)

    return result


# =============================================================================
# GOODNESS OF FIT TESTING
# =============================================================================

def test_goodness_of_fit(
    values: np.ndarray,
    params: DistributionParams
) -> GoodnessOfFit:
    """
    Test goodness-of-fit for a fitted distribution.

    Performs:
    - Kolmogorov-Smirnov test
    - Anderson-Darling test
    - Computes AIC and BIC

    :param values: original data values
    :param params: fitted distribution parameters
    :return: GoodnessOfFit named tuple with test statistics
    """
    valid_values = values[~np.isnan(values)]
    n = len(valid_values)

    if n < MIN_VALUES_FOR_FIT or not params.is_valid():
        return GoodnessOfFit(np.nan, np.nan, np.nan, np.nan, np.nan)

    # Compute CDF values
    cdf_values = compute_cdf(valid_values, params)
    cdf_values = cdf_values[~np.isnan(cdf_values)]

    if len(cdf_values) == 0:
        return GoodnessOfFit(np.nan, np.nan, np.nan, np.nan, np.nan)

    # Kolmogorov-Smirnov test (against uniform distribution)
    try:
        ks_stat, ks_pvalue = stats.kstest(cdf_values, 'uniform')
    except Exception:
        ks_stat, ks_pvalue = np.nan, np.nan

    # Anderson-Darling test
    try:
        ad_result = stats.anderson(cdf_values, 'norm')
        ad_stat = ad_result.statistic
    except Exception:
        ad_stat = np.nan

    # Number of parameters
    k = len(params.params)

    # Log-likelihood approximation (using normal approximation to CDF)
    try:
        z_values = cdf_to_standard_normal(cdf_values)
        log_likelihood = np.sum(stats.norm.logpdf(z_values))
    except Exception:
        log_likelihood = np.nan

    # AIC and BIC
    if not np.isnan(log_likelihood):
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
    else:
        aic, bic = np.nan, np.nan

    return GoodnessOfFit(ks_stat, ks_pvalue, ad_stat, aic, bic)


def compare_distributions(
    values: np.ndarray,
    distributions: Optional[List[DistributionType]] = None,
    method: FittingMethod = FittingMethod.LMOMENTS
) -> Dict[str, Dict]:
    """
    Compare multiple distributions for best fit.

    :param values: 1-D array of values
    :param distributions: list of distributions to compare
    :param method: fitting method
    :return: dictionary with fit results for each distribution

    Example:
        >>> results = compare_distributions(precip_data)
        >>> best = min(results.items(), key=lambda x: x[1].get('aic', np.inf))
        >>> print(f"Best distribution: {best[0]}")
    """
    if distributions is None:
        distributions = list(DistributionType)

    results = {}

    for dist in distributions:
        try:
            params = fit_distribution(values, dist, method)
            gof = test_goodness_of_fit(values, params)

            results[dist.value] = {
                'params': params.to_dict(),
                'ks_statistic': gof.ks_statistic,
                'ks_pvalue': gof.ks_pvalue,
                'ad_statistic': gof.ad_statistic,
                'aic': gof.aic,
                'bic': gof.bic,
                'valid': params.is_valid()
            }
        except Exception as e:
            results[dist.value] = {'error': str(e), 'valid': False}

    return results


def select_best_distribution(
    values: np.ndarray,
    distributions: Optional[List[DistributionType]] = None,
    criterion: str = 'aic'
) -> Tuple[DistributionType, DistributionParams]:
    """
    Automatically select the best-fitting distribution.

    :param values: 1-D array of values
    :param distributions: list of distributions to compare
    :param criterion: selection criterion ('aic', 'bic', 'ks')
    :return: tuple of (best distribution type, fitted parameters)
    """
    comparison = compare_distributions(values, distributions)

    # Filter valid results
    valid_results = {
        k: v for k, v in comparison.items()
        if v.get('valid', False) and not np.isnan(v.get(criterion, np.nan))
    }

    if not valid_results:
        # Fallback to gamma
        params = fit_distribution(values, DistributionType.GAMMA)
        return DistributionType.GAMMA, params

    # Select best based on criterion
    if criterion == 'ks':
        # For KS, higher p-value is better
        best_name = max(valid_results.keys(), key=lambda k: valid_results[k]['ks_pvalue'])
    else:
        # For AIC/BIC, lower is better
        best_name = min(valid_results.keys(), key=lambda k: valid_results[k][criterion])

    best_dist = DistributionType(best_name)
    best_params = fit_distribution(values, best_dist)

    return best_dist, best_params


# =============================================================================
# DATA DIAGNOSTICS
# =============================================================================

class DataDiagnostics(NamedTuple):
    """Diagnostics for distribution fitting suitability."""
    n_total: int
    n_valid: int
    n_zeros: int
    n_positive: int
    n_negative: int
    zero_proportion: float
    mean: float
    std: float
    variance: float
    skewness: float
    min_value: float
    max_value: float
    is_suitable_for_gamma: bool
    is_suitable_for_pearson3: bool
    warnings: List[str]
    recommendation: str


def diagnose_data(values: np.ndarray) -> DataDiagnostics:
    """
    Diagnose data for distribution fitting suitability.

    Use this function before fitting to understand potential issues
    with your data, especially for large AOIs with many zeros.

    :param values: 1-D array of values to diagnose
    :return: DataDiagnostics with statistics and recommendations

    Example:
        >>> diag = diagnose_data(precip_data)
        >>> if diag.warnings:
        ...     print("Warnings:", diag.warnings)
        >>> print(f"Recommendation: {diag.recommendation}")
    """
    warnings_list = []

    # Basic counts
    valid_mask = ~np.isnan(values) & np.isfinite(values)
    valid_values = values[valid_mask]
    n_total = len(values)
    n_valid = len(valid_values)
    n_invalid = n_total - n_valid

    if n_invalid > 0:
        warnings_list.append(f"{n_invalid} NaN/Inf values will be excluded")

    if n_valid == 0:
        return DataDiagnostics(
            n_total=n_total, n_valid=0, n_zeros=0, n_positive=0, n_negative=0,
            zero_proportion=1.0, mean=np.nan, std=np.nan, variance=np.nan,
            skewness=np.nan, min_value=np.nan, max_value=np.nan,
            is_suitable_for_gamma=False, is_suitable_for_pearson3=False,
            warnings=["No valid data points"], recommendation="Cannot fit - no valid data"
        )

    # Zero analysis
    n_zeros = np.sum(np.abs(valid_values) < EPSILON)
    n_positive = np.sum(valid_values > EPSILON)
    n_negative = np.sum(valid_values < -EPSILON)
    zero_proportion = n_zeros / n_valid

    # Statistics
    mean = np.mean(valid_values)
    std = np.std(valid_values, ddof=1) if n_valid > 1 else 0.0
    variance = std ** 2
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)

    # Skewness
    if std > EPSILON and n_valid > 2:
        centered = valid_values - mean
        m3 = np.mean(centered ** 3)
        skewness = m3 / (std ** 3) if std ** 3 > EPSILON else 0.0
    else:
        skewness = 0.0

    # Suitability checks
    is_suitable_gamma = True
    is_suitable_pearson3 = True

    # Check for insufficient data
    if n_valid < MIN_VALUES_FOR_FIT:
        warnings_list.append(f"Only {n_valid} valid values (minimum {MIN_VALUES_FOR_FIT} required)")
        is_suitable_gamma = False
        is_suitable_pearson3 = False

    # Check for excessive zeros
    if zero_proportion > MAX_ZERO_PROPORTION:
        warnings_list.append(
            f"Very high zero proportion ({zero_proportion:.1%}) - "
            f"fitting will be unreliable"
        )
        is_suitable_gamma = False
        is_suitable_pearson3 = False
    elif zero_proportion > 0.8:
        warnings_list.append(
            f"High zero proportion ({zero_proportion:.1%}) - "
            f"fitting may be unstable"
        )

    # Check for insufficient non-zero values
    if n_positive < MIN_NONZERO_VALUES:
        warnings_list.append(
            f"Only {n_positive} positive values (minimum {MIN_NONZERO_VALUES} for reliable fitting)"
        )
        is_suitable_gamma = False

    # Check for near-constant data
    if variance < MIN_VARIANCE:
        warnings_list.append(
            f"Near-constant data (variance={variance:.2e}) - "
            f"normal approximation will be used"
        )

    # Check for negative values (Gamma issue)
    if n_negative > 0:
        warnings_list.append(
            f"{n_negative} negative values present - "
            f"Gamma distribution cannot be used directly"
        )
        is_suitable_gamma = False

    # Generate recommendation
    if not is_suitable_gamma and not is_suitable_pearson3:
        if zero_proportion > MAX_ZERO_PROPORTION:
            recommendation = (
                "Data has too many zeros for reliable fitting. "
                "Consider: (1) using a longer accumulation scale, "
                "(2) aggregating to coarser spatial resolution, or "
                "(3) masking this location as 'arid/unsuitable'."
            )
        elif n_valid < MIN_VALUES_FOR_FIT:
            recommendation = (
                f"Insufficient data ({n_valid} values). "
                f"Need at least {MIN_VALUES_FOR_FIT} values for reliable fitting."
            )
        else:
            recommendation = "Data characteristics prevent reliable fitting."
    elif is_suitable_gamma and is_suitable_pearson3:
        if zero_proportion > 0.5:
            recommendation = (
                "Data is suitable for fitting but has many zeros. "
                "L-moments method recommended for robustness."
            )
        else:
            recommendation = "Data is suitable for both Gamma and Pearson III fitting."
    elif is_suitable_pearson3:
        recommendation = (
            "Data has negative values - use Pearson III (not Gamma) for SPEI."
        )
    else:
        recommendation = "Use Gamma distribution with L-moments method."

    return DataDiagnostics(
        n_total=n_total,
        n_valid=n_valid,
        n_zeros=n_zeros,
        n_positive=n_positive,
        n_negative=n_negative,
        zero_proportion=zero_proportion,
        mean=mean,
        std=std,
        variance=variance,
        skewness=skewness,
        min_value=min_val,
        max_value=max_val,
        is_suitable_for_gamma=is_suitable_gamma,
        is_suitable_for_pearson3=is_suitable_pearson3,
        warnings=warnings_list,
        recommendation=recommendation
    )


def diagnose_fitting_failure(
    values: np.ndarray,
    distribution: DistributionType,
    params: DistributionParams
) -> str:
    """
    Diagnose why distribution fitting failed or produced invalid parameters.

    :param values: original data values
    :param distribution: distribution type that was attempted
    :param params: resulting parameters (may be invalid)
    :return: diagnostic message explaining the failure
    """
    diag = diagnose_data(values)

    if params.is_valid():
        return "Fitting succeeded - no failure to diagnose."

    # Build diagnostic message
    reasons = []

    if diag.n_valid < MIN_VALUES_FOR_FIT:
        reasons.append(f"insufficient data ({diag.n_valid} < {MIN_VALUES_FOR_FIT})")

    if diag.zero_proportion > MAX_ZERO_PROPORTION:
        reasons.append(f"excessive zeros ({diag.zero_proportion:.1%})")

    if diag.n_positive < MIN_NONZERO_VALUES:
        reasons.append(f"too few positive values ({diag.n_positive})")

    if diag.variance < MIN_VARIANCE:
        reasons.append(f"near-constant data (var={diag.variance:.2e})")

    if distribution == DistributionType.GAMMA and diag.n_negative > 0:
        reasons.append(f"negative values present ({diag.n_negative})")

    if not reasons:
        reasons.append("numerical issues during parameter estimation")

    return f"Fitting failed for {distribution.value}: " + ", ".join(reasons)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _invalid_params(
    distribution: DistributionType,
    method: FittingMethod,
    prob_zero: float = 0.0,
    n_samples: int = 0
) -> DistributionParams:
    """Create invalid parameters object for failed fits."""
    # Default NaN params based on distribution
    if distribution == DistributionType.GAMMA:
        params = {'alpha': np.nan, 'beta': np.nan}
    elif distribution == DistributionType.PEARSON3:
        params = {'skew': np.nan, 'loc': np.nan, 'scale': np.nan}
    elif distribution == DistributionType.LOG_LOGISTIC:
        params = {'alpha': np.nan, 'beta': np.nan}
    elif distribution in (DistributionType.GEV, DistributionType.GEN_LOGISTIC):
        params = {'shape': np.nan, 'loc': np.nan, 'scale': np.nan}
    else:
        params = {}

    return DistributionParams(
        distribution=distribution,
        params=params,
        prob_zero=prob_zero,
        n_samples=n_samples,
        fitting_method=method
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Enums
    'DistributionType',
    'FittingMethod',

    # Data classes
    'DistributionParams',
    'GoodnessOfFit',
    'DataDiagnostics',

    # Main functions
    'fit_distribution',
    'compute_cdf',
    'cdf_to_standard_normal',

    # Individual fitting functions
    'fit_gamma',
    'fit_pearson3',
    'fit_log_logistic',
    'fit_gev',
    'fit_gen_logistic',

    # Individual CDF functions
    'gamma_cdf',
    'pearson3_cdf',
    'log_logistic_cdf',
    'gev_cdf',
    'gen_logistic_cdf',

    # L-moments
    'compute_lmoments',
    'compute_lmoment_ratios',

    # Comparison and selection
    'test_goodness_of_fit',
    'compare_distributions',
    'select_best_distribution',

    # Diagnostics
    'diagnose_data',
    'diagnose_fitting_failure',

    # Constants
    'RECOMMENDED_DISTRIBUTIONS',
    'MIN_VALUES_FOR_FIT',
    'MIN_NONZERO_VALUES',
    'MAX_ZERO_PROPORTION',
]
