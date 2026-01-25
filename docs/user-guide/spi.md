# SPI - Standardized Precipitation Index

> **Note:** This guide references "drought" in some sections for historical context with meteorological literature, but SPI is a bidirectional index that monitors **both dry and wet extremes** equally. All analysis functions work for both directions.

## Overview

The Standardized Precipitation Index (SPI) is a widely used climate index that characterizes both dry (drought) and wet (flood/excess) conditions based on precipitation.

**SPI Values:**

- **Negative values:** Indicate dry conditions (drought)
- **Positive values:** Indicate wet conditions (flooding/excess precipitation)

**Key Features:**

- Based on precipitation only
- Multiple time scales (1, 3, 6, 12, 24 months)
- Gamma distribution fitting
- CF-compliant NetCDF output
- **Monitors both drought and wet extremes**

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

import xarray as xr
from indices import spi

# Load precipitation data
precip = xr.open_dataset('input/precipitation.nc')['precip']

# Calculate SPI-12 (monitors both dry and wet conditions)
spi_12 = spi(precip, scale=12, distribution='gamma')

# Negative values = drought (dry)
# Positive values = wet conditions (flooding/excess)

# Save output
spi_12.to_netcdf('output/netcdf/spi_12.nc')
```

## Parameters

### Required

- **`values`** (xarray.DataArray): Precipitation data
  - Dimensions: `(time, lat, lon)` - CF convention
  - Units: mm/month (or any consistent unit)
  - Missing values: NaN supported

### Optional

- **`scale`** (int): Time scale in months
  - Default: 3
  - Common values: 1, 3, 6, 12, 24
  - Range: 1-72 months

- **`distribution`** (str): Statistical distribution
  - Default: 'gamma'
  - Options: 'gamma', 'pearson'
  - Recommendation: Use 'gamma' for precipitation

- **`data_start_year`** (int): Calibration start year
  - Default: 1991
  - Use for climate normals (e.g., 1991-2020)

- **`data_end_year`** (int): Calibration end year
  - Default: 2020
  - WMO recommends 30-year periods

- **`calibration_year_initial`** (int): Initial calibration start
  - Default: Same as `data_start_year`

- **`calibration_year_final`** (int): Final calibration end
  - Default: Same as `data_end_year`

- **`periodicity`** (Periodicity): Monthly or daily
  - Default: Periodicity.monthly
  - Options: Periodicity.monthly, Periodicity.daily

## Output

**Returns:** xarray.DataArray with:

- **Dimensions:** `(time, lat, lon)`
- **Values:** Standardized index (-3 to +3 typically)
- **Attributes:**
  - `long_name`: "Standardized Precipitation Index"
  - `scale`: Time scale used
  - `distribution`: Distribution type
  - `calibration_period`: Years used for calibration

**SPI Value Interpretation:**

| SPI Value | Category | Probability |
| ----------- | -------- | ----------- |
| ≤ -2.00 | Exceptionally Dry | Bottom 2.3% |
| -2.00 to -1.50 | Extremely Dry | Bottom 6.7% |
| -1.50 to -1.20 | Severely Dry | Bottom 9.7% |
| -1.20 to -0.70 | Moderately Dry | Bottom 24.2% |
| -0.70 to -0.50 | Abnormally Dry | Bottom 30.9% |
| -0.50 to +0.50 | Near Normal | Middle 38.2% |
| +0.50 to +0.70 | Abnormally Moist | Top 30.9% |
| +0.70 to +1.20 | Moderately Moist | Top 24.2% |
| +1.20 to +1.50 | Very Moist | Top 9.7% |
| +1.50 to +2.00 | Extremely Moist | Top 6.7% |
| ≥ +2.00 | Exceptionally Moist | Top 2.3% |

## Examples

### Example 1: Single Time Scale

```python
import xarray as xr
from indices import spi

# Load CHIRPS data
precip = xr.open_dataset('input/chirps_monthly.nc')['precip']

# Calculate SPI-12
spi_12 = spi(precip, scale=12, distribution='gamma',
             data_start_year=1991, data_end_year=2020)

# Save
spi_12.to_netcdf('output/netcdf/spi_12.nc')

# Quick preview
print(spi_12)
spi_12.isel(time=-1).plot()
```

### Example 2: Multiple Time Scales

```python
from indices import spi_multi_scale

# Calculate SPI for multiple scales
scales = [1, 3, 6, 12, 24]
spi_multi = spi_multi_scale(precip, scales=scales, distribution='gamma')

# Access different scales
spi_1 = spi_multi['spi_gamma_1_month']
spi_3 = spi_multi['spi_gamma_3_month']
spi_12 = spi_multi['spi_gamma_12_month']

# Save all
spi_multi.to_netcdf('output/netcdf/spi_multi.nc')
```

### Example 3: With Parameter Saving

```python
from indices import spi, save_fitting_params

# Calculate SPI and save fitting parameters
spi_12, params = spi(precip, scale=12, return_params=True)

# Save parameters for reuse
save_fitting_params(params, 'spi_12_params.nc',
                    scale=12, periodicity='monthly')

# Later: Load and reuse parameters
from indices import load_fitting_params
params_loaded = load_fitting_params('spi_12_params.nc')
```

## Time Scales Guide

### SPI-1 (1-month)

- **Use:** Short-term precipitation anomalies
- **Applications:** Agricultural drought (growing season), flash droughts
- **Responds to:** Recent rainfall

### SPI-3 (3-month)

- **Use:** Seasonal precipitation patterns
- **Applications:** Agricultural drought, soil moisture
- **Responds to:** Short-term dry spells

### SPI-6 (6-month)

- **Use:** Medium-term precipitation trends
- **Applications:** Agricultural + hydrological drought
- **Responds to:** Seasonal to inter-seasonal patterns

### SPI-12 (12-month)

- **Use:** Annual precipitation patterns
- **Applications:** Hydrological drought, reservoir levels
- **Responds to:** Long-term trends

### SPI-24 (24-month)

- **Use:** Multi-year precipitation trends
- **Applications:** Long-term water resource planning
- **Responds to:** Multi-year dry periods

## Best Practices

### 1. Calibration Period

- Use WMO-recommended 30-year normals (e.g., 1991-2020)
- Ensure calibration period has complete data
- Update calibration periodically

### 2. Quality Control

```python
# Check for missing data
missing_pct = (precip.isnull().sum() / precip.size * 100).values
print(f"Missing data: {missing_pct:.2f}%")

# Check data range
print(f"Precip range: [{precip.min().values:.1f}, {precip.max().values:.1f}] mm")
```

### 3. Dimension Order

Data must follow CF convention: `(time, lat, lon)`

```python
from utils import ensure_cf_compliant

# Auto-fix dimension order
precip = ensure_cf_compliant(precip)
```

## Common Issues

### Issue 1: Dimension Order Error

**Problem:** `ValueError: cannot reshape array...`    
**Solution:**

```python
# Check dimensions
print(precip.dims)  # Should be ('time', 'lat', 'lon')

# Fix if needed
if precip.dims != ('time', 'lat', 'lon'):
    precip = precip.transpose('time', 'lat', 'lon')
```

### Issue 2: All NaN Output

**Problem:** SPI values are all NaN    
**Causes:**

- Insufficient calibration data
- All-zero precipitation (arid regions)
- Wrong units (need mm/month, not mm/day)

**Solution:**

```python
# Check precipitation values
print(precip.sel(time='2020').mean(['lat', 'lon']).values)

# Should be 30-300 mm/month for most regions
# If values are 1-10, likely mm/day - convert to monthly
```

### Issue 3: Memory Error

**Problem:** Out of memory for large datasets    
**Solution:** Use Dask-enabled version

```python
# Load with chunks
precip = xr.open_dataset('precip.nc', chunks={'time': 100})['precip']

# Calculate with Dask
spi_12 = spi(precip, scale=12)  # Automatically uses Dask if input is chunked
```

## Visualization

### Quick Map

```python
import matplotlib.pyplot as plt

# Latest month
spi_12.isel(time=-1).plot(cmap='RdBu', vmin=-3, vmax=3)
plt.title('SPI-12 (Latest Month)')
plt.show()
```

### Time Series

```python
# Single location
location_spi = spi_12.isel(lat=50, lon=100)
location_spi.plot()
plt.axhline(y=-1.2, color='red', linestyle='--', label='Drought threshold')
plt.title('SPI-12 Time Series')
plt.legend()
plt.show()
```

### Using Built-in Functions

```python
from visualization import plot_index

# Color-coded time series
plot_index(location_spi, threshold=-1.2,
           title='SPI-12 Climate Analysis')
```

## Performance Tips

### 1. Multi-Scale Calculation

Use `spi_multi_scale()` instead of calling `spi()` multiple times:

```python
# Better (single pass)
spi_multi = spi_multi_scale(precip, scales=[3, 6, 12])

# Slower (three passes)
spi_3 = spi(precip, scale=3)
spi_6 = spi(precip, scale=6)
spi_12 = spi(precip, scale=12)
```

### 2. Parameter Reuse

Save and reuse fitting parameters for operational forecasting:

```python
# One-time: Calculate and save parameters
spi_historical, params = spi(precip_historical, scale=12, return_params=True)
save_fitting_params(params, 'spi_12_params.nc')

# Operational: Reuse parameters
params = load_fitting_params('spi_12_params.nc')
spi_latest = spi(precip_latest, scale=12, fitting_params=params)
```

### 3. Regional Subsets

Process smaller regions when possible:

```python
# Subset to region of interest
precip_morocco = precip.sel(lat=slice(27, 36), lon=slice(-13, -1))
spi_12 = spi(precip_morocco, scale=12)
```

## References

- McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought frequency and duration to time scales. 8th Conference on Applied Climatology.

- WMO (2012). Standardized Precipitation Index User Guide. (WMO-No. 1090), Geneva.

- Lloyd‐Hughes, B., & Saunders, M. A. (2002). A drought climatology for Europe. International Journal of Climatology, 22(13), 1571-1592.

## See Also

- [SPEI Guide](spei.md) - For temperature-influenced drought
- [Drought Characteristics](runtheory.md) - Event analysis
- [Visualization Guide](visualization.md) - Plotting options
