# Quick Start Guide

Get started with SPI and SPEI calculations in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/precip-index.git
cd precip-index

# Install dependencies
pip install -r requirements.txt
```

## Setup Python Path

Before using the package, add the `src` folder to your Python path:

```python
import sys
sys.path.insert(0, 'src')  # Or use absolute path: '/path/to/precip-index/src'
```

## Basic Usage

### 1. Calculate SPI

```python
import sys
sys.path.insert(0, 'src')

import xarray as xr
from indices import spi

# Load your precipitation data
ds = xr.open_dataset('your_precip_data.nc')
precip = ds['precip']  # or 'precipitation', 'prcp', 'pr'

# Calculate SPI-12 (12-month scale)
spi_12 = spi(
    precip,
    scale=12,
    periodicity='monthly',
    calibration_start_year=1991,
    calibration_end_year=2020
)

# Save results
from indices import save_index_to_netcdf
save_index_to_netcdf(spi_12, 'spi_12_output.nc')
```

### 2. Calculate SPEI

```python
from indices import spei

# Load precipitation and temperature
precip = ds['precip']
temp = ds['temperature']
lat = ds['lat']

# Calculate SPEI-12 (auto-calculates PET from temperature)
spei_12 = spei(
    precip,
    temperature=temp,
    latitude=lat,
    scale=12,
    periodicity='monthly',
    calibration_start_year=1991,
    calibration_end_year=2020
)

save_index_to_netcdf(spei_12, 'spei_12_output.nc')
```

### 3. Multi-Scale Calculation

```python
from indices import spi_multi_scale

# Calculate SPI for multiple time scales at once
spi_multi = spi_multi_scale(
    precip,
    scales=[1, 3, 6, 12],
    periodicity='monthly',
    calibration_start_year=1991,
    calibration_end_year=2020
)

# Access individual scales
spi_1 = spi_multi['spi_gamma_1_month']
spi_3 = spi_multi['spi_gamma_3_month']
spi_12 = spi_multi['spi_gamma_12_month']
```

### 4. Save and Reuse Parameters

```python
from indices import save_fitting_params, load_fitting_params

# Calculate SPI and save parameters
spi_12, params = spi(precip, scale=12, return_params=True)

# Save parameters for future use
save_fitting_params(
    params,
    'spi_params.nc',
    scale=12,
    periodicity='monthly',
    index_type='spi'
)

# Later: Load and reuse parameters (much faster!)
params = load_fitting_params('spi_params.nc', scale=12, periodicity='monthly')
spi_12_fast = spi(new_precip, scale=12, fitting_params=params)
```

### 5. Climate Extremes Event Analysis

Identify and analyze complete extreme events using run theory:

```python
from runtheory import identify_events, calculate_period_statistics
from visualization import plot_events

# Identify drought events (negative threshold)
spi_location = spi_12.isel(lat=50, lon=100)
drought_events = identify_events(spi_location, threshold=-1.2, min_duration=3)

print(f"Found {len(drought_events)} drought events")
print(drought_events[['start_date', 'end_date', 'duration', 'magnitude', 'peak']])

# Or identify wet events (positive threshold) - same function!
wet_events = identify_events(spi_location, threshold=+1.2, min_duration=3)

# Visualize events
from visualization import plot_events
plot_events(spi_location, drought_events, threshold=-1.2)

# Gridded statistics for a time period
stats = calculate_period_statistics(spi_12, threshold=-1.2,
                                    start_year=2020, end_year=2024)
stats.num_events.plot(title='Number of Drought Events 2020-2024')
```

## Data Requirements

### Format
- NetCDF files (.nc)
- Dimensions: Any order is supported (auto-transposes to CF Convention)
  - Preferred: `(time, lat, lon)`
  - Also works: `(lat, lon, time)` ← Auto-detected and transposed!
- Monthly or daily temporal resolution

### Variables
- **For SPI:**
  - Precipitation in mm/month or mm/day

- **For SPEI:**
  - Precipitation in mm/month or mm/day
  - Temperature in °C (for PET calculation)
  - OR pre-computed PET in mm/month or mm/day

### Calibration Period
- Default: 1991-2020 (WMO recommendation)
- Minimum: 30 years for robust statistics
- Should overlap with your data period

## Understanding the Output

### SPI/SPEI Values
| Value Range | Category | Meaning |
|-------------|----------|---------|
| ≥ 2.0 | Extremely wet | Top 2.3% |
| 1.5 to 2.0 | Very wet | Top 6.7% |
| 1.0 to 1.5 | Moderately wet | Top 15.9% |
| -1.0 to 1.0 | Near normal | Middle 68.2% |
| -1.5 to -1.0 | Moderately dry | Bottom 15.9% |
| -2.0 to -1.5 | Severely dry | Bottom 6.7% |
| ≤ -2.0 | Extremely dry | Bottom 2.3% |

### Time Scales
- **1-month:** Short-term meteorological drought
- **3-month:** Seasonal agricultural drought
- **6-month:** Medium-term agricultural/hydrological drought
- **12-month:** Long-term water resources/hydrological drought

## Common Issues

### Issue: "ValueError: cannot reshape array"
**Solution:** This was a bug, now fixed! The code automatically handles any dimension order.

### Issue: "ModuleNotFoundError: No module named 'indices'"
**Solution:** Make sure to add the src directory to Python path:
```python
import sys
sys.path.insert(0, 'src')
```

### Issue: All NaN output
**Possible causes:**
- Calibration period doesn't overlap with data
- All zero or negative precipitation values
- Check: `precip.min()`, `precip.max()`, data year range

## Run the Test Script

```bash
# Test with your own data
python test_spi.py
```

This will:
1. Load CHIRPS data from `input/` folder
2. Calculate SPI-3 and multi-scale SPI
3. Save results to `output/` folder
4. Display statistics and validation

## Learn More

- 📓 **Jupyter Notebooks:** See `notebook/` for detailed tutorials
  - `01_calculate_spi.ipynb` - Complete SPI guide
  - `02_calculate_spei.ipynb` - Complete SPEI guide

- 📖 **Documentation:** See `README.md` for full API reference

- 🧪 **Test Results:** See `TEST_RESULTS.md` for validation with real data

- 📝 **Changes:** See `CHANGELOG.md` for version history

## Support

Found a bug or have a question?
1. Check the Jupyter notebooks for examples
2. Review the test results
3. Open an issue on GitHub

---

**Ready to start? Try the test script or open the Jupyter notebooks!**
