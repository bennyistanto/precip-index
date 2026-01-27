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

## Global-Scale Processing

For large datasets (e.g., global CHIRPS, ERA5), use memory-efficient chunked processing:

```python
from indices import spi_global, estimate_memory_requirements

# First, estimate memory requirements
mem = estimate_memory_requirements('global_chirps_monthly.nc')
print(f"Input size: {mem['input_size_gb']:.1f} GB")
print(f"Peak memory: {mem['peak_memory_gb']:.1f} GB")
print(f"Recommended chunk: {mem['recommended_chunk_size']}")

# Process with automatic chunking
result = spi_global(
    'global_chirps_monthly.nc',
    'spi_12_global.nc',
    scale=12,
    calibration_start_year=1991,
    calibration_end_year=2020,
    chunk_size=500,  # Adjust based on available RAM
    save_params=True  # Save parameters for reuse
)
```

::: {.callout-tip}
## Chunk Size Guidelines

| Available RAM | Recommended Chunk Size |
|--------------|------------------------|
| 16 GB | 200 × 200 |
| 32 GB | 300 × 300 |
| 64 GB | 400 × 400 |
| 128 GB | 600 × 600 |

Larger chunks = faster processing, but require more memory.
:::

For more control, use the `ChunkedProcessor` class:

```python
from chunked import ChunkedProcessor

processor = ChunkedProcessor(chunk_lat=500, chunk_lon=500)
result = processor.compute_spi_chunked(
    precip='global_precip.nc',
    output_path='spi_12_global.nc',
    scale=12,
    save_params=True,
    callback=lambda cur, tot, msg: print(f"[{cur}/{tot}] {msg}")
)
```

See the [Global-Scale Processing Tutorial](../tutorials/05-global-scale-processing.qmd) for detailed examples.

## Climate Extremes Event Analysis

Identify and analyze complete extreme events using run theory.

The same functions work for both drought (negative threshold) and wet events (positive threshold). This unified approach makes analysis consistent and straightforward.

### Drought Events

```python
from runtheory import identify_events
from visualization import plot_events

# Identify drought events (negative threshold)
spi_location = spi_12.isel(lat=50, lon=100)
drought_events = identify_events(spi_location, threshold=-1.2, min_duration=3)

print(f"Found {len(drought_events)} drought events")
print(drought_events[['start_date', 'end_date', 'duration', 'magnitude', 'peak']])

# Visualize events
plot_events(spi_location, drought_events, threshold=-1.2)
```

### Wet Events

```python
# Identify wet events (positive threshold) - same function!
wet_events = identify_events(spi_location, threshold=+1.2, min_duration=3)

print(f"Found {len(wet_events)} wet events")
plot_events(spi_location, wet_events, threshold=+1.2)
```

### Gridded Statistics

```python
from runtheory import calculate_period_statistics

# Calculate gridded statistics for a time period
stats = calculate_period_statistics(
    spi_12,
    threshold=-1.2,
    start_year=2020,
    end_year=2024
)

# Plot number of events per grid cell
stats.num_events.plot(title='Number of Drought Events 2020-2024')
```

## Data Requirements

### Format

- **File Type:** NetCDF files (.nc)
- **Dimensions:** Any order is supported (auto-transposes to CF Convention)
  - Preferred: `(time, lat, lon)`
  - Also works: `(lat, lon, time)` ← Auto-detected and transposed!
- **Temporal Resolution:** Monthly or daily

### Variables

- **For SPI:**

  - Precipitation in mm/month or mm/day

- **For SPEI:**

  - Precipitation in mm/month or mm/day
  - Temperature in °C (for PET calculation)
  - OR pre-computed PET in mm/month or mm/day

### Calibration Period

- **Default:** 1991-2020 (WMO recommendation)
- **Minimum:** 30 years for robust statistics
- **Requirement:** Should overlap with your data period

## Understanding the Output

### SPI/SPEI Classification

| Value Range | Category | Probability | Interpretation |
|-------------|----------|-------------|----------------|
| ≤ -2.00 | Exceptionally Dry | Bottom 2.3% | Exceptional drought |
| -2.00 to -1.50 | Extremely Dry | Bottom 6.7% | Severe drought |
| -1.50 to -1.20 | Severely Dry | Bottom 9.7% | Severe drought |
| -1.20 to -0.70 | Moderately Dry | Bottom 24.2% | Moderate drought |
| -0.70 to -0.50 | Abnormally Dry | Bottom 30.9% | Below normal |
| -0.50 to +0.50 | Near Normal | Middle 38.2% | Normal conditions |
| +0.50 to +0.70 | Abnormally Moist | Top 30.9% | Above normal |
| +0.70 to +1.20 | Moderately Moist | Top 24.2% | Moderately wet |
| +1.20 to +1.50 | Very Moist | Top 9.7% | Very wet conditions |
| +1.50 to +2.00 | Extremely Moist | Top 6.7% | Extremely wet conditions |
| ≥ +2.00 | Exceptionally Moist | Top 2.3% | Extreme flooding risk |

### Time Scales

| Scale | Application | Use Case |
|-------|-------------|----------|
| 1-month | Meteorological | Short-term precipitation deficits |
| 3-month | Agricultural | Seasonal crop stress |
| 6-month | Agricultural/Hydrological | Medium-term water resources |
| 12-month | Hydrological | Long-term water supply |
| 24-month | Socio-economic | Multi-year water planning |

## Common Issues

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

Test with the included example data:

```bash
python tests/run_all_tests.py
```

This will:

1. Load TerraClimate Bali data from `input/` folder
2. Calculate SPI-3 and multi-scale SPI
3. Calculate SPEI with temperature
4. Test run theory functions
5. Display statistics and validation

Expected runtime: ~30 seconds

## Learn More

- 📓 **Jupyter Notebooks:** See `notebook/` for detailed tutorials
  - `01_calculate_spi.ipynb` - Complete SPI guide
  - `02_calculate_spei.ipynb` - Complete SPEI guide

- 📖 **Documentation:** See `README.md` for full API reference

- 📝 **Changes:** See `CHANGELOG.md` for version history

## Support

Found a bug or have a question?

1. Check the Jupyter notebooks for examples
2. Review the test results
3. Open an issue on GitHub

---

**Ready to start? Try the test script or open the Jupyter notebooks!**
