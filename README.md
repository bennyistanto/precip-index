# Precipitation Index (SPI and SPEI)

A minimal, efficient Python implementation of **Standardized Precipitation Index (SPI)** and **Standardized Precipitation Evapotranspiration Index (SPEI)** for drought monitoring.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

This project is a streamlined version of the original [climate-indices](https://github.com/monocongo/climate-indices) package by James Adams (monocongo), redesigned with the following goals:

| Feature | This Package | Original |
|---------|--------------|----------|
| **Indices** | SPI, SPEI only | SPI, SPEI, PET, PDSI, PHDI, PMDI, Z-Index, PNP |
| **Distribution** | Gamma only | Gamma + Pearson Type III |
| **Interface** | Notebook/API first | CLI first |
| **Data format** | CF Convention (time, lat, lon) | Flexible |
| **Parameters** | Save/load support | Save/load support |
| **Optimization** | NumPy vectorized + Numba JIT | Multiprocessing |

### Why Gamma Only?

The Pearson Type III distribution often fails on global datasets due to:
- L-moments estimation errors in arid regions (division by zero)
- Insufficient variability when many values are zero/near-zero
- Numerical instability with extreme skewness

**Gamma distribution is the original SPI distribution** (McKee et al., 1993) and is robust for global coverage, including arid regions.

## Installation

### Requirements

```bash
pip install numpy scipy xarray netcdf4 numba
```

### Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/precip-index.git
cd precip-index
```

Add the `src` folder to your Python path in notebooks:

```python
import sys
sys.path.insert(0, '/path/to/precip-index/src')
```

## Quick Start

### Calculate SPI

```python
import xarray as xr
from indices import spi, save_fitting_params

# Load precipitation data (CF Convention: time, lat, lon)
ds = xr.open_dataset('precipitation.nc')
precip = ds['precip']

# Calculate SPI-12 with parameter saving
spi_12, params = spi(
    precip,
    scale=12,
    periodicity='monthly',
    calibration_start_year=1991,
    calibration_end_year=2020,
    return_params=True
)

# Save parameters for future use
save_fitting_params(
    params, 
    'spi_gamma_params_12_month.nc',
    scale=12,
    periodicity='monthly',
    index_type='spi',
    calibration_start_year=1991,
    calibration_end_year=2020,
    coords={'lat': precip.lat, 'lon': precip.lon}
)
```

### Calculate SPEI

```python
from indices import spei

# Option 1: With pre-computed PET
spei_12 = spei(precip, pet=pet_da, scale=12)

# Option 2: With temperature (auto-compute PET using Thornthwaite)
spei_12 = spei(
    precip,
    temperature=temp_da,
    latitude=lat_da,
    scale=12
)
```

### Reuse Saved Parameters

```python
from indices import spi, load_fitting_params

# Load previously saved parameters
params = load_fitting_params('spi_gamma_params_12_month.nc', scale=12, periodicity='monthly')

# Calculate SPI with pre-computed parameters (faster)
spi_12 = spi(new_precip_data, scale=12, fitting_params=params)
```

### Multi-Scale Calculation

```python
from indices import spi_multi_scale, spei_multi_scale

# Calculate SPI for multiple scales at once
spi_ds = spi_multi_scale(precip, scales=[1, 3, 6, 12])

# Access individual scales
spi_1 = spi_ds['spi_gamma_1_month']
spi_12 = spi_ds['spi_gamma_12_month']
```

### Drought Classification

```python
from indices import classify_drought, get_drought_area_percentage

# Classify into drought categories (McKee et al., 1993)
drought_class = classify_drought(spi_12)

# Calculate drought area percentage time series
drought_pct = get_drought_area_percentage(spi_12, threshold=-1.0)
```

## Project Structure

```
precip-index/
├── src/
│   ├── __init__.py      # Package initialization and public API
│   ├── config.py        # Enums, constants, logging setup
│   ├── utils.py         # Data transforms, PET calculation
│   ├── compute.py       # Gamma fitting, parallel processing
│   └── indices.py       # SPI, SPEI functions, save/load params
├── notebook/
│   ├── 01_calculate_spi.ipynb
│   └── 02_calculate_spei.ipynb
├── README.md
├── LICENSE
└── requirements.txt
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `spi(precip, scale, ...)` | Calculate SPI for single scale |
| `spei(precip, pet, scale, ...)` | Calculate SPEI for single scale |
| `spi_multi_scale(precip, scales, ...)` | Calculate SPI for multiple scales |
| `spei_multi_scale(precip, pet, scales, ...)` | Calculate SPEI for multiple scales |

### Parameter I/O

| Function | Description |
|----------|-------------|
| `save_fitting_params(params, filepath, ...)` | Save gamma parameters to NetCDF |
| `load_fitting_params(filepath, scale, periodicity)` | Load gamma parameters from NetCDF |

### Utilities

| Function | Description |
|----------|-------------|
| `save_index_to_netcdf(data, filepath, ...)` | Save results with CF-compliant encoding |
| `classify_drought(index_values)` | Classify into drought categories |
| `get_drought_area_percentage(index_values, threshold)` | Calculate % area under drought |
| `calculate_pet(temperature, latitude, data_start_year)` | Thornthwaite PET calculation |

### Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `FITTED_INDEX_VALID_MIN` | -3.09 | Minimum valid SPI/SPEI value |
| `FITTED_INDEX_VALID_MAX` | 3.09 | Maximum valid SPI/SPEI value |
| `DEFAULT_CALIBRATION_START_YEAR` | 1991 | WMO standard calibration start |
| `DEFAULT_CALIBRATION_END_YEAR` | 2020 | WMO standard calibration end |

## Drought Classification

McKee et al. (1993) classification used by `classify_drought()`:

| Category | SPI/SPEI Range | Code |
|----------|----------------|------|
| Extremely wet | ≥ 2.0 | 4 |
| Very wet | 1.5 to 2.0 | 3 |
| Moderately wet | 1.0 to 1.5 | 2 |
| Near normal | -1.0 to 1.0 | 1 |
| Moderately dry | -1.5 to -1.0 | 0 |
| Severely dry | -2.0 to -1.5 | -1 |
| Extremely dry | ≤ -2.0 | -2 |

## Performance

Optimized for global-scale gridded data:

- **Vectorized NumPy** operations for spatial processing
- **Numba JIT** compilation for core loops
- **Dask support** for out-of-core processing (datasets larger than RAM)
- **CF Convention** compliance (time, lat, lon) — no transpose needed

Typical performance on global CHIRPS data (2400 × 7200 grid):
- SPI-12 calculation: ~10-30 minutes (depending on hardware)
- With pre-computed parameters: ~5-10 minutes

## Credits

This project is based on the excellent [climate-indices](https://github.com/monocongo/climate-indices) package by **James Adams** ([@monocongo](https://github.com/monocongo)).

Key modifications:
- Simplified to SPI and SPEI only
- Gamma distribution only (removed Pearson Type III)
- Notebook-first interface (removed CLI)
- CF Convention compliance
- Enhanced parameter save/load functionality

## References

- McKee, T.B., Doesken, N.J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *8th Conference on Applied Climatology*, American Meteorological Society.

- Vicente-Serrano, S.M., Beguería, S., & López-Moreno, J.I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. *Journal of Climate*, 23(7), 1696-1718.

- Thornthwaite, C.W. (1948). An approach toward a rational classification of climate. *Geographical Review*, 38(1), 55-94.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) file for details.

## Author

**Benny Istanto**    
Climate Geographer    
GOST/DEC Data Group, The World Bank

---

*This package is developed for operational drought monitoring applications. For research requiring multiple distribution options, consider using the original [climate-indices](https://github.com/monocongo/climate-indices) package.*
