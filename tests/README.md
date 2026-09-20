# Test Suite for precip-index Package

This directory contains a structured integration test suite for the `precip-index` package using **real TerraClimate data from Bali, Indonesia (1950-2025)**.

## Test Structure

```
tests/
├── conftest.py              # Shared utilities and configuration
├── run_all_tests.py         # Master test runner
│
├── 01_data_quality.py       # Data loading & quality checks
├── 02_spi_calculation.py    # SPI across distributions & scales
├── 03_pet_comparison.py     # Thornthwaite vs Hargreaves PET
├── 04_spei_calculation.py   # SPEI across methods & distributions
├── 05_spi_spei_comparison.py# SPI vs SPEI analysis
├── 06_visualization.py      # All plots generation
│
└── output/                  # Test outputs (auto-created)
    ├── netcdf/              # Calculated indices
    ├── plots/               # Visualization outputs
    └── reports/             # Quality reports
```

## Test Data

All tests use NetCDF files from the `input/` folder:

| File | Variable | Purpose |
|------|----------|---------|
| `terraclimate_bali_ppt_1950_2025.nc` | Precipitation | SPI/SPEI calculation |
| `terraclimate_bali_tmean_1950_2025.nc` | Mean Temperature | PET calculation |
| `terraclimate_bali_tmin_1950_2025.nc` | Min Temperature | Hargreaves PET |
| `terraclimate_bali_tmax_1950_2025.nc` | Max Temperature | Hargreaves PET |
| `terraclimate_bali_pet_1950_2025.nc` | PET (reference) | SPEI, PET comparison |

**Dataset Information:**

- **Source**: TerraClimate v1.1 (http://www.climatologylab.org/terraclimate.html)
- **Location**: Bali, Indonesia (8-9°S, 114-116°E)
- **Period**: January 1950 - December 2025 (912 months, 76 years)
- **Resolution**: ~4km (1/24 degree)

## Test Files

### 01_data_quality.py

Validates input data files and generates quality reports.

**What it tests:**

- File existence and sizes
- Data dimensions and coordinates
- Value ranges and statistics
- Missing data patterns
- Calibration period coverage
- Data readiness for SPI/SPEI

**Output:** `output/reports/01_data_quality_report.txt`

### 02_spi_calculation.py

Tests SPI calculation with multiple distributions and scales.

**What it tests:**

- SPI-3 and SPI-12 calculation
- Gamma, Pearson III, Log-Logistic distributions
- Statistical validation (mean≈0, std≈1)
- Cross-distribution correlation

**Output:** `output/netcdf/spi_*.nc`

### 03_pet_comparison.py

Compares PET calculation methods.

**What it tests:**

- Thornthwaite PET (temperature-based)
- Hargreaves-Samani PET (Tmin/Tmax-based)
- Comparison with TerraClimate Penman-Monteith
- Correlation and bias analysis

**Output:**

- `output/netcdf/pet_thornthwaite.nc`
- `output/netcdf/pet_hargreaves.nc`
- `output/plots/pet_comparison_*.png`

### 04_spei_calculation.py

Tests SPEI with different PET methods and distributions.

**What it tests:**

- SPEI with pre-computed PET
- SPEI with Thornthwaite
- SPEI with Hargreaves
- SPEI-3 and SPEI-12 scales
- Multiple distributions

**Output:** `output/netcdf/spei_*.nc`

### 05_spi_spei_comparison.py

Analyzes differences between SPI and SPEI.

**What it tests:**

- SPI vs SPEI correlation
- Drought detection agreement
- Impact of temperature trends
- Drought area percentage comparison

**Output:**

- `output/reports/05_spi_spei_comparison_report.txt`
- `output/plots/spi_spei_*.png`

### 06_visualization.py

Generates comprehensive visualizations.

**What it tests:**

- WMO color scheme time series
- Distribution comparison plots
- Spatial maps
- Summary figures
- PET method comparison plots

**Output:** `output/plots/*.png`

## Quick Start

### Prerequisites

1. **Install dependencies:**

```bash
pip install numpy scipy xarray netcdf4 numba matplotlib
```

2. **Ensure input data exists:**

The `input/` folder should contain the TerraClimate Bali NetCDF files.

### Run All Tests

```bash
# From repository root
python tests/run_all_tests.py
```

### Run Individual Tests

```bash
# Data quality check
python tests/01_data_quality.py

# SPI calculation
python tests/02_spi_calculation.py

# PET comparison
python tests/03_pet_comparison.py

# SPEI calculation
python tests/04_spei_calculation.py

# SPI vs SPEI comparison
python tests/05_spi_spei_comparison.py

# Visualization
python tests/06_visualization.py
```

## Expected Output

```
======================================================================
 PRECIP-INDEX TEST SUITE
======================================================================
Running integration tests with TerraClimate Bali data (1950-2025)
Source: input/ folder

======================================================================
 CHECKING INPUT DATA
======================================================================
  [OK] Precipitation (required): terraclimate_bali_ppt_1950_2025.nc (0.8 MB)
  [OK] Temperature mean (required): terraclimate_bali_tmean_1950_2025.nc (1.7 MB)
  [OK] PET (optional): terraclimate_bali_pet_1950_2025.nc (0.8 MB)
  [OK] Temperature min (for Hargreaves): terraclimate_bali_tmin_1950_2025.nc (1.7 MB)
  [OK] Temperature max (for Hargreaves): terraclimate_bali_tmax_1950_2025.nc (1.7 MB)

======================================================================
 RUNNING TESTS
======================================================================
Found 6 test files:
  - 01_data_quality.py
  - 02_spi_calculation.py
  - 03_pet_comparison.py
  - 04_spei_calculation.py
  - 05_spi_spei_comparison.py
  - 06_visualization.py

... [tests run] ...

======================================================================
 TEST SUMMARY
======================================================================
  [OK]   01_data_quality.py (2.1s)
  [OK]   02_spi_calculation.py (45.3s)
  [OK]   03_pet_comparison.py (12.8s)
  [OK]   04_spei_calculation.py (89.2s)
  [OK]   05_spi_spei_comparison.py (34.5s)
  [OK]   06_visualization.py (28.7s)
----------------------------------------------------------------------
  Total: 6/6 tests passed
  Elapsed time: 212.6s
======================================================================

Outputs created:
  - NetCDF files: 24
  - Plot files: 15
  - Report files: 5
  - Location: tests/output

[OK] All tests passed!
```

## Configuration

Test parameters are defined in `conftest.py`:

```python
# Calibration period (WMO standard)
CALIBRATION_START = 1991
CALIBRATION_END = 2020

# Data start year
DATA_START_YEAR = 1950

# Distributions to test
TEST_DISTRIBUTIONS = ['gamma', 'pearson3', 'log_logistic']

# Scales to test
TEST_SCALES = [3, 12]
```

## Legacy Test Files

The following legacy test files are preserved for backward compatibility but are not part of the main test suite:

- `test_spi.py` - Original SPI test
- `test_spei_with_pet.py` - Original SPEI test
- `test_drought_characteristics.py` - Event analysis test
- `test_complete_analysis.py` - Comprehensive function test
- `test_hargreaves_pet.py` - Hargreaves implementation test

## License

BSD 3-Clause License - See LICENSE file in repository root
