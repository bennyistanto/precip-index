# Test Suite for precip-index Package

This directory contains integration tests for the `precip-index` package using **real TerraClimate data from Bali, Indonesia (1958-2024)**.

## Test Data

All tests use NetCDF files from the `input/` folder:

| File | Variable | Size | Purpose |
|------|----------|------|---------|
| `terraclimate_bali_ppt_1958_2024.nc` | Precipitation | 848 KB | SPI calculation |
| `terraclimate_bali_tmean_1958_2024.nc` | Temperature | 1.7 MB | SPEI (PET from temp) |
| `terraclimate_bali_pet_1958_2024.nc` | PET | 778 KB | SPEI (pre-computed PET) |

**Dataset Information:**
- **Source**: TerraClimate (http://www.climatologylab.org/terraclimate.html)
- **Location**: Bali, Indonesia (8-9°S, 114-116°E)
- **Period**: January 1958 - December 2024 (67 years)
- **Resolution**: ~4km (1/24 degree)
- **Grid Size**: 3 lat × 4 lon (12 grid points)

## Test Files

### 1. `test_spi.py`
Tests SPI (Standardized Precipitation Index) calculation.

**What it tests:**
- Loading TerraClimate precipitation data
- Single-scale SPI calculation (SPI-3)
- Multi-scale SPI calculation (SPI-3, 6, 9, 12)
- Parameter saving
- NetCDF output

**Output:**
- `test_output/spi_3_bali_test.nc`
- `test_output/spi_multi_bali_test.nc`

**Run:**
```bash
python tests/test_spi.py
```

### 2. `test_spei_with_pet.py`
Tests SPEI (Standardized Precipitation Evapotranspiration Index) calculation.

**What it tests:**
- SPEI with pre-computed PET
- SPEI with temperature (auto-compute PET using Thornthwaite)
- Comparison between both methods
- PET calculation validation

**Output:**
- `test_output/spei_6_with_pet_test.nc`
- `test_output/spei_6_with_temp_test.nc`
- `test_output/pet_thornthwaite_bali_test.nc`

**Run:**
```bash
python tests/test_spei_with_pet.py
```

### 3. `test_drought_characteristics.py`
Minimal test of run theory functions - verifies functionality without creating extensive outputs.

**What it tests:**
- Event identification (dry and wet)
- Time-series calculation
- Event summarization
- Event state detection
- Period statistics (single period)

**Output:**
- None - this is a minimal functionality test

**Run:**
```bash
python tests/test_drought_characteristics.py
```

### 4. `test_complete_analysis.py`
Comprehensive test of all renamed functions with bidirectional thresholds.

**What it tests:**
- All run theory functions (dry and wet)
- All visualization functions
- Package neutralization (drought → event terminology)
- Function consistency across positive/negative thresholds

**Output:**
- 2 minimal test plots in `test_output/`

**Run:**
```bash
python tests/test_complete_analysis.py
```

### 5. `run_all_tests.py`
Master script that runs all tests in sequence.

**Features:**
- Checks for required input data
- Runs all 4 test files
- Reports pass/fail status
- Total elapsed time

**Run:**
```bash
python tests/run_all_tests.py
```

## Quick Start

### Prerequisites

1. **Install dependencies:**
```bash
pip install numpy scipy xarray netcdf4 numba matplotlib
```

2. **Ensure input data exists:**
The `input/` folder should contain the three TerraClimate Bali NetCDF files.

### Run All Tests

```bash
# From repository root
python tests/run_all_tests.py
```

Expected output:
```
==================================================================
PRECIP-INDEX TEST SUITE
==================================================================
Running all integration tests...
Dataset: TerraClimate Bali (1958-2024)

Checking input data files...
  ✓ input/terraclimate_bali_ppt_1958_2024.nc (0.8 MB)
  ✓ input/terraclimate_bali_tmean_1958_2024.nc (1.7 MB)
  ✓ input/terraclimate_bali_pet_1958_2024.nc (0.8 MB)

... [tests run] ...

✅ All tests passed!
```

## License

BSD 3-Clause License - See LICENSE file in repository root
