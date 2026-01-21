# Tests

This directory contains integration tests for the precip-index package.

## Test Files

### 1. `test_spi.py`
**Tests:** SPI calculation functionality

**What it tests:**
- Basic SPI calculation
- Single and multi-scale SPI
- Gamma distribution fitting
- Parameter saving and loading
- Output validation
- Edge cases

**Run:**
```bash
python tests/test_spi.py
```

**Expected output:**
- SPI NetCDF files in `output/netcdf/`
- Test validation messages
- Execution time

---

### 2. `test_spei_with_pet.py`
**Tests:** SPEI calculation with PET

**What it tests:**
- PET calculation (Thornthwaite method)
- SPEI calculation
- Multi-scale SPEI
- PET-precipitation integration
- Temperature data handling
- Output validation

**Run:**
```bash
python tests/test_spei_with_pet.py
```

**Expected output:**
- PET NetCDF files
- SPEI NetCDF files in `output/netcdf/`
- Validation messages

---

### 3. `test_drought_characteristics.py`
**Tests:** Drought event analysis using run theory

**What it tests:**
- Event identification
- Time-series monitoring
- Period statistics (gridded)
- Annual statistics
- Period comparison
- All 3 analysis modes
- Visualization functions
- Output organization

**Run:**
```bash
python tests/test_drought_characteristics.py
```

**Expected output:**
- CSV files in `output/csv/`
- NetCDF files in `output/netcdf/`
- Plots in `output/plots/single/` and `output/plots/spatial/`
- Comprehensive test summary

---

## Running All Tests

### Sequential Execution
```bash
cd precip-index
python tests/test_spi.py
python tests/test_spei_with_pet.py
python tests/test_drought_characteristics.py
```

### Using Python Module
```bash
cd precip-index
python -m tests.test_spi
python -m tests.test_spei_with_pet
python -m tests.test_drought_characteristics
```

---

## Test Data

**Current approach:** Tests use **synthetic data** generated within each test file

**Why synthetic data?**
- No external dependencies
- Reproducible results
- Fast execution
- Known properties for validation

**Synthetic data specifications:**
- 30 years of monthly data (360 months)
- Gamma-distributed precipitation
- ~10% zero values (dry periods)
- Seasonal cycle included
- Spatial variation included

---

## Expected Test Durations

| Test File | Grid Size | Approximate Time |
|-----------|-----------|------------------|
| `test_spi.py` | 20×30 | 10-15 seconds |
| `test_spei_with_pet.py` | 20×30 | 15-20 seconds |
| `test_drought_characteristics.py` | 165×244* | 2-3 minutes |

*Loads pre-calculated SPI or uses smaller test grid

---

## Output Validation

Each test performs validation checks:

### SPI/SPEI Tests
- ✅ Output dimensions match input
- ✅ Mean ≈ 0 (over calibration period)
- ✅ Std ≈ 1 (over calibration period)
- ✅ Range typically -3 to +3
- ✅ No excessive NaN values
- ✅ Files saved successfully

### Drought Characteristics Tests
- ✅ Events identified correctly
- ✅ Characteristics computed accurately
- ✅ Time series has correct structure
- ✅ Period statistics calculated
- ✅ Visualizations generated
- ✅ Files organized properly

---

## Test Coverage

### Modules Tested
- ✅ `indices.py` - SPI and SPEI calculation
- ✅ `utils.py` - PET calculation and validation
- ✅ `runtheory.py` - All 3 drought analysis modes
- ✅ `visualization.py` - All plot types

### Not Currently Tested (Future Work)
- ⬜ Unit tests for individual functions
- ⬜ Edge cases (all zeros, all NaN, etc.)
- ⬜ Performance benchmarks
- ⬜ Cross-validation with published results
- ⬜ Error handling and exceptions

---

## Test Best Practices

### Before Running Tests
1. Ensure all dependencies installed
2. Check `output/` folder exists (auto-created)
3. Ensure sufficient disk space (~50 MB)

### After Tests Complete
1. Review test output messages
2. Check `output/` folders for generated files
3. Visually inspect sample plots
4. Verify file organization

### Troubleshooting

**Issue:** "Module not found"
```bash
# Solution: Add src to Python path or run from project root
cd precip-index
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%\src         # Windows
```

**Issue:** "File not found"
```bash
# Solution: Ensure running from project root
cd precip-index
python tests/test_spi.py
```

**Issue:** "Out of memory"
```bash
# Solution: Tests use small synthetic data, but if issues:
# - Close other applications
# - Reduce grid size in test file
# - Run tests one at a time
```

---

## Future Enhancements

### Planned Additions
1. **Unit Tests** - Test individual functions in isolation
2. **pytest Framework** - Structured test suite with fixtures
3. **Continuous Integration** - GitHub Actions for automated testing
4. **Coverage Reports** - Track code coverage metrics
5. **Performance Tests** - Benchmark execution times
6. **Regression Tests** - Ensure consistency across versions

### Example pytest Structure (Future)
```
tests/
├── __init__.py
├── conftest.py              # pytest fixtures
├── unit/                    # Unit tests
│   ├── test_spi_unit.py
│   ├── test_spei_unit.py
│   └── test_runtheory_unit.py
├── integration/             # Integration tests
│   ├── test_spi_integration.py
│   ├── test_spei_integration.py
│   └── test_drought_integration.py
└── performance/             # Performance benchmarks
    └── test_performance.py
```

---

## Contributing

When adding new functionality:
1. Add corresponding test case
2. Ensure test passes
3. Update this README
4. Document expected behavior

---

## Test Philosophy

**Current approach:** Integration tests that verify end-to-end workflows

**Purpose:**
- Ensure package works as expected for users
- Validate complete workflows
- Generate example outputs
- Serve as executable documentation

**Trade-offs:**
- ✅ Tests real-world usage patterns
- ✅ Catches integration issues
- ✅ Produces useful outputs
- ⚠️ Longer execution times
- ⚠️ Less granular than unit tests

---

## Quick Reference

```bash
# Run specific test
python tests/test_spi.py

# Run with verbose output
python tests/test_drought_characteristics.py -v

# Clean output before testing
rm -rf output/*
python tests/test_spi.py

# Check test outputs
ls output/netcdf/
ls output/csv/
ls output/plots/single/
ls output/plots/spatial/
```

---

## Contact

For test-related issues or questions:
- Check test output messages
- Review documentation in `docs/`
- Open issue on GitHub repository

---

**Last Updated:** 2026-01-21
**Version:** 2026.1
