# Jupyter Notebooks for SPI and SPEI

This directory contains example Jupyter notebooks demonstrating how to calculate Standardized Precipitation Index (SPI) and Standardized Precipitation Evapotranspiration Index (SPEI) using the `precip-index` package.

## Notebooks

### 01_calculate_spi.ipynb
Complete tutorial for calculating SPI (Standardized Precipitation Index):
- Loading precipitation data
- Calculating SPI for single and multiple scales (1, 3, 6, 12 months)
- Saving and loading gamma distribution fitting parameters
- Visualizing SPI time series and spatial maps
- Drought classification and analysis
- Calculating drought area percentage

### 02_calculate_spei.ipynb
Complete tutorial for calculating SPEI (Standardized Precipitation Evapotranspiration Index):
- Loading precipitation and temperature data
- Calculating PET using Thornthwaite method
- Computing SPEI for multiple scales
- Using pre-computed PET vs calculating from temperature
- Advanced visualization techniques
- Comparing SPI vs SPEI results

## Getting Started

### Prerequisites

Make sure you have installed all required packages:

```bash
pip install numpy scipy xarray netcdf4 numba matplotlib cartopy
```

### Data Requirements

The notebooks expect NetCDF files with the following characteristics:

**For SPI:**
- Precipitation data in mm
- Dimensions: `(time, lat, lon)` following CF Convention
- Time coordinate with monthly or daily frequency

**For SPEI:**
- Precipitation data in mm
- Temperature data in °C (or PET in mm)
- Same dimensional structure as SPI

### Sample Data

The notebooks include code to:
1. Generate synthetic test data for demonstration
2. Load real climate data from common sources (CHIRPS, ERA5, etc.)

You can use your own data by modifying the file paths in the notebooks.

## Usage

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `notebook/` directory

3. Open either:
   - `01_calculate_spi.ipynb` for SPI examples
   - `02_calculate_spei.ipynb` for SPEI examples

4. Run cells sequentially (Shift + Enter)

## Key Features Demonstrated

### Basic Operations
- ✅ Loading NetCDF climate data
- ✅ Single-scale index calculation
- ✅ Multi-scale index calculation
- ✅ Parameter saving and reuse

### Advanced Features
- ✅ CF Convention compliance checking
- ✅ Custom calibration periods
- ✅ Parallel processing for gridded data
- ✅ Memory-efficient processing with Dask

### Analysis & Visualization
- ✅ Time series plotting
- ✅ Spatial maps with Cartopy
- ✅ Drought classification (McKee et al., 1993)
- ✅ Drought area percentage time series
- ✅ Comparative analysis (SPI vs SPEI)

## Directory Structure

```
notebook/
├── README.md                      # This file
├── 01_calculate_spi.ipynb         # SPI tutorial
├── 02_calculate_spei.ipynb        # SPEI tutorial
├── data/                          # Sample data (optional, git-ignored)
│   ├── precip_sample.nc
│   ├── temp_sample.nc
│   └── pet_sample.nc
└── output/                        # Output files (git-ignored)
    ├── spi_results/
    └── spei_results/
```

## Tips

### Performance Optimization
- Use pre-computed fitting parameters when processing multiple files with the same calibration period
- For large global datasets, consider using the Dask-enabled functions
- Chunk your data appropriately: `{'time': -1, 'lat': 100, 'lon': 100}`

### Common Issues

**Issue:** Import errors
```python
# Solution: Add src directory to Python path
import sys
sys.path.insert(0, '../src')
```

**Issue:** Memory errors with large datasets
```python
# Solution: Use chunked processing
ds = xr.open_dataset('large_file.nc', chunks={'time': 12, 'lat': 200, 'lon': 200})
```

**Issue:** Invalid values in output
```python
# Check calibration period matches your data range
# Ensure precipitation values are non-negative
# Verify CF Convention dimension order: (time, lat, lon)
```

## References

- McKee, T.B., Doesken, N.J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *8th Conference on Applied Climatology*.

- Vicente-Serrano, S.M., Beguería, S., & López-Moreno, J.I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. *Journal of Climate*, 23(7), 1696-1718.

## Support

For issues or questions:
1. Check the main README.md in the repository root
2. Review the docstrings in the source code
3. Open an issue on GitHub

## License

BSD 3-Clause License - See LICENSE file in repository root
