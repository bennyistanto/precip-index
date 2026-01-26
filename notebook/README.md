# Jupyter Notebooks for SPI, SPEI, and Climate Extremes Analysis

This directory contains example Jupyter notebooks demonstrating how to calculate climate indices (SPI, SPEI) and analyze extreme events (both drought and flooding) using the `precip-index` package.

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

### 03_event_characteristics.ipynb

Climate extremes event analysis using run theory:

- Identifying extreme events (both drought and wet conditions)
- Calculating event characteristics: duration, magnitude, intensity, peak
- Time-series monitoring with varying characteristics
- Period statistics for decision-making (gridded analysis)
- Event comparison and trend analysis
- Works for both dry (negative threshold) and wet (positive threshold) events

### 04_visualization_gallery.ipynb

Comprehensive visualization examples for climate extremes:

- Event timeline plots with highlighted events
- Spatial maps of drought/wet characteristics
- Event characteristic analysis plots
- Period comparison visualizations
- All plot types demonstrated with both dry and wet examples

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

3. Open any notebook:

   - `01_calculate_spi.ipynb` - SPI calculation basics
   - `02_calculate_spei.ipynb` - SPEI with temperature
   - `03_event_characteristics.ipynb` - Climate extremes analysis
   - `04_visualization_gallery.ipynb` - All visualization examples

4. Run cells sequentially (Shift + Enter)

**Recommended Learning Path:**

1. Start with **01** (SPI basics)
2. Then **02** (SPEI with PET)
3. Move to **03** (event analysis)
4. Explore **04** (visualization gallery)

## Run Theory for Climate Extremes

The notebooks demonstrate **run theory**, a framework for identifying and analyzing climate extreme events. This methodology works for **both dry (drought) and wet (flooding/excess) conditions**.

![Run Theory Framework](../docs/images/runtheory.svg)

**Run Theory Framework:** Events are identified when an index crosses a threshold. This example shows **dry events** (below threshold), but the identical analysis applies to **wet events** (above threshold). Key metrics—Duration (D), Magnitude (M), Intensity (I), and Inter-arrival Time (T)—are calculated the same way for both extremes.

### Bidirectional Application

| Extreme Type | Threshold | Example | Notebooks |
|--------------|-----------|---------|-----------|
| **Drought (Dry)** | Negative (e.g., -1.2) | SPI/SPEI < 0 | 01, 02, 03, 04 |
| **Flooding (Wet)** | Positive (e.g., +1.2) | SPI/SPEI > 0 | 03, 04 |

**💡 Key Point:** The same functions and analysis work for both extremes - only the threshold sign changes!

For detailed methodology, see [../docs/user-guide/runtheory.md](../docs/user-guide/runtheory.md)

---

## Key Features Demonstrated

### Climate Indices (Notebooks 01-02)

- ✅ Loading NetCDF climate data
- ✅ Single-scale and multi-scale index calculation
- ✅ Parameter saving and reuse
- ✅ CF Convention compliance checking
- ✅ Custom calibration periods
- ✅ Drought classification (McKee et al., 1993)

### Climate Extremes Analysis (Notebooks 03-04)

- ✅ Event identification (both drought and wet conditions)
- ✅ Duration, magnitude, intensity, peak calculation
- ✅ Time-series monitoring with varying characteristics
- ✅ Period statistics (gridded analysis for decision-making)
- ✅ Event comparison and trend analysis
- ✅ Comprehensive visualization suite

### Technical Features (All Notebooks)

- ✅ Parallel processing for gridded data
- ✅ Memory-efficient processing with Dask
- ✅ Time series and spatial visualization
- ✅ Comparative analysis (SPI vs SPEI, dry vs wet)

## Directory Structure

```
notebook/
├── README.md                           # This file
├── 01_calculate_spi.ipynb              # SPI calculation tutorial
├── 02_calculate_spei.ipynb             # SPEI calculation tutorial
├── 03_event_characteristics.ipynb      # Climate extremes analysis (run theory)
├── 04_visualization_gallery.ipynb      # Visualization examples
├── data/                               # Sample data (optional, git-ignored)
│   ├── precip_sample.nc
│   ├── temp_sample.nc
│   └── pet_sample.nc
└── output/                             # Output files (git-ignored)
    ├── csv/                            # Event analysis results
    ├── netcdf/                         # Gridded statistics
    └── figures/                        # Plots and visualizations
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

## Learn More

**Documentation:**

- [Run Theory Guide](../docs/user-guide/runtheory.md) - Complete framework explanation
- [SPI Guide](../docs/user-guide/spi.md) - SPI calculation details
- [SPEI Guide](../docs/user-guide/spei.md) - SPEI methodology
- [Magnitude Explained](../docs/user-guide/magnitude.md) - Cumulative vs instantaneous
- [Visualization Guide](../docs/user-guide/visualization.md) - All plot types

**Quick References:**

- [../QUICK_START.md](../QUICK_START.md) - Quick start examples
- [../README.md](../README.md) - Main package overview

## References

- McKee, T.B., Doesken, N.J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *8th Conference on Applied Climatology*.

- Vicente-Serrano, S.M., Beguería, S., & López-Moreno, J.I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. *Journal of Climate*, 23(7), 1696-1718.

- Yevjevich, V. (1967). An objective approach to definitions and investigations of continental hydrologic droughts. Hydrology Papers 23, Colorado State University.

## Support

For issues or questions:

1. Check the main README.md in the repository root
2. Review the docstrings in the source code
3. Open an issue on GitHub

## License

BSD 3-Clause License - See LICENSE file in repository root
