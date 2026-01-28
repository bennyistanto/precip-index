# Precipitation Index - SPI & SPEI for Climate Extremes Monitoring

A minimal, efficient Python implementation of **Standardized Precipitation Index (SPI)** and **Standardized Precipitation Evapotranspiration Index (SPEI)** for monitoring **both drought and wet conditions** using run theory.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Features

**Climate Indices:**

- SPI (Standardized Precipitation Index) - precipitation-based
- SPEI (Standardized Precipitation Evapotranspiration Index) - temperature-inclusive
- Multiple time scales (1, 3, 6, 12, 24 months)
- CF-compliant NetCDF output

**Bidirectional Analysis**

- Monitor **drought** (dry conditions)
- Monitor **floods** (wet conditions)
- Unified framework for both extremes
- Consistent methodology

**Multi-Distribution Support**

- **Gamma** - Standard for SPI (McKee et al. 1993)
- **Pearson Type III** - Recommended for SPEI
- **Log-Logistic** - Better tail behavior
- Automatic method selection per distribution

**Run Theory Implementation**

- Event identification & characterization
- Duration, magnitude, intensity, peak
- Time-series monitoring
- Gridded period statistics

**Chunked Processing**

- Memory-efficient spatial tiling
- Process global-scale data (CHIRPS, ERA5)
- Automatic memory estimation
- Streaming I/O for datasets exceeding RAM

**Visualization Suite**

- Time series plots with event highlighting
- 11-category WMO SPI/SPEI classification
- Spatial maps of event characteristics
- Cross-distribution comparison charts

## Quick Example

Calculate SPI with multiple distributions and identify drought events:

```python
import xarray as xr
from indices import spi, spi_multi_scale
from runtheory import identify_events
from visualization import plot_index

# Load precipitation data
ds = xr.open_dataset('precipitation.nc')
precip = ds['precip']

# Calculate SPI-12 with Gamma (default)
spi_12 = spi(precip, scale=12, periodicity='monthly')

# Or use Pearson III / Log-Logistic
spi_12_p3 = spi(precip, scale=12, distribution='pearson3')
spi_12_ll = spi(precip, scale=12, distribution='log_logistic')

# Multi-scale SPI in one call
spi_multi = spi_multi_scale(precip, scales=[3, 6, 9, 12])

# Identify drought events (threshold -1.2)
events = identify_events(spi_12.isel(lat=0, lon=0), threshold=-1.2)

# Visualize
plot_index(spi_12.isel(lat=0, lon=0), threshold=-1.2)
```

## What Makes This Package Different?

**Bidirectional by Design**

Unlike traditional drought-only tools, **precip-index** treats dry and wet extremes equally. Use negative thresholds for droughts, positive thresholds for floods --- same functions, same methodology.

**Multi-Distribution Fitting**

Choose the probability distribution that best fits your data. Gamma, Pearson III, and Log-Logistic each use their optimal fitting method (Method of Moments, MLE) --- validated to produce correct SPI/SPEI across all grid cells. See the [validation results](technical/validation.qmd).

**Run Theory Framework**

Goes beyond simple threshold exceedance. Implements full run theory to characterize event duration, magnitude (cumulative & instantaneous), intensity, and peak values.

**Scalable Architecture**

Built for datasets of any size. Small regional grids run in-memory; global datasets (CHIRPS, ERA5, TerraClimate) use the chunked processing module with automatic spatial tiling and streaming I/O.

**Global-Scale Performance**

Tested with **CHIRPS v3 global** (0.05°, 2400 x 7200 grid, ~69 GB, 539 months). SPI-12 Gamma computation with 12 spatial chunks completed in **~2 hours 47 minutes** including parameter saving, on a workstation with 343 GB RAM. Processing time scales with grid size and available memory --- regional subsets complete in seconds to minutes.

## Example Applications

- **Operational Drought Monitoring** - Track ongoing droughts with real-time updates
- **Flood Risk Assessment** - Identify wet extremes and excess precipitation
- **Climate Impact Studies** - Analyze historical trends in extremes
- **Early Warning Systems** - Generate alerts based on evolving event characteristics
- **Decision Support** - Gridded statistics for regional planning

## Getting Started

**📖 See [QUICK_START.md](QUICK_START.md) for detailed installation and usage examples.**

The Quick Start guide covers:

- Installation and setup
- Calculating SPI and SPEI
- Analyzing climate extremes (both dry and wet)
- Visualization examples
- Working with your own data

## Installation

```bash
# Install dependencies
pip install numpy scipy xarray netCDF4 numba matplotlib

# Clone repository
git clone https://github.com/bennyistanto/precip-index.git
cd precip-index
```

## Documentation

**User Guides:**

- [SPI Guide](docs/user-guide/spi.md) - Calculation, parameters, examples
- [SPEI Guide](docs/user-guide/spei.md) - Temperature-inclusive index
- [Climate Extremes Analysis](docs/user-guide/runtheory.md) - Event analysis (dry & wet)
- [Magnitude Explained](docs/user-guide/magnitude.md) - Cumulative vs instantaneous
- [Visualization Guide](docs/user-guide/visualization.md) - Plotting functions

**Tutorials (Jupyter Notebooks):**

- `notebook/01_calculate_spi.ipynb` - SPI calculation
- `notebook/02_calculate_spei.ipynb` - SPEI with PET
- `notebook/03_event_characteristics.ipynb` - Climate extremes event analysis
- `notebook/04_visualization_gallery.ipynb` - All plot types

**Technical Documentation:**

- [Implementation Details](docs/technical/implementation.md) - Architecture and design
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Credits

**Modified/adapted from:** [climate-indices](https://github.com/monocongo/climate_indices) by James Adams (monocongo)

**Author:** Benny Istanto    
**Organization:** GOST/DEC Data Group, The World Bank    
**Email:** bistanto@worldbank.org    

## References

- McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought frequency and duration to time scales. 8th Conference on Applied Climatology.

- Vicente-Serrano, S.M., Beguería, S., López-Moreno, J.I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. Journal of Climate, 23(7), 1696-1718.

- Yevjevich, V. (1967). An objective approach to definitions and investigations of continental hydrologic droughts. Hydrology Papers 23, Colorado State University.

## License

BSD 3-Clause License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For bugs or feature requests, open an issue on GitHub.

---

> **⚠️ Active Development:** This package is under active development. Some features may not work as expected and bugs may be present. Please report issues on GitHub.
