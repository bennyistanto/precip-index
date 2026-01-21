# Precipitation Index - SPI & SPEI for Climate Extremes Monitoring

> **⚠️ Active Development:** This package is under active development. Some features may not work as expected and bugs may be present. Please report issues on GitHub.

A minimal, efficient Python implementation of **Standardized Precipitation Index (SPI)** and **Standardized Precipitation Evapotranspiration Index (SPEI)** for monitoring **both drought and wet conditions** using run theory.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Features

**Climate Indices:**
- SPI (Standardized Precipitation Index) - precipitation-based
- SPEI (Standardized Precipitation Evapotranspiration Index) - temperature-inclusive
- Multiple time scales (1, 3, 6, 12, 24 months)
- CF-compliant NetCDF output
- **Monitor both dry (drought) and wet (flood/excess) conditions**

**Climate Extremes Analysis (Run Theory):**
- Event identification for **both drought and wet conditions**
- Duration, magnitude, intensity, peak for any extreme
- Time-series monitoring with varying characteristics
- Period aggregation (gridded statistics for decision-making)
- Comprehensive visualization suite
- Works with negative thresholds (dry) or positive thresholds (wet)

**Optimized for:**
- Global-scale gridded data
- CF Convention (time, lat, lon)
- Operational climate monitoring (drought, flooding, extremes)

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
- `notebooks/01_calculate_spi.ipynb` - SPI calculation
- `notebooks/02_calculate_spei.ipynb` - SPEI with PET
- `notebooks/03_event_characteristics.ipynb` - Climate extremes event analysis
- `notebooks/04_visualization_gallery.ipynb` - All plot types

**Technical Documentation:**
- [Implementation Details](docs/technical/implementation.md) - Architecture and design
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Key Capabilities

### Neutral Climate Extremes Analysis

All event analysis functions work for **both dry and wet conditions**:

| Function | Dry Events (Drought) | Wet Events (Flooding) |
|----------|----------------------|----------------------|
| `identify_events()` | `threshold=-1.2` | `threshold=+1.2` |
| `calculate_timeseries()` | Monitors dry periods | Monitors wet periods |
| `calculate_period_statistics()` | SPI/SPEI < 0 | SPI/SPEI > 0 |
| `plot_index()` | Full range with WMO colors | Both extremes |

**The threshold direction determines which extreme to analyze** - the same tools work for both ends of the precipitation spectrum!

### Three Analysis Modes

**1. Event-Based** - Identify complete extreme events
```python
events = identify_events(spi_ts, threshold=-1.2)
# Returns: DataFrame with event_id, duration, magnitude, intensity, peak
```

**2. Time-Series** - Month-by-month monitoring
```python
ts = calculate_timeseries(spi_ts, threshold=-1.2)
# Returns: DataFrame with varying characteristics over time
```

**3. Period Statistics** - Gridded decision support
```python
stats = calculate_period_statistics(spi, threshold=-1.2,
                                    start_year=2020, end_year=2024)
# Returns: xarray Dataset (lat, lon) with 9 variables
```

### Dual Magnitude Approach

Provides **both** magnitude types for comprehensive analysis:
- **Cumulative** - Total water deficit (standard run theory)
- **Instantaneous** - Current severity (NDVI-like pattern)

See [Magnitude Explained](docs/user-guide/magnitude.md) for details.

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

**Version:** 2026.1
**Last Updated:** 2026-01-21
