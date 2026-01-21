# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Calendar Versioning](https://calver.org/) (YYYY.M).

---

## [2026.1] - 2026-01-21

### Initial Release

First public release of the precip-index package - a streamlined implementation of SPI and SPEI for climate extremes monitoring.

**Modified/adapted from:** [climate-indices](https://github.com/monocongo/climate_indices) by James Adams (monocongo)

### Features

**Climate Indices:**
- SPI (Standardized Precipitation Index) calculation using Gamma distribution
- SPEI (Standardized Precipitation Evapotranspiration Index) with Thornthwaite PET
- Multi-scale support (1, 3, 6, 12, 24 months)
- Parameter save/load functionality for faster recomputation
- CF-compliant NetCDF output

**Climate Extremes Analysis (Run Theory):**
- Event identification for both drought (dry) and wet conditions
- Time-series monitoring with dual magnitude (cumulative & instantaneous)
- Gridded period statistics for decision support
- Event characteristics: duration, magnitude, intensity, peak
- Spatial aggregation and period comparison tools

**Visualization:**
- WMO standard 11-category SPI/SPEI color classification
- Event plots with automatic dry/wet differentiation
- Timeline visualizations with multiple characteristics
- Spatial statistics maps

### Technical Details

- Optimized for global-scale gridded data
- NumPy vectorization + Numba JIT compilation
- CF Convention compliance (time, lat, lon)
- Gamma distribution only (robust for arid regions)
- Python 3.8+ support

### Documentation

- Comprehensive user guides for SPI, SPEI, and run theory
- Jupyter notebook tutorials
- Technical implementation documentation
- Quick start guide

---

## Future Releases

Version 2026.2 and beyond will include:
- Bug fixes and performance improvements
- Additional test coverage
- Enhanced documentation and examples
- Community-requested features

---

**For detailed technical implementation notes, see [docs/technical/implementation.md](docs/technical/implementation.md)**
