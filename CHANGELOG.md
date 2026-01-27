# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Calendar Versioning](https://calver.org/) (YYYY.M).

---

## [2026.1.1] - 2026-01-27

### Global-Scale Processing

Major update adding memory-efficient processing for global-scale datasets.

**New Module: `chunked.py`**

- `ChunkedProcessor` class for memory-efficient computation
- `compute_spi_chunked()` - Chunked SPI calculation with progress tracking
- `compute_spei_chunked()` - Chunked SPEI calculation
- `estimate_memory()` - Pre-computation memory estimation
- `iter_chunks()` - Spatial chunk coordinate generator

**New Functions in `indices.py`**

- `spi_global()` - Global-scale SPI with automatic chunking
- `spei_global()` - Global-scale SPEI with automatic chunking
- `estimate_memory_requirements()` - Memory estimation utility

**New Functions in `utils.py`**

- `get_optimal_chunk_size()` - Calculate optimal chunk dimensions
- `format_bytes()` - Human-readable byte formatting
- `get_array_memory_size()` - Array memory footprint calculation
- `print_memory_info()` - System memory status display

**New Functions in `compute.py`**

- `_rolling_sum_3d()` - O(n) cumulative sum algorithm
- `_compute_gamma_params_vectorized()` - Memory-optimized parameter fitting
- `_transform_to_normal_vectorized()` - Period-by-period transformation
- `compute_index_dask()` - Fixed Dask integration
- `compute_index_dask_to_zarr()` - Stream output to Zarr format

**Memory Optimizations**

- ~12x reduction in peak memory usage
- Float32 internal processing (50% memory reduction)
- Cumulative sum algorithm for O(n) rolling windows
- Period-by-period processing to limit peak memory
- Explicit garbage collection between chunks

**New Dependencies**

- `zarr>=2.10.0` - Efficient chunked array storage
- `psutil>=5.8.0` - Memory detection and monitoring

**New Notebook**

- `05_global_scale_processing.ipynb` - Tutorial for global datasets

**Configuration Updates**

- Added memory constants in `config.py`
- `MEMORY_MULTIPLIER`, `MEMORY_SAFETY_FACTOR`
- `DEFAULT_CHUNK_LAT`, `DEFAULT_CHUNK_LON`

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
