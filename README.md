# Precipitation Index - SPI & SPEI for Climate Extremes Monitoring

**precip-index** is a lightweight set of Python scripts for calculating precipitation-based climate indices (SPI and SPEI) and analyzing **dry and wet extremes** using **run theory**, designed for gridded `xarray` workflows.  

📚 Documentation: https://bennyistanto.github.io/precip-index/

## Key features

- **SPI / SPEI** at 1, 3, 6, 12, 24-month scales (xarray + CF-style NetCDF outputs)
- **Bidirectional extremes**: drought (dry) and flood-prone (wet) conditions in one framework
- **Multi-distribution fitting**: Gamma, Pearson Type III, Log-Logistic
- **Run theory events**: duration, magnitude, intensity, peak, interarrival + gridded summaries
- **Scalable processing**: chunked tiling, memory estimation, streaming I/O for global datasets
- **Visualization**: event-highlighted time series, 11-category classification, maps, comparisons

## What makes precip-index different?

- **Dry + wet symmetry**: same API and methodology for negative (drought) and positive (wet) thresholds
- **Distribution-aware SPI/SPEI**: choose the best-fit distribution per workflow (Gamma / P-III / Log-Logistic)
- **Event analytics included**: run theory metrics beyond simple threshold exceedance
- **Designed for large grids**: practical for CHIRPS / ERA5-Land / TerraClimate via chunked processing

## Credits

SPI/SPEI components are modified/adapted from `climate-indices` by James Adams ([monocongo](https://github.com/monocongo/climate_indices)).
