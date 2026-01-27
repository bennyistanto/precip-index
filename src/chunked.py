"""
Chunked processing module for memory-efficient SPI/SPEI computation.

Designed for global-scale datasets (e.g., CHIRPS, ERA5) that exceed available RAM.
Uses spatial tiling with streaming I/O to process arbitrarily large grids.

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank
"""

import gc
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from config import (
    DEFAULT_DISTRIBUTION,
    DISTRIBUTION_PARAM_NAMES,
    Periodicity,
    get_logger,
)

_logger = get_logger(__name__)


# =============================================================================
# MEMORY ESTIMATION
# =============================================================================

@dataclass
class MemoryEstimate:
    """Memory estimation for SPI/SPEI computation."""
    input_size_gb: float
    peak_memory_gb: float
    recommended_chunk_size: Tuple[int, int]
    n_chunks: int
    available_memory_gb: float
    fits_in_memory: bool

    def __repr__(self) -> str:
        status = "✓ Fits in memory" if self.fits_in_memory else "✗ Requires chunking"
        return (
            f"MemoryEstimate(\n"
            f"  Input size: {self.input_size_gb:.2f} GB\n"
            f"  Peak memory needed: {self.peak_memory_gb:.2f} GB\n"
            f"  Available memory: {self.available_memory_gb:.2f} GB\n"
            f"  Status: {status}\n"
            f"  Recommended chunk size: {self.recommended_chunk_size} (lat, lon)\n"
            f"  Number of chunks: {self.n_chunks}\n"
            f")"
        )


def estimate_memory(
    n_time: int,
    n_lat: int,
    n_lon: int,
    available_memory_gb: float = None,
    memory_multiplier: float = 12.0,
    safety_factor: float = 0.7
) -> MemoryEstimate:
    """
    Estimate memory requirements for SPI/SPEI computation.

    :param n_time: Number of time steps
    :param n_lat: Number of latitude points
    :param n_lon: Number of longitude points
    :param available_memory_gb: Available RAM in GB (auto-detected if None)
    :param memory_multiplier: Peak memory as multiple of input (default 12x)
    :param safety_factor: Fraction of available memory to use (default 0.7)
    :return: MemoryEstimate object with recommendations
    """
    # Auto-detect available memory
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            _logger.warning("psutil not installed, assuming 16GB available memory")
            available_memory_gb = 16.0

    # Calculate sizes
    bytes_per_element = 8  # float64
    input_bytes = n_time * n_lat * n_lon * bytes_per_element
    input_gb = input_bytes / (1024**3)
    peak_gb = input_gb * memory_multiplier

    # Usable memory
    usable_memory_gb = available_memory_gb * safety_factor

    fits_in_memory = peak_gb <= usable_memory_gb

    # Calculate optimal chunk size
    if fits_in_memory:
        chunk_lat, chunk_lon = n_lat, n_lon
        n_chunks = 1
    else:
        # Target chunk size that fits in memory
        target_cells = int((usable_memory_gb / memory_multiplier) * (1024**3) / (n_time * bytes_per_element))

        # Try to make roughly square chunks
        chunk_size = int(np.sqrt(target_cells))
        chunk_lat = min(chunk_size, n_lat)
        chunk_lon = min(chunk_size, n_lon)

        # Ensure minimum chunk size for efficiency
        chunk_lat = max(chunk_lat, 100)
        chunk_lon = max(chunk_lon, 100)

        # Calculate number of chunks
        n_lat_chunks = int(np.ceil(n_lat / chunk_lat))
        n_lon_chunks = int(np.ceil(n_lon / chunk_lon))
        n_chunks = n_lat_chunks * n_lon_chunks

    return MemoryEstimate(
        input_size_gb=input_gb,
        peak_memory_gb=peak_gb,
        recommended_chunk_size=(chunk_lat, chunk_lon),
        n_chunks=n_chunks,
        available_memory_gb=available_memory_gb,
        fits_in_memory=fits_in_memory
    )


def estimate_memory_from_data(
    data: Union[xr.DataArray, xr.Dataset],
    var_name: Optional[str] = None,
    available_memory_gb: float = None
) -> MemoryEstimate:
    """
    Estimate memory requirements from actual data.

    :param data: xarray DataArray or Dataset
    :param var_name: Variable name if Dataset
    :param available_memory_gb: Available RAM in GB
    :return: MemoryEstimate object
    """
    if isinstance(data, xr.Dataset):
        if var_name is None:
            var_name = list(data.data_vars)[0]
        data = data[var_name]

    shape = data.shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D data (time, lat, lon), got shape {shape}")

    return estimate_memory(
        n_time=shape[0],
        n_lat=shape[1],
        n_lon=shape[2],
        available_memory_gb=available_memory_gb
    )


# =============================================================================
# CHUNK ITERATOR
# =============================================================================

@dataclass
class ChunkInfo:
    """Information about a spatial chunk."""
    chunk_idx: int
    total_chunks: int
    lat_slice: slice
    lon_slice: slice
    lat_start: int
    lat_end: int
    lon_start: int
    lon_end: int

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (lat_size, lon_size) for this chunk."""
        return (self.lat_end - self.lat_start, self.lon_end - self.lon_start)

    def __repr__(self) -> str:
        return (
            f"Chunk {self.chunk_idx + 1}/{self.total_chunks} "
            f"[lat {self.lat_start}:{self.lat_end}, lon {self.lon_start}:{self.lon_end}]"
        )


def iter_chunks(
    n_lat: int,
    n_lon: int,
    chunk_lat: int,
    chunk_lon: int
) -> Iterator[ChunkInfo]:
    """
    Iterate over spatial chunks.

    :param n_lat: Total latitude points
    :param n_lon: Total longitude points
    :param chunk_lat: Chunk size in latitude dimension
    :param chunk_lon: Chunk size in longitude dimension
    :yields: ChunkInfo objects for each chunk
    """
    n_lat_chunks = int(np.ceil(n_lat / chunk_lat))
    n_lon_chunks = int(np.ceil(n_lon / chunk_lon))
    total_chunks = n_lat_chunks * n_lon_chunks

    chunk_idx = 0
    for lat_chunk_idx in range(n_lat_chunks):
        lat_start = lat_chunk_idx * chunk_lat
        lat_end = min(lat_start + chunk_lat, n_lat)

        for lon_chunk_idx in range(n_lon_chunks):
            lon_start = lon_chunk_idx * chunk_lon
            lon_end = min(lon_start + chunk_lon, n_lon)

            yield ChunkInfo(
                chunk_idx=chunk_idx,
                total_chunks=total_chunks,
                lat_slice=slice(lat_start, lat_end),
                lon_slice=slice(lon_start, lon_end),
                lat_start=lat_start,
                lat_end=lat_end,
                lon_start=lon_start,
                lon_end=lon_end
            )
            chunk_idx += 1


# =============================================================================
# CHUNKED COMPUTATION ENGINE
# =============================================================================

class ChunkedProcessor:
    """
    Memory-efficient chunked processor for SPI/SPEI computation.

    Processes large datasets in spatial tiles, streaming results to disk.

    Example:
        >>> processor = ChunkedProcessor(
        ...     chunk_lat=500,
        ...     chunk_lon=500,
        ...     n_workers=8
        ... )
        >>> result = processor.compute_spi(
        ...     precip_path='chirps_global.nc',
        ...     output_path='spi_global.nc',
        ...     scale=12,
        ...     calibration_start_year=1991,
        ...     calibration_end_year=2020
        ... )
    """

    def __init__(
        self,
        chunk_lat: int = 500,
        chunk_lon: int = 500,
        n_workers: int = None,
        temp_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize chunked processor.

        :param chunk_lat: Chunk size in latitude dimension
        :param chunk_lon: Chunk size in longitude dimension
        :param n_workers: Number of parallel workers (default: CPU count)
        :param temp_dir: Directory for temporary files
        :param verbose: Print progress information
        """
        self.chunk_lat = chunk_lat
        self.chunk_lon = chunk_lon
        self.n_workers = n_workers or os.cpu_count()
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.verbose = verbose

        _logger.info(
            f"ChunkedProcessor initialized: chunk_size=({chunk_lat}, {chunk_lon}), "
            f"workers={self.n_workers}"
        )

    def _log(self, msg: str):
        """Log message if verbose mode is on."""
        if self.verbose:
            _logger.info(msg)

    def compute_spi_chunked(
        self,
        precip: Union[str, Path, xr.DataArray, xr.Dataset],
        output_path: Union[str, Path],
        scale: int,
        periodicity: Union[str, Periodicity] = Periodicity.monthly,
        calibration_start_year: int = 1991,
        calibration_end_year: int = 2020,
        var_name: Optional[str] = None,
        save_params: bool = True,
        params_path: Optional[str] = None,
        compress: bool = True,
        complevel: int = 4,
        callback: Optional[Callable[[ChunkInfo, float], None]] = None,
        distribution: str = DEFAULT_DISTRIBUTION
    ) -> xr.Dataset:
        """
        Compute SPI using chunked processing for large datasets.

        :param precip: Precipitation data (file path or xarray object)
        :param output_path: Output NetCDF file path
        :param scale: Accumulation scale (e.g., 1, 3, 6, 12)
        :param periodicity: 'monthly' or 'daily'
        :param calibration_start_year: Start year of calibration period
        :param calibration_end_year: End year of calibration period
        :param var_name: Variable name (required if Dataset with multiple vars)
        :param save_params: Whether to save fitting parameters
        :param params_path: Path for fitting parameters file
        :param compress: Use compression for output
        :param complevel: Compression level (1-9)
        :param callback: Optional callback function(chunk_info, progress_pct)
        :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
            'gev', 'gen_logistic'). Default: 'gamma'
        :return: Dataset with computed SPI
        """
        from compute import compute_index_parallel
        from indices import save_fitting_params
        from utils import get_data_year_range

        dist = distribution.lower()

        # Convert periodicity
        if isinstance(periodicity, str):
            periodicity = Periodicity.from_string(periodicity)

        # Load data lazily
        self._log(f"Opening data: {precip if isinstance(precip, (str, Path)) else 'xarray object'}")

        if isinstance(precip, (str, Path)):
            ds = xr.open_dataset(precip, chunks={'time': -1, 'lat': self.chunk_lat, 'lon': self.chunk_lon})
            if var_name is None:
                # Auto-detect precipitation variable
                precip_vars = [v for v in ds.data_vars
                              if any(x in v.lower() for x in ['precip', 'prcp', 'pr', 'ppt', 'rainfall'])]
                if len(precip_vars) == 1:
                    var_name = precip_vars[0]
                elif len(precip_vars) == 0:
                    var_name = list(ds.data_vars)[0]
                else:
                    raise ValueError(f"Multiple precipitation variables found: {precip_vars}. Specify var_name.")
            precip_da = ds[var_name]
        elif isinstance(precip, xr.Dataset):
            if var_name is None:
                var_name = list(precip.data_vars)[0]
            precip_da = precip[var_name]
            ds = precip
        else:
            precip_da = precip
            ds = precip.to_dataset()

        # Ensure correct dimension order (time, lat, lon)
        if precip_da.dims != ('time', 'lat', 'lon'):
            self._log(f"Transposing dimensions from {precip_da.dims} to ('time', 'lat', 'lon')")
            precip_da = precip_da.transpose('time', 'lat', 'lon')

        # Get dimensions
        n_time, n_lat, n_lon = precip_da.shape

        # Get year range
        data_start_year, data_end_year = get_data_year_range(ds)

        # Memory estimation
        mem_est = estimate_memory(n_time, n_lat, n_lon)
        self._log(f"\n{mem_est}")

        # Adjust chunk sizes if needed
        chunk_lat = min(self.chunk_lat, n_lat)
        chunk_lon = min(self.chunk_lon, n_lon)

        if mem_est.fits_in_memory and n_lat <= chunk_lat and n_lon <= chunk_lon:
            self._log("Data fits in memory, using single-chunk processing")
            return self._compute_single_chunk(
                precip_da, output_path, scale, periodicity,
                data_start_year, calibration_start_year, calibration_end_year,
                save_params, params_path, compress, complevel,
                distribution=dist
            )

        # Prepare output file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create output dataset structure
        self._log(f"Creating output file: {output_path}")

        coords = {
            'time': precip_da.time,
            'lat': precip_da.lat,
            'lon': precip_da.lon
        }

        # Initialize output arrays with fill value
        from config import NC_FILL_VALUE, get_variable_name, get_variable_attributes

        var_name_out = get_variable_name('spi', scale, periodicity, distribution=dist)

        # Create empty dataset
        out_ds = xr.Dataset(
            {
                var_name_out: xr.DataArray(
                    data=np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32),
                    dims=['time', 'lat', 'lon'],
                    coords=coords,
                    attrs=get_variable_attributes('spi', scale, periodicity, distribution=dist)
                )
            },
            attrs={
                'title': f'Standardized Precipitation Index (SPI-{scale})',
                'institution': 'GOST/DEC Data Group, The World Bank',
                'source': 'precip-index package (chunked processing)',
                'distribution': dist,
                'calibration_start_year': calibration_start_year,
                'calibration_end_year': calibration_end_year,
            }
        )

        # Set encoding
        encoding = {
            var_name_out: {
                'dtype': 'float32',
                '_FillValue': NC_FILL_VALUE,
                'zlib': compress,
                'complevel': complevel,
                'chunksizes': (min(12, n_time), min(chunk_lat, n_lat), min(chunk_lon, n_lon))
            }
        }

        # Save initial structure
        out_ds.to_netcdf(output_path, mode='w', encoding=encoding)
        out_ds.close()

        # Initialize parameter arrays if saving (distribution-aware)
        param_names = DISTRIBUTION_PARAM_NAMES.get(dist, ('alpha', 'beta', 'prob_zero'))
        if save_params:
            periods = periodicity.value
            all_params_arrays = {}
            for pname in param_names:
                all_params_arrays[pname] = np.full((periods, n_lat, n_lon), np.nan, dtype=np.float32)

        # Process chunks
        chunks = list(iter_chunks(n_lat, n_lon, chunk_lat, chunk_lon))
        total_chunks = len(chunks)

        self._log(f"Processing {total_chunks} chunks with size ({chunk_lat}, {chunk_lon}), distribution={dist}")

        for chunk_info in chunks:
            progress_pct = (chunk_info.chunk_idx + 1) / total_chunks * 100
            self._log(f"Processing {chunk_info} ({progress_pct:.1f}%)")

            # Extract chunk data
            chunk_data = precip_da.isel(
                lat=chunk_info.lat_slice,
                lon=chunk_info.lon_slice
            ).values

            # Clip negative values
            chunk_data = np.clip(chunk_data, 0, None)

            # Compute SPI for chunk
            try:
                result_chunk, params = compute_index_parallel(
                    chunk_data,
                    scale=scale,
                    data_start_year=data_start_year,
                    calibration_start_year=calibration_start_year,
                    calibration_end_year=calibration_end_year,
                    periodicity=periodicity,
                    distribution=dist
                )

                # Write result chunk to output file
                with xr.open_dataset(output_path, mode='r+') as out_ds:
                    out_ds[var_name_out].values[
                        :,
                        chunk_info.lat_start:chunk_info.lat_end,
                        chunk_info.lon_start:chunk_info.lon_end
                    ] = result_chunk.astype(np.float32)
                    out_ds.to_netcdf(output_path, mode='a')

                # Store parameters (generic for any distribution)
                if save_params:
                    for pname in param_names:
                        if pname in params and isinstance(params[pname], np.ndarray):
                            all_params_arrays[pname][
                                :,
                                chunk_info.lat_start:chunk_info.lat_end,
                                chunk_info.lon_start:chunk_info.lon_end
                            ] = params[pname]

            except Exception as e:
                _logger.error(f"Error processing {chunk_info}: {e}")
                raise

            # Callback for progress tracking
            if callback:
                callback(chunk_info, progress_pct)

            # Force garbage collection after each chunk
            del chunk_data, result_chunk
            gc.collect()

        # Save parameters if requested
        if save_params:
            if params_path is None:
                params_path = str(output_path).replace('.nc', '_params.nc')

            self._log(f"Saving fitting parameters to: {params_path}")
            save_fitting_params(
                all_params_arrays,
                params_path,
                scale=scale,
                periodicity=periodicity,
                index_type='spi',
                calibration_start_year=calibration_start_year,
                calibration_end_year=calibration_end_year,
                coords={'lat': precip_da.lat.values, 'lon': precip_da.lon.values},
                distribution=dist
            )

        self._log(f"Chunked SPI computation complete: {output_path}")

        # Return the result
        return xr.open_dataset(output_path)

    def _compute_single_chunk(
        self,
        precip_da: xr.DataArray,
        output_path: Union[str, Path],
        scale: int,
        periodicity: Periodicity,
        data_start_year: int,
        calibration_start_year: int,
        calibration_end_year: int,
        save_params: bool,
        params_path: Optional[str],
        compress: bool,
        complevel: int,
        distribution: str = DEFAULT_DISTRIBUTION
    ) -> xr.Dataset:
        """Process data that fits in memory as a single chunk."""
        from indices import spi, save_fitting_params, save_index_to_netcdf

        result, params = spi(
            precip_da,
            scale=scale,
            periodicity=periodicity,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            return_params=True,
            distribution=distribution
        )

        # Save result
        save_index_to_netcdf(result, str(output_path), compress=compress, complevel=complevel)

        # Save parameters if requested
        if save_params:
            if params_path is None:
                params_path = str(output_path).replace('.nc', '_params.nc')

            save_fitting_params(
                params,
                params_path,
                scale=scale,
                periodicity=periodicity,
                index_type='spi',
                calibration_start_year=calibration_start_year,
                calibration_end_year=calibration_end_year,
                coords={'lat': precip_da.lat.values, 'lon': precip_da.lon.values},
                distribution=distribution
            )

        return xr.open_dataset(output_path)

    def compute_spei_chunked(
        self,
        precip: Union[str, Path, xr.DataArray],
        pet: Union[str, Path, xr.DataArray],
        output_path: Union[str, Path],
        scale: int,
        periodicity: Union[str, Periodicity] = Periodicity.monthly,
        calibration_start_year: int = 1991,
        calibration_end_year: int = 2020,
        precip_var_name: Optional[str] = None,
        pet_var_name: Optional[str] = None,
        save_params: bool = True,
        params_path: Optional[str] = None,
        compress: bool = True,
        complevel: int = 4,
        callback: Optional[Callable[[ChunkInfo, float], None]] = None,
        distribution: str = DEFAULT_DISTRIBUTION
    ) -> xr.Dataset:
        """
        Compute SPEI using chunked processing for large datasets.

        :param precip: Precipitation data (file path or xarray object)
        :param pet: PET data (file path or xarray object)
        :param output_path: Output NetCDF file path
        :param scale: Accumulation scale
        :param periodicity: 'monthly' or 'daily'
        :param calibration_start_year: Start year of calibration period
        :param calibration_end_year: End year of calibration period
        :param precip_var_name: Precipitation variable name
        :param pet_var_name: PET variable name
        :param save_params: Whether to save fitting parameters
        :param params_path: Path for fitting parameters file
        :param compress: Use compression for output
        :param complevel: Compression level
        :param callback: Optional callback for progress
        :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
            'gev', 'gen_logistic'). Default: 'gamma'.
            Note: Pearson III or Log-Logistic are recommended for SPEI.
        :return: Dataset with computed SPEI
        """
        from compute import compute_index_parallel
        from indices import save_fitting_params
        from utils import get_data_year_range

        dist = distribution.lower()

        if isinstance(periodicity, str):
            periodicity = Periodicity.from_string(periodicity)

        # Load precipitation
        self._log(f"Opening precipitation data")
        if isinstance(precip, (str, Path)):
            precip_ds = xr.open_dataset(precip, chunks={'time': -1, 'lat': self.chunk_lat, 'lon': self.chunk_lon})
            if precip_var_name is None:
                precip_var_name = self._find_var(precip_ds, ['precip', 'prcp', 'pr', 'ppt'])
            precip_da = precip_ds[precip_var_name]
        elif isinstance(precip, xr.Dataset):
            if precip_var_name is None:
                precip_var_name = list(precip.data_vars)[0]
            precip_da = precip[precip_var_name]
            precip_ds = precip
        else:
            precip_da = precip
            precip_ds = precip.to_dataset()

        # Load PET
        self._log(f"Opening PET data")
        if isinstance(pet, (str, Path)):
            pet_ds = xr.open_dataset(pet, chunks={'time': -1, 'lat': self.chunk_lat, 'lon': self.chunk_lon})
            if pet_var_name is None:
                pet_var_name = self._find_var(pet_ds, ['pet', 'eto', 'et', 'evap'])
            pet_da = pet_ds[pet_var_name]
        elif isinstance(pet, xr.Dataset):
            if pet_var_name is None:
                pet_var_name = list(pet.data_vars)[0]
            pet_da = pet[pet_var_name]
        else:
            pet_da = pet

        # Ensure correct dimension order
        if precip_da.dims != ('time', 'lat', 'lon'):
            precip_da = precip_da.transpose('time', 'lat', 'lon')
        if pet_da.dims != ('time', 'lat', 'lon'):
            pet_da = pet_da.transpose('time', 'lat', 'lon')

        # Get dimensions
        n_time, n_lat, n_lon = precip_da.shape

        # Get year range
        data_start_year, _ = get_data_year_range(precip_ds)

        # Memory estimation (multiply by 2 for both precip and PET)
        mem_est = estimate_memory(n_time, n_lat, n_lon)
        mem_est.peak_memory_gb *= 1.5  # Account for both inputs
        self._log(f"\n{mem_est}")

        # Prepare output file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from config import NC_FILL_VALUE, get_variable_name, get_variable_attributes

        var_name_out = get_variable_name('spei', scale, periodicity, distribution=dist)

        # Create output dataset
        coords = {'time': precip_da.time, 'lat': precip_da.lat, 'lon': precip_da.lon}

        out_ds = xr.Dataset(
            {
                var_name_out: xr.DataArray(
                    data=np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32),
                    dims=['time', 'lat', 'lon'],
                    coords=coords,
                    attrs=get_variable_attributes('spei', scale, periodicity, distribution=dist)
                )
            },
            attrs={
                'title': f'Standardized Precipitation Evapotranspiration Index (SPEI-{scale})',
                'institution': 'GOST/DEC Data Group, The World Bank',
                'source': 'precip-index package (chunked processing)',
                'distribution': dist,
                'calibration_start_year': calibration_start_year,
                'calibration_end_year': calibration_end_year,
            }
        )

        chunk_lat = min(self.chunk_lat, n_lat)
        chunk_lon = min(self.chunk_lon, n_lon)

        encoding = {
            var_name_out: {
                'dtype': 'float32',
                '_FillValue': NC_FILL_VALUE,
                'zlib': compress,
                'complevel': complevel,
                'chunksizes': (min(12, n_time), min(chunk_lat, n_lat), min(chunk_lon, n_lon))
            }
        }

        out_ds.to_netcdf(output_path, mode='w', encoding=encoding)
        out_ds.close()

        # Initialize parameter arrays (distribution-aware)
        param_names = DISTRIBUTION_PARAM_NAMES.get(dist, ('alpha', 'beta', 'prob_zero'))
        if save_params:
            periods = periodicity.value
            all_params_arrays = {}
            for pname in param_names:
                all_params_arrays[pname] = np.full((periods, n_lat, n_lon), np.nan, dtype=np.float32)

        # Process chunks
        chunks = list(iter_chunks(n_lat, n_lon, chunk_lat, chunk_lon))
        total_chunks = len(chunks)

        self._log(f"Processing {total_chunks} chunks, distribution={dist}")

        for chunk_info in chunks:
            progress_pct = (chunk_info.chunk_idx + 1) / total_chunks * 100
            self._log(f"Processing {chunk_info} ({progress_pct:.1f}%)")

            # Extract chunk data
            precip_chunk = precip_da.isel(lat=chunk_info.lat_slice, lon=chunk_info.lon_slice).values
            pet_chunk = pet_da.isel(lat=chunk_info.lat_slice, lon=chunk_info.lon_slice).values

            # Compute water balance with offset
            water_balance = (precip_chunk - pet_chunk) + 1000.0

            try:
                result_chunk, params = compute_index_parallel(
                    water_balance,
                    scale=scale,
                    data_start_year=data_start_year,
                    calibration_start_year=calibration_start_year,
                    calibration_end_year=calibration_end_year,
                    periodicity=periodicity,
                    distribution=dist
                )

                # Write result
                with xr.open_dataset(output_path, mode='r+') as out_ds:
                    out_ds[var_name_out].values[
                        :,
                        chunk_info.lat_start:chunk_info.lat_end,
                        chunk_info.lon_start:chunk_info.lon_end
                    ] = result_chunk.astype(np.float32)
                    out_ds.to_netcdf(output_path, mode='a')

                # Store parameters (generic for any distribution)
                if save_params:
                    for pname in param_names:
                        if pname in params and isinstance(params[pname], np.ndarray):
                            all_params_arrays[pname][
                                :,
                                chunk_info.lat_start:chunk_info.lat_end,
                                chunk_info.lon_start:chunk_info.lon_end
                            ] = params[pname]

            except Exception as e:
                _logger.error(f"Error processing {chunk_info}: {e}")
                raise

            if callback:
                callback(chunk_info, progress_pct)

            del precip_chunk, pet_chunk, water_balance, result_chunk
            gc.collect()

        # Save parameters
        if save_params:
            if params_path is None:
                params_path = str(output_path).replace('.nc', '_params.nc')

            self._log(f"Saving fitting parameters to: {params_path}")
            save_fitting_params(
                all_params_arrays,
                params_path,
                scale=scale,
                periodicity=periodicity,
                index_type='spei',
                calibration_start_year=calibration_start_year,
                calibration_end_year=calibration_end_year,
                coords={'lat': precip_da.lat.values, 'lon': precip_da.lon.values},
                distribution=dist
            )

        self._log(f"Chunked SPEI computation complete: {output_path}")
        return xr.open_dataset(output_path)

    def _find_var(self, ds: xr.Dataset, patterns: List[str]) -> str:
        """Find variable matching patterns."""
        for var in ds.data_vars:
            if any(p in var.lower() for p in patterns):
                return var
        return list(ds.data_vars)[0]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_spi_global(
    precip_path: Union[str, Path],
    output_path: Union[str, Path],
    scale: int = 12,
    calibration_start_year: int = 1991,
    calibration_end_year: int = 2020,
    chunk_size: int = 500,
    n_workers: int = None,
    var_name: Optional[str] = None,
    distribution: str = DEFAULT_DISTRIBUTION
) -> xr.Dataset:
    """
    Compute SPI for global dataset with automatic memory management.

    Convenience function that handles chunking automatically.

    :param precip_path: Path to precipitation NetCDF file
    :param output_path: Path for output SPI NetCDF file
    :param scale: Accumulation scale (default: 12)
    :param calibration_start_year: Calibration start year
    :param calibration_end_year: Calibration end year
    :param chunk_size: Spatial chunk size (default: 500)
    :param n_workers: Number of parallel workers
    :param var_name: Precipitation variable name
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :return: Dataset with computed SPI

    Example:
        >>> result = compute_spi_global(
        ...     'chirps_global_monthly.nc',
        ...     'spi_12_global.nc',
        ...     scale=12
        ... )
    """
    processor = ChunkedProcessor(
        chunk_lat=chunk_size,
        chunk_lon=chunk_size,
        n_workers=n_workers
    )

    return processor.compute_spi_chunked(
        precip=precip_path,
        output_path=output_path,
        scale=scale,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        var_name=var_name,
        distribution=distribution
    )


def compute_spei_global(
    precip_path: Union[str, Path],
    pet_path: Union[str, Path],
    output_path: Union[str, Path],
    scale: int = 12,
    calibration_start_year: int = 1991,
    calibration_end_year: int = 2020,
    chunk_size: int = 500,
    n_workers: int = None,
    precip_var_name: Optional[str] = None,
    pet_var_name: Optional[str] = None,
    distribution: str = DEFAULT_DISTRIBUTION
) -> xr.Dataset:
    """
    Compute SPEI for global dataset with automatic memory management.

    :param precip_path: Path to precipitation NetCDF file
    :param pet_path: Path to PET NetCDF file
    :param output_path: Path for output SPEI NetCDF file
    :param scale: Accumulation scale
    :param calibration_start_year: Calibration start year
    :param calibration_end_year: Calibration end year
    :param chunk_size: Spatial chunk size
    :param n_workers: Number of parallel workers
    :param precip_var_name: Precipitation variable name
    :param pet_var_name: PET variable name
    :param distribution: distribution type ('gamma', 'pearson3', 'log_logistic',
        'gev', 'gen_logistic'). Default: 'gamma'
    :return: Dataset with computed SPEI
    """
    processor = ChunkedProcessor(
        chunk_lat=chunk_size,
        chunk_lon=chunk_size,
        n_workers=n_workers
    )

    return processor.compute_spei_chunked(
        precip=precip_path,
        pet=pet_path,
        output_path=output_path,
        scale=scale,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        precip_var_name=precip_var_name,
        pet_var_name=pet_var_name,
        distribution=distribution
    )
