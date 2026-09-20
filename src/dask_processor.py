"""
Dask-parallel SPI/SPEI processing with fitting-parameter capture.

Complements chunked.ChunkedProcessor, which walks spatial tiles one at a time
in the main process. This module hands each tile to a Dask worker so tiles are
computed concurrently, while still collecting the distribution fitting
parameters that compute.compute_index_parallel() produces per tile - something
compute.compute_index_dask() cannot do, because dask.array.map_blocks discards
everything but the returned block.

Two output backends:

``output_format='zarr'``
    Workers write their own region straight into a Zarr store. Regions are
    disjoint, so writes are lock-free and fully parallel, and a run that dies
    part-way keeps the tiles already written. Requires ``zarr``.

``output_format='netcdf'``
    Workers write their own slice into a pre-allocated NetCDF file under a
    distributed lock. HDF5 has no parallel-write support here, so writes are
    serialized - but they are short compared with the fitting, and compute
    still overlaps across workers.

Either way only the tile is ever in memory; the full grid is never
materialized. The fitting parameters are assembled on the client, where they
are small ((periods, lat, lon) per parameter), and written with
indices.save_fitting_params() so they reload with indices.load_fitting_params().

---
Author: Benny Istanto, GOST/DEC Data Group/The World Bank

Built upon the foundation of climate-indices by James Adams,
with substantial modifications for multi-distribution support,
bidirectional event analysis, and scalable processing.
---
"""

import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from config import (
    DASK_TILE_MULTIPLIER_SPEI,
    DASK_TILE_MULTIPLIER_SPI,
    DISTRIBUTION_PARAM_NAMES,
    MAX_DASK_TILE,
    MEMORY_SAFETY_FACTOR,
    NC_FILL_VALUE,
    PET_VAR_PATTERNS,
    Periodicity,
    PRECIP_VAR_PATTERNS,
    SPEI_WATER_BALANCE_OFFSET,
)
from utils import find_variable, get_global_attributes, get_logger

_logger = get_logger(__name__)


# =============================================================================
# TILE LAYOUT PLANNING
# =============================================================================

@dataclass
class DaskLayout:
    """A tile size / worker count pair that fits in available memory."""
    tile: int
    n_workers: int
    n_tiles: int
    per_tile_gb: float
    total_gb: float
    available_gb: float

    def __repr__(self) -> str:
        fit = "ok" if self.total_gb <= self.available_gb else "TOO BIG"
        return (
            f"DaskLayout(\n"
            f"  Tile size: {self.tile} x {self.tile}\n"
            f"  Workers: {self.n_workers}   Tiles: {self.n_tiles}\n"
            f"  Peak per worker: {self.per_tile_gb:.1f} GB\n"
            f"  Peak total: {self.total_gb:.1f} GB / {self.available_gb:.1f} GB available [{fit}]\n"
            f")"
        )


def plan_layout(
    n_time: int,
    n_lat: int,
    n_lon: int,
    n_workers: int,
    index: str = 'spei',
    available_memory_gb: Optional[float] = None,
    safety_factor: float = MEMORY_SAFETY_FACTOR,
    max_tile: int = MAX_DASK_TILE,
) -> DaskLayout:
    """
    Pick a tile size such that ``n_workers`` tiles fit in memory at once.

    This is the step that is easy to get wrong when moving from serial chunking
    to Dask: a tile size that is fine one-at-a-time becomes ``n_workers`` times
    larger in aggregate. A 1440x1440 SPEI tile over 912 months peaks near 60 GB,
    so 40 concurrent workers would need roughly 2.4 TB.

    :param n_time: number of time steps
    :param n_lat: number of latitude points
    :param n_lon: number of longitude points
    :param n_workers: number of concurrent Dask workers
    :param index: 'spi' or 'spei' - SPEI holds precip, PET and water balance
    :param available_memory_gb: RAM budget (auto-detected if None)
    :param safety_factor: fraction of available memory to target
    :param max_tile: upper bound on tile edge length
    :return: DaskLayout with a tile size that fits
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            _logger.warning("psutil not installed, assuming 16 GB available")
            available_memory_gb = 16.0

    # Peak working set per tile, as a multiple of one float32 tile array.
    # Measured against the intermediates in compute._rolling_sum_3d plus the
    # scaled/result arrays; SPEI additionally holds precip, PET and P-PET.
    multiplier = (DASK_TILE_MULTIPLIER_SPEI if index.lower() == 'spei'
                  else DASK_TILE_MULTIPLIER_SPI)
    budget_gb = available_memory_gb * safety_factor
    per_worker_gb = budget_gb / max(n_workers, 1)

    bytes_per_tile_cell = n_time * 4 * multiplier
    target_cells = (per_worker_gb * (1024 ** 3)) / bytes_per_tile_cell
    tile = int(math.sqrt(max(target_cells, 1.0)))
    tile = max(64, min(tile, max_tile, max(n_lat, n_lon)))

    n_tiles = math.ceil(n_lat / tile) * math.ceil(n_lon / tile)
    per_tile_gb = (tile * tile * bytes_per_tile_cell) / (1024 ** 3)

    return DaskLayout(
        tile=tile,
        n_workers=n_workers,
        n_tiles=n_tiles,
        per_tile_gb=per_tile_gb,
        total_gb=per_tile_gb * n_workers,
        available_gb=available_memory_gb,
    )


def _iter_tiles(n_lat: int, n_lon: int, tile_lat: int, tile_lon: int
                ) -> List[Tuple[int, int, int, int]]:
    """Return (lat_start, lat_end, lon_start, lon_end) for every tile."""
    out = []
    for y in range(0, n_lat, tile_lat):
        for x in range(0, n_lon, tile_lon):
            out.append((y, min(y + tile_lat, n_lat), x, min(x + tile_lon, n_lon)))
    return out


def _run_signature(
    index_type: str,
    scale: int,
    distribution: str,
    periodicity_value: int,
    calibration_start_year: int,
    calibration_end_year: int,
    shape: Tuple[int, int, int],
    tile: int,
) -> Dict:
    """
    Describe a run precisely enough that a resume cannot mix incompatible work.

    A manifest is only honoured when every field matches, so changing the
    scale, distribution, calibration or tiling starts from scratch instead of
    silently stitching tiles computed under different settings.
    """
    return {
        'index_type': index_type,
        'scale': scale,
        'distribution': distribution,
        'periodicity': periodicity_value,
        'calibration_start_year': calibration_start_year,
        'calibration_end_year': calibration_end_year,
        'shape': list(shape),
        'tile': tile,
    }


def _manifest_path(output_path: str) -> Path:
    """Sidecar file recording which tiles are finished."""
    return Path(str(output_path) + '.manifest.json')


def _tileparams_dir(output_path: str) -> Path:
    """Directory holding one small parameter file per completed tile."""
    return Path(str(output_path) + '_tileparams')


def _load_manifest(output_path: str, signature: Dict) -> set:
    """
    Return the set of already-completed tile bounds for this exact run.

    :param output_path: output store or file path
    :param signature: expected run signature from _run_signature()
    :return: set of (y0, y1, x0, x1) tuples, empty if there is nothing usable
    """
    path = _manifest_path(output_path)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        _logger.warning(f"Ignoring unreadable manifest {path.name}: {exc}")
        return set()

    if data.get('signature') != signature:
        _logger.warning(
            f"Manifest {path.name} was written for different settings; "
            "ignoring it and recomputing every tile."
        )
        return set()
    return {tuple(b) for b in data.get('completed', [])}


def _write_manifest(output_path: str, signature: Dict, completed: set) -> None:
    """Persist the completed-tile set. Called after every tile, so a crash keeps progress."""
    path = _manifest_path(output_path)
    payload = {
        'signature': signature,
        'completed': sorted(list(b) for b in completed),
    }
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        tmp.write_text(json.dumps(payload), encoding='utf-8')
        tmp.replace(path)
    except OSError as exc:
        _logger.warning(f"Could not update manifest {path.name}: {exc}")


def _save_tile_params(output_path: str, bounds: Tuple[int, int, int, int],
                      params: Dict[str, np.ndarray]) -> None:
    """
    Write one tile's fitting parameters beside the output.

    Keeping them on disk rather than in a client-side accumulator means a
    crash (or an unclean cluster shutdown) never loses completed work, and a
    resumed run does not have to recompute tiles just to recover parameters.
    """
    d = _tileparams_dir(output_path)
    d.mkdir(parents=True, exist_ok=True)
    y0, y1, x0, x1 = bounds
    out = d / f"tile_{y0}_{y1}_{x0}_{x1}.npz"
    tmp = out.with_suffix('.npz.tmp')
    # Write through a file handle: given a path, np.savez_compressed appends
    # '.npz' when the name does not already end in it, which would leave the
    # data at '<name>.npz.tmp.npz' and make the rename below fail.
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **{k: v for k, v in params.items()})
    tmp.replace(out)


def _load_tile_params(output_path: str, bounds: Tuple[int, int, int, int]
                      ) -> Optional[Dict[str, np.ndarray]]:
    """Read back one tile's parameters, or None if absent."""
    y0, y1, x0, x1 = bounds
    p = _tileparams_dir(output_path) / f"tile_{y0}_{y1}_{x0}_{x1}.npz"
    if not p.exists():
        return None
    try:
        with np.load(p) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError) as exc:
        _logger.warning(f"Could not read {p.name}: {exc}")
        return None


def _write_with_retry(
    write_fn,
    bounds: Tuple[int, int, int, int],
    attempts: int = 6,
    base_delay: float = 1.0,
) -> None:
    """
    Run a tile write, retrying transient filesystem locks with backoff.

    On Windows an antivirus or indexing service routinely holds a newly created
    file open for a fraction of a second. Zarr writes each chunk to
    ``<chunk>.<uuid>.partial`` and renames it into place, and that rename then
    fails with ``WinError 32``. The condition clears on its own, so retrying is
    the correct response - failing would throw away a tile that took minutes of
    compute and leave a NaN hole in the output.

    Only OSError is retried; anything else is a real error and propagates
    immediately.

    :param write_fn: zero-argument callable performing the write
    :param bounds: tile bounds, for log messages
    :param attempts: total tries before giving up
    :param base_delay: first backoff in seconds, doubled each retry
    :raises OSError: if every attempt fails
    """
    import time

    for attempt in range(1, attempts + 1):
        try:
            write_fn()
            if attempt > 1:
                _logger.info(f"Tile {bounds} written on attempt {attempt}")
            return
        except OSError as exc:
            if attempt == attempts:
                _logger.error(
                    f"Tile {bounds} still unwritable after {attempts} attempts: {exc}"
                )
                raise
            delay = base_delay * (2 ** (attempt - 1))
            _logger.warning(
                f"Tile {bounds} write blocked (attempt {attempt}/{attempts}): {exc}. "
                f"Retrying in {delay:.0f}s"
            )
            time.sleep(delay)


def _find_var(ds: xr.Dataset, patterns: List[str], kind: str = 'variable') -> str:
    """
    Find the data variable matching the given name patterns.

    Delegates to utils.find_variable, which matches most-specific-first so
    short patterns like 'pr' and 'et' cannot hijack 'pressure' or 'wetdays'.
    """
    return find_variable(ds, patterns, kind=kind)


def _encode_time(time_coord: xr.DataArray) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Encode a datetime64 time coordinate as CF numeric values plus attributes.

    Needed because the NetCDF template is written with netCDF4 directly rather
    than through xarray, so CF time encoding has to be done explicitly.

    :param time_coord: time coordinate DataArray
    :return: (numeric values, attribute dict with units and calendar)
    """
    from xarray.coding.times import encode_cf_datetime

    values, units, calendar = encode_cf_datetime(time_coord.values)
    return np.asarray(values), {'units': units, 'calendar': calendar}


# =============================================================================
# WORKER TASK
# =============================================================================

def _process_tile(
    bounds: Tuple[int, int, int, int],
    precip_path: str,
    precip_var: str,
    pet_path: Optional[str],
    pet_var: Optional[str],
    output_path: str,
    output_format: str,
    var_name_out: str,
    scale: int,
    data_start_year: int,
    calibration_start_year: int,
    calibration_end_year: int,
    periodicity_value: int,
    distribution: str,
    lock_name: Optional[str],
) -> Tuple[Tuple[int, int, int, int], Dict[str, np.ndarray]]:
    """
    Compute one spatial tile on a worker and write it to the output store.

    Runs in a separate process, so every argument is a plain picklable value and
    the worker opens the input files itself. Only the fitting parameters travel
    back to the client - the index values go straight to disk.

    :param bounds: (lat_start, lat_end, lon_start, lon_end)
    :param precip_path: path to precipitation NetCDF
    :param precip_var: precipitation variable name
    :param pet_path: path to PET NetCDF, or None for SPI
    :param pet_var: PET variable name, or None for SPI
    :param output_path: Zarr store or NetCDF file to write into
    :param output_format: 'zarr' or 'netcdf'
    :param var_name_out: output variable name
    :param scale: accumulation scale
    :param data_start_year: first year of the data
    :param calibration_start_year: calibration start year
    :param calibration_end_year: calibration end year
    :param periodicity_value: Periodicity enum value (12 or 366)
    :param distribution: distribution name
    :param lock_name: distributed lock name for NetCDF writes, else None
    :return: (bounds, params dict of (periods, tile_lat, tile_lon) arrays)
    """
    import numpy as _np
    import xarray as _xr

    from compute import compute_index_parallel
    from config import Periodicity as _Periodicity

    y0, y1, x0, x1 = bounds
    periodicity = _Periodicity(periodicity_value)

    # --- read this tile only ---
    with _xr.open_dataset(precip_path) as pds:
        precip = pds[precip_var].isel(
            lat=slice(y0, y1), lon=slice(x0, x1)
        ).values.astype(_np.float32)

    if pet_path is not None:
        with _xr.open_dataset(pet_path) as eds:
            pet = eds[pet_var].isel(
                lat=slice(y0, y1), lon=slice(x0, x1)
            ).values.astype(_np.float32)
        values = (precip - pet) + SPEI_WATER_BALANCE_OFFSET
        del pet
    else:
        values = _np.clip(precip, 0, None)
    del precip
    gc.collect()

    # --- compute ---
    result, params = compute_index_parallel(
        values,
        scale=scale,
        data_start_year=data_start_year,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        periodicity=periodicity,
        distribution=distribution,
    )
    del values
    gc.collect()

    result = result.astype(_np.float32)

    # --- write this tile ---
    #
    # Retried, because on Windows a freshly written chunk is regularly held
    # open for a moment by an antivirus or indexing service. Zarr writes each
    # chunk to '<chunk>.<uuid>.partial' and then renames it, and that rename
    # fails with WinError 32 ("being used by another process") if the scanner
    # still has the file. It is transient - a short backoff clears it - but
    # without a retry it discards a tile that took minutes to compute.
    if output_format == 'zarr':
        def _write():
            tile_ds = _xr.Dataset(
                {var_name_out: (('time', 'lat', 'lon'), result)}
            )
            # Region writes must not carry coordinates spanning the full dims
            tile_ds.to_zarr(
                output_path,
                region={'time': slice(None),
                        'lat': slice(y0, y1),
                        'lon': slice(x0, x1)},
            )

        _write_with_retry(_write, bounds)
    else:
        import netCDF4

        def _write():
            lock = None
            if lock_name is not None:
                from dask.distributed import Lock
                lock = Lock(lock_name)
                lock.acquire()
            try:
                with netCDF4.Dataset(output_path, mode='r+') as nc:
                    nc.variables[var_name_out][:, y0:y1, x0:x1] = result
            finally:
                if lock is not None:
                    lock.release()

        _write_with_retry(_write, bounds)

    del result
    gc.collect()

    # Drop the non-array 'distribution' marker before shipping back
    params_out = {k: v for k, v in params.items() if isinstance(v, _np.ndarray)}
    return bounds, params_out


# =============================================================================
# CORE DRIVER
# =============================================================================

def _compute_dask(
    precip_path: Union[str, Path],
    pet_path: Optional[Union[str, Path]],
    output_path: Union[str, Path],
    index_type: str,
    scale: int,
    periodicity: Union[str, Periodicity],
    calibration_start_year: int,
    calibration_end_year: int,
    distribution: str,
    output_format: str,
    tile: Optional[int],
    n_workers: Optional[int],
    save_params: bool,
    params_path: Optional[str],
    compress: bool,
    complevel: int,
    precip_var_name: Optional[str],
    pet_var_name: Optional[str],
    global_attrs: Optional[Dict],
    memory_limit: Optional[str],
    dashboard: bool,
    resume: bool,
) -> str:
    """Shared driver behind compute_spi_dask / compute_spei_dask."""
    from dask.distributed import Client, LocalCluster, Lock, as_completed

    from indices import save_fitting_params
    from utils import get_data_year_range, get_variable_attributes, get_variable_name

    if isinstance(periodicity, str):
        periodicity = Periodicity.from_string(periodicity)

    output_format = output_format.lower()
    if output_format not in ('zarr', 'netcdf'):
        raise ValueError(f"output_format must be 'zarr' or 'netcdf', got: {output_format}")

    dist = distribution.lower()
    precip_path = str(precip_path)
    pet_path = str(pet_path) if pet_path is not None else None
    output_path = str(output_path)

    # --- inspect inputs ---
    with xr.open_dataset(precip_path) as pds:
        if precip_var_name is None:
            precip_var_name = _find_var(pds, PRECIP_VAR_PATTERNS,
                                        kind='precipitation variable')
        pvar = pds[precip_var_name]
        if pvar.dims != ('time', 'lat', 'lon'):
            raise ValueError(
                f"Expected dims ('time','lat','lon'), got {pvar.dims}. "
                "Transpose the file before processing."
            )
        n_time, n_lat, n_lon = pvar.shape
        data_start_year, _ = get_data_year_range(pds)
        time_coord = pds['time'].copy(deep=True)
        lat_coord = pds['lat'].copy(deep=True)
        lon_coord = pds['lon'].copy(deep=True)

    if pet_path is not None and pet_var_name is None:
        with xr.open_dataset(pet_path) as eds:
            pet_var_name = _find_var(eds, PET_VAR_PATTERNS, kind='PET variable')

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) // 2)

    # --- plan tiles ---
    layout = plan_layout(n_time, n_lat, n_lon, n_workers, index=index_type)
    if tile is not None:
        # Honour an explicit tile size, but re-report the memory it implies
        multiplier = (DASK_TILE_MULTIPLIER_SPEI if index_type == 'spei'
                      else DASK_TILE_MULTIPLIER_SPI)
        layout.tile = tile
        layout.per_tile_gb = (tile * tile * n_time * 4 * multiplier) / (1024 ** 3)
        layout.total_gb = layout.per_tile_gb * n_workers
        layout.n_tiles = math.ceil(n_lat / tile) * math.ceil(n_lon / tile)
    tile = layout.tile

    _logger.info(f"\n{layout}")
    if layout.total_gb > layout.available_gb:
        _logger.warning(
            f"Planned peak {layout.total_gb:.1f} GB exceeds {layout.available_gb:.1f} GB "
            f"available. Reduce tile= or n_workers=."
        )

    tiles = _iter_tiles(n_lat, n_lon, tile, tile)
    var_name_out = get_variable_name(index_type, scale, periodicity, distribution=dist)

    title = (f'Standardized Precipitation Index (SPI-{scale})' if index_type == 'spi'
             else f'Standardized Precipitation Evapotranspiration Index (SPEI-{scale})')
    attrs = get_global_attributes(
        title=title,
        distribution=dist,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        global_attrs=global_attrs,
    )

    # --- decide whether to (re)create the output template ---
    # Creating it is destructive: Zarr uses mode='w' and netCDF4 mode='w',
    # both of which erase any tiles already written. So only do it when there
    # is nothing to resume from, otherwise a resumed run would silently wipe
    # the work it was supposed to continue.
    signature = _run_signature(
        index_type, scale, dist, periodicity.value,
        calibration_start_year, calibration_end_year,
        (n_time, n_lat, n_lon), tile,
    )
    prior = _load_manifest(output_path, signature) if resume else set()
    output_exists = Path(output_path).exists()
    fresh_template = not (resume and prior and output_exists)

    if not fresh_template:
        _logger.info(
            f"Reusing existing {output_format} output with "
            f"{len(prior)} completed tile(s): {output_path}"
        )

    var_attrs = get_variable_attributes(index_type, scale, periodicity, distribution=dist)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not fresh_template:
        pass
    elif output_format == 'zarr':
        _logger.info(f"Creating zarr template: {output_path}")
        import dask.array as dsa
        empty = dsa.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32,
                         chunks=(n_time, tile, tile))
        template = xr.Dataset(
            {var_name_out: (('time', 'lat', 'lon'), empty)},
            coords={'time': time_coord, 'lat': lat_coord, 'lon': lon_coord},
            attrs=attrs,
        )
        template[var_name_out].attrs = var_attrs
        # compute=False writes only metadata and chunk layout, no data
        template.to_zarr(output_path, mode='w', compute=False, consolidated=True)
        del template, empty
    else:
        import netCDF4

        _logger.info(f"Creating netcdf template: {output_path}")
        with netCDF4.Dataset(output_path, mode='w', format='NETCDF4') as nc:
            nc.createDimension('time', n_time)
            nc.createDimension('lat', n_lat)
            nc.createDimension('lon', n_lon)

            for name, coord in (('time', time_coord), ('lat', lat_coord), ('lon', lon_coord)):
                if name == 'time':
                    vals, enc_attrs = _encode_time(coord)
                    v = nc.createVariable(name, 'f8', (name,))
                    v[:] = vals
                    v.setncatts(enc_attrs)
                else:
                    v = nc.createVariable(name, 'f8', (name,))
                    v[:] = coord.values
                    v.setncatts({k: str(x) for k, x in coord.attrs.items()})

            dv = nc.createVariable(
                var_name_out, 'f4', ('time', 'lat', 'lon'),
                zlib=compress, complevel=complevel,
                fill_value=NC_FILL_VALUE,
                chunksizes=(min(12, n_time), min(tile, n_lat), min(tile, n_lon)),
            )
            dv.setncatts({k: v for k, v in var_attrs.items() if v is not None})
            nc.setncatts({k: v for k, v in attrs.items() if v is not None})

    param_names = DISTRIBUTION_PARAM_NAMES.get(dist, ("alpha", "beta", "prob_zero"))

    # --- work out what is left to do ---
    completed = prior
    if completed and not fresh_template:
        todo = [b for b in tiles if tuple(b) not in completed]
        _logger.info(
            f"Resuming: {len(completed)} of {len(tiles)} tiles already done, "
            f"{len(todo)} remaining"
        )
    else:
        completed = set()
        todo = list(tiles)

    if not todo:
        _logger.info("All tiles already present; nothing to compute")

    # --- run ---
    cluster_kwargs = dict(n_workers=n_workers, threads_per_worker=1,
                          processes=True)
    if memory_limit is not None:
        cluster_kwargs['memory_limit'] = memory_limit
    if not dashboard:
        cluster_kwargs['dashboard_address'] = None

    lock_name = None if output_format == 'zarr' else f'ncwrite-{os.getpid()}'

    cluster = client = None
    failures: List[Tuple[Tuple[int, int, int, int], str]] = []

    if todo:
        _logger.info(f"Starting Dask cluster: {n_workers} workers, {len(todo)} tiles")
        try:
            cluster = LocalCluster(**cluster_kwargs)
            client = Client(cluster)
            if dashboard:
                _logger.info(f"Dashboard: {client.dashboard_link}")
            if lock_name is not None:
                Lock(lock_name)  # register before workers reference it

            future_to_bounds = {}
            for b in todo:
                fut = client.submit(
                    _process_tile,
                    b, precip_path, precip_var_name, pet_path, pet_var_name,
                    output_path, output_format, var_name_out, scale,
                    data_start_year, calibration_start_year, calibration_end_year,
                    periodicity.value, dist, lock_name,
                    pure=False,
                )
                future_to_bounds[fut] = tuple(b)

            done = 0
            for future in as_completed(list(future_to_bounds)):
                submitted = future_to_bounds.get(future)
                try:
                    bounds, params = future.result()
                except Exception as exc:
                    # One bad tile must not abandon the other 49. Record it,
                    # leave it out of the manifest, and carry on.
                    failures.append((submitted, f"{type(exc).__name__}: {exc}"))
                    _logger.error(f"Tile {submitted} failed: {exc}")
                    del future
                    gc.collect()
                    continue

                done += 1
                y0, y1, x0, x1 = bounds
                _logger.info(
                    f"[{done}/{len(todo)}] tile lat {y0}:{y1}, lon {x0}:{x1} "
                    f"({done / len(todo) * 100:.1f}%)"
                )

                # Persist this tile's parameters and mark it done immediately,
                # so progress survives a crash or an unclean shutdown.
                if save_params:
                    _save_tile_params(output_path, tuple(bounds), params)
                completed.add(tuple(bounds))
                _write_manifest(output_path, signature, completed)

                del params, future
                gc.collect()
        finally:
            # Best-effort shutdown. On Windows the nannies regularly miss the
            # 4-second kill deadline and Cluster.__exit__ raises TimeoutError,
            # which previously destroyed an otherwise-successful run and took
            # the parameter file with it. A shutdown hiccup is not a failure.
            for obj, label in ((client, 'Dask client'), (cluster, 'Dask cluster')):
                if obj is None:
                    continue
                try:
                    obj.close()
                except Exception as exc:
                    _logger.warning(
                        f"{label} did not shut down cleanly ({type(exc).__name__}: "
                        f"{exc}). Computed tiles are unaffected."
                    )

    if failures:
        _logger.error(f"{len(failures)} tile(s) failed:")
        for b, msg in failures:
            _logger.error(f"  lat {b[0]}:{b[1]}, lon {b[2]}:{b[3]} -> {msg}")

    # --- assemble and save parameters from the per-tile sidecars ---
    if save_params:
        if params_path is None:
            base = output_path[:-5] if output_path.endswith('.zarr') else output_path
            params_path = str(base).replace('.nc', '') + '_params.nc'

        missing = [b for b in tiles if tuple(b) not in completed]
        if missing:
            _logger.warning(
                f"Writing parameters with {len(missing)} tile(s) still missing; "
                "those regions stay NaN. Re-run to fill them in."
            )

        periods = periodicity.value
        gb = len(param_names) * periods * n_lat * n_lon * 4 / (1024 ** 3)
        _logger.info(f"Assembling {len(param_names)} parameters ({gb:.2f} GB) from tile files")
        all_params = {
            p: np.full((periods, n_lat, n_lon), np.nan, dtype=np.float32)
            for p in param_names
        }
        n_loaded = 0
        for b in tiles:
            bt = tuple(b)
            if bt not in completed:
                continue
            tp = _load_tile_params(output_path, bt)
            if tp is None:
                _logger.warning(f"Parameters missing for tile {bt}; region stays NaN")
                continue
            y0, y1, x0, x1 = bt
            for p in param_names:
                if p in tp:
                    all_params[p][:, y0:y1, x0:x1] = tp[p]
            n_loaded += 1
            del tp

        _logger.info(f"Saving fitting parameters from {n_loaded} tiles: {params_path}")
        save_fitting_params(
            all_params,
            params_path,
            scale=scale,
            periodicity=periodicity,
            index_type=index_type,
            calibration_start_year=calibration_start_year,
            calibration_end_year=calibration_end_year,
            coords={'lat': lat_coord.values, 'lon': lon_coord.values},
            distribution=dist,
        )
        del all_params
        gc.collect()

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(todo)} tile(s) failed; the rest were written and "
            f"recorded in {_manifest_path(output_path).name}. Re-run with resume=True "
            f"to retry only the failures. First error: {failures[0][1]}"
        )

    _logger.info(f"Dask {index_type.upper()} computation complete: {output_path}")
    return output_path


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_spi_dask(
    precip_path: Union[str, Path],
    output_path: Union[str, Path],
    scale: int = 12,
    periodicity: Union[str, Periodicity] = Periodicity.monthly,
    calibration_start_year: int = 1991,
    calibration_end_year: int = 2020,
    distribution: str = 'gamma',
    output_format: str = 'zarr',
    tile: Optional[int] = None,
    n_workers: Optional[int] = None,
    save_params: bool = True,
    params_path: Optional[str] = None,
    compress: bool = True,
    complevel: int = 4,
    var_name: Optional[str] = None,
    global_attrs: Optional[Dict] = None,
    memory_limit: Optional[str] = None,
    dashboard: bool = True,
    resume: bool = True,
) -> str:
    """
    Compute global SPI with Dask workers, saving fitting parameters.

    :param precip_path: path to precipitation NetCDF, dims (time, lat, lon)
    :param output_path: '.zarr' store or '.nc' file, matching output_format
    :param scale: accumulation scale in time steps
    :param periodicity: 'monthly' or 'daily'
    :param calibration_start_year: calibration start year
    :param calibration_end_year: calibration end year
    :param distribution: 'gamma', 'pearson3', 'log_logistic', ...
    :param output_format: 'zarr' (parallel writes) or 'netcdf' (locked writes)
    :param tile: tile edge in grid cells; auto-planned from RAM when None
    :param n_workers: concurrent workers; half the CPU count when None
    :param save_params: collect and write distribution fitting parameters
    :param params_path: parameter file path, derived from output_path when None
    :param compress: zlib compression for NetCDF output
    :param complevel: compression level 1-9
    :param var_name: precipitation variable name, auto-detected when None
    :param global_attrs: extra global attributes
    :param memory_limit: per-worker limit, e.g. '16GB'
    :param dashboard: start the Dask dashboard
    :param resume: reuse an existing output and skip tiles already recorded in
        its ``.manifest.json``. Set False to recompute everything from scratch,
        which recreates the output and erases any prior tiles.
    :return: output_path

    Example:
        >>> compute_spi_dask(
        ...     'wld_chirps_ppt.nc', 'spi_12.zarr',
        ...     scale=12, distribution='gamma',
        ...     output_format='zarr', n_workers=8,
        ...     params_path='spi_12_params.nc',
        ... )
    """
    return _compute_dask(
        precip_path=precip_path, pet_path=None, output_path=output_path,
        index_type='spi', scale=scale, periodicity=periodicity,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        distribution=distribution, output_format=output_format,
        tile=tile, n_workers=n_workers, save_params=save_params,
        params_path=params_path, compress=compress, complevel=complevel,
        precip_var_name=var_name, pet_var_name=None,
        global_attrs=global_attrs, memory_limit=memory_limit,
        dashboard=dashboard, resume=resume,
    )


def compute_spei_dask(
    precip_path: Union[str, Path],
    pet_path: Union[str, Path],
    output_path: Union[str, Path],
    scale: int = 12,
    periodicity: Union[str, Periodicity] = Periodicity.monthly,
    calibration_start_year: int = 1991,
    calibration_end_year: int = 2020,
    distribution: str = 'pearson3',
    output_format: str = 'zarr',
    tile: Optional[int] = None,
    n_workers: Optional[int] = None,
    save_params: bool = True,
    params_path: Optional[str] = None,
    compress: bool = True,
    complevel: int = 4,
    precip_var_name: Optional[str] = None,
    pet_var_name: Optional[str] = None,
    global_attrs: Optional[Dict] = None,
    memory_limit: Optional[str] = None,
    dashboard: bool = True,
    resume: bool = True,
) -> str:
    """
    Compute global SPEI with Dask workers, saving fitting parameters.

    :param precip_path: path to precipitation NetCDF, dims (time, lat, lon)
    :param pet_path: path to PET NetCDF with matching grid and time axis
    :param output_path: '.zarr' store or '.nc' file, matching output_format
    :param scale: accumulation scale in time steps
    :param periodicity: 'monthly' or 'daily'
    :param calibration_start_year: calibration start year
    :param calibration_end_year: calibration end year
    :param distribution: 'pearson3' recommended for SPEI
    :param output_format: 'zarr' (parallel writes) or 'netcdf' (locked writes)
    :param tile: tile edge in grid cells; auto-planned from RAM when None
    :param n_workers: concurrent workers; half the CPU count when None
    :param save_params: collect and write distribution fitting parameters
    :param params_path: parameter file path, derived from output_path when None
    :param compress: zlib compression for NetCDF output
    :param complevel: compression level 1-9
    :param precip_var_name: precipitation variable, auto-detected when None
    :param pet_var_name: PET variable, auto-detected when None
    :param global_attrs: extra global attributes
    :param memory_limit: per-worker limit, e.g. '16GB'
    :param dashboard: start the Dask dashboard
    :param resume: reuse an existing output and skip tiles already recorded in
        its ``.manifest.json``. Set False to recompute everything from scratch,
        which recreates the output and erases any prior tiles.
    :return: output_path

    Example:
        >>> compute_spei_dask(
        ...     'terraclimate_ppt.nc', 'terraclimate_pet.nc',
        ...     'spei_12.zarr', scale=12, distribution='pearson3',
        ...     output_format='zarr', n_workers=8,
        ... )
    """
    return _compute_dask(
        precip_path=precip_path, pet_path=pet_path, output_path=output_path,
        index_type='spei', scale=scale, periodicity=periodicity,
        calibration_start_year=calibration_start_year,
        calibration_end_year=calibration_end_year,
        distribution=distribution, output_format=output_format,
        tile=tile, n_workers=n_workers, save_params=save_params,
        params_path=params_path, compress=compress, complevel=complevel,
        precip_var_name=precip_var_name, pet_var_name=pet_var_name,
        global_attrs=global_attrs, memory_limit=memory_limit,
        dashboard=dashboard, resume=resume,
    )


def zarr_to_netcdf(
    zarr_path: Union[str, Path],
    netcdf_path: Union[str, Path],
    compress: bool = True,
    complevel: int = 4,
    time_chunk: int = 12,
) -> str:
    """
    Stream a Zarr store out to NetCDF without loading it all into memory.

    :param zarr_path: source Zarr store
    :param netcdf_path: destination NetCDF file
    :param compress: zlib compression
    :param complevel: compression level 1-9
    :param time_chunk: NetCDF chunk length along time
    :return: netcdf_path
    """
    ds = xr.open_zarr(str(zarr_path))
    try:
        encoding = {
            v: {
                'dtype': 'float32',
                '_FillValue': NC_FILL_VALUE,
                'zlib': compress,
                'complevel': complevel,
            }
            for v in ds.data_vars
        }
        for v in ds.data_vars:
            shp = ds[v].shape
            if len(shp) == 3:
                encoding[v]['chunksizes'] = (
                    min(time_chunk, shp[0]), min(512, shp[1]), min(512, shp[2])
                )
        _logger.info(f"Streaming {zarr_path} -> {netcdf_path}")
        ds.to_netcdf(str(netcdf_path), encoding=encoding)
    finally:
        ds.close()
    _logger.info(f"Wrote {netcdf_path}")
    return str(netcdf_path)
