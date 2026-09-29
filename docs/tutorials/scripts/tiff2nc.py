#!/usr/bin/env python3
"""Convert a folder of GeoTIFFs into one CF-compliant NetCDF with a time axis.

Lineage: this is a generalisation of a script by Benny Istanto (WFP 2020,
World Bank 2023), itself based on Rich Signell's answer at
https://gis.stackexchange.com/a/70487. The 2023 revision fixed the half-pixel
shift, which is the single most important thing this script does and is
preserved here verbatim in intent.

The half-pixel problem
----------------------
A GeoTIFF's geotransform describes the **outer corner** of the top-left pixel.
A NetCDF coordinate variable describes the **centre** of each cell. Copy the
geotransform origin straight into a coordinate array and the whole grid is
offset by half a pixel, which on a 4 km grid is 2 km. Nothing errors; the data
just quietly sits in the wrong place, and it will not line up with any other
dataset you compare it against.

    lon = origin_x + arange(nx) * res_x + res_x / 2
    lat = origin_y + arange(ny) * res_y + res_y / 2      # res_y is negative

What this adds over the original
--------------------------------
* every path, name and attribute is a parameter, nothing is hardcoded
* the grid of every file is checked against the first one, so a stray raster
  on a different grid raises instead of silently corrupting a time step
* per-file nodata is read and converted, rather than assuming -9999
* geospatial bounds are computed and written as global attributes, which is
  where ACDD puts them, rather than typed by hand onto the variable
* chunking is set for time-series access
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = ["tiffs_to_netcdf", "cell_centres"]

EPS = 1e-9


def cell_centres(transform, width: int, height: int):
    """Cell-centre coordinates from a rasterio/affine transform.

    This is the half-pixel correction. `transform.c` / `transform.f` are the
    corner of the top-left pixel; adding half a pixel moves us to its centre.
    """
    res_x, res_y = transform.a, transform.e          # res_y is normally negative
    lon = transform.c + np.arange(width) * res_x + res_x / 2.0
    lat = transform.f + np.arange(height) * res_y + res_y / 2.0
    return lon, lat


def _parse_date(name: str, pattern: str, fmt: str) -> datetime:
    m = re.search(pattern, name)
    if not m:
        raise ValueError(f"{name}: no date matched {pattern!r}")
    return datetime.strptime(m.group(1), fmt)


def tiffs_to_netcdf(
    src_dir: os.PathLike | str,
    output_path: os.PathLike | str,
    var_name: str,
    *,
    glob: str = "*.tif",
    date_pattern: str = r"(\d{8})",
    date_format: str = "%Y%m%d",
    units: str = "mm",
    standard_name: Optional[str] = None,
    long_name: Optional[str] = None,
    time_units: Optional[str] = None,
    calendar: str = "standard",
    lat_order: str = "descending",
    dtype: str = "float32",
    nodata: Optional[float] = None,
    fill_value: float = -9999.0,
    complevel: int = 5,
    chunks: Optional[Sequence[int]] = None,
    global_attrs: Optional[dict] = None,
    report: bool = True,
) -> Path:
    """Stack GeoTIFFs into a NetCDF time series.

    Parameters
    ----------
    date_pattern, date_format
        How to read the timestamp out of each filename. The pattern needs one
        capture group; the format is handed to `strptime`.
    lat_order
        `"descending"` keeps the GeoTIFF's natural north-to-south row order and
        copies rows as they are. `"ascending"` flips both the coordinate and
        the data. Neither is more correct; pick one and be consistent, because
        a mismatch between the two is a silent north-south flip.
    nodata
        Value in the source rasters meaning "missing", used when the file does
        not declare one. **CHIRPS GeoTIFFs are exactly this case**: they carry
        no nodata tag but use -9999, so without this every ocean cell arrives
        as -9999 mm of rain. If the file does declare a nodata value, that one
        is used and this is ignored.
    fill_value
        Written as `_FillValue`. Source nodata is converted to it.
    """
    import netCDF4 as nc
    import rasterio

    src_dir, output_path = Path(src_dir), Path(output_path)
    files = sorted(src_dir.glob(glob))
    if not files:
        raise FileNotFoundError(f"no files matched {glob!r} in {src_dir}")

    dated = sorted(((_parse_date(f.name, date_pattern, date_format), f)
                    for f in files), key=lambda t: t[0])
    dates = [d for d, _ in dated]
    if len(set(dates)) != len(dates):
        dupes = sorted({d for d in dates if dates.count(d) > 1})
        raise ValueError(f"duplicate timestamps: {[str(d)[:10] for d in dupes]}")

    # ---- grid from the first file, then enforced on the rest --------------- #
    with rasterio.open(dated[0][1]) as r0:
        transform, width, height = r0.transform, r0.width, r0.height
        crs = r0.crs
        declared_nodata = r0.nodata
        if transform.b or transform.d:
            raise ValueError("rotated/sheared rasters are not supported")
        lon, lat = cell_centres(transform, width, height)

    if declared_nodata is None and nodata is None and report:
        print("warning   : the source declares no nodata value. If it uses a "
              "sentinel such as -9999,")
        print("            pass nodata=-9999 or it will be read as real data.")

    flip = lat_order == "ascending"
    if flip:
        lat = lat[::-1]
    elif lat_order != "descending":
        raise ValueError("lat_order must be 'descending' or 'ascending'")

    if report:
        print(f"files     : {len(files)}  {dates[0]:%Y-%m} .. {dates[-1]:%Y-%m}")
        print(f"grid      : {height} x {width}  res "
              f"{abs(transform.a):.6f} x {abs(transform.e):.6f}  crs {crs}")
        print(f"lon       : {lon[0]:.6f} .. {lon[-1]:.6f}  (cell centres)")
        print(f"lat       : {lat[0]:.6f} .. {lat[-1]:.6f}  ({lat_order})")

    if time_units is None:
        time_units = f"days since {dates[0]:%Y-%m-%d} 00:00:00"

    if chunks is None:
        chunks = (min(len(dates), 512), min(height, 256), min(width, 256))
    chunks = tuple(int(c) for c in chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    ds = nc.Dataset(output_path, "w", format="NETCDF4")
    try:
        ds.createDimension("time", None)
        ds.createDimension("lat", height)
        ds.createDimension("lon", width)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units, tv.calendar = time_units, calendar
        tv.standard_name, tv.long_name, tv.axis = "time", "time", "T"

        yv = ds.createVariable("lat", "f8", ("lat",))
        yv.units, yv.axis = "degrees_north", "Y"
        yv.standard_name = yv.long_name = "latitude"
        yv[:] = lat

        xv = ds.createVariable("lon", "f8", ("lon",))
        xv.units, xv.axis = "degrees_east", "X"
        xv.standard_name = xv.long_name = "longitude"
        xv[:] = lon

        cv = ds.createVariable("crs", "i4")
        cv.grid_mapping_name = "latitude_longitude"
        cv.longitude_of_prime_meridian = 0.0
        cv.semi_major_axis = 6378137.0
        cv.inverse_flattening = 298.257223563
        cv.long_name = "CRS definition"
        if crs is not None:
            cv.spatial_ref = crs.to_wkt()
            if crs.to_epsg():
                cv.epsg_code = f"EPSG:{crs.to_epsg()}"

        dv = ds.createVariable(var_name, dtype, ("time", "lat", "lon"),
                               zlib=True, complevel=complevel,
                               chunksizes=chunks, fill_value=fill_value)
        dv.units = units
        dv.standard_name = standard_name or var_name
        dv.long_name = long_name or var_name
        dv.grid_mapping = "crs"

        res_y = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 0.0
        res_x = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0.0
        attrs = {
            "Conventions": "CF-1.8",
            "cdm_data_type": "GRID",
            "geospatial_lat_min": float(lat.min()),
            "geospatial_lat_max": float(lat.max()),
            "geospatial_lon_min": float(lon.min()),
            "geospatial_lon_max": float(lon.max()),
            "geospatial_lat_resolution": res_y,
            "geospatial_lon_resolution": res_x,
            "geospatial_lat_units": "degrees_north",
            "geospatial_lon_units": "degrees_east",
            "time_coverage_start": f"{dates[0]:%Y-%m-%d}",
            "time_coverage_end": f"{dates[-1]:%Y-%m-%d}",
            "date_created": f"{datetime.now(timezone.utc):%Y-%m-%d}",
            "history": (f"{datetime.now(timezone.utc):%a %b %d %H:%M:%S %Y} UTC: "
                        f"built from {len(files)} GeoTIFFs using tiff2nc.py"),
        }
        attrs.update(global_attrs or {})
        ds.setncatts(attrs)

        base = nc.date2num(dates[0], time_units, calendar)
        for i, (date, path) in enumerate(dated):
            with rasterio.open(path) as r:
                if (r.width, r.height) != (width, height):
                    raise ValueError(
                        f"{path.name}: grid is {r.height}x{r.width}, "
                        f"expected {height}x{width}")
                if max(abs(a - b) for a, b in zip(r.transform[:6], transform[:6])) > EPS:
                    raise ValueError(
                        f"{path.name}: geotransform differs from the first file")
                arr = r.read(1).astype("float64")
                file_nodata = r.nodata

            # A value the file declares wins; otherwise use the override.
            nd = file_nodata if file_nodata is not None else nodata
            if nd is not None and not np.isnan(nd):
                arr = np.where(arr == nd, np.nan, arr)
            if flip:
                arr = np.flipud(arr)

            arr = np.where(np.isnan(arr), fill_value, arr).astype(dtype)
            tv[i] = nc.date2num(date, time_units, calendar)
            dv[i, :, :] = arr
            if report and (i + 1) % 50 == 0:
                print(f"\r  {i + 1}/{len(dated)}", end="", flush=True)
        if report:
            print(f"\r  {len(dated)}/{len(dated)}")
    finally:
        ds.close()

    if report:
        mb = output_path.stat().st_size / 1024 ** 2
        print(f"wrote     : {output_path.name} "
              f"({mb/1024:.2f} GB)" if mb > 1024 else
              f"wrote     : {output_path.name} ({mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stack GeoTIFFs into a CF NetCDF")
    ap.add_argument("src_dir")
    ap.add_argument("output")
    ap.add_argument("var_name")
    ap.add_argument("--glob", default="*.tif")
    ap.add_argument("--date-pattern", default=r"(\d{8})")
    ap.add_argument("--date-format", default="%Y%m%d")
    ap.add_argument("--units", default="mm")
    ap.add_argument("--lat-order", default="descending",
                    choices=["descending", "ascending"])
    ap.add_argument("--nodata", type=float, default=None,
                    help="sentinel meaning missing, when the file declares none "
                         "(CHIRPS GeoTIFFs need --nodata -9999)")
    a = ap.parse_args()
    tiffs_to_netcdf(a.src_dir, a.output, a.var_name, glob=a.glob,
                    date_pattern=a.date_pattern, date_format=a.date_format,
                    units=a.units, lat_order=a.lat_order, nodata=a.nodata)
