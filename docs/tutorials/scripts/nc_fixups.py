#!/usr/bin/env python3
"""Small, composable fixes for NetCDF inputs, each preserving CF metadata.

Each function takes and returns an `xarray.Dataset`, so they chain:

    ds = (standardise_names(xr.open_dataset(path))
          .pipe(order_dims)
          .pipe(convert_precip_units, source_units="kg m-2 s-1")
          .pipe(clean_missing)
          .pipe(write_cf_metadata, title="CHIRPS monthly"))

The reason these are functions rather than four lines of inline xarray is that
each one has a detail that is easy to get wrong and silent when you do. Those
details are in the docstrings.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional

import numpy as np
import xarray as xr

__all__ = ["standardise_names", "order_dims", "convert_precip_units",
           "replace_zeros", "clean_missing", "write_cf_metadata",
           "CF_COORD_ATTRS", "COMMON_SENTINELS"]

CF_COORD_ATTRS = {
    "lat": dict(standard_name="latitude", long_name="latitude",
                units="degrees_north", axis="Y"),
    "lon": dict(standard_name="longitude", long_name="longitude",
                units="degrees_east", axis="X"),
    "time": dict(standard_name="time", long_name="time", axis="T"),
}

COMMON_SENTINELS = (-9999.0, -999.0, -99.0, -32768.0, -2147483648.0,
                    9999.0, 1e20, 1e36)

_NAME_MAP = {
    "latitude": "lat", "Latitude": "lat", "y": "lat", "Y": "lat",
    "nav_lat": "lat", "LAT": "lat",
    "longitude": "lon", "Longitude": "lon", "x": "lon", "X": "lon",
    "nav_lon": "lon", "LON": "lon",
    "t": "time", "T": "time", "TIME": "time", "Time": "time",
}


# --------------------------------------------------------------------------- #
def standardise_names(ds: xr.Dataset, extra: Optional[Mapping[str, str]] = None,
                      var_renames: Optional[Mapping[str, str]] = None
                      ) -> xr.Dataset:
    """Rename coordinates to `time`, `lat`, `lon` and refill CF attributes.

    Renaming alone is not enough. A file that called its axis `y` usually also
    lacks `standard_name`, `units` and `axis`, and those are what let another
    tool know which way is north. This restores them where absent without
    overwriting anything the file already states.
    """
    mapping = dict(_NAME_MAP)
    mapping.update(extra or {})
    rename = {k: v for k, v in mapping.items()
              if k in ds.variables and v not in ds.variables}
    if rename:
        ds = ds.rename(rename)
    if var_renames:
        present = {k: v for k, v in var_renames.items() if k in ds.variables}
        if present:
            ds = ds.rename(present)

    for name, attrs in CF_COORD_ATTRS.items():
        if name in ds.coords:
            for key, value in attrs.items():
                ds[name].attrs.setdefault(key, value)

    # Longitude convention: 0..360 sorts and slices differently from -180..180.
    if "lon" in ds.coords and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
        ds.lon.attrs.setdefault("units", "degrees_east")
    return ds


# --------------------------------------------------------------------------- #
def order_dims(ds: xr.Dataset, order: Iterable[str] = ("time", "lat", "lon")
               ) -> xr.Dataset:
    """Transpose every 3-D variable to `(time, lat, lon)`.

    Note this changes the logical order only. The bytes on disk keep whatever
    chunking they had, so a transpose that looks free in memory can be very
    expensive on the next read. If you transpose, write the result out rather
    than working lazily from it.
    """
    order = tuple(order)
    for name, da in ds.data_vars.items():
        if set(order).issubset(da.dims) and da.dims != order:
            rest = [d for d in da.dims if d not in order]
            ds[name] = da.transpose(*rest, *order)
    return ds


# --------------------------------------------------------------------------- #
_PER_SECOND = {"kg m-2 s-1", "kg/m2/s", "kg m**-2 s**-1", "mm/s", "mm s-1"}
_PER_HOUR = {"mm/hr", "mm/h", "mm hr-1", "mm h-1"}
_PER_DAY = {"mm/day", "mm/d", "mm day-1", "mm d-1"}
_METRES = {"m", "metre", "meter", "metres", "meters"}


def convert_precip_units(ds: xr.Dataset, var: str = "precip",
                         source_units: Optional[str] = None,
                         target: str = "mm") -> xr.Dataset:
    """Convert a precipitation variable to a monthly total in mm.

    Rate units are multiplied by the **actual length of each month**, never by
    an average. A fixed 30.44-day month overstates February by 8.7% and
    understates a 31-day month by 1.8%: a systematic seasonal error of 10.5
    percentage points, which lands squarely on the seasonal cycle a drought
    index is trying to measure.
    """
    da = ds[var]
    units = (source_units or da.attrs.get("units", "")).strip()
    if not units:
        raise ValueError(
            f"{var} has no units attribute and none was supplied. Do not guess: "
            "the same variable name ships in different units by product.")

    if "time" not in da.dims:
        raise ValueError(f"{var} has no time dimension")
    days = ds["time"].dt.days_in_month

    u = units.lower()
    if u in {s.lower() for s in _PER_SECOND}:
        out, note = da * days * 86400.0, f"{units} x seconds in month"
    elif u in {s.lower() for s in _PER_HOUR}:
        out, note = da * days * 24.0, f"{units} x hours in month"
    elif u in {s.lower() for s in _PER_DAY}:
        out, note = da * days, f"{units} x days in month"
    elif u in {s.lower() for s in _METRES}:
        out, note = da * 1000.0, "m x 1000"
    elif u in {"mm", "millimetre", "millimeter", "kg m-2", "kg/m2"}:
        out, note = da, "already a depth"
    else:
        raise ValueError(f"unhandled units {units!r} for {var}")

    out.attrs = dict(da.attrs)
    out.attrs["units"] = target
    prior = out.attrs.get("history", "")
    out.attrs["history"] = f"{note} -> {target}; {prior}".strip("; ")
    ds[var] = out
    return ds


# --------------------------------------------------------------------------- #
def replace_zeros(ds: xr.Dataset, var: str = "precip", value: float = 0.001
                  ) -> xr.Dataset:
    """Replace exact zeros with a small positive value, leaving NaN untouched.

    Only do this if your fitting code cannot handle zeros. `precip-index` can:
    it fits a mixed distribution and carries the zero probability separately as
    `prob_zero`, which is the statistically correct treatment. Substituting a
    small constant biases the fitted distribution and throws that away.

    The idiom matters. `da.where(cond, other)` keeps values where `cond` is
    True, so the condition must be the values you want to KEEP:

        da.where(da != 0, 0.001)      # correct: keeps non-zeros, NaN survives
        da.where(da == 0, 0.001)      # wrong: keeps only the zeros

    NaN survives the correct form because `NaN != 0` is True. Verified, not
    assumed.
    """
    da = ds[var]
    n_zero = int((da == 0).sum())
    out = da.where(da != 0, value)
    out.attrs = dict(da.attrs)
    out.attrs["comment"] = (
        f"{n_zero} exact zeros replaced with {value} {da.attrs.get('units','')}"
        .strip())
    ds[var] = out
    return ds


# --------------------------------------------------------------------------- #
def clean_missing(ds: xr.Dataset, var: str = "precip",
                  sentinels: Iterable[float] = COMMON_SENTINELS,
                  minimum: Optional[float] = 0.0,
                  report: bool = True) -> xr.Dataset:
    """Turn declared and undeclared no-data values into NaN.

    Order matters: sentinels first, physical bounds second. Applying a
    `>= 0` filter first would leave a -9999 in place if you then only checked
    for exact matches, and applying it to already-masked data is free.
    """
    da = ds[var]

    declared = da.encoding.get("_FillValue", da.attrs.get("missing_value"))
    if declared is not None and not (isinstance(declared, float)
                                     and np.isnan(declared)):
        da = da.where(da != declared)

    for s in sentinels:
        da = da.where(da != s)

    if minimum is not None:
        da = da.where(da >= minimum)

    if report:
        total = int(da.size)
        missing = int(da.isnull().sum())
        print(f"{var}: {missing:,} of {total:,} missing ({100*missing/total:.1f}%)")
        if "time" in da.dims:
            always = da.isnull().all("time")
            sometimes = da.isnull().any("time") & ~always
            print(f"  never observed : {int(always.sum()):,} cells")
            print(f"  gaps in time   : {int(sometimes.sum()):,} cells")

    da.attrs = dict(ds[var].attrs)
    for junk in ("missing_value", "_FillValue"):
        da.attrs.pop(junk, None)
    ds[var] = da
    return ds


# --------------------------------------------------------------------------- #
def write_cf_metadata(ds: xr.Dataset, title: Optional[str] = None,
                      institution: Optional[str] = None,
                      source: Optional[str] = None,
                      references: Optional[str] = None,
                      history_note: Optional[str] = None) -> xr.Dataset:
    """Add the CRS variable, grid mapping and geospatial bounds."""
    ds = ds.copy()
    if "crs" not in ds.variables:
        ds["crs"] = xr.DataArray(np.int32(0))
    ds["crs"].attrs.update(dict(
        grid_mapping_name="latitude_longitude",
        longitude_of_prime_meridian=0.0,
        semi_major_axis=6378137.0,
        inverse_flattening=298.257223563,
        epsg_code="EPSG:4326",
        long_name="CRS definition",
    ))
    for name, da in ds.data_vars.items():
        if name != "crs" and {"lat", "lon"}.issubset(da.dims):
            da.attrs["grid_mapping"] = "crs"

    attrs = {"Conventions": "CF-1.8", "cdm_data_type": "GRID"}
    if "lat" in ds.coords and "lon" in ds.coords:
        lat, lon = ds["lat"].values, ds["lon"].values
        attrs.update({
            "geospatial_lat_min": float(np.min(lat)),
            "geospatial_lat_max": float(np.max(lat)),
            "geospatial_lon_min": float(np.min(lon)),
            "geospatial_lon_max": float(np.max(lon)),
            "geospatial_lat_units": "degrees_north",
            "geospatial_lon_units": "degrees_east",
        })
        if lat.size > 1:
            attrs["geospatial_lat_resolution"] = float(abs(lat[1] - lat[0]))
        if lon.size > 1:
            attrs["geospatial_lon_resolution"] = float(abs(lon[1] - lon[0]))
    if "time" in ds.coords and ds["time"].size:
        attrs["time_coverage_start"] = str(ds["time"].values[0])[:10]
        attrs["time_coverage_end"] = str(ds["time"].values[-1])[:10]
    for key, value in (("title", title), ("institution", institution),
                       ("source", source), ("references", references)):
        if value:
            attrs[key] = value

    stamp = f"{datetime.now(timezone.utc):%a %b %d %H:%M:%S %Y} UTC"
    note = history_note or "standardised with nc_fixups.py"
    attrs["history"] = f"{stamp}: {note}\n{ds.attrs.get('history', '')}".strip()
    ds.attrs.update(attrs)
    return ds
