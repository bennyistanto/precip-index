#!/usr/bin/env python
"""
Test 08: Tutorial helper scripts

Exercises the modules published in docs/tutorials/scripts/:

    tiff2nc.py     GeoTIFF stack -> CF NetCDF, cell-centre coordinates
    nc_fixups.py   rename, reorder, unit conversion, zeros, missing values
    nc_clip.py     shapefile clip keeping every touched cell
    tc_prepare.py  unpack packed integers, merge a yearly series

Everything here is built from synthetic fixtures in a temporary directory, so
the suite needs no input data, no network and no particular machine. That is
deliberate: these modules are documentation, and documentation that quietly
stops working is worse than none.

Author: Benny Istanto
"""

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from conftest import print_header, print_subheader, print_ok, print_fail, print_info

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'docs' / 'tutorials' / 'scripts'))

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print_ok(f"{label}{f'  ({detail})' if detail else ''}")
    else:
        FAILED += 1
        print_fail(f"{label}{f'  ({detail})' if detail else ''}")


# ===========================================================================
# Fixtures
# ===========================================================================
RES = 0.05
ORIGIN_X, ORIGIN_Y = 100.0, 10.0        # OUTER corner of the top-left pixel
NX, NY = 12, 8
NODATA = -9999.0


def make_geotiffs(dest: Path, stamps=("20200101", "20200201", "20200301")):
    """Monthly GeoTIFFs on a grid whose origin makes a half-pixel error obvious."""
    import rasterio
    from rasterio.transform import from_origin

    dest.mkdir(parents=True, exist_ok=True)
    transform = from_origin(ORIGIN_X, ORIGIN_Y, RES, RES)
    for i, stamp in enumerate(stamps):
        arr = np.full((NY, NX), float(i + 1), dtype='float32')
        arr[0, 0] = NODATA          # no-data cell
        arr[1, 1] = 0.0             # exact zero
        with rasterio.open(dest / f"rain_{stamp}.tif", 'w', driver='GTiff',
                           height=NY, width=NX, count=1, dtype='float32',
                           crs='EPSG:4326', transform=transform,
                           nodata=NODATA) as dst:
            dst.write(arr, 1)
    return sorted(dest.glob('*.tif'))


def make_packed_years(dest: Path):
    """Yearly files packed the way TerraClimate packs them."""
    dest.mkdir(parents=True, exist_ok=True)
    spec = {'ppt': dict(dtype='int32', fill=-2147483648, scale=0.1),
            'pet': dict(dtype='int16', fill=-32768, scale=0.1)}
    rng = np.random.default_rng(0)
    truth = {}
    for var, s in spec.items():
        for year in (2000, 2001):
            vals = np.round(rng.uniform(0, 300, size=(12, 6, 9)), 1)
            vals[:, :2, :] = np.nan                       # an "ocean" band
            truth.setdefault(var, []).append(vals)
            da = xr.DataArray(
                vals, dims=('time', 'lat', 'lon'),
                coords=dict(
                    time=xr.date_range(f'{year}-01-01', periods=12, freq='MS'),
                    lat=np.linspace(50, 45, 6), lon=np.linspace(0, 8, 9)),
                name=var)
            da.attrs = {'units': 'mm', 'long_name': f'{var}_amount'}
            da.to_dataset().to_netcdf(
                dest / f'TerraClimate_{var}_{year}.nc',
                encoding={var: {'dtype': s['dtype'], 'scale_factor': s['scale'],
                                'add_offset': 0.0, '_FillValue': s['fill']}})
    return spec, truth


def make_polygon_shapefile(path: Path):
    """A deliberately ragged polygon, so the boundary ring is non-trivial."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    # A cross/plus shape: lots of perimeter for its area. The centre and the
    # arm lengths are deliberately NOT multiples of the cell size, so the
    # edges cut through cells rather than landing on cell boundaries. A
    # grid-aligned polygon would make all_touched a no-op and the test
    # vacuous.
    cx = ORIGIN_X + NX * RES / 2 + RES * 0.23
    cy = ORIGIN_Y - NY * RES / 2 - RES * 0.37
    w, h = RES * 1.45, RES * 2.85
    poly = Polygon([
        (cx - w, cy - h), (cx + w, cy - h), (cx + w, cy - w), (cx + h, cy - w),
        (cx + h, cy + w), (cx + w, cy + w), (cx + w, cy + h), (cx - w, cy + h),
        (cx - w, cy + w), (cx - h, cy + w), (cx - h, cy - w), (cx - w, cy - w),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({'id': [1]}, geometry=[poly], crs='EPSG:4326').to_file(path)
    return path


# ===========================================================================
# tiff2nc
# ===========================================================================
def test_tiff2nc(tmp: Path):
    print_subheader("tiff2nc: GeoTIFF stack to CF NetCDF")
    from tiff2nc import tiffs_to_netcdf, cell_centres
    import rasterio
    from rasterio.transform import from_origin

    tif_dir = tmp / 'tif'
    make_geotiffs(tif_dir)
    out = tiffs_to_netcdf(tif_dir, tmp / 'stacked.nc', 'precip',
                          date_pattern=r'(\d{8})', date_format='%Y%m%d',
                          units='mm', standard_name='precipitation_amount',
                          report=False)

    ds = xr.open_dataset(out)
    try:
        # The half-pixel correction: coordinates are cell CENTRES.
        check("half-pixel: lon[0] is the cell centre",
              abs(float(ds.lon[0]) - (ORIGIN_X + RES / 2)) < 1e-9,
              f"{float(ds.lon[0])} vs corner {ORIGIN_X}")
        check("half-pixel: lat[0] is the cell centre",
              abs(float(ds.lat[0]) - (ORIGIN_Y - RES / 2)) < 1e-9,
              f"{float(ds.lat[0])} vs corner {ORIGIN_Y}")

        check("shape is (time, lat, lon)",
              ds['precip'].dims == ('time', 'lat', 'lon'))
        check("all time steps written", ds.sizes['time'] == 3)
        check("nodata became NaN", bool(np.isnan(ds['precip'].values[0, 0, 0])))
        check("exact zero preserved", float(ds['precip'].values[0, 1, 1]) == 0.0)

        check("Conventions is CF", str(ds.attrs.get('Conventions', '')).startswith('CF-'))
        check("variable references a grid mapping",
              ds['precip'].attrs.get('grid_mapping') == 'crs')
        check("crs variable exists", 'crs' in ds.variables)
        check("lat has CF attributes",
              ds['lat'].attrs.get('units') == 'degrees_north'
              and ds['lat'].attrs.get('axis') == 'Y')
        check("lon has CF attributes",
              ds['lon'].attrs.get('units') == 'degrees_east'
              and ds['lon'].attrs.get('axis') == 'X')
        check("geospatial bounds match the data",
              abs(ds.attrs['geospatial_lat_max'] - float(ds.lat.max())) < 1e-9)
    finally:
        ds.close()

    # cell_centres in isolation
    lon, lat = cell_centres(from_origin(0.0, 0.0, 1.0, 1.0), 3, 3)
    check("cell_centres offsets by half a pixel",
          np.allclose(lon, [0.5, 1.5, 2.5]) and np.allclose(lat, [-0.5, -1.5, -2.5]))

    # A raster on a different grid must be refused, not silently stacked.
    odd = tif_dir / 'rain_20200401.tif'
    with rasterio.open(odd, 'w', driver='GTiff', height=NY, width=NX + 1,
                       count=1, dtype='float32', crs='EPSG:4326',
                       transform=from_origin(ORIGIN_X, ORIGIN_Y, RES, RES)) as d:
        d.write(np.zeros((NY, NX + 1), dtype='float32'), 1)
    try:
        tiffs_to_netcdf(tif_dir, tmp / 'bad.nc', 'precip', report=False)
        check("mismatched grid is rejected", False, "no error raised")
    except ValueError:
        check("mismatched grid is rejected", True)
    odd.unlink()

    # An undeclared sentinel must be maskable via the override. CHIRPS
    # GeoTIFFs are exactly this case: no nodata tag, but -9999 in the ocean.
    import rasterio as _rio
    sentinel_dir = tmp / 'sentinel'
    sentinel_dir.mkdir(exist_ok=True)
    arr = np.full((NY, NX), 5.0, dtype='float32')
    arr[0, :] = -9999.0
    with _rio.open(sentinel_dir / 'rain_20210101.tif', 'w', driver='GTiff',
                   height=NY, width=NX, count=1, dtype='float32',
                   crs='EPSG:4326',
                   transform=from_origin(ORIGIN_X, ORIGIN_Y, RES, RES)) as d:
        d.write(arr, 1)          # note: no nodata= argument, as CHIRPS does
    with _rio.open(sentinel_dir / 'rain_20210101.tif') as chk:
        check("fixture declares no nodata, like CHIRPS", chk.nodata is None)

    # Write with a DIFFERENT fill value, so the source sentinel cannot be
    # masked by coincidence. With the default fill_value=-9999 the CHIRPS
    # sentinel happens to land on the declared fill and is masked by luck;
    # that luck disappears the moment anyone changes fill_value.
    o1 = tiffs_to_netcdf(sentinel_dir, tmp / 'sent_raw.nc', 'precip',
                         fill_value=-32768.0, report=False)
    d1 = xr.open_dataset(o1)
    check("without the override the sentinel survives as real data",
          float(d1['precip'].values[0, 0, 0]) == -9999.0,
          "-9999 read back as precipitation")
    d1.close()

    o2 = tiffs_to_netcdf(sentinel_dir, tmp / 'sent_fix.nc', 'precip',
                         nodata=-9999.0, fill_value=-32768.0, report=False)
    d2 = xr.open_dataset(o2)
    check("nodata override turns the sentinel into NaN",
          bool(np.isnan(d2['precip'].values[0, 0, 0])))
    check("real values untouched by the override",
          float(d2['precip'].values[0, 4, 4]) == 5.0)
    d2.close()

    # Duplicate timestamps must be refused.
    shutil.copy(tif_dir / 'rain_20200101.tif', tif_dir / 'copy_20200101.tif')
    try:
        tiffs_to_netcdf(tif_dir, tmp / 'bad2.nc', 'precip', report=False)
        check("duplicate timestamps are rejected", False, "no error raised")
    except ValueError:
        check("duplicate timestamps are rejected", True)
    (tif_dir / 'copy_20200101.tif').unlink()


# ===========================================================================
# nc_fixups
# ===========================================================================
def test_nc_fixups(tmp: Path):
    print_subheader("nc_fixups: rename, reorder, units, zeros, missing")
    from nc_fixups import (standardise_names, order_dims, convert_precip_units,
                           replace_zeros, clean_missing, write_cf_metadata)

    ds = xr.open_dataset(tmp / 'stacked.nc').load()
    ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    ds['precip'] = ds['precip'].transpose('longitude', 'time', 'latitude')
    ds['precip'].attrs['units'] = 'mm/day'

    ds = standardise_names(ds)
    check("coordinates renamed to lat/lon/time",
          {'lat', 'lon', 'time'}.issubset(set(ds.coords)))
    check("CF attributes restored after rename",
          ds['lat'].attrs.get('standard_name') == 'latitude')

    ds = order_dims(ds)
    check("dimensions reordered to (time, lat, lon)",
          ds['precip'].dims == ('time', 'lat', 'lon'))

    # Unit conversion must use the real month length, including leap years.
    before = float(ds['precip'].isel(time=1, lat=4, lon=4))
    ds = convert_precip_units(ds, 'precip', target='mm')
    after = float(ds['precip'].isel(time=1, lat=4, lon=4))
    check("mm/day -> mm uses actual days in month",
          abs(after - before * 29) < 1e-4,
          f"Feb 2020 is a leap month: {before} -> {after}, expected {before * 29}")
    check("units attribute updated", ds['precip'].attrs['units'] == 'mm')

    # A missing units attribute must raise rather than be guessed.
    probe = ds.copy(deep=True)
    probe['precip'].attrs.pop('units', None)
    try:
        convert_precip_units(probe, 'precip')
        check("missing units raises", False, "no error raised")
    except ValueError:
        check("missing units raises", True)

    # Zero replacement must not touch NaN.
    nan_before = int(ds['precip'].isnull().sum())
    zeros_before = int((ds['precip'] == 0).sum())
    ds = replace_zeros(ds, 'precip', 0.001)
    check("zeros replaced", int((ds['precip'] == 0).sum()) == 0,
          f"{zeros_before} zeros -> 0")
    check("NaN preserved through zero replacement",
          int(ds['precip'].isnull().sum()) == nan_before,
          f"{nan_before} NaN before and after")

    # The reversed idiom must be visibly wrong, which is why the docs warn.
    probe = xr.DataArray([0.0, 1.5, np.nan, 0.0, 3.2])
    right = probe.where(probe != 0, 0.001).values
    wrong = probe.where(probe == 0, 0.001).values
    check("where(!=0) keeps data and NaN",
          right[1] == 1.5 and np.isnan(right[2]) and right[0] == 0.001)
    check("where(==0) is destructive, as documented",
          wrong[1] == 0.001 and wrong[0] == 0.0)

    ds = clean_missing(ds, 'precip', report=False)
    check("clean_missing leaves NaN in place",
          int(ds['precip'].isnull().sum()) >= nan_before)

    ds = write_cf_metadata(ds, title='synthetic')
    check("write_cf_metadata sets Conventions",
          str(ds.attrs.get('Conventions', '')).startswith('CF-'))
    check("write_cf_metadata sets grid_mapping",
          ds['precip'].attrs.get('grid_mapping') == 'crs')
    check("write_cf_metadata computes geospatial bounds",
          'geospatial_lat_resolution' in ds.attrs)
    ds.close()


# ===========================================================================
# nc_clip
# ===========================================================================
def test_nc_clip(tmp: Path):
    print_subheader("nc_clip: shapefile clip keeping touched cells")
    from nc_clip import clip_netcdf, build_mask
    import geopandas as gpd

    shp = make_polygon_shapefile(tmp / 'shp' / 'aoi.shp')
    src = tmp / 'stacked.nc'

    ds = xr.open_dataset(src)
    lons, lats = ds['lon'].values, ds['lat'].values
    ds.close()
    geom = gpd.read_file(shp).union_all()

    centre = build_mask(geom, lons, lats, all_touched=False)
    touched = build_mask(geom, lons, lats, all_touched=True)
    check("all_touched keeps at least as many cells",
          int(touched.sum()) >= int(centre.sum()))
    check("all_touched recovers boundary cells on a ragged polygon",
          int(touched.sum()) > int(centre.sum()),
          f"{int(centre.sum())} -> {int(touched.sum())}")
    check("every centre-selected cell is also touched",
          bool((centre & ~touched).sum() == 0))

    out = clip_netcdf(src, shp, tmp / 'clipped.nc', all_touched=True,
                      crop=True, time_batch=2, report=False)
    ds = xr.open_dataset(out)
    try:
        check("clip output is float32", ds['precip'].dtype == np.float32)
        check("clip output is CF",
              str(ds.attrs.get('Conventions', '')).startswith('CF-')
              and ds['precip'].attrs.get('grid_mapping') == 'crs')
        check("clip records the operation in history",
              'clipped to' in ds.attrs.get('history', ''))
        check("clip updates geospatial bounds to the crop",
              abs(ds.attrs['geospatial_lat_max'] - float(ds['lat'].max())) < 1e-9)
        check("crop is no larger than the source",
              ds.sizes['lat'] <= len(lats) and ds.sizes['lon'] <= len(lons))
        check("time axis preserved", ds.sizes['time'] == 3)

        finite = int(np.isfinite(ds['precip'].isel(time=0).values).sum())
        check("clipped cells are inside the mask",
              0 < finite <= int(touched.sum()),
              f"{finite} finite, mask has {int(touched.sum())}")

        for junk in ('scale_factor', 'add_offset', 'missing_value'):
            check(f"clip drops {junk}", junk not in ds['precip'].attrs)
    finally:
        ds.close()

    # A non-overlapping polygon must raise, not write an empty file.
    from shapely.geometry import box
    far = tmp / 'shp' / 'far.shp'
    gpd.GeoDataFrame({'id': [1]}, geometry=[box(0, 0, 1, 1)],
                     crs='EPSG:4326').to_file(far)
    try:
        clip_netcdf(src, far, tmp / 'nope.nc', report=False)
        check("non-overlapping polygon raises", False, "no error raised")
    except ValueError:
        check("non-overlapping polygon raises", True)


# ===========================================================================
# tc_prepare
# ===========================================================================
def test_tc_prepare(tmp: Path):
    print_subheader("tc_prepare: unpack and merge packed years")
    from tc_prepare import unpack_file, merge_years

    raw = tmp / 'raw'
    spec, truth = make_packed_years(raw)

    # Confirm the fixtures really are packed the way upstream packs them.
    import netCDF4
    f = netCDF4.Dataset(raw / 'TerraClimate_ppt_2000.nc')
    check("fixture ppt is packed int32",
          str(f.variables['ppt'].dtype) == 'int32'
          and getattr(f.variables['ppt'], 'scale_factor', None) == 0.1)
    f.close()
    f = netCDF4.Dataset(raw / 'TerraClimate_pet_2000.nc')
    check("fixture pet is packed int16",
          str(f.variables['pet'].dtype) == 'int16')
    f.close()

    unp = tmp / 'unpacked'
    for p in sorted(raw.glob('*.nc')):
        unpack_file(p, unp / p.name)

    for var in spec:
        merged = merge_years(sorted(unp.glob(f'TerraClimate_{var}_*.nc')),
                             tmp / f'merged_{var}.nc', var)
        expect = np.concatenate(truth[var], axis=0)
        ds = xr.open_dataset(merged)
        got = ds[var].values
        check(f"{var}: merged shape", got.shape == expect.shape)
        check(f"{var}: output is float32", got.dtype == np.float32)
        check(f"{var}: NaN mask preserved",
              np.array_equal(np.isnan(expect), np.isnan(got)))
        m = np.isfinite(expect)
        maxdiff = float(np.nanmax(np.abs(expect[m] - got[m])))
        check(f"{var}: values round-trip within float32",
              maxdiff < 1e-4, f"max |diff| {maxdiff:.2e}")
        ds.close()

        g = netCDF4.Dataset(merged)
        attrs = g.variables[var].ncattrs()
        check(f"{var}: packing attributes dropped",
              not any(a in attrs for a in ('scale_factor', 'add_offset')))
        g.close()


# ===========================================================================
def main() -> int:
    print_header("TEST 08: TUTORIAL HELPER SCRIPTS")
    print_info("Synthetic fixtures only: no input data, no network required")

    tmp = Path(tempfile.mkdtemp(prefix='precip_tut_'))
    print_info(f"Working directory: {tmp}")
    try:
        for fn in (test_tiff2nc, test_nc_fixups, test_nc_clip, test_tc_prepare):
            try:
                fn(tmp)
            except Exception:
                global FAILED
                FAILED += 1
                print_fail(f"{fn.__name__} raised:")
                traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print_header("TEST 08 COMPLETE")
    total = PASSED + FAILED
    if FAILED:
        print_fail(f"{FAILED} of {total} checks failed")
        return 1
    print_ok(f"All {total} checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
