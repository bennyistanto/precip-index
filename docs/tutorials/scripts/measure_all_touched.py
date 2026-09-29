#!/usr/bin/env python3
"""How many grid cells does `all_touched=True` recover for your area of interest?

The default rasterisation rule keeps a cell only when its centre falls inside
the polygon, so every cell the boundary merely crosses is dropped. This counts
the difference on the grid you actually intend to use, before you commit to a
clip.

    python measure_all_touched.py boundary.shp --res 0.0416667
    python measure_all_touched.py shp_dir/ --glob "WB_GAD_ADM0_*.shp"

The loss scales with the perimeter-to-area ratio rather than with size, so a
large compact country loses a fraction of a per cent while an island or a long
thin basin can lose a quarter of its cells. Run it on your own boundary rather
than assuming a published figure applies.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

# Common grids, as a convenience for --res
PRESETS = {
    "terraclimate": 1.0 / 24.0,   # ~4 km
    "chirps": 0.05,               # ~5 km
    "era5-land": 0.1,             # ~9 km
    "imerg": 0.1,
}


def snapped_grid(bounds, res: float, pad: int = 2):
    """A lat/lon grid aligned to a global origin, covering `bounds` with padding.

    Snapping matters: a grid invented around the polygon's own bounds does not
    sit where the real product's cells sit, and the counts would not transfer.
    """
    minx, miny, maxx, maxy = bounds
    i0 = np.floor((minx + 180) / res) - pad
    i1 = np.ceil((maxx + 180) / res) + pad
    j0 = np.floor((miny + 90) / res) - pad
    j1 = np.ceil((maxy + 90) / res) + pad
    lons = (np.arange(i0, i1) + 0.5) * res - 180
    lats = (np.arange(j0, j1) + 0.5) * res - 90
    return lons, lats[::-1]          # north to south, as most products store it


def rasterise(geom, lons, lats, all_touched: bool) -> np.ndarray:
    from affine import Affine
    from rasterio import features

    lon_res = float(abs(lons[1] - lons[0]))
    lat_res = float(abs(lats[1] - lats[0]))
    if lats[0] > lats[-1]:
        tr = (Affine.translation(lons[0] - lon_res / 2, lats[0] + lat_res / 2)
              * Affine.scale(lon_res, -lat_res))
    else:
        tr = (Affine.translation(lons[0] - lon_res / 2, lats[0] - lat_res / 2)
              * Affine.scale(lon_res, lat_res))
    return features.rasterize([(geom, 1)], out_shape=(len(lats), len(lons)),
                              transform=tr, fill=0, dtype=np.uint8,
                              all_touched=all_touched).astype(bool)


def _boundary_ring_share(centre: np.ndarray, extra: np.ndarray) -> float:
    """Fraction of the gained cells that sit directly against the centre mask."""
    if not extra.any():
        return float("nan")
    pad = np.pad(centre, 1, constant_values=False)
    nbr = np.zeros_like(centre)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nbr |= pad[1 + dy:1 + dy + centre.shape[0],
                           1 + dx:1 + dx + centre.shape[1]]
    return float((extra & nbr).sum()) / float(extra.sum())


def compare(shapefiles: Iterable[Path], res: float) -> None:
    import geopandas as gpd

    print(f"grid resolution: {res:.7f} deg\n")
    print(f"{'area':22s} {'centre-only':>12s} {'all_touched':>12s} "
          f"{'gained':>9s} {'gain %':>8s} {'perim/area':>11s}")
    print("-" * 80)

    for shp in shapefiles:
        gdf = gpd.read_file(shp)
        if gdf.crs is None:
            print(f"{shp.stem:22s} skipped: no CRS")
            continue
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        geom = gdf.union_all()

        lons, lats = snapped_grid(gdf.total_bounds, res)
        centre = rasterise(geom, lons, lats, all_touched=False)
        touched = rasterise(geom, lons, lats, all_touched=True)

        n_c, n_t = int(centre.sum()), int(touched.sum())
        gained = n_t - n_c
        ratio = geom.length / geom.area if geom.area else float("nan")
        print(f"{shp.stem:22s} {n_c:12,d} {n_t:12,d} {gained:9,d} "
              f"{100 * gained / max(n_c, 1):7.1f}% {ratio:11.2f}")

        share = _boundary_ring_share(centre, touched & ~centre)
        if gained and not np.isnan(share):
            print(f"{'':22s} {100 * share:.0f}% of the gained cells border the "
                  f"retained mask, i.e. a boundary ring")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path,
                    help="a .shp file, or a directory to search")
    ap.add_argument("--glob", default="*.shp",
                    help="pattern when PATH is a directory (default: *.shp)")
    ap.add_argument("--res", default="terraclimate",
                    help="cell size in degrees, or one of: "
                         + ", ".join(PRESETS))
    a = ap.parse_args()

    res = PRESETS.get(str(a.res).lower())
    if res is None:
        res = float(a.res)

    files = ([a.path] if a.path.is_file()
             else sorted(a.path.glob(a.glob)))
    if not files:
        raise SystemExit(f"no shapefiles found at {a.path}")
    compare(files, res)


if __name__ == "__main__":
    main()
