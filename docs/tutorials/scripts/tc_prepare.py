"""TerraClimate preparation helpers: download, unpack, merge.

These are the functions published in docs/tutorials/00b-terraclimate.qmd.
"""
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import xarray as xr

FILL = -9999.0


# --------------------------------------------------------------------- download
def download_year(var: str, year: int, out_dir: Path,
                  base: str = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA",
                  timeout: int = 600) -> Path:
    """Download one yearly TerraClimate file, resuming and skipping completed ones.

    Returns the local path. Raises on a size mismatch rather than leaving a
    truncated file that would fail much later in the pipeline.
    """
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base}/TerraClimate_{var}_{year}.nc"
    dest = out_dir / f"TerraClimate_{var}_{year}.nc"

    head = requests.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    remote_size = int(head.headers.get("Content-Length", 0))

    have = dest.stat().st_size if dest.exists() else 0
    if remote_size and have == remote_size:
        print(f"  {dest.name}: already complete ({have/1e6:.0f} MB)")
        return dest

    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    if have:
        print(f"  {dest.name}: resuming at {have/1e6:.0f} of {remote_size/1e6:.0f} MB")

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, mode) as fh:
            for block in r.iter_content(chunk_size=1 << 20):
                fh.write(block)

    got = dest.stat().st_size
    if remote_size and got != remote_size:
        raise IOError(f"{dest.name}: got {got} bytes, expected {remote_size}")
    print(f"  {dest.name}: {got/1e6:.0f} MB")
    return dest


# ----------------------------------------------------------------------- unpack
def unpack_file(src: Path, dest: Path, dtype: str = "float32",
                complevel: int = 5) -> Path:
    """Unpack one packed TerraClimate file to real numbers.

    xarray applies scale_factor/add_offset and the fill mask on read, so the
    work here is writing the result back WITHOUT packing attributes, and with
    one fill value regardless of what integer type the source used.
    """
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(src, mask_and_scale=True) as ds:
        ds = ds.load()
        enc = {}
        for name, da in ds.data_vars.items():
            if name == "crs" or da.ndim < 2:
                enc[name] = {}
                continue
            ds[name] = da.astype(dtype)
            # Drop anything inherited from the packed representation.
            for junk in ("scale_factor", "add_offset", "_Unsigned",
                         "missing_value", "_FillValue"):
                ds[name].attrs.pop(junk, None)
            enc[name] = {"dtype": dtype, "_FillValue": FILL,
                         "zlib": True, "complevel": complevel}
        ds.to_netcdf(dest, encoding=enc)
    return dest


# ------------------------------------------------------------------------ merge
def merge_years(files: Iterable[Path], dest: Path, var: str,
                dtype: str = "float32", complevel: int = 5,
                chunks: Optional[tuple] = None) -> Path:
    """Concatenate yearly files along time into one series.

    `chunks` is the on-disk chunk shape (time, lat, lon). A chunk that spans
    the whole time axis means a per-pixel series is one read, which is what
    SPI/SPEI wants; keep the spatial extent small so a chunk stays a sensible
    size. Default picks a time-complete chunk near 16 MB.
    """
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("no input files")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4",
                           parallel=False, mask_and_scale=True)
    ds = ds.sortby("time")

    nt = ds.sizes["time"]
    if chunks is None:
        side = max(16, int(np.sqrt((16e6 / 4) / max(nt, 1))))
        chunks = (nt, min(side, ds.sizes["lat"]), min(side * 2, ds.sizes["lon"]))

    enc = {var: {"dtype": dtype, "_FillValue": FILL, "zlib": True,
                 "complevel": complevel, "chunksizes": tuple(chunks)}}
    if "crs" in ds.variables:
        enc["crs"] = {}

    ds[var] = ds[var].astype(dtype)
    for junk in ("scale_factor", "add_offset", "_Unsigned", "missing_value"):
        ds[var].attrs.pop(junk, None)

    ds.to_netcdf(dest, encoding=enc, engine="netcdf4")
    ds.close()
    return dest
