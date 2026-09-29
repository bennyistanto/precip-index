"""Is TerraClimate's packing identical across variables and years?

If scale_factor or add_offset differ between yearly files, concatenating the
packed integers without unpacking first silently corrupts the series. This
checks the question directly against the THREDDS OPeNDAP endpoint, which
returns metadata without transferring the arrays.
"""
import netCDF4

BASE = ("http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
        "TERRACLIMATE_ALL/data/TerraClimate_{v}_{y}.nc")

CASES = [("ppt", 1950), ("ppt", 1990), ("ppt", 2024), ("ppt", 2025),
         ("pet", 1950), ("pet", 2024)]

print(f"{'file':34s} {'dtype':7s} {'scale':>7s} {'offset':>7s} {'fill':>13s} {'units':>6s}")
print("-" * 82)
seen = {}
for v, y in CASES:
    name = f"TerraClimate_{v}_{y}.nc"
    try:
        f = netCDF4.Dataset(BASE.format(v=v, y=y))
        d = f.variables[v]
        sc = getattr(d, "scale_factor", None)
        off = getattr(d, "add_offset", None)
        fill = getattr(d, "_FillValue", None)
        un = getattr(d, "units", None)
        print(f"{name:34s} {str(d.dtype):7s} {str(sc):>7s} {str(off):>7s} "
              f"{str(fill):>13s} {str(un):>6s}")
        seen.setdefault(v, set()).add((str(d.dtype), str(sc), str(off)))
        f.close()
    except Exception as e:
        print(f"{name:34s} FAILED {type(e).__name__}")

print()
for v, s in seen.items():
    verdict = "consistent" if len(s) == 1 else "DIFFERS ACROSS YEARS"
    print(f"{v}: {len(s)} distinct (dtype, scale, offset) combination(s) -> {verdict}")
    for combo in sorted(s):
        print(f"     {combo}")
