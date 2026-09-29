"""How current is each product, and how long after a period does it appear?

Probes with HEAD requests only. Reports which files exist and their
Last-Modified, which is the publication date on these servers.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.request

NOW = datetime.now(timezone.utc)
TC = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{v}_{y}.nc"
CH_TIF = ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/"
          "chirps-v3.0.{y}.{m:02d}.tif")
CH_NC = ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/netcdf/"
         "by_year/chirps-v3.0.{y}.monthly.nc")


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            lm = r.headers.get("Last-Modified")
            return (r.status,
                    int(r.headers.get("Content-Length", 0)),
                    parsedate_to_datetime(lm) if lm else None)
    except urllib.error.HTTPError as e:
        return (e.code, 0, None)
    except Exception as e:
        return (type(e).__name__, 0, None)


def show(label, url):
    status, size, lm = head(url)
    if status == 200:
        age = f"{(NOW - lm).days:>4d} d ago" if lm else "  ?"
        when = f"{lm:%Y-%m-%d}" if lm else "?"
        print(f"  {label:34s} present  {size/1e6:8.1f} MB  published {when}  {age}")
        return lm
    print(f"  {label:34s} ABSENT ({status})")
    return None


print(f"checked {NOW:%Y-%m-%d %H:%M} UTC\n")

print("TerraClimate, yearly files (ppt)")
print("-" * 76)
tc = {}
for y in (2023, 2024, 2025, 2026):
    tc[y] = show(f"TerraClimate_ppt_{y}.nc", TC.format(v="ppt", y=y))

print()
print("CHIRPS v3.0 monthly GeoTIFF, recent months")
print("-" * 76)
ch = {}
for y, m in ((2026, 5), (2026, 6), (2026, 7), (2026, 8), (2026, 9)):
    ch[(y, m)] = show(f"chirps-v3.0.{y}.{m:02d}.tif", CH_TIF.format(y=y, m=m))

print()
print("CHIRPS v3.0 monthly NetCDF, by year")
print("-" * 76)
for y in (2025, 2026):
    show(f"chirps-v3.0.{y}.monthly.nc", CH_NC.format(y=y))

print()
print("=" * 76)
print("LATENCY, from the end of the period to the publication date")
print("=" * 76)

for (y, m), lm in ch.items():
    if lm is None:
        continue
    # end of that month
    end = datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=timezone.utc)
    print(f"  CHIRPS {y}-{m:02d}: month ended {end:%Y-%m-%d}, "
          f"published {lm:%Y-%m-%d}  ->  {(lm - end).days} days")

latest_tc = max((y for y, v in tc.items() if v), default=None)
if latest_tc:
    end = datetime(latest_tc + 1, 1, 1, tzinfo=timezone.utc)
    lm = tc[latest_tc]
    print(f"  TerraClimate {latest_tc}: year ended {end:%Y-%m-%d}, "
          f"file dated {lm:%Y-%m-%d}  ->  {(lm - end).days} days")
    print("    caveat: v1.1 was a full re-release, so Last-Modified on these")
    print("    files reflects that release, not the original publication date.")
