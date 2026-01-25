#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for SPI calculation with TerraClimate Bali data
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, 'src')

import numpy as np
import xarray as xr

# Import SPI functions
from indices import spi, spi_multi_scale, save_index_to_netcdf

print("=" * 80)
print("SPI Calculation Test with TerraClimate Bali Data")
print("=" * 80)

# Load data
print("\n1. Loading TerraClimate precipitation data...")
try:
    ds = xr.open_dataset('input/terraclimate_bali_ppt_1958_2024.nc')
    print("   [OK] Data loaded successfully")

    # Display dataset info
    print(f"\n   Dataset dimensions: {dict(ds.dims)}")
    print(f"   Variables: {list(ds.data_vars)}")
    print(f"   Coordinates: {list(ds.coords)}")

    # Identify precipitation variable
    precip_var_names = ['ppt', 'precip', 'precipitation', 'prcp', 'pr', 'rain', 'rainfall']
    precip_var = None

    for var_name in ds.data_vars:
        if var_name.lower() in precip_var_names or any(p in var_name.lower() for p in precip_var_names):
            precip_var = var_name
            break

    if precip_var is None:
        # Just take the first data variable
        precip_var = list(ds.data_vars)[0]

    print(f"\n   Using variable: '{precip_var}'")
    precip = ds[precip_var]

    print(f"   Shape: {precip.shape}")
    print(f"   Dimensions: {precip.dims}")

    # Check time range
    if 'time' in precip.coords:
        print(f"   Time range: {precip.time[0].values} to {precip.time[-1].values}")
        print(f"   Number of time steps: {len(precip.time)}")

    # Check spatial extent
    if 'lat' in precip.coords or 'latitude' in precip.coords:
        lat_coord = 'lat' if 'lat' in precip.coords else 'latitude'
        lon_coord = 'lon' if 'lon' in precip.coords else 'longitude'
        print(f"   Latitude range: [{float(precip[lat_coord].min()):.2f}, {float(precip[lat_coord].max()):.2f}]")
        print(f"   Longitude range: [{float(precip[lon_coord].min()):.2f}, {float(precip[lon_coord].max()):.2f}]")

    # Check for NaN values
    total_values = precip.size
    nan_values = int(np.isnan(precip.values).sum())
    print(f"   NaN values: {nan_values:,} ({100*nan_values/total_values:.1f}%)")

    # Basic statistics
    print(f"\n   Precipitation statistics:")
    print(f"   Mean: {float(precip.mean()):.2f} mm/month")
    print(f"   Min: {float(precip.min()):.2f} mm/month")
    print(f"   Max: {float(precip.max()):.2f} mm/month")

except Exception as e:
    print(f"   [ERROR] Error loading data: {e}")
    sys.exit(1)

# Calculate SPI-3 (single scale)
print("\n2. Calculating SPI-3 (single scale)...")
try:
    spi_3, params = spi(
        precip,
        scale=3,
        periodicity='monthly',
        calibration_start_year=1991,
        calibration_end_year=2020,
        return_params=True
    )
    print("   [OK] SPI-3 calculation successful!")
    print(f"   Output shape: {spi_3.shape}")
    print(f"   Valid range: [{float(spi_3.min()):.2f}, {float(spi_3.max()):.2f}]")
    print(f"   Mean: {float(spi_3.mean()):.3f}")
    print(f"   Std: {float(spi_3.std()):.3f}")

    # Save result
    os.makedirs('test_output', exist_ok=True)
    save_index_to_netcdf(spi_3, 'test_output/spi_3_bali_test.nc', compress=True)
    print("   [OK] Saved to: test_output/spi_3_bali_test.nc")

except Exception as e:
    print(f"   [ERROR] Error calculating SPI-3: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Calculate multi-scale SPI (3, 6, 9, 12)
print("\n3. Calculating multi-scale SPI (3, 6, 9, 12)...")
try:
    spi_multi = spi_multi_scale(
        precip,
        scales=[3, 6, 9, 12],
        periodicity='monthly',
        calibration_start_year=1991,
        calibration_end_year=2020,
        return_params=False
    )
    print("   [OK] Multi-scale SPI calculation successful!")
    print(f"   Variables: {list(spi_multi.data_vars)}")

    for scale in [3, 6, 9, 12]:
        var_name = f'spi_gamma_{scale}_month'
        if var_name in spi_multi:
            spi_data = spi_multi[var_name]
            print(f"\n   SPI-{scale}:")
            print(f"     Shape: {spi_data.shape}")
            print(f"     Range: [{float(spi_data.min()):.2f}, {float(spi_data.max()):.2f}]")
            print(f"     Mean: {float(spi_data.mean()):.3f}")

    # Save results
    save_index_to_netcdf(spi_multi, 'test_output/spi_multi_bali_test.nc', compress=True)
    print("\n   [OK] Saved to: test_output/spi_multi_bali_test.nc")

except Exception as e:
    print(f"   [ERROR] Error calculating multi-scale SPI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("[OK] All SPI tests completed successfully!")
print("=" * 80)
print("\nDataset: TerraClimate Bali (1958-2024)")
print("Location: Bali, Indonesia")
print("Grid size: 3x4 cells (~4km resolution)")
