#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for SPI calculation with TerraClimate Bali data.

Tests SPI with multiple probability distributions (Gamma, Pearson III,
Log-Logistic) and compares results visually.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import SPI functions
from indices import spi, spi_multi_scale, save_index_to_netcdf
from config import DISTRIBUTION_DISPLAY_NAMES
from utils import summarize_data_completeness, print_data_completeness

# Output directory
OUTPUT_DIR = 'test_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Distributions to test
TEST_DISTRIBUTIONS = ['gamma', 'pearson3', 'log_logistic']

print("=" * 80)
print("SPI Calculation Test with TerraClimate Bali Data")
print("Testing distributions:", ', '.join(
    DISTRIBUTION_DISPLAY_NAMES[d] for d in TEST_DISTRIBUTIONS
))
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
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

    # Data completeness (land-aware)
    report = summarize_data_completeness(precip)
    print_data_completeness(report, indent="   ")

    # Basic statistics
    print(f"\n   Precipitation statistics:")
    print(f"   Mean: {float(precip.mean()):.2f} mm/month")
    print(f"   Min: {float(precip.min()):.2f} mm/month")
    print(f"   Max: {float(precip.max()):.2f} mm/month")

except Exception as e:
    print(f"   [ERROR] Error loading data: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Single-scale SPI-3 with each distribution
# ============================================================================
print("\n2. Calculating SPI-3 (single scale) with each distribution...")

spi3_results = {}

for dist in TEST_DISTRIBUTIONS:
    dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
    print(f"\n   --- {dist_name} distribution ---")
    try:
        spi_3, params = spi(
            precip,
            scale=3,
            periodicity='monthly',
            calibration_start_year=1991,
            calibration_end_year=2020,
            return_params=True,
            distribution=dist
        )
        spi3_results[dist] = spi_3
        print(f"   [OK] SPI-3 ({dist_name}) calculation successful!")
        print(f"   Output shape: {spi_3.shape}")
        print(f"   Valid range: [{float(spi_3.min()):.2f}, {float(spi_3.max()):.2f}]")
        print(f"   Mean: {float(spi_3.mean()):.3f}")
        print(f"   Std: {float(spi_3.std()):.3f}")

        # Save result
        out_path = os.path.join(OUTPUT_DIR, f'spi_3_{dist}_bali_test.nc')
        save_index_to_netcdf(spi_3, out_path, compress=True)
        print(f"   [OK] Saved to: {out_path}")

    except Exception as e:
        print(f"   [ERROR] Error calculating SPI-3 ({dist_name}): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# STEP 3: Multi-scale SPI with default (gamma) distribution
# ============================================================================
print("\n3. Calculating multi-scale SPI (3, 6, 9, 12) with Gamma...")
try:
    spi_multi = spi_multi_scale(
        precip,
        scales=[3, 6, 9, 12],
        periodicity='monthly',
        calibration_start_year=1991,
        calibration_end_year=2020,
        return_params=False,
        distribution='gamma'
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
    out_path = os.path.join(OUTPUT_DIR, 'spi_multi_bali_test.nc')
    save_index_to_netcdf(spi_multi, out_path, compress=True)
    print(f"\n   [OK] Saved to: {out_path}")

except Exception as e:
    print(f"   [ERROR] Error calculating multi-scale SPI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: Multi-scale SPI with Pearson III
# ============================================================================
print("\n4. Calculating multi-scale SPI (3, 6, 9, 12) with Pearson III...")
try:
    spi_multi_p3 = spi_multi_scale(
        precip,
        scales=[3, 6, 9, 12],
        periodicity='monthly',
        calibration_start_year=1991,
        calibration_end_year=2020,
        return_params=False,
        distribution='pearson3'
    )
    print("   [OK] Multi-scale SPI (Pearson III) calculation successful!")
    print(f"   Variables: {list(spi_multi_p3.data_vars)}")

    for scale in [3, 6, 9, 12]:
        var_name = f'spi_pearson3_{scale}_month'
        if var_name in spi_multi_p3:
            spi_data = spi_multi_p3[var_name]
            print(f"\n   SPI-{scale} (Pearson III):")
            print(f"     Range: [{float(spi_data.min()):.2f}, {float(spi_data.max()):.2f}]")
            print(f"     Mean: {float(spi_data.mean()):.3f}")

    out_path = os.path.join(OUTPUT_DIR, 'spi_multi_pearson3_bali_test.nc')
    save_index_to_netcdf(spi_multi_p3, out_path, compress=True)
    print(f"\n   [OK] Saved to: {out_path}")

except Exception as e:
    print(f"   [ERROR] Error calculating multi-scale SPI (Pearson III): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 5: Cross-distribution comparison
# ============================================================================
print("\n5. Comparing SPI-3 across distributions...")
try:
    # Pick a single grid cell for comparison
    lat_idx, lon_idx = 5, 5
    lat_val = float(precip.lat.values[lat_idx])
    lon_val = float(precip.lon.values[lon_idx])
    print(f"   Comparison at grid cell: lat={lat_val:.2f}, lon={lon_val:.2f}")

    ref_dist = 'gamma'
    for dist in TEST_DISTRIBUTIONS:
        if dist == ref_dist:
            continue
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        ref_name = DISTRIBUTION_DISPLAY_NAMES[ref_dist]

        a = spi3_results[ref_dist].isel(lat=lat_idx, lon=lon_idx).values
        b = spi3_results[dist].isel(lat=lat_idx, lon=lon_idx).values

        valid = ~(np.isnan(a) | np.isnan(b))
        if np.sum(valid) > 0:
            diff = np.abs(a[valid] - b[valid])
            corr = np.corrcoef(a[valid], b[valid])[0, 1]
            print(f"\n   {ref_name} vs {dist_name}:")
            print(f"     Max absolute difference: {np.max(diff):.4f}")
            print(f"     Mean absolute difference: {np.mean(diff):.4f}")
            print(f"     Correlation: {corr:.6f}")

except Exception as e:
    print(f"   [ERROR] Error comparing distributions: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 6: Visual comparison plot
# ============================================================================
print("\n6. Creating visual comparison plots...")
try:
    lat_idx, lon_idx = 5, 5

    # --- Plot 1: SPI-3 time series comparison ---
    fig, axes = plt.subplots(len(TEST_DISTRIBUTIONS), 1, figsize=(14, 3.5 * len(TEST_DISTRIBUTIONS)),
                             sharex=True, sharey=True)
    if len(TEST_DISTRIBUTIONS) == 1:
        axes = [axes]

    time_vals = spi3_results['gamma'].time.values

    for ax, dist in zip(axes, TEST_DISTRIBUTIONS):
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        data = spi3_results[dist].isel(lat=lat_idx, lon=lon_idx).values

        # Bar plot: green for positive, red for negative
        colors = np.where(data >= 0, '#2166ac', '#b2182b')
        ax.bar(range(len(data)), data, color=colors, width=1.0, edgecolor='none')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=-1, color='orange', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=-2, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='skyblue', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=2, color='blue', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_ylabel('SPI-3')
        ax.set_title(f'SPI-3 - {dist_name} Distribution')
        ax.set_ylim(-3.5, 3.5)

    # Set time labels on bottom axis
    n = len(time_vals)
    tick_step = max(1, n // 10)
    tick_positions = range(0, n, tick_step)
    tick_labels = [str(np.datetime_as_string(time_vals[i], unit='M')) for i in tick_positions]
    axes[-1].set_xticks(list(tick_positions))
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[-1].set_xlabel('Time')

    fig.suptitle(f'SPI-3 Distribution Comparison - Bali ({lat_val:.2f}N, {lon_val:.2f}E)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spi3_distribution_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

    # --- Plot 2: Scatter comparison between distributions ---
    n_compare = len(TEST_DISTRIBUTIONS) - 1
    fig, axes = plt.subplots(1, n_compare, figsize=(6 * n_compare, 5))
    if n_compare == 1:
        axes = [axes]

    ref_data = spi3_results['gamma'].isel(lat=lat_idx, lon=lon_idx).values
    ref_name = DISTRIBUTION_DISPLAY_NAMES['gamma']

    for ax, dist in zip(axes, [d for d in TEST_DISTRIBUTIONS if d != 'gamma']):
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        comp_data = spi3_results[dist].isel(lat=lat_idx, lon=lon_idx).values

        valid = ~(np.isnan(ref_data) | np.isnan(comp_data))
        if np.sum(valid) > 0:
            ax.scatter(ref_data[valid], comp_data[valid], alpha=0.3, s=8, c='steelblue')
            # 1:1 line
            lims = [-3.5, 3.5]
            ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            corr = np.corrcoef(ref_data[valid], comp_data[valid])[0, 1]
            ax.set_title(f'{ref_name} vs {dist_name}\nr = {corr:.4f}')

        ax.set_xlabel(f'SPI-3 ({ref_name})')
        ax.set_ylabel(f'SPI-3 ({dist_name})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle('SPI-3 Cross-Distribution Scatter Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spi3_distribution_scatter.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

    # --- Plot 3: Multi-scale SPI overview (Gamma) ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, sharey=True)
    for ax, scale in zip(axes, [3, 6, 9, 12]):
        var_name = f'spi_gamma_{scale}_month'
        if var_name in spi_multi:
            data = spi_multi[var_name].isel(lat=lat_idx, lon=lon_idx).values
            time_arr = spi_multi[var_name].time.values
            colors = np.where(data >= 0, '#2166ac', '#b2182b')
            ax.bar(range(len(data)), data, color=colors, width=1.0, edgecolor='none')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel(f'SPI-{scale}')
            ax.set_title(f'SPI-{scale} (Gamma)')
            ax.set_ylim(-3.5, 3.5)

    n = len(time_arr)
    tick_step = max(1, n // 10)
    tick_positions = range(0, n, tick_step)
    tick_labels = [str(np.datetime_as_string(time_arr[i], unit='M')) for i in tick_positions]
    axes[-1].set_xticks(list(tick_positions))
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[-1].set_xlabel('Time')

    fig.suptitle(f'Multi-Scale SPI (Gamma) - Bali ({lat_val:.2f}N, {lon_val:.2f}E)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spi_multiscale_gamma.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

except Exception as e:
    print(f"   [ERROR] Error creating plots: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 7: Spatial maps - SPI across distributions at a single timestep
# ============================================================================
print("\n7. Creating spatial distribution comparison maps...")
try:
    # Find a timestep with notable drought (use a recent dry period)
    # Use the last timestep for simplicity
    time_idx = -1
    time_str = str(np.datetime_as_string(spi3_results['gamma'].time[time_idx].values, unit='M'))

    n_dist = len(TEST_DISTRIBUTIONS)
    fig, axes = plt.subplots(1, n_dist, figsize=(6 * n_dist, 5))
    if n_dist == 1:
        axes = [axes]

    for ax, dist in zip(axes, TEST_DISTRIBUTIONS):
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        data = spi3_results[dist].isel(time=time_idx)
        im = data.plot(ax=ax, cmap='RdYlBu', vmin=-3, vmax=3, add_colorbar=False)
        ax.set_title(f'SPI-3 ({dist_name})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.colorbar(im, ax=axes, label='SPI-3', shrink=0.8, pad=0.02,
                 orientation='vertical', aspect=30)
    fig.suptitle(f'SPI-3 Spatial Distribution Comparison - {time_str}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    out_path = os.path.join(OUTPUT_DIR, 'spi3_spatial_distribution_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

    # --- Spatial difference map: Gamma vs Pearson III ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, dist in zip(axes, ['pearson3', 'log_logistic']):
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        diff = spi3_results[dist].isel(time=time_idx) - spi3_results['gamma'].isel(time=time_idx)
        im = diff.plot(ax=ax, cmap='PiYG', vmin=-1.5, vmax=1.5, add_colorbar=True,
                        cbar_kwargs={'label': f'SPI-3 ({dist_name}) - SPI-3 (Gamma)'})
        ax.set_title(f'{dist_name} minus Gamma')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.suptitle(f'SPI-3 Spatial Difference from Gamma - {time_str}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spi3_spatial_difference_from_gamma.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

except Exception as e:
    print(f"   [ERROR] Error creating spatial maps: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("[OK] All SPI tests completed successfully!")
print("=" * 80)
print(f"\nDistributions tested: {', '.join(DISTRIBUTION_DISPLAY_NAMES[d] for d in TEST_DISTRIBUTIONS)}")
print("Dataset: TerraClimate Bali (1958-2024)")
print("Location: Bali, Indonesia")
print("Grid size: 3x4 cells (~4km resolution)")
print(f"\nOutputs in: {OUTPUT_DIR}/")
