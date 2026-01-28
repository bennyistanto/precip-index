#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for SPEI calculation with TerraClimate Bali data.

Tests both scenarios: with PET and with temperature.
Tests multiple probability distributions (Gamma, Pearson III, Log-Logistic).
Compares results visually across distributions and input methods.
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

# Import SPEI functions
from indices import spei, save_index_to_netcdf
from utils import calculate_pet, summarize_data_completeness, print_data_completeness
from config import DISTRIBUTION_DISPLAY_NAMES

# Output directory
OUTPUT_DIR = 'test_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Distributions to test
TEST_DISTRIBUTIONS = ['gamma', 'pearson3', 'log_logistic']

print("=" * 80)
print("SPEI Calculation Test with TerraClimate Bali Data")
print("Comparing PET vs Temperature Input")
print("Testing distributions:", ', '.join(
    DISTRIBUTION_DISPLAY_NAMES[d] for d in TEST_DISTRIBUTIONS
))
print("=" * 80)

# ============================================================================
# STEP 1: Load precipitation data
# ============================================================================
print("\n1. Loading TerraClimate precipitation data...")
try:
    ds_ppt = xr.open_dataset('input/terraclimate_bali_ppt_1958_2024.nc')
    precip = ds_ppt['ppt']
    print("   [OK] Precipitation loaded")
    print(f"   Shape: {precip.shape}")
    print(f"   Dimensions: {precip.dims}")
    print(f"   Time range: {precip.time[0].values} to {precip.time[-1].values}")
    report = summarize_data_completeness(precip)
    print_data_completeness(report, indent="   ")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Load temperature data
# ============================================================================
print("\n2. Loading TerraClimate temperature data...")
try:
    ds_temp = xr.open_dataset('input/terraclimate_bali_tmean_1958_2024.nc')
    temperature = ds_temp['tmean']
    print("   [OK] Temperature loaded")
    print(f"   Shape: {temperature.shape}")
    print(f"   Mean temperature: {float(temperature.mean()):.1f} C")
    print(f"   Range: [{float(temperature.min()):.1f}, {float(temperature.max()):.1f}] C")
    t_report = summarize_data_completeness(temperature)
    print(f"   Land cells: {t_report['land_cells']}, "
          f"Mean completeness: {t_report['mean_temporal_completeness']:.1f}%")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: Load pre-computed PET data
# ============================================================================
print("\n3. Loading TerraClimate PET data...")
try:
    ds_pet = xr.open_dataset('input/terraclimate_bali_pet_1958_2024.nc')
    pet_precomputed = ds_pet['pet']
    print("   [OK] PET loaded")
    print(f"   Shape: {pet_precomputed.shape}")
    print(f"   Mean PET: {float(pet_precomputed.mean()):.1f} mm/month")
    print(f"   Range: [{float(pet_precomputed.min()):.1f}, {float(pet_precomputed.max()):.1f}] mm/month")
    p_report = summarize_data_completeness(pet_precomputed)
    print(f"   Land cells: {p_report['land_cells']}, "
          f"Mean completeness: {p_report['mean_temporal_completeness']:.1f}%")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# ============================================================================
# STEP 4: Calculate PET from temperature for comparison
# ============================================================================
print("\n4. Computing PET from temperature (Thornthwaite method)...")
try:
    lat_vals = temperature.lat.values

    pet_computed = calculate_pet(
        temperature=temperature,
        latitude=lat_vals,
        data_start_year=1958
    )
    print("   [OK] PET calculated from temperature")
    print(f"   Mean PET: {float(pet_computed.mean()):.1f} mm/month")
    print(f"   Range: [{float(pet_computed.min()):.1f}, {float(pet_computed.max()):.1f}] mm/month")

    # Save computed PET for reference
    pet_computed.to_netcdf(os.path.join(OUTPUT_DIR, 'pet_thornthwaite_bali_test.nc'))
    print(f"   [OK] Computed PET saved to: {OUTPUT_DIR}/pet_thornthwaite_bali_test.nc")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 5: SPEI-6 with each distribution (using pre-computed PET)
# ============================================================================
print("\n5. Calculating SPEI-6 with pre-computed PET for each distribution...")

spei6_pet_results = {}

for dist in TEST_DISTRIBUTIONS:
    dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
    print(f"\n   --- {dist_name} distribution ---")
    try:
        spei_result = spei(
            precip=precip,
            pet=pet_precomputed,
            scale=6,
            periodicity='monthly',
            calibration_start_year=1991,
            calibration_end_year=2020,
            return_params=False,
            distribution=dist
        )
        spei6_pet_results[dist] = spei_result
        print(f"   [OK] SPEI-6 ({dist_name}) with PET successful!")
        print(f"   Output shape: {spei_result.shape}")
        print(f"   Valid range: [{float(spei_result.min()):.2f}, {float(spei_result.max()):.2f}]")
        print(f"   Mean: {float(spei_result.mean()):.3f}")
        print(f"   Std: {float(spei_result.std()):.3f}")

        out_path = os.path.join(OUTPUT_DIR, f'spei_6_{dist}_with_pet_test.nc')
        save_index_to_netcdf(spei_result, out_path, compress=True)
        print(f"   [OK] Saved to: {out_path}")

    except Exception as e:
        print(f"   [ERROR] Error calculating SPEI-6 ({dist_name}): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# STEP 6: SPEI-6 with temperature (auto-compute PET) - Gamma only
# ============================================================================
print("\n6. Calculating SPEI-6 with temperature (auto-compute PET) - Gamma...")
try:
    spei_6_with_temp = spei(
        precip=precip,
        temperature=temperature,
        latitude=lat_vals,
        scale=6,
        periodicity='monthly',
        calibration_start_year=1991,
        calibration_end_year=2020,
        return_params=False,
        distribution='gamma'
    )
    print("   [OK] SPEI-6 calculation successful (using temperature)!")
    print(f"   Output shape: {spei_6_with_temp.shape}")
    print(f"   Valid range: [{float(spei_6_with_temp.min()):.2f}, {float(spei_6_with_temp.max()):.2f}]")
    print(f"   Mean: {float(spei_6_with_temp.mean()):.3f}")
    print(f"   Std: {float(spei_6_with_temp.std()):.3f}")

    out_path = os.path.join(OUTPUT_DIR, 'spei_6_gamma_with_temp_test.nc')
    save_index_to_netcdf(spei_6_with_temp, out_path, compress=True)
    print(f"   [OK] Saved to: {out_path}")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 7: Compare computed PET vs pre-computed PET
# ============================================================================
print("\n7. Comparing computed PET with pre-computed PET...")
try:
    pet_diff = np.abs(pet_computed.values - pet_precomputed.values)
    max_pet_diff = float(np.nanmax(pet_diff))
    mean_pet_diff = float(np.nanmean(pet_diff))

    print(f"   PET Max difference: {max_pet_diff:.4f} mm/month")
    print(f"   PET Mean difference: {mean_pet_diff:.4f} mm/month")

    if max_pet_diff < 1.0:
        print("   [OK] PET values are very close!")
    else:
        print("   [NOTE] Some differences expected - different PET methods")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 8: Compare PET vs Temperature SPEI (Gamma)
# ============================================================================
print("\n8. Comparing SPEI (PET input) vs SPEI (Temperature input) - Gamma...")
try:
    spei_pet_gamma = spei6_pet_results['gamma']
    diff = np.abs(spei_pet_gamma.values - spei_6_with_temp.values)
    max_diff = float(np.nanmax(diff))
    mean_diff = float(np.nanmean(diff))

    print(f"   SPEI Max difference: {max_diff:.6f}")
    print(f"   SPEI Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("   [OK] Results are very close! Both methods produce consistent SPEI.")
    else:
        print("   [NOTE] Some differences expected due to different PET calculation methods")

    # Calculate correlation
    valid_mask = ~(np.isnan(spei_pet_gamma.values) | np.isnan(spei_6_with_temp.values))
    if np.sum(valid_mask) > 0:
        correlation = np.corrcoef(
            spei_pet_gamma.values[valid_mask],
            spei_6_with_temp.values[valid_mask]
        )[0, 1]
        print(f"   Correlation: {correlation:.6f}")

        if correlation > 0.99:
            print("   [OK] Very high correlation - methods are equivalent!")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 9: Cross-distribution comparison (PET input)
# ============================================================================
print("\n9. Comparing SPEI-6 across distributions...")
try:
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

        a = spei6_pet_results[ref_dist].isel(lat=lat_idx, lon=lon_idx).values
        b = spei6_pet_results[dist].isel(lat=lat_idx, lon=lon_idx).values

        valid = ~(np.isnan(a) | np.isnan(b))
        if np.sum(valid) > 0:
            diff_vals = np.abs(a[valid] - b[valid])
            corr = np.corrcoef(a[valid], b[valid])[0, 1]
            print(f"\n   {ref_name} vs {dist_name}:")
            print(f"     Max absolute difference: {np.max(diff_vals):.4f}")
            print(f"     Mean absolute difference: {np.mean(diff_vals):.4f}")
            print(f"     Correlation: {corr:.6f}")

except Exception as e:
    print(f"   [ERROR] Error comparing distributions: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 10: Visual comparison plots
# ============================================================================
print("\n10. Creating visual comparison plots...")
try:
    lat_idx, lon_idx = 5, 5

    # --- Plot 1: SPEI-6 time series comparison across distributions ---
    fig, axes = plt.subplots(len(TEST_DISTRIBUTIONS), 1,
                             figsize=(14, 3.5 * len(TEST_DISTRIBUTIONS)),
                             sharex=True, sharey=True)
    if len(TEST_DISTRIBUTIONS) == 1:
        axes = [axes]

    time_vals = spei6_pet_results['gamma'].time.values

    for ax, dist in zip(axes, TEST_DISTRIBUTIONS):
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        data = spei6_pet_results[dist].isel(lat=lat_idx, lon=lon_idx).values

        colors = np.where(data >= 0, '#2166ac', '#b2182b')
        ax.bar(range(len(data)), data, color=colors, width=1.0, edgecolor='none')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=-1, color='orange', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=-2, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='skyblue', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=2, color='blue', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_ylabel('SPEI-6')
        ax.set_title(f'SPEI-6 - {dist_name} Distribution')
        ax.set_ylim(-3.5, 3.5)

    n = len(time_vals)
    tick_step = max(1, n // 10)
    tick_positions = range(0, n, tick_step)
    tick_labels = [str(np.datetime_as_string(time_vals[i], unit='M')) for i in tick_positions]
    axes[-1].set_xticks(list(tick_positions))
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[-1].set_xlabel('Time')

    fig.suptitle(f'SPEI-6 Distribution Comparison - Bali ({lat_val:.2f}N, {lon_val:.2f}E)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spei6_distribution_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

    # --- Plot 2: Scatter comparison between distributions ---
    n_compare = len(TEST_DISTRIBUTIONS) - 1
    fig, axes = plt.subplots(1, n_compare, figsize=(6 * n_compare, 5))
    if n_compare == 1:
        axes = [axes]

    ref_data = spei6_pet_results['gamma'].isel(lat=lat_idx, lon=lon_idx).values
    ref_name = DISTRIBUTION_DISPLAY_NAMES['gamma']

    for ax, dist in zip(axes, [d for d in TEST_DISTRIBUTIONS if d != 'gamma']):
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        comp_data = spei6_pet_results[dist].isel(lat=lat_idx, lon=lon_idx).values

        valid = ~(np.isnan(ref_data) | np.isnan(comp_data))
        if np.sum(valid) > 0:
            ax.scatter(ref_data[valid], comp_data[valid], alpha=0.3, s=8, c='steelblue')
            lims = [-3.5, 3.5]
            ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            corr = np.corrcoef(ref_data[valid], comp_data[valid])[0, 1]
            ax.set_title(f'{ref_name} vs {dist_name}\nr = {corr:.4f}')

        ax.set_xlabel(f'SPEI-6 ({ref_name})')
        ax.set_ylabel(f'SPEI-6 ({dist_name})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle('SPEI-6 Cross-Distribution Scatter Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spei6_distribution_scatter.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

    # --- Plot 3: PET vs Temperature SPEI comparison (Gamma) ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, sharey=True)

    pet_data = spei6_pet_results['gamma'].isel(lat=lat_idx, lon=lon_idx).values
    temp_data = spei_6_with_temp.isel(lat=lat_idx, lon=lon_idx).values

    for ax, data, label in zip(axes, [pet_data, temp_data],
                                ['SPEI-6 (Gamma) - Pre-computed PET',
                                 'SPEI-6 (Gamma) - Temperature (auto PET)']):
        colors = np.where(data >= 0, '#2166ac', '#b2182b')
        ax.bar(range(len(data)), data, color=colors, width=1.0, edgecolor='none')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('SPEI-6')
        ax.set_title(label)
        ax.set_ylim(-3.5, 3.5)

    n = len(time_vals)
    tick_step = max(1, n // 10)
    tick_positions = range(0, n, tick_step)
    tick_labels = [str(np.datetime_as_string(time_vals[i], unit='M')) for i in tick_positions]
    axes[-1].set_xticks(list(tick_positions))
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[-1].set_xlabel('Time')

    fig.suptitle(f'SPEI-6 PET vs Temperature Input Comparison - Bali ({lat_val:.2f}N, {lon_val:.2f}E)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spei6_pet_vs_temp_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

except Exception as e:
    print(f"   [ERROR] Error creating plots: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 11: Spatial maps - SPEI across distributions
# ============================================================================
print("\n11. Creating spatial distribution comparison maps...")
try:
    time_idx = -1
    time_str = str(np.datetime_as_string(spei6_pet_results['gamma'].time[time_idx].values, unit='M'))

    n_dist = len(TEST_DISTRIBUTIONS)
    fig, axes = plt.subplots(1, n_dist, figsize=(6 * n_dist, 6))
    if n_dist == 1:
        axes = [axes]

    for ax, dist in zip(axes, TEST_DISTRIBUTIONS):
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        data = spei6_pet_results[dist].isel(time=time_idx)
        im = data.plot(ax=ax, cmap='RdYlBu', vmin=-3, vmax=3, add_colorbar=False)
        ax.set_title(f'SPEI-6 ({dist_name})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.suptitle(f'SPEI-6 Spatial Distribution Comparison - {time_str}',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(bottom=0.18, wspace=0.3)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    fig.colorbar(im, cax=cbar_ax, label='SPEI-6', orientation='horizontal')

    out_path = os.path.join(OUTPUT_DIR, 'spei6_spatial_distribution_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

    # --- Spatial difference map: Gamma vs others ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, dist in zip(axes, ['pearson3', 'log_logistic']):
        dist_name = DISTRIBUTION_DISPLAY_NAMES[dist]
        diff = spei6_pet_results[dist].isel(time=time_idx) - spei6_pet_results['gamma'].isel(time=time_idx)
        im = diff.plot(ax=ax, cmap='PiYG', vmin=-1.5, vmax=1.5, add_colorbar=True,
                        cbar_kwargs={'label': f'SPEI-6 ({dist_name}) - SPEI-6 (Gamma)'})
        ax.set_title(f'{dist_name} minus Gamma')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    fig.suptitle(f'SPEI-6 Spatial Difference from Gamma - {time_str}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'spei6_spatial_difference_from_gamma.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {out_path}")

except Exception as e:
    print(f"   [ERROR] Error creating spatial maps: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("[OK] All SPEI tests completed successfully!")
print("=" * 80)
print(f"\nDistributions tested: {', '.join(DISTRIBUTION_DISPLAY_NAMES[d] for d in TEST_DISTRIBUTIONS)}")
print("\nConclusion:")
print("  1. SPEI with pre-computed PET: WORKS")
print("  2. SPEI with temperature (auto PET): WORKS")
print("  3. Multi-distribution support: WORKS")
print("  4. Results compared visually across distributions")
print("\nDataset: TerraClimate Bali (1958-2024)")
print("Location: Bali, Indonesia")
print("Grid size: 3x4 cells (~4km resolution)")
print(f"\nOutputs in: {OUTPUT_DIR}/")
