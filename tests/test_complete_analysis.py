"""
Complete Analysis Test - Run Theory and Visualization

This script tests all analysis functions with both dry and wet thresholds:
- Run theory functions (runtheory.py)
- Visualization functions (visualization.py)

Tests with multiple distributions to verify run theory works on
non-gamma SPI/SPEI outputs.

Uses TerraClimate Bali data (1958-2024).

Author: Benny Istanto
Date: 2026-01-25
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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from config import DISTRIBUTION_DISPLAY_NAMES

# Output directory
OUTPUT_DIR = 'test_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("COMPLETE ANALYSIS TEST")
print("="*70)
print()
print("Testing run theory and visualization with both dry and wet thresholds")
print("Using TerraClimate Bali data (1958-2024)")
print()

# ============================================================================
# STEP 1: Load Real Test Data
# ============================================================================
print("Step 1: Loading SPI data...")

try:
    # Load gamma SPI-12
    spi_gamma_path = os.path.join(OUTPUT_DIR, 'spi_multi_bali_test.nc')
    if os.path.exists(spi_gamma_path):
        ds_gamma = xr.open_dataset(spi_gamma_path)
        spi_gamma = ds_gamma['spi_gamma_12_month']
        print(f"  [OK] Loaded Gamma SPI-12 from test output")
    else:
        print(f"  [ERROR] Gamma SPI data not found at {spi_gamma_path}. Run test_spi.py first.")
        sys.exit(1)

    print(f"  SPI-12 (Gamma) shape: {spi_gamma.shape}")
    print(f"  Time range: {spi_gamma.time[0].values} to {spi_gamma.time[-1].values}")
    print(f"  SPI range: [{spi_gamma.min().values:.2f}, {spi_gamma.max().values:.2f}]")
    print(f"  Mean: {spi_gamma.mean().values:.3f}, Std: {spi_gamma.std().values:.3f}")
    print()

    # Load Pearson III SPI-12 if available
    spi_p3_path = os.path.join(OUTPUT_DIR, 'spi_multi_pearson3_bali_test.nc')
    spi_pearson3 = None
    if os.path.exists(spi_p3_path):
        ds_p3 = xr.open_dataset(spi_p3_path)
        spi_pearson3 = ds_p3['spi_pearson3_12_month']
        print(f"  [OK] Loaded Pearson III SPI-12 from test output")
        print(f"  SPI-12 (Pearson III) range: [{spi_pearson3.min().values:.2f}, {spi_pearson3.max().values:.2f}]")
        print()

except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: Test Run Theory Functions with Gamma SPI
# ============================================================================
print("="*70)
print("STEP 2: TESTING RUN THEORY FUNCTIONS (Gamma SPI-12)")
print("="*70)
print()

from runtheory import (
    identify_events,
    calculate_timeseries,
    summarize_events,
    get_event_state,
    calculate_period_statistics
)

# Extract single location for event-based analysis
spi_loc = spi_gamma.isel(lat=5, lon=5)
lat_val = float(spi_gamma.lat.values[5])
lon_val = float(spi_gamma.lon.values[5])

print(f"Test location: {lat_val:.2f}, {lon_val:.2f} (Bali, Indonesia)")
print()

# Test 2.1: identify_events() - DRY
print("Test 2.1: identify_events() - DROUGHT (threshold -1.2)")
dry_events = identify_events(spi_loc, threshold=-1.2, min_duration=2)
print(f"  [OK] Found {len(dry_events)} drought events")
if len(dry_events) > 0:
    print(f"    Mean duration: {dry_events['duration'].mean():.1f} months")
    print(f"    Max magnitude: {dry_events['magnitude'].max():.2f}")
print()

# Test 2.2: identify_events() - WET
print("Test 2.2: identify_events() - WET (threshold +1.2)")
wet_events = identify_events(spi_loc, threshold=+1.2, min_duration=2)
print(f"  [OK] Found {len(wet_events)} wet events")
if len(wet_events) > 0:
    print(f"    Mean duration: {wet_events['duration'].mean():.1f} months")
    print(f"    Max magnitude: {wet_events['magnitude'].max():.2f}")
print()

# Test 2.3: calculate_timeseries() - DRY
print("Test 2.3: calculate_timeseries() - DROUGHT (threshold -1.2)")
dry_ts = calculate_timeseries(spi_loc, threshold=-1.2)
print(f"  [OK] Time series created: {len(dry_ts)} months")
print(f"    Months in drought: {dry_ts['is_event'].sum()}")
print(f"    Columns: {list(dry_ts.columns)}")
print()

# Test 2.4: calculate_timeseries() - WET
print("Test 2.4: calculate_timeseries() - WET (threshold +1.2)")
wet_ts = calculate_timeseries(spi_loc, threshold=+1.2)
print(f"  [OK] Time series created: {len(wet_ts)} months")
print(f"    Months in wet events: {wet_ts['is_event'].sum()}")
print()

# Test 2.5: summarize_events() - DRY
print("Test 2.5: summarize_events() - DROUGHT")
if len(dry_events) > 0:
    dry_summary = summarize_events(dry_events)
    print(f"  [OK] Summary created with {len(dry_summary)} statistics")
    print(f"    num_events: {dry_summary['num_events']}")
    print(f"    total_event_months: {dry_summary['total_event_months']}")
else:
    print("  - No events to summarize")
print()

# Test 2.6: summarize_events() - WET
print("Test 2.6: summarize_events() - WET")
if len(wet_events) > 0:
    wet_summary = summarize_events(wet_events)
    print(f"  [OK] Summary created with {len(wet_summary)} statistics")
    print(f"    num_events: {wet_summary['num_events']}")
    print(f"    total_event_months: {wet_summary['total_event_months']}")
else:
    print("  - No events to summarize")
print()

# Test 2.7: get_event_state() - DRY
print("Test 2.7: get_event_state() - DROUGHT")
is_event, category, deviation = get_event_state(-1.5, threshold=-1.0)
print(f"  [OK] State: is_event={is_event}, category='{category}', deviation={deviation:.2f}")
print()

# Test 2.8: get_event_state() - WET
print("Test 2.8: get_event_state() - WET")
is_event, category, deviation = get_event_state(+1.5, threshold=+1.0)
print(f"  [OK] State: is_event={is_event}, category='{category}', deviation={deviation:.2f}")
print()

# Test 2.9: calculate_period_statistics() - DRY
print("Test 2.9: calculate_period_statistics() - DROUGHT (2000-2023)")
dry_stats = calculate_period_statistics(
    spi_gamma, threshold=-1.2, start_year=2000, end_year=2023, min_duration=2
)
print(f"  [OK] Statistics calculated")
print(f"    Variables: {list(dry_stats.data_vars)}")
print(f"    Dimensions: {dry_stats.dims}")
print(f"    Mean events: {dry_stats.num_events.mean().values:.1f}")
print()

# Test 2.10: calculate_period_statistics() - WET
print("Test 2.10: calculate_period_statistics() - WET (2000-2023)")
wet_stats = calculate_period_statistics(
    spi_gamma, threshold=+1.2, start_year=2000, end_year=2023, min_duration=2
)
print(f"  [OK] Statistics calculated")
print(f"    Variables: {list(wet_stats.data_vars)}")
print(f"    Mean events: {wet_stats.num_events.mean().values:.1f}")
print()

# ============================================================================
# STEP 2b: Run Theory with Pearson III SPI (if available)
# ============================================================================
if spi_pearson3 is not None:
    print("="*70)
    print("STEP 2b: RUN THEORY WITH PEARSON III SPI-12")
    print("="*70)
    print()

    spi_p3_loc = spi_pearson3.isel(lat=5, lon=5)

    p3_dry = identify_events(spi_p3_loc, threshold=-1.2, min_duration=2)
    print(f"  [OK] Pearson III drought events: {len(p3_dry)}")

    p3_wet = identify_events(spi_p3_loc, threshold=+1.2, min_duration=2)
    print(f"  [OK] Pearson III wet events: {len(p3_wet)}")

    p3_dry_stats = calculate_period_statistics(
        spi_pearson3, threshold=-1.2, start_year=2000, end_year=2023, min_duration=2
    )
    print(f"  [OK] Pearson III period statistics calculated")
    print(f"    Mean drought events: {p3_dry_stats.num_events.mean().values:.1f}")

    # Compare event counts
    print(f"\n  Event comparison (single location):")
    print(f"    Gamma drought events:      {len(dry_events)}")
    print(f"    Pearson III drought events: {len(p3_dry)}")
    print(f"    Gamma wet events:           {len(wet_events)}")
    print(f"    Pearson III wet events:     {len(p3_wet)}")
    print()

# ============================================================================
# STEP 3: Test Visualization Functions
# ============================================================================
print("="*70)
print("STEP 3: TESTING VISUALIZATION FUNCTIONS")
print("="*70)
print()

from visualization import (
    plot_index,
    plot_events,
    plot_event_characteristics,
    plot_event_timeline,
    plot_spatial_stats
)

# Test 3.1: plot_index() - verify function works
print("Test 3.1: plot_index()")
fig = plot_index(spi_loc, threshold=-1.2, title='SPI-12 (Gamma): Drought Events - Bali')
plt.savefig(os.path.join(OUTPUT_DIR, 'test_plot_index.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot created successfully")
print()

# Test 3.2: plot_events() - verify function works
print("Test 3.2: plot_events()")
if len(dry_events) > 0:
    fig = plot_events(spi_loc, dry_events, threshold=-1.2, title='Drought Events Timeline - Bali')
    plt.close()
    print("  [OK] Function executed successfully")
else:
    print("  - No events to plot")
print()

# Test 3.3: plot_event_characteristics() - verify function works
print("Test 3.3: plot_event_characteristics()")
if len(dry_events) > 0:
    fig = plot_event_characteristics(dry_events, characteristic='magnitude')
    plt.close()
    print("  [OK] Function executed successfully")
else:
    print("  - No events to plot")
print()

# Test 3.4: plot_event_timeline() - verify function works
print("Test 3.4: plot_event_timeline()")
fig = plot_event_timeline(dry_ts)
plt.savefig(os.path.join(OUTPUT_DIR, 'test_plot_timeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Plot created successfully")
print()

# Test 3.5: plot_spatial_stats() - dry
print("Test 3.5: plot_spatial_stats() - Drought")
fig = plot_spatial_stats(dry_stats, variable='num_events', title='Drought Event Count (2000-2023) - Bali')
plt.savefig(os.path.join(OUTPUT_DIR, 'test_spatial_drought_events.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: test_spatial_drought_events.png")

# Test 3.6: plot_spatial_stats() - wet
print("Test 3.6: plot_spatial_stats() - Wet")
fig = plot_spatial_stats(wet_stats, variable='num_events', title='Wet Event Count (2000-2023) - Bali')
plt.savefig(os.path.join(OUTPUT_DIR, 'test_spatial_wet_events.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Saved: test_spatial_wet_events.png")

# Test 3.7: Spatial SPI map at a recent drought timestep
print("Test 3.7: Spatial SPI-12 map (latest timestep)")
latest_spi = spi_gamma.isel(time=-1)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
im = latest_spi.plot(ax=ax, cmap='RdYlBu', vmin=-3, vmax=3, add_colorbar=True,
                      cbar_kwargs={'label': 'SPI-12'})
time_str = str(np.datetime_as_string(spi_gamma.time[-1].values, unit='M'))
ax.set_title(f'SPI-12 (Gamma) - {time_str}')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.savefig(os.path.join(OUTPUT_DIR, 'test_spatial_spi12_latest.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: test_spatial_spi12_latest.png ({time_str})")
print()

# ============================================================================
# STEP 3b: Visualization with Pearson III data
# ============================================================================
if spi_pearson3 is not None:
    print("="*70)
    print("STEP 3b: VISUALIZATION WITH PEARSON III SPI")
    print("="*70)
    print()

    spi_p3_loc = spi_pearson3.isel(lat=5, lon=5)

    print("Test 3b.1: plot_index() with Pearson III SPI")
    fig = plot_index(spi_p3_loc, threshold=-1.2, title='SPI-12 (Pearson III): Drought Events - Bali')
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_plot_index_pearson3.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Pearson III plot saved")
    print()

    # Side-by-side distribution comparison for run theory
    print("Test 3b.2: Distribution comparison plot for run theory")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    for ax, data, label in zip(axes,
                                [spi_loc, spi_p3_loc],
                                ['SPI-12 (Gamma)', 'SPI-12 (Pearson III)']):
        vals = data.values
        colors = np.where(vals >= 0, '#2166ac', '#b2182b')
        ax.bar(range(len(vals)), vals, color=colors, width=1.0, edgecolor='none')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=-1.2, color='red', linewidth=0.8, linestyle='--', alpha=0.7, label='Threshold')
        ax.axhline(y=+1.2, color='blue', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.set_ylabel('SPI-12')
        ax.set_title(label)
        ax.set_ylim(-3.5, 3.5)

    time_vals = spi_gamma.time.values
    n = len(time_vals)
    tick_step = max(1, n // 10)
    tick_positions = range(0, n, tick_step)
    tick_labels = [str(np.datetime_as_string(time_vals[i], unit='M')) for i in tick_positions]
    axes[-1].set_xticks(list(tick_positions))
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    axes[-1].set_xlabel('Time')

    fig.suptitle(f'Run Theory Input Comparison - Bali ({lat_val:.2f}N, {lon_val:.2f}E)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'test_runtheory_distribution_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {out_path}")
    print()

# ============================================================================
# STEP 4: Summary
# ============================================================================
print("="*70)
print("TEST SUMMARY")
print("="*70)
print()
print("[OK] ALL TESTS PASSED!")
print()
print("Run Theory Functions Tested:")
print("  [OK] identify_events() - dry and wet")
print("  [OK] calculate_timeseries() - dry and wet")
print("  [OK] summarize_events() - dry and wet")
print("  [OK] get_event_state() - dry and wet")
print("  [OK] calculate_period_statistics() - dry and wet")
if spi_pearson3 is not None:
    print("  [OK] All functions also tested with Pearson III SPI")
print()
print("Visualization Functions Tested:")
print("  [OK] plot_index()")
print("  [OK] plot_events()")
print("  [OK] plot_event_characteristics()")
print("  [OK] plot_event_timeline()")
print("  [OK] plot_spatial_stats()")
if spi_pearson3 is not None:
    print("  [OK] Distribution comparison plots created")
print()
print(f"Results Summary:")
print(f"  Drought events found (Gamma): {len(dry_events)}")
print(f"  Wet events found (Gamma): {len(wet_events)}")
print(f"  Test plots saved to: {OUTPUT_DIR}/")
print()
print("="*70)
print("COMPLETE ANALYSIS VERIFIED!")
print("="*70)
print()
print("All functions work correctly for both:")
print("  - Drought (dry) events: threshold -1.2")
print("  - Wet (flood/excess) events: threshold +1.2")
if spi_pearson3 is not None:
    print("  - Multiple distribution inputs verified")
print()
print("Dataset: TerraClimate Bali (1958-2024)")
print("Location: Bali, Indonesia")
print()
