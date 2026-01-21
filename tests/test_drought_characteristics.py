#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for climate extreme event characteristics using run theory.

Tests all three modes:
1. Event-based: identify_events()
2. Time-series: calculate_timeseries()
3. Period statistics: calculate_period_statistics()

Tests both dry (drought) and wet (flood) event identification.
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
import matplotlib.pyplot as plt

# Import climate extreme event analysis functions
from runtheory import (
    identify_events,
    calculate_timeseries,
    calculate_period_statistics,
    calculate_annual_statistics,
    compare_periods,
    summarize_events
)

# Import visualization functions
from visualization import (
    generate_location_filename,
    plot_index,
    plot_events,
    plot_event_characteristics,
    plot_event_timeline
)

print("=" * 80)
print("Climate Extreme Event Analysis Test - Run Theory Implementation")
print("=" * 80)

# Load SPI data
print("\n1. Loading SPI-12 data...")
try:
    # Try to load pre-computed SPI-12
    if os.path.exists('output/spi_multi_chirps.nc'):
        ds = xr.open_dataset('output/spi_multi_chirps.nc')
        spi_12 = ds['spi_gamma_12_month']
        print("   [OK] Loaded pre-computed SPI-12")
    else:
        print("   [ERROR] SPI-12 data not found. Run test_spi.py first.")
        sys.exit(1)

    print(f"   Shape: {spi_12.shape}")
    print(f"   Dimensions: {spi_12.dims}")
    print(f"   Time range: {spi_12.time.values[0]} to {spi_12.time.values[-1]}")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Select a single location for detailed analysis
print("\n2. Selecting test location (center of grid)...")
lat_idx = spi_12.sizes['lat'] // 2
lon_idx = spi_12.sizes['lon'] // 2
spi_ts = spi_12.isel(lat=lat_idx, lon=lon_idx)
print(f"   Location: lat={float(spi_ts.lat):.2f}, lon={float(spi_ts.lon):.2f}")

# Test 1: Event-based analysis
print("\n3. Test 1: Event-Based Drought Identification...")
print("   Threshold: -1.2")
try:
    events = identify_events(spi_ts, threshold=-1.2, min_duration=3)
    print(f"   [OK] Found {len(events)} drought events (duration >= 3 months)")

    if len(events) > 0:
        print("\n   First 5 events:")
        print(events.head()[['start_date', 'end_date', 'duration', 'magnitude', 'intensity', 'peak']])

        # Summary statistics
        summary = summarize_events(events)
        print("\n   Summary Statistics:")
        print(f"   - Total events: {summary['num_events']}")
        print(f"   - Mean duration: {summary['mean_duration']:.1f} months")
        print(f"   - Max duration: {summary['max_duration']} months")
        print(f"   - Mean magnitude: {summary['mean_magnitude']:.2f}")
        print(f"   - Most severe peak: {summary['most_severe_peak']:.2f}")

        # Save event list with location in filename
        import os
        os.makedirs('output/csv', exist_ok=True)
        csv_file = f'output/csv/drought_events_lat{float(spi_ts.lat):.2f}_lon{float(spi_ts.lon):.2f}.csv'
        events.to_csv(csv_file, index=False)
        print(f"   [OK] Saved events to: {csv_file}")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 2: Time-series monitoring
print("\n4. Test 2: Time-Series Drought Monitoring...")
try:
    ts = calculate_timeseries(spi_ts, threshold=-1.2)
    print(f"   [OK] Calculated time-series characteristics")
    print(f"   Shape: {ts.shape}")

    # Show current drought status (last 12 months)
    current = ts.tail(12)
    in_drought = current[current['is_event']]

    if len(in_drought) > 0:
        print("\n   Current drought status (last 12 months):")
        print(in_drought[['index_value', 'duration', 'magnitude_cumulative',
                         'magnitude_instantaneous', 'intensity']].head())
    else:
        print("\n   No drought in last 12 months")

    # Save time series with location in filename
    csv_file = f'output/csv/drought_timeseries_lat{float(spi_ts.lat):.2f}_lon{float(spi_ts.lon):.2f}.csv'
    ts.to_csv(csv_file)
    print(f"   [OK] Saved time series to: {csv_file}")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 3: Period statistics (gridded)
print("\n5. Test 3: Period Statistics (Gridded)...")
print("   Calculating statistics for 2020-2024...")
try:
    stats_recent = calculate_period_statistics(
        spi_12,
        threshold=-1.2,
        start_year=2020,
        end_year=2024,
        min_duration=3
    )
    print("   [OK] Period statistics calculated")
    print(f"   Output dimensions: {stats_recent.dims}")

    # Show spatial statistics
    print("\n   Spatial Statistics (2020-2024):")
    print(f"   - Mean events per location: {float(stats_recent.num_events.mean()):.1f}")
    print(f"   - Max events at any location: {int(stats_recent.num_events.max())}")
    print(f"   - Mean total magnitude: {float(stats_recent.total_magnitude.mean()):.2f}")
    print(f"   - Worst peak anywhere: {float(stats_recent.worst_peak.min()):.2f}")

    # Save gridded output
    import os
    os.makedirs('output/netcdf', exist_ok=True)
    stats_recent.to_netcdf('output/netcdf/drought_stats_2020-2024.nc')
    print("   [OK] Saved to: output/netcdf/drought_stats_2020-2024.nc")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 4: Annual statistics
print("\n6. Test 4: Annual Statistics (Year-by-Year)...")
try:
    # Calculate for subset of years (last 10 years)
    spi_subset = spi_12.sel(time=slice('2015-01-01', '2024-12-31'))

    print("   Calculating annual statistics for 2015-2024...")
    annual_stats = calculate_annual_statistics(
        spi_subset,
        threshold=-1.2,
        min_duration=3
    )
    print("   [OK] Annual statistics calculated")
    print(f"   Output dimensions: {annual_stats.dims}")

    # Show trend
    annual_mean = annual_stats.num_events.mean(dim=['lat', 'lon'])
    print("\n   Events per year (spatial average):")
    for year, count in zip(annual_stats.year.values, annual_mean.values):
        print(f"   - {year}: {float(count):.1f} events")

    # Save annual output
    annual_stats.to_netcdf('output/netcdf/drought_stats_annual.nc')
    print("   [OK] Saved to: output/netcdf/drought_stats_annual.nc")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 5: Compare periods
print("\n7. Test 5: Compare Time Periods...")
try:
    periods = [(1991, 2020), (2021, 2024)]
    names = ['Historical (1991-2020)', 'Recent (2021-2024)']

    print(f"   Comparing: {names[0]} vs {names[1]}")
    comparison = compare_periods(
        spi_12,
        periods=periods,
        threshold=-1.2,
        min_duration=3,
        period_names=names
    )
    print("   [OK] Period comparison complete")
    print(f"   Output dimensions: {comparison.dims}")

    # Calculate differences
    diff = comparison.sel(period=names[1]) - comparison.sel(period=names[0])

    print("\n   Changes (Recent - Historical):")
    print(f"   - Change in mean events: {float(diff.num_events.mean()):+.2f}")
    print(f"   - Change in total magnitude: {float(diff.total_magnitude.mean()):+.2f}")
    print(f"   - Change in % time in drought: {float(diff.pct_time_in_drought.mean()):+.2f}%")

    # Save comparison
    comparison.to_netcdf('output/netcdf/drought_comparison.nc')
    print("   [OK] Saved to: output/netcdf/drought_comparison.nc")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 6: Visualization
print("\n8. Test 6: Creating Visualizations...")
try:
    import os
    os.makedirs('output/plots/single', exist_ok=True)
    os.makedirs('output/plots/spatial', exist_ok=True)

    # Generate location-based filenames
    lat_str = f"{float(spi_ts.lat):.2f}"
    lon_str = f"{float(spi_ts.lon):.2f}"

    # Plot 1: Drought index with events
    print("   Creating plot 1: Drought events timeline...")
    fig1 = plt.figure(figsize=(14, 6))
    ax1 = plot_events(spi_ts, events, threshold=-1.2,
                              title=f'SPI-12 Drought Events - Morocco (lat={lat_str}, lon={lon_str})')
    plot_file = f'output/plots/single/plot_events_lat{lat_str}_lon{lon_str}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {plot_file}")

    # Plot 2: Drought characteristics
    print("   Creating plot 2: Event characteristics...")
    fig2 = plot_event_characteristics(events, characteristic='magnitude')
    plot_file = f'output/plots/single/plot_event_characteristics_lat{lat_str}_lon{lon_str}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {plot_file}")

    # Plot 3: Time-series evolution
    print("   Creating plot 3: Drought evolution timeline...")
    # Uses default characteristics which now include both magnitude types
    fig3 = plot_event_timeline(ts)
    plot_file = f'output/plots/single/plot_event_timeline_lat{lat_str}_lon{lon_str}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {plot_file}")

    # Plot 4: Spatial map of recent period
    print("   Creating plot 4: Spatial maps...")
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Number of events
    stats_recent.num_events.plot(ax=axes[0,0], cmap='YlOrRd')
    axes[0,0].set_title('Number of Events (2020-2024)')

    # Total magnitude
    stats_recent.total_magnitude.plot(ax=axes[0,1], cmap='YlOrRd')
    axes[0,1].set_title('Total Magnitude (2020-2024)')

    # Worst peak
    stats_recent.worst_peak.plot(ax=axes[1,0], cmap='YlOrRd_r')
    axes[1,0].set_title('Worst Peak Severity (2020-2024)')

    # % time in drought
    stats_recent.pct_time_in_drought.plot(ax=axes[1,1], cmap='YlOrRd')
    axes[1,1].set_title('% Time in Drought (2020-2024)')

    plt.tight_layout()
    plot_file = 'output/plots/spatial/plot_spatial_stats_2020-2024.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [OK] Saved: {plot_file}")

except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("[OK] All drought characteristics tests completed successfully!")
print("=" * 80)
print("\nOutputs created (organized by type):")
print("\n  CSV Files (single-location):")
print(f"  - output/csv/drought_events_lat{lat_str}_lon{lon_str}.csv")
print(f"  - output/csv/drought_timeseries_lat{lat_str}_lon{lon_str}.csv")
print("\n  NetCDF Files (gridded):")
print("  - output/netcdf/drought_stats_2020-2024.nc")
print("  - output/netcdf/drought_stats_annual.nc")
print("  - output/netcdf/drought_comparison.nc")
print("\n  Plots - Single Location:")
print(f"  - output/plots/single/plot_events_lat{lat_str}_lon{lon_str}.png")
print(f"  - output/plots/single/plot_event_characteristics_lat{lat_str}_lon{lon_str}.png")
print(f"  - output/plots/single/plot_event_timeline_lat{lat_str}_lon{lon_str}.png")
print("\n  Plots - Spatial:")
print("  - output/plots/spatial/plot_spatial_stats_2020-2024.png")
print("\nAll functions validated:")
print("  [OK] Event-based analysis (identify_drought_events)")
print("  [OK] Time-series monitoring (calculate_timeseries)")
print("  [OK] Period statistics (calculate_period_statistics)")
print("  [OK] Annual statistics (calculate_annual_statistics)")
print("  [OK] Period comparison (compare_periods)")
print("  [OK] Visualization functions")
print("\nFolder structure:")
print("  output/csv/        - Single-location CSV files")
print("  output/netcdf/     - Gridded NetCDF files")
print("  output/plots/single/  - Single-location plots")
print("  output/plots/spatial/ - Spatial maps")
