#!/usr/bin/env python
"""
Test 06: Visualization

This test generates comprehensive visualizations:
- Time series plots with WMO color scheme
- Distribution comparison plots
- Spatial maps
- Multi-panel summary figures
- Seasonal drought heatmap
- Historical extreme events with run theory
- Multi-scale SPI comparison
- Decadal trend analysis
- Exceedance probability plots

Author: Benny Istanto
"""

from conftest import (
    print_header, print_subheader, print_ok, print_fail, print_info,
    setup_output_dirs, load_data, check_input_files, setup_matplotlib,
    OUTPUT_NETCDF, OUTPUT_PLOTS,
    CALIBRATION_START, CALIBRATION_END, DATA_START_YEAR,
    TEST_DISTRIBUTIONS, SAMPLE_LAT_IDX, SAMPLE_LON_IDX
)

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# Import functions
from indices import spi, spei
from utils import calculate_pet
from config import DISTRIBUTION_DISPLAY_NAMES
from visualization import plot_index, plot_events
from runtheory import identify_events


def load_or_calculate_indices(precip, tmean, tmin, tmax, lat):
    """Load pre-calculated indices or calculate them."""
    print_subheader("Loading/Calculating Indices")

    results = {}

    # Check if we have pre-calculated files
    spi_12_file = OUTPUT_NETCDF / 'spi_12_gamma.nc'
    spei_12_file = OUTPUT_NETCDF / 'spei_12_thornthwaite_gamma.nc'

    if spi_12_file.exists():
        results['spi_12'] = xr.open_dataarray(spi_12_file)
        print_ok("Loaded SPI-12 from cache")
    else:
        print_info("Calculating SPI-12...")
        results['spi_12'] = spi(precip, scale=12, periodicity='monthly',
                                calibration_start_year=CALIBRATION_START,
                                calibration_end_year=CALIBRATION_END)
        print_ok("SPI-12 calculated")

    if spei_12_file.exists() and tmean is not None:
        results['spei_12'] = xr.open_dataarray(spei_12_file)
        print_ok("Loaded SPEI-12 from cache")
    elif tmean is not None:
        print_info("Calculating SPEI-12...")
        results['spei_12'] = spei(precip, temperature=tmean, latitude=lat, scale=12,
                                  periodicity='monthly', data_start_year=DATA_START_YEAR,
                                  calibration_start_year=CALIBRATION_START,
                                  calibration_end_year=CALIBRATION_END,
                                  pet_method='thornthwaite')
        print_ok("SPEI-12 calculated")

    return results


def create_wmo_timeseries_plots(results: dict):
    """Create time series plots with WMO color scheme."""
    print_subheader("Creating WMO Time Series Plots")

    plt = setup_matplotlib()

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, results['spi_12'].shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, results['spi_12'].shape[2] - 1)

    for name, da in results.items():
        # Extract point time series
        ts = da[:, sample_lat, sample_lon]

        # Use the built-in visualization function
        ax = plot_index(ts, threshold=-1.0, title=f'{name.upper()} Time Series - Bali')
        fig = ax.get_figure()

        filepath = OUTPUT_PLOTS / f'{name}_wmo_timeseries.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print_ok(f"Saved: {filepath.name}")


def create_distribution_comparison_plot(precip):
    """Create a plot comparing SPI across different distributions."""
    print_subheader("Creating Distribution Comparison Plot")

    plt = setup_matplotlib()

    # Calculate SPI-12 for each distribution
    spis = {}
    for dist in TEST_DISTRIBUTIONS:
        print_info(f"Calculating SPI-12 ({DISTRIBUTION_DISPLAY_NAMES[dist]})...")
        spis[dist] = spi(precip, scale=12, periodicity='monthly',
                         calibration_start_year=CALIBRATION_START,
                         calibration_end_year=CALIBRATION_END,
                         distribution=dist)

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, precip.shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, precip.shape[2] - 1)

    # Create comparison plot
    fig, axes = plt.subplots(len(TEST_DISTRIBUTIONS), 1, figsize=(14, 3 * len(TEST_DISTRIBUTIONS)), sharex=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (dist, da) in enumerate(spis.items()):
        ts = da[:, sample_lat, sample_lon]

        ax = axes[i]
        ax.fill_between(da.time, 0, ts.where(ts >= 0), color='#2166ac', alpha=0.7)
        ax.fill_between(da.time, 0, ts.where(ts < 0), color='#b2182b', alpha=0.7)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axhline(y=-1, color='orange', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.axhline(y=1, color='orange', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_ylabel(f'SPI-12\n({DISTRIBUTION_DISPLAY_NAMES[dist]})')
        ax.set_ylim(-3.5, 3.5)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    axes[0].set_title('SPI-12 Distribution Comparison - Bali')

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / 'spi_distribution_comparison.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def create_spatial_maps(results: dict):
    """Create spatial maps of mean and recent indices."""
    print_subheader("Creating Spatial Maps")

    plt = setup_matplotlib()

    for name, da in results.items():
        # Mean map
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Mean over entire period
        mean_da = da.mean(dim='time')
        im1 = axes[0].pcolormesh(da.lon, da.lat, mean_da, cmap='RdYlBu', vmin=-1, vmax=1, shading='auto')
        axes[0].set_title(f'Mean {name.upper()} (1950-2025)')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0], label=name.upper())

        # Most recent value
        recent_da = da[-1, :, :]
        recent_time = str(da.time[-1].values)[:7]
        im2 = axes[1].pcolormesh(da.lon, da.lat, recent_da, cmap='RdYlBu', vmin=-2.5, vmax=2.5, shading='auto')
        axes[1].set_title(f'{name.upper()} ({recent_time})')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[1], label=name.upper())

        plt.tight_layout()
        filepath = OUTPUT_PLOTS / f'{name}_spatial_maps.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        print_ok(f"Saved: {filepath.name}")


def create_summary_figure(results: dict, precip, tmean):
    """Create a comprehensive multi-panel summary figure."""
    print_subheader("Creating Summary Figure")

    plt = setup_matplotlib()

    fig = plt.figure(figsize=(16, 12))

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, results['spi_12'].shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, results['spi_12'].shape[2] - 1)

    # Panel 1: Precipitation climatology (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    precip_ts = precip[:, sample_lat, sample_lon]

    # Monthly climatology
    monthly_clim = precip_ts.groupby('time.month').mean()
    ax1.bar(range(1, 13), monthly_clim.values, color='steelblue', edgecolor='navy', alpha=0.8)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Precipitation (mm)')
    ax1.set_title('Monthly Precipitation Climatology')
    ax1.set_xticks(range(1, 13))
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Temperature trend (top middle)
    if tmean is not None:
        ax2 = fig.add_subplot(2, 3, 2)
        temp_ts = tmean[:, sample_lat, sample_lon]

        # Annual mean
        annual_temp = temp_ts.groupby('time.year').mean()
        years = annual_temp.year.values
        temps = annual_temp.values

        ax2.plot(years, temps, 'o-', color='orangered', markersize=3, alpha=0.7)

        # Trend line
        z = np.polyfit(years, temps, 1)
        p = np.poly1d(z)
        ax2.plot(years, p(years), 'k--', linewidth=2,
                label=f'Trend: {z[0]*10:.2f}°C/decade')

        ax2.set_xlabel('Year')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Annual Mean Temperature Trend')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Panel 3: SPI-12 histogram (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    spi_values = results['spi_12'].values.flatten()
    valid_spi = spi_values[~np.isnan(spi_values)]

    ax3.hist(valid_spi, bins=50, color='steelblue', edgecolor='navy', alpha=0.8, density=True)

    # Standard normal reference
    x = np.linspace(-4, 4, 100)
    ax3.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi), 'r-', linewidth=2, label='Standard Normal')

    ax3.set_xlabel('SPI-12')
    ax3.set_ylabel('Density')
    ax3.set_title('SPI-12 Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: SPI-12 time series (bottom left, wider)
    ax4 = fig.add_subplot(2, 3, (4, 5))
    spi_ts = results['spi_12'][:, sample_lat, sample_lon]

    ax4.fill_between(results['spi_12'].time, 0, spi_ts.where(spi_ts >= 0), color='#2166ac', alpha=0.7)
    ax4.fill_between(results['spi_12'].time, 0, spi_ts.where(spi_ts < 0), color='#b2182b', alpha=0.7)
    ax4.axhline(y=0, color='k', linewidth=0.5)
    ax4.axhline(y=-1, color='orange', linestyle='--', linewidth=0.5, alpha=0.7)
    ax4.axhline(y=-2, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    ax4.set_xlabel('Time')
    ax4.set_ylabel('SPI-12')
    ax4.set_title('SPI-12 Time Series (1950-2025)')
    ax4.set_ylim(-3.5, 3.5)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Spatial map (bottom right)
    ax5 = fig.add_subplot(2, 3, 6)
    mean_spi = results['spi_12'].mean(dim='time')
    im = ax5.pcolormesh(results['spi_12'].lon, results['spi_12'].lat, mean_spi,
                       cmap='RdYlBu', vmin=-0.5, vmax=0.5, shading='auto')
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    ax5.set_title('Mean SPI-12 (1950-2025)')
    plt.colorbar(im, ax=ax5, label='SPI-12')

    plt.suptitle('Precipitation Index Analysis - Bali, Indonesia', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_PLOTS / 'summary_figure.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def create_pet_method_summary(precip, tmean, tmin, tmax, lat):
    """Create PET method comparison summary."""
    print_subheader("Creating PET Method Summary")

    plt = setup_matplotlib()

    # Calculate PET with different methods
    pet_methods = {}

    print_info("Calculating Thornthwaite PET...")
    pet_methods['Thornthwaite'] = calculate_pet(tmean, lat, DATA_START_YEAR, method='thornthwaite')

    if tmin is not None and tmax is not None:
        print_info("Calculating Hargreaves PET...")
        pet_methods['Hargreaves'] = calculate_pet(tmean, lat, DATA_START_YEAR, method='hargreaves',
                                                  temp_min=tmin, temp_max=tmax)

    # Load reference PET if available
    status = check_input_files()
    if status['pet']['exists']:
        pet_methods['TerraClimate'] = load_data('pet')

    if len(pet_methods) < 2:
        print_info("Not enough PET methods for comparison plot")
        return

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, tmean.shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, tmean.shape[2] - 1)

    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = {'Thornthwaite': '#ff7f0e', 'Hargreaves': '#2ca02c', 'TerraClimate': '#1f77b4'}

    # Time series (last 5 years)
    n_months = 60
    ax1 = axes[0]

    for name, pet_da in pet_methods.items():
        ts = pet_da[-n_months:, sample_lat, sample_lon].values
        ax1.plot(range(n_months), ts, label=name, color=colors.get(name, 'gray'), linewidth=1.5)

    ax1.set_xlabel('Months (last 5 years)')
    ax1.set_ylabel('PET (mm/month)')
    ax1.set_title('PET Method Comparison - Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter comparison
    ax2 = axes[1]

    if 'TerraClimate' in pet_methods:
        ref = pet_methods['TerraClimate'].values.flatten()

        for name, pet_da in pet_methods.items():
            if name == 'TerraClimate':
                continue

            calc = pet_da.values.flatten()
            valid = ~(np.isnan(ref) | np.isnan(calc))

            r = np.corrcoef(ref[valid], calc[valid])[0, 1]
            ax2.scatter(ref[valid], calc[valid], alpha=0.1, s=5,
                       color=colors.get(name, 'gray'), label=f'{name} (r={r:.3f})')

        max_val = np.nanmax(ref)
        ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')

        ax2.set_xlabel('TerraClimate PET (mm/month)')
        ax2.set_ylabel('Calculated PET (mm/month)')
        ax2.set_title('PET Method vs TerraClimate Reference')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / 'pet_method_summary.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def create_seasonal_drought_heatmap(results: dict):
    """Create a month vs year heatmap showing drought patterns."""
    print_subheader("Creating Seasonal Drought Heatmap")

    plt = setup_matplotlib()

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, results['spi_12'].shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, results['spi_12'].shape[2] - 1)

    for name, da in results.items():
        # Extract time series
        ts = da[:, sample_lat, sample_lon]

        # Create DataFrame with year and month
        times = pd.to_datetime(ts.time.values)
        df = pd.DataFrame({
            'value': ts.values,
            'year': times.year,
            'month': times.month
        })

        # Pivot to create heatmap matrix
        pivot = df.pivot_table(values='value', index='month', columns='year', aggfunc='mean')

        # Create figure
        fig, ax = plt.subplots(figsize=(18, 6))

        # Create heatmap
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlBu',
                      vmin=-2.5, vmax=2.5, interpolation='nearest')

        # Set ticks
        years = pivot.columns.values
        ax.set_xticks(range(0, len(years), 5))
        ax.set_xticklabels(years[::5], rotation=45, ha='right')
        ax.set_yticks(range(12))
        ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Month', fontsize=12)
        ax.set_title(f'{name.upper()} Seasonal Drought Heatmap - Bali (1950-2025)\n'
                    'Red = Drought, Blue = Wet', fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(name.upper(), fontsize=11)

        plt.tight_layout()
        filepath = OUTPUT_PLOTS / f'{name}_seasonal_heatmap.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        print_ok(f"Saved: {filepath.name}")


def create_historical_events_plot(results: dict):
    """Create plot showing historical extreme events with run theory."""
    print_subheader("Creating Historical Events Plot")

    plt = setup_matplotlib()

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, results['spi_12'].shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, results['spi_12'].shape[2] - 1)

    for name, da in results.items():
        # Extract time series
        ts = da[:, sample_lat, sample_lon]

        # Identify drought events (negative threshold)
        drought_events = identify_events(ts, threshold=-1.0)
        # Identify wet events (positive threshold)
        wet_events = identify_events(ts, threshold=1.0)

        # Create figure with 2 panels sharing x-axis
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                 gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.15})

        # Panel 1: Drought events
        ax1 = axes[0]
        times = pd.to_datetime(ts.time.values)
        values = ts.values

        ax1.fill_between(times, 0, np.where(values < 0, values, 0),
                        color='#b2182b', alpha=0.5, label='Dry')
        ax1.fill_between(times, 0, np.where(values >= 0, values, 0),
                        color='#2166ac', alpha=0.5, label='Wet')
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Drought threshold')
        ax1.axhline(y=-2, color='darkred', linestyle='--', linewidth=1, alpha=0.7, label='Extreme drought')
        ax1.axhline(y=1, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Wet threshold')
        ax1.axhline(y=2, color='darkblue', linestyle='--', linewidth=1, alpha=0.7, label='Extreme wet')

        # Annotate top 5 drought events
        if len(drought_events) > 0:
            top_droughts = drought_events.nlargest(5, 'magnitude')
            for idx, event in top_droughts.iterrows():
                peak_idx = int(event['peak_idx'])
                if peak_idx < len(times):
                    peak_time = times[peak_idx]
                    peak_val = values[peak_idx]
                    ax1.annotate(f"{peak_time.year}",
                               xy=(peak_time, peak_val),
                               xytext=(0, -20), textcoords='offset points',
                               fontsize=9, ha='center', color='darkred',
                               arrowprops=dict(arrowstyle='->', color='darkred', lw=0.5))

        # Annotate top 5 wet events
        if len(wet_events) > 0:
            top_wets = wet_events.nlargest(5, 'magnitude')
            for idx, event in top_wets.iterrows():
                peak_idx = int(event['peak_idx'])
                if peak_idx < len(times):
                    peak_time = times[peak_idx]
                    peak_val = values[peak_idx]
                    ax1.annotate(f"{peak_time.year}",
                               xy=(peak_time, peak_val),
                               xytext=(0, 20), textcoords='offset points',
                               fontsize=9, ha='center', color='darkblue',
                               arrowprops=dict(arrowstyle='->', color='darkblue', lw=0.5))

        ax1.set_ylabel(name.upper(), fontsize=12)
        ax1.set_title(f'{name.upper()} Historical Extreme Events - Bali (1950-2025)\n'
                     f'Identified {len(drought_events)} drought and {len(wet_events)} wet events',
                     fontsize=14, fontweight='bold')
        # Headroom for the event-year callouts. They sit 20 points off the
        # peak, and the index itself reaches the +/-3.09 clip, so a +/-3.5
        # limit pushed the top labels into the title and behind the legend.
        # The band above +3.5 is reserved for labels, and the legend above it.
        ax1.set_ylim(-4.3, 5.4)
        ax1.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax1.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Event statistics (using datetime x-axis to align with top panel)
        ax2 = axes[1]

        if len(drought_events) > 0:
            # Create bar chart using datetime positions to align with top panel
            bar_width = np.timedelta64(180, 'D')  # ~6 months width for bars

            drought_times = []
            drought_mags = []
            for idx, event in drought_events.iterrows():
                start_idx = int(event['start_idx'])
                if start_idx < len(times):
                    drought_times.append(times[start_idx])
                    drought_mags.append(event['magnitude'])

            if drought_times:
                ax2.bar(drought_times, drought_mags, color='#b2182b', alpha=0.7,
                       width=bar_width, label='Drought magnitude')

            # Add wet events
            if len(wet_events) > 0:
                wet_times = []
                wet_mags = []
                for idx, event in wet_events.iterrows():
                    start_idx = int(event['start_idx'])
                    if start_idx < len(times):
                        wet_times.append(times[start_idx])
                        wet_mags.append(event['magnitude'])
                if wet_times:
                    ax2.bar(wet_times, wet_mags, color='#2166ac', alpha=0.7,
                           width=bar_width, label='Wet magnitude')

            # Annotate top 5 drought events on bottom panel
            top_droughts = drought_events.nlargest(5, 'magnitude')
            for idx, event in top_droughts.iterrows():
                start_idx = int(event['start_idx'])
                if start_idx < len(times):
                    event_time = times[start_idx]
                    event_mag = event['magnitude']
                    ax2.annotate(f"{event_time.year}",
                               xy=(event_time, event_mag),
                               xytext=(0, 12), textcoords='offset points',
                               fontsize=9, ha='center', color='darkred', fontweight='bold',
                               arrowprops=dict(arrowstyle='->', color='darkred', lw=0.5))

            # Annotate top 5 wet events on bottom panel
            if len(wet_events) > 0:
                top_wets = wet_events.nlargest(5, 'magnitude')
                for idx, event in top_wets.iterrows():
                    start_idx = int(event['start_idx'])
                    if start_idx < len(times):
                        event_time = times[start_idx]
                        event_mag = event['magnitude']
                        ax2.annotate(f"{event_time.year}",
                                   xy=(event_time, event_mag),
                                   xytext=(0, 12), textcoords='offset points',
                                   fontsize=9, ha='center', color='darkblue', fontweight='bold',
                                   arrowprops=dict(arrowstyle='->', color='darkblue', lw=0.5))

        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Event Magnitude', fontsize=12)
        ax2.set_title('Event Magnitude by Year', fontsize=12)
        # Same reasoning as the panel above: leave room for the bar callouts
        # and the legend instead of letting them sit on the top spine.
        _top = ax2.get_ylim()[1]
        ax2.set_ylim(0, _top * 1.28)
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = OUTPUT_PLOTS / f'{name}_historical_events.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        print_ok(f"Saved: {filepath.name}")


def create_multiscale_comparison(precip):
    """Create multi-scale SPI comparison (1, 3, 6, 12, 24 months)."""
    print_subheader("Creating Multi-Scale SPI Comparison")

    plt = setup_matplotlib()

    scales = [1, 3, 6, 12, 24]
    spis = {}

    for scale in scales:
        print_info(f"Calculating SPI-{scale}...")
        spis[scale] = spi(precip, scale=scale, periodicity='monthly',
                         calibration_start_year=CALIBRATION_START,
                         calibration_end_year=CALIBRATION_END,
                         distribution='gamma')

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, precip.shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, precip.shape[2] - 1)

    # Create figure
    fig, axes = plt.subplots(len(scales), 1, figsize=(16, 3 * len(scales)), sharex=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(scales)))

    for i, (scale, da) in enumerate(spis.items()):
        ax = axes[i]
        ts = da[:, sample_lat, sample_lon]
        times = pd.to_datetime(da.time.values)

        ax.fill_between(times, 0, ts.where(ts >= 0), color='#2166ac', alpha=0.6)
        ax.fill_between(times, 0, ts.where(ts < 0), color='#b2182b', alpha=0.6)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axhline(y=-1, color='orange', linestyle='--', linewidth=0.5, alpha=0.7)

        ax.set_ylabel(f'SPI-{scale}', fontsize=11)
        ax.set_ylim(-3.5, 3.5)
        ax.grid(True, alpha=0.3)

        # Add description
        if scale == 1:
            desc = "Short-term meteorological"
        elif scale == 3:
            desc = "Agricultural drought"
        elif scale == 6:
            desc = "Hydrological onset"
        elif scale == 12:
            desc = "Hydrological drought"
        else:
            desc = "Long-term water resources"

        ax.text(0.02, 0.95, desc, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', style='italic', alpha=0.7)

    axes[0].set_title('Multi-Scale SPI Comparison - Bali (1950-2025)\n'
                     'Shorter scales capture recent anomalies; longer scales reveal persistent conditions',
                     fontsize=14, fontweight='bold')
    axes[-1].set_xlabel('Time', fontsize=12)

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / 'spi_multiscale_comparison.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def create_decadal_trend_analysis(results: dict):
    """Create decadal trend analysis showing how drought patterns changed over time."""
    print_subheader("Creating Decadal Trend Analysis")

    plt = setup_matplotlib()

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, results['spi_12'].shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, results['spi_12'].shape[2] - 1)

    for name, da in results.items():
        ts = da[:, sample_lat, sample_lon]
        times = pd.to_datetime(ts.time.values)
        values = ts.values

        # Define decades
        decades = {
            '1950s': (1950, 1959),
            '1960s': (1960, 1969),
            '1970s': (1970, 1979),
            '1980s': (1980, 1989),
            '1990s': (1990, 1999),
            '2000s': (2000, 2009),
            '2010s': (2010, 2019),
            '2020s': (2020, 2025)
        }

        # Create figure with multiple panels
        fig = plt.figure(figsize=(16, 12))

        # Panel 1: Drought frequency by decade (bar chart)
        ax1 = fig.add_subplot(2, 2, 1)
        drought_freq = []
        severe_freq = []
        decade_labels = []

        for decade, (start, end) in decades.items():
            mask = (times.year >= start) & (times.year <= end)
            decade_values = values[mask]
            valid = decade_values[~np.isnan(decade_values)]

            if len(valid) > 0:
                drought_pct = (valid < -1).sum() / len(valid) * 100
                severe_pct = (valid < -1.5).sum() / len(valid) * 100
            else:
                drought_pct = 0
                severe_pct = 0

            decade_labels.append(decade)
            drought_freq.append(drought_pct)
            severe_freq.append(severe_pct)

        x = np.arange(len(decade_labels))
        width = 0.35

        ax1.bar(x - width/2, drought_freq, width, label='Moderate (< -1.0)', color='#fdae61')
        ax1.bar(x + width/2, severe_freq, width, label='Severe (< -1.5)', color='#d7191c')
        ax1.set_xticks(x)
        ax1.set_xticklabels(decade_labels)
        ax1.set_ylabel('Frequency (%)', fontsize=11)
        ax1.set_title('Drought Frequency by Decade', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Panel 2: Mean index by decade
        ax2 = fig.add_subplot(2, 2, 2)
        decade_means = []
        decade_stds = []

        for decade, (start, end) in decades.items():
            mask = (times.year >= start) & (times.year <= end)
            decade_values = values[mask]
            valid = decade_values[~np.isnan(decade_values)]

            if len(valid) > 0:
                decade_means.append(np.mean(valid))
                decade_stds.append(np.std(valid))
            else:
                decade_means.append(0)
                decade_stds.append(0)

        colors = ['#b2182b' if m < 0 else '#2166ac' for m in decade_means]
        ax2.bar(decade_labels, decade_means, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='k', linewidth=1)
        ax2.set_ylabel(f'Mean {name.upper()}', fontsize=11)
        ax2.set_title('Mean Index by Decade', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Panel 3: Box plots by decade
        ax3 = fig.add_subplot(2, 2, 3)
        decade_data = []
        for decade, (start, end) in decades.items():
            mask = (times.year >= start) & (times.year <= end)
            decade_values = values[mask]
            valid = decade_values[~np.isnan(decade_values)]
            decade_data.append(valid)

        bp = ax3.boxplot(decade_data, tick_labels=decade_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#2166ac')
            patch.set_alpha(0.5)
        ax3.axhline(y=-1, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax3.axhline(y=0, color='k', linewidth=0.5)
        ax3.set_ylabel(name.upper(), fontsize=11)
        ax3.set_title('Distribution by Decade', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Panel 4: Running mean trend
        ax4 = fig.add_subplot(2, 2, 4)

        # 10-year running mean
        df = pd.DataFrame({'value': values}, index=times)
        running_mean = df['value'].rolling(window=120, min_periods=60).mean()

        ax4.plot(times, running_mean, color='darkblue', linewidth=2, label='10-year running mean')
        ax4.fill_between(times, 0, running_mean.where(running_mean >= 0), color='#2166ac', alpha=0.3)
        ax4.fill_between(times, 0, running_mean.where(running_mean < 0), color='#b2182b', alpha=0.3)
        ax4.axhline(y=0, color='k', linewidth=0.5)

        # Linear trend
        valid_idx = ~np.isnan(running_mean.values)
        if valid_idx.sum() > 10:
            x_numeric = np.arange(len(running_mean))[valid_idx]
            y_values = running_mean.values[valid_idx]
            z = np.polyfit(x_numeric, y_values, 1)
            p = np.poly1d(z)
            ax4.plot(times[valid_idx], p(x_numeric), 'r--', linewidth=2,
                    label=f'Trend: {z[0]*1200:.3f}/decade')

        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel(f'{name.upper()} (10-yr mean)', fontsize=11)
        ax4.set_title('Long-term Trend', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{name.upper()} Decadal Trend Analysis - Bali', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        filepath = OUTPUT_PLOTS / f'{name}_decadal_trends.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print_ok(f"Saved: {filepath.name}")


def create_exceedance_probability_plot(results: dict):
    """Create exceedance probability plots for different drought thresholds."""
    print_subheader("Creating Exceedance Probability Plot")

    plt = setup_matplotlib()

    # Get sample location
    sample_lat = min(SAMPLE_LAT_IDX, results['spi_12'].shape[1] - 1)
    sample_lon = min(SAMPLE_LON_IDX, results['spi_12'].shape[2] - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (name, da) in enumerate(results.items()):
        ax = axes[ax_idx]
        ts = da[:, sample_lat, sample_lon]
        values = ts.values[~np.isnan(ts.values)]

        # Sort values
        sorted_vals = np.sort(values)
        n = len(sorted_vals)

        # Calculate exceedance probability (Weibull plotting position)
        exceedance_prob = (np.arange(1, n + 1)) / (n + 1) * 100

        # Plot
        ax.plot(sorted_vals, exceedance_prob, color='darkblue', linewidth=2)

        # Mark key thresholds
        thresholds = [
            (-2.0, 'Extreme Drought', '#760005'),
            (-1.5, 'Severe Drought', '#ec0013'),
            (-1.0, 'Moderate Drought', '#ffa938'),
            (0, 'Normal', 'gray'),
            (1.0, 'Moderately Wet', '#00b44a'),
            (1.5, 'Severely Wet', '#008180'),
            (2.0, 'Extremely Wet', '#2a23eb')
        ]

        for thresh, label, color in thresholds:
            # Find exceedance probability for this threshold
            prob = (sorted_vals <= thresh).sum() / n * 100
            ax.axvline(x=thresh, color=color, linestyle='--', alpha=0.7, linewidth=1)
            ax.annotate(f'{label}\n({prob:.1f}%)',
                       xy=(thresh, 50), fontsize=8, ha='center',
                       color=color, alpha=0.8)

        ax.set_xlabel(name.upper(), fontsize=12)
        ax.set_ylabel('Cumulative Probability (%)', fontsize=12)
        ax.set_title(f'{name.upper()} Exceedance Probability - Bali', fontsize=12, fontweight='bold')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        stats_text = f'n = {n}\nmean = {np.mean(values):.2f}\nstd = {np.std(values):.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    filepath = OUTPUT_PLOTS / 'exceedance_probability.png'
    plt.savefig(filepath, dpi=150)
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def create_climate_stripes(results: dict):
    """Create climate stripes visualization (warming stripes style for drought)."""
    print_subheader("Creating Climate Stripes Visualization")

    plt = setup_matplotlib()

    for name, da in results.items():
        # Get spatial mean for each timestep
        spatial_mean = da.mean(dim=['lat', 'lon'])
        times = pd.to_datetime(spatial_mean.time.values)
        values = spatial_mean.values

        # Group by year
        df = pd.DataFrame({'value': values, 'year': times.year})
        annual_mean = df.groupby('year')['value'].mean()

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 4))

        # Create stripes
        years = annual_mean.index.values
        vals = annual_mean.values

        # Normalize to colormap
        for i, (year, val) in enumerate(zip(years, vals)):
            if np.isnan(val):
                color = 'white'
            else:
                # Map value to color
                norm_val = (val + 3) / 6  # Normalize from [-3, 3] to [0, 1]
                norm_val = np.clip(norm_val, 0, 1)
                color = plt.cm.RdYlBu(norm_val)

            ax.axvspan(i - 0.5, i + 0.5, color=color, linewidth=0)

        # Clean up axes
        ax.set_xlim(-0.5, len(years) - 0.5)
        ax.set_ylim(0, 1)

        # Add year labels
        ax.set_xticks(range(0, len(years), 10))
        ax.set_xticklabels(years[::10], fontsize=10)
        ax.set_yticks([])

        ax.set_title(f'{name.upper()} Climate Stripes - Bali (1950-2025)\n'
                    'Blue = Wet Years, Red = Dry Years', fontsize=14, fontweight='bold')

        plt.tight_layout()
        filepath = OUTPUT_PLOTS / f'{name}_climate_stripes.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        print_ok(f"Saved: {filepath.name}")


def main():
    """Run all visualization tests."""
    print_header("TEST 06: VISUALIZATION")

    setup_output_dirs()

    # Load data
    print_subheader("Loading Input Data")

    status = check_input_files()

    try:
        precip = load_data('precip')
        print_ok(f"Precipitation: {precip.shape}")
    except Exception as e:
        print_fail(f"Could not load precipitation: {e}")
        return False

    tmean = load_data('tmean') if status['tmean']['exists'] else None
    tmin = load_data('tmin') if status['tmin']['exists'] else None
    tmax = load_data('tmax') if status['tmax']['exists'] else None
    lat = precip.lat

    if tmean is not None:
        print_ok(f"Temperature (mean): {tmean.shape}")
    if tmin is not None:
        print_ok(f"Temperature (min): {tmin.shape}")
    if tmax is not None:
        print_ok(f"Temperature (max): {tmax.shape}")

    # Load or calculate indices
    results = load_or_calculate_indices(precip, tmean, tmin, tmax, lat)

    # ===== BASIC VISUALIZATIONS =====
    create_wmo_timeseries_plots(results)
    create_distribution_comparison_plot(precip)
    create_spatial_maps(results)
    create_summary_figure(results, precip, tmean)

    if tmean is not None:
        create_pet_method_summary(precip, tmean, tmin, tmax, lat)

    # ===== ADVANCED VISUALIZATIONS =====
    # Seasonal drought heatmap - visually striking month x year pattern
    create_seasonal_drought_heatmap(results)

    # Historical events with run theory - demonstrates package capability
    create_historical_events_plot(results)

    # Multi-scale comparison - educational for users
    create_multiscale_comparison(precip)

    # Decadal trend analysis - scientifically interesting
    create_decadal_trend_analysis(results)

    # Exceedance probability - useful for risk assessment
    create_exceedance_probability_plot(results)

    # Climate stripes - modern, visually compelling
    create_climate_stripes(results)

    # Summary
    print_header("TEST 06 COMPLETE")
    print_ok(f"All visualizations saved to: {OUTPUT_PLOTS}")

    # List created files
    plot_files = list(OUTPUT_PLOTS.glob('*.png'))
    print_info(f"Created {len(plot_files)} plot files:")
    for f in sorted(plot_files):
        print(f"    - {f.name}")

    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
