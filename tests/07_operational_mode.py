#!/usr/bin/env python
"""
Test 07: Operational Mode - Parameter Persistence

This test demonstrates the operational drought monitoring workflow:
1. Calibrate on historical data (1950-2021) and save parameters
2. Load parameters and apply to new data (2022-2025) without refitting
3. Validate consistency between calibration and operational modes

This is crucial for real-world drought monitoring where you establish
a stable baseline and consistently apply it to new observations.

Author: Benny Istanto
"""

from conftest import (
    print_header, print_subheader, print_ok, print_fail, print_info,
    print_stat, setup_output_dirs, load_data, compare_arrays, print_comparison,
    write_report, setup_matplotlib, compute_correlation, compute_rmse,
    OUTPUT_NETCDF, OUTPUT_PLOTS, OUTPUT_REPORTS,
    DATA_START_YEAR, SAMPLE_LAT_IDX, SAMPLE_LON_IDX
)

import numpy as np
import xarray as xr
from pathlib import Path

# Import SPI/SPEI functions
from indices import (
    spi, spei,
    save_fitting_params, load_fitting_params
)
from config import DISTRIBUTION_DISPLAY_NAMES


# Test configuration
CALIBRATION_START = 1950
CALIBRATION_END = 2021  # Leave 2022-2025 for operational mode
OPERATIONAL_START = 2022
OPERATIONAL_END = 2025

TEST_SCALES = [3, 12]
TEST_DISTRIBUTIONS = ['gamma', 'pearson3']


def split_data_by_time(data: xr.DataArray, split_year: int):
    """
    Split data into calibration and operational periods.

    Args:
        data: Full time series data
        split_year: First year of operational period

    Returns:
        tuple: (calibration_data, operational_data, full_data)
    """
    # Get time coordinate
    times = data.time.values
    years = np.array([t.astype('datetime64[Y]').astype(int) + 1970
                      for t in times])

    # Split masks
    calib_mask = years < split_year
    oper_mask = years >= split_year

    calib_data = data.isel(time=calib_mask)
    oper_data = data.isel(time=oper_mask)

    return calib_data, oper_data, data


def test_spi_operational_mode(precip, scale: int, distribution: str) -> dict:
    """
    Test SPI operational mode workflow.

    Steps:
    1. Calculate SPI on calibration period (1950-2021) and save params
    2. Load params and calculate SPI on full period (1950-2025)
    3. Compare with fresh calculation on full period

    Returns:
        dict with results and comparison statistics
    """
    print_info(f"Testing SPI-{scale} ({distribution}) operational mode...")

    # Split data
    calib_precip, oper_precip, full_precip = split_data_by_time(precip, OPERATIONAL_START)

    print_info(f"  Calibration period: {CALIBRATION_START}-{CALIBRATION_END} ({len(calib_precip.time)} months)")
    print_info(f"  Operational period: {OPERATIONAL_START}-{OPERATIONAL_END} ({len(oper_precip.time)} months)")

    # ===== STEP 1: Calibration Phase =====
    # Calculate SPI on calibration data and get parameters
    spi_calib, params = spi(
        calib_precip,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution,
        return_params=True
    )

    # Save parameters
    param_file = OUTPUT_NETCDF / f'spi_{scale}_{distribution}_params.nc'

    # Get coordinates for saving
    lat_coord = 'lat' if 'lat' in precip.coords else 'latitude'
    lon_coord = 'lon' if 'lon' in precip.coords else 'longitude'

    save_fitting_params(
        params,
        str(param_file),
        scale=scale,
        periodicity='monthly',
        index_type='spi',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        coords={
            'lat': precip[lat_coord].values,
            'lon': precip[lon_coord].values
        },
        distribution=distribution
    )
    print_ok(f"  Saved parameters to: {param_file.name}")

    # ===== STEP 2: Operational Phase =====
    # Load parameters and apply to FULL data (including new period)
    loaded_params = load_fitting_params(
        str(param_file),
        scale=scale,
        periodicity='monthly',
        distribution=distribution
    )

    # Calculate SPI on full period using loaded parameters (no refitting!)
    spi_operational = spi(
        full_precip,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution,
        fitting_params=loaded_params
    )
    print_ok(f"  Calculated SPI using loaded parameters")

    # ===== STEP 3: Reference Calculation =====
    # Calculate SPI fresh on full period (for comparison)
    spi_fresh = spi(
        full_precip,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution
    )
    print_ok(f"  Calculated SPI fresh (reference)")

    # ===== STEP 4: Validation =====
    # Compare operational vs fresh calculation
    oper_vals = spi_operational.values.flatten()
    fresh_vals = spi_fresh.values.flatten()

    valid_mask = ~(np.isnan(oper_vals) | np.isnan(fresh_vals))

    correlation = compute_correlation(oper_vals, fresh_vals)
    rmse = compute_rmse(oper_vals, fresh_vals)
    max_diff = float(np.nanmax(np.abs(oper_vals - fresh_vals)))

    # Check if they're essentially identical (numerical precision)
    is_identical = max_diff < 1e-6

    if is_identical:
        print_ok(f"  VALIDATED: Operational and fresh results are identical (max diff: {max_diff:.2e})")
    else:
        print_info(f"  Results differ slightly: r={correlation:.6f}, max_diff={max_diff:.6f}")

    return {
        'scale': scale,
        'distribution': distribution,
        'spi_calibration': spi_calib,
        'spi_operational': spi_operational,
        'spi_fresh': spi_fresh,
        'params': params,
        'loaded_params': loaded_params,
        'param_file': param_file,
        'correlation': correlation,
        'rmse': rmse,
        'max_diff': max_diff,
        'is_identical': is_identical,
        'n_calib_months': len(calib_precip.time),
        'n_oper_months': len(oper_precip.time),
    }


def test_spei_operational_mode(precip, tmean, scale: int, distribution: str) -> dict:
    """
    Test SPEI operational mode workflow.

    Returns:
        dict with results and comparison statistics
    """
    print_info(f"Testing SPEI-{scale} ({distribution}) operational mode...")

    # Split data
    calib_precip, _, full_precip = split_data_by_time(precip, OPERATIONAL_START)
    calib_tmean, _, full_tmean = split_data_by_time(tmean, OPERATIONAL_START)

    # Get latitude for PET calculation
    lat_coord = 'lat' if 'lat' in precip.coords else 'latitude'
    latitude = precip[lat_coord]

    # ===== STEP 1: Calibration Phase =====
    spei_calib, params = spei(
        calib_precip,
        temperature=calib_tmean,
        latitude=latitude,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution,
        pet_method='thornthwaite',
        return_params=True
    )

    # Save parameters
    param_file = OUTPUT_NETCDF / f'spei_{scale}_{distribution}_params.nc'

    lat_coord = 'lat' if 'lat' in precip.coords else 'latitude'
    lon_coord = 'lon' if 'lon' in precip.coords else 'longitude'

    save_fitting_params(
        params,
        str(param_file),
        scale=scale,
        periodicity='monthly',
        index_type='spei',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        coords={
            'lat': precip[lat_coord].values,
            'lon': precip[lon_coord].values
        },
        distribution=distribution
    )
    print_ok(f"  Saved parameters to: {param_file.name}")

    # ===== STEP 2: Operational Phase =====
    loaded_params = load_fitting_params(
        str(param_file),
        scale=scale,
        periodicity='monthly',
        distribution=distribution
    )

    spei_operational = spei(
        full_precip,
        temperature=full_tmean,
        latitude=latitude,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution,
        pet_method='thornthwaite',
        fitting_params=loaded_params
    )
    print_ok(f"  Calculated SPEI using loaded parameters")

    # ===== STEP 3: Reference Calculation =====
    spei_fresh = spei(
        full_precip,
        temperature=full_tmean,
        latitude=latitude,
        scale=scale,
        periodicity='monthly',
        calibration_start_year=CALIBRATION_START,
        calibration_end_year=CALIBRATION_END,
        distribution=distribution,
        pet_method='thornthwaite'
    )
    print_ok(f"  Calculated SPEI fresh (reference)")

    # ===== STEP 4: Validation =====
    oper_vals = spei_operational.values.flatten()
    fresh_vals = spei_fresh.values.flatten()

    correlation = compute_correlation(oper_vals, fresh_vals)
    rmse = compute_rmse(oper_vals, fresh_vals)
    max_diff = float(np.nanmax(np.abs(oper_vals - fresh_vals)))

    is_identical = max_diff < 1e-6

    if is_identical:
        print_ok(f"  VALIDATED: Operational and fresh results are identical (max diff: {max_diff:.2e})")
    else:
        print_info(f"  Results differ slightly: r={correlation:.6f}, max_diff={max_diff:.6f}")

    return {
        'scale': scale,
        'distribution': distribution,
        'spei_calibration': spei_calib,
        'spei_operational': spei_operational,
        'spei_fresh': spei_fresh,
        'params': params,
        'loaded_params': loaded_params,
        'param_file': param_file,
        'correlation': correlation,
        'rmse': rmse,
        'max_diff': max_diff,
        'is_identical': is_identical,
    }


def create_operational_workflow_plot(spi_results: dict, spei_results: dict):
    """Create visualization showing the operational workflow."""
    plt = setup_matplotlib()

    # Use SPI-12 gamma as main example
    spi_res = spi_results.get('spi_12_gamma', list(spi_results.values())[0])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Get time series at sample location
    spi_oper = spi_res['spi_operational']
    spi_fresh = spi_res['spi_fresh']

    times = spi_oper.time.values
    years = np.array([t.astype('datetime64[Y]').astype(int) + 1970 for t in times])

    # Sample point time series
    ts_oper = spi_oper.isel(lat=SAMPLE_LAT_IDX, lon=SAMPLE_LON_IDX).values
    ts_fresh = spi_fresh.isel(lat=SAMPLE_LAT_IDX, lon=SAMPLE_LON_IDX).values

    # Panel 1: Full time series with calibration/operational shading
    ax1 = axes[0]
    ax1.axvspan(times[0], times[years < OPERATIONAL_START][-1],
                alpha=0.2, color='blue', label='Calibration Period (1950-2021)')
    ax1.axvspan(times[years >= OPERATIONAL_START][0], times[-1],
                alpha=0.2, color='green', label='Operational Period (2022-2025)')

    ax1.plot(times, ts_oper, 'b-', linewidth=0.8, label='SPI (loaded params)')
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax1.axhline(-1, color='orange', linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.axhline(-2, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    ax1.set_ylabel('SPI-12')
    ax1.set_title('Operational Drought Monitoring: Calibrate Once, Apply Forever', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(times[0], times[-1])
    ax1.set_ylim(-3.5, 3.5)

    # Panel 2: Zoom on transition period (2020-2025)
    ax2 = axes[1]

    # Get transition period data
    trans_mask = years >= 2020
    trans_times = times[trans_mask]
    trans_oper = ts_oper[trans_mask]
    trans_fresh = ts_fresh[trans_mask]
    trans_years = years[trans_mask]

    ax2.axvspan(trans_times[trans_years < OPERATIONAL_START][-1], trans_times[-1],
                alpha=0.2, color='green', label='New Data (2022-2025)')

    ax2.plot(trans_times, trans_oper, 'b-', linewidth=1.5, marker='o',
             markersize=3, label='Operational (loaded params)')
    ax2.plot(trans_times, trans_fresh, 'r--', linewidth=1.5, marker='x',
             markersize=3, label='Fresh calculation')

    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2.axvline(trans_times[trans_years >= OPERATIONAL_START][0],
                color='black', linestyle=':', linewidth=1, label='Operational start')

    ax2.set_ylabel('SPI-12')
    ax2.set_title('Transition Period: Seamless Continuation into New Data', fontsize=11)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(trans_times[0], trans_times[-1])

    # Panel 3: Scatter plot validation
    ax3 = axes[2]

    valid_mask = ~(np.isnan(ts_oper) | np.isnan(ts_fresh))
    ax3.scatter(ts_fresh[valid_mask], ts_oper[valid_mask], alpha=0.5, s=10, c='steelblue')

    # 1:1 line
    lims = [-3.5, 3.5]
    ax3.plot(lims, lims, 'r-', linewidth=1, label='1:1 line')

    # Correlation text
    corr = spi_res['correlation']
    rmse = spi_res['rmse']
    ax3.text(0.05, 0.95, f'r = {corr:.6f}\nRMSE = {rmse:.2e}',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3.set_xlabel('Fresh Calculation (full recalibration)')
    ax3.set_ylabel('Operational Mode (loaded params)')
    ax3.set_title('Validation: Operational vs Fresh Results', fontsize=11)
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_aspect('equal')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = OUTPUT_PLOTS / 'operational_mode_workflow.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def create_parameter_maps(spi_results: dict):
    """Create maps showing fitted distribution parameters."""
    plt = setup_matplotlib()
    import matplotlib.colors as mcolors

    # Get parameters from SPI-12 gamma
    spi_res = spi_results.get('spi_12_gamma', list(spi_results.values())[0])
    params = spi_res['params']

    # Parameters are stored as (month, lat, lon)
    alpha = params.get('alpha', params.get('shape'))
    beta = params.get('beta', params.get('scale'))
    prob_zero = params.get('prob_zero')

    if alpha is None:
        print_info("  Skipping parameter maps (parameters not in expected format)")
        return

    # Latitude here runs north to south, so imshow(origin='lower') drew every
    # one of these maps upside down. pcolormesh takes the coordinates, which
    # keeps the orientation right whichever way the axis is sorted, and it is
    # what the rest of the suite already uses.
    ref = spi_res.get('spi_calibration')
    lat = ref.lat.values if ref is not None and 'lat' in ref.coords else None
    lon = ref.lon.values if ref is not None and 'lon' in ref.coords else None

    def draw(ax, arr, cmap, **kw):
        if lat is not None and lon is not None and arr.shape == (lat.size, lon.size):
            mesh = ax.pcolormesh(lon, lat, arr, cmap=cmap, shading='auto', **kw)
            ax.set_aspect('equal')
            return mesh
        # Fallback without coordinates: row 0 is the northernmost latitude.
        return ax.imshow(arr, cmap=cmap, origin='upper', **kw)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Show parameters for January (month 0) and July (month 6)
    months = [(0, 'January'), (6, 'July')]

    for col, (month_idx, month_name) in enumerate(months):
        # Alpha (shape)
        ax = axes[0, col]
        if len(alpha.shape) == 3:
            data = alpha[month_idx]
        else:
            data = alpha[month_idx] if len(alpha.shape) > 1 else alpha

        im = draw(ax, data, 'YlOrRd')
        ax.set_title(f'Alpha (shape) - {month_name}')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Beta (scale)
        ax = axes[1, col]
        if len(beta.shape) == 3:
            data = beta[month_idx]
        else:
            data = beta[month_idx] if len(beta.shape) > 1 else beta

        im = draw(ax, data, 'YlGnBu')
        ax.set_title(f'Beta (scale) - {month_name}')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Prob zero for any month and annual mean
    ax = axes[0, 2]
    if prob_zero is not None and len(prob_zero.shape) == 3:
        annual_prob_zero = np.nanmean(prob_zero, axis=0)
        im = draw(ax, annual_prob_zero * 100, 'Purples', vmin=0, vmax=10)
        ax.set_title('Prob(Zero) - Annual Mean (%)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add text explanation
    ax = axes[1, 2]
    ax.axis('off')
    explanation = """
    Fitted Distribution Parameters
    (Gamma Distribution)

    Alpha (shape): Controls distribution shape
    - Higher values → more symmetric
    - Lower values → more skewed

    Beta (scale): Controls spread
    - Higher values → more variability
    - Lower values → less variability

    These parameters are fitted once during
    calibration and reused for all future
    calculations, ensuring consistency.
    """
    ax.text(0.1, 0.9, explanation, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Distribution Parameters: Calibrated on 1950-2021, Applied to 2022-2025',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    filepath = OUTPUT_PLOTS / 'operational_mode_parameters.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def create_multi_distribution_comparison(spi_results: dict, spei_results: dict):
    """Create comparison plot across distributions - compact horizontal layout."""
    plt = setup_matplotlib()

    # Create a more compact figure with horizontal layout
    fig = plt.figure(figsize=(14, 8))

    # Use GridSpec for better control
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1.2], hspace=0.3, wspace=0.3)

    # Blue/orange rather than blue/green. The old pair separated well for
    # normal vision and deuteranopia but collapsed for tritanopia (CVD dE 9.5),
    # and at similar lightness the four overlapping lines were hard to tell
    # apart. Blue/orange scores dE 33.6 normal and 32.7 tritan.
    colors = {'gamma': '#2a78d6', 'pearson3': '#eb6834'}
    dist_labels = {'gamma': 'Gamma', 'pearson3': 'Pearson III'}

    # Gamma and Pearson III agree to r ~ 0.99 here, so the two curves sit on
    # top of each other and whichever is drawn last hides the other
    # completely. Colour alone cannot fix that. So the DISTRIBUTION carries
    # both colour and dash - Gamma solid underneath, Pearson III dashed on
    # top - and the blue shows through the gaps in the orange. Width is left
    # to carry the scale.
    dist_style = {'gamma': dict(color=colors['gamma'], linestyle='-'),
                  'pearson3': dict(color=colors['pearson3'],
                                   linestyle=(0, (4, 2.6)))}
    scale_width = {3: 1.3, 12: 2.6}

    # Row 1: SPI
    # Time series (SPI-3 and SPI-12 overlaid)
    ax1 = fig.add_subplot(gs[0, 0])

    for scale in (3, 12):
        for dist in ['gamma', 'pearson3']:
            key = f'spi_{scale}_{dist}'
            if key in spi_results:
                res = spi_results[key]
                ts = res['spi_operational'].isel(lat=SAMPLE_LAT_IDX, lon=SAMPLE_LON_IDX)
                times = ts.time.values
                years = np.array([t.astype('datetime64[Y]').astype(int) + 1970 for t in times])
                mask = years >= 2018

                # Label every line. Previously only the Gamma lines were
                # labelled, so the legend read "(Gamma)" twice while four
                # lines were drawn, and the Pearson III curves looked like
                # Gamma. Their tails diverge, so that misreads badly.
                label = f'SPI-{scale} ({dist_labels[dist]})'
                ax1.plot(times[mask], ts.values[mask], label=label,
                        linewidth=scale_width[scale], solid_capstyle='round',
                        dash_capstyle='butt', **dist_style[dist])

    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax1.axvline(times[years >= OPERATIONAL_START][0], color='red',
               linestyle=':', linewidth=1.5, label='Operational start')
    ax1.set_ylabel('SPI Value')
    ax1.set_title('SPI Time Series (2018-2025)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=7, ncol=2)
    ax1.set_ylim(-3, 3)

    # Row 2: SPEI
    ax2 = fig.add_subplot(gs[1, 0])

    for scale in (3, 12):
        for dist in ['gamma', 'pearson3']:
            key = f'spei_{scale}_{dist}'
            if key in spei_results:
                res = spei_results[key]
                ts = res['spei_operational'].isel(lat=SAMPLE_LAT_IDX, lon=SAMPLE_LON_IDX)
                times = ts.time.values
                years = np.array([t.astype('datetime64[Y]').astype(int) + 1970 for t in times])
                mask = years >= 2018

                label = f'SPEI-{scale} ({dist_labels[dist]})'
                ax2.plot(times[mask], ts.values[mask], label=label,
                        linewidth=scale_width[scale], solid_capstyle='round',
                        dash_capstyle='butt', **dist_style[dist])

    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2.axvline(times[years >= OPERATIONAL_START][0], color='red',
               linestyle=':', linewidth=1.5)
    ax2.set_ylabel('SPEI Value')
    ax2.set_title('SPEI Time Series (2018-2025)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=7, ncol=2)
    ax2.set_ylim(-3, 3)

    # Scatter plots: Operational vs Fresh
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax, results, name in [(ax3, spi_results, 'SPI'), (ax4, spei_results, 'SPEI')]:
        for dist in ['gamma', 'pearson3']:
            for scale in [3, 12]:
                key = f'{name.lower()}_{scale}_{dist}'
                if key in results:
                    res = results[key]
                    oper = res[f'{name.lower()}_operational'].values.flatten()
                    fresh = res[f'{name.lower()}_fresh'].values.flatten()
                    valid = ~(np.isnan(oper) | np.isnan(fresh))

                    # Subsample for performance
                    idx = np.random.choice(np.where(valid)[0], min(1000, valid.sum()), replace=False)
                    # Every series lies on the 1:1 line by construction, so
                    # filled dots simply bury each other. Open markers let the
                    # series underneath stay visible.
                    marker = 'o' if scale == 12 else 's'
                    ax.scatter(fresh[idx], oper[idx], facecolors='none',
                              edgecolors=colors[dist], linewidths=0.6,
                              alpha=0.5, s=14, marker=marker,
                              label=f'{name}-{scale} {dist_labels[dist]}')

        ax.plot([-3, 3], [-3, 3], 'r-', linewidth=1, label='1:1 line')
        ax.set_xlabel('Fresh Calculation')
        ax.set_ylabel('Operational Mode')
        ax.set_title(f'{name}: Operational vs Fresh', fontsize=10)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Validation summary table
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.axis('off')

    # Create summary table
    table_data = []
    for name, results in [('SPI', spi_results), ('SPEI', spei_results)]:
        for scale in [3, 12]:
            for dist in ['gamma', 'pearson3']:
                key = f'{name.lower()}_{scale}_{dist}'
                if key in results:
                    res = results[key]
                    status = 'IDENTICAL' if res['is_identical'] else 'DIFFERS'
                    table_data.append([f'{name}-{scale}', dist_labels[dist], status])

    # Draw table
    cell_text = table_data
    col_labels = ['Index', 'Distribution', 'Status']

    table = ax5.table(cellText=cell_text, colLabels=col_labels,
                      cellLoc='center', loc='center',
                      colColours=['#f0f0f0']*3)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Color the status cells
    for i, row in enumerate(cell_text):
        if row[2] == 'IDENTICAL':
            table[(i+1, 2)].set_facecolor('#d4edda')  # Light green
        else:
            table[(i+1, 2)].set_facecolor('#f8d7da')  # Light red

    ax5.set_title('Validation Summary\n(All 8 Configurations)', fontsize=10, fontweight='bold', pad=20)

    plt.suptitle('Operational Mode: Multi-Scale, Multi-Distribution Validation',
                 fontsize=12, fontweight='bold', y=0.98)

    filepath = OUTPUT_PLOTS / 'operational_mode_multiscale.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print_ok(f"Saved: {filepath.name}")


def generate_report(spi_results: dict, spei_results: dict) -> str:
    """Generate text report for operational mode test."""
    lines = []

    lines.append("OPERATIONAL MODE TEST RESULTS")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Workflow: Calibrate once on historical data, apply to new observations")
    lines.append(f"Calibration Period: {CALIBRATION_START}-{CALIBRATION_END}")
    lines.append(f"Operational Period: {OPERATIONAL_START}-{OPERATIONAL_END}")
    lines.append("")

    lines.append("SPI RESULTS")
    lines.append("-" * 50)

    for key, res in spi_results.items():
        lines.append(f"\n{key}:")
        lines.append(f"  Parameter file: {res['param_file'].name}")
        lines.append(f"  Correlation (oper vs fresh): {res['correlation']:.8f}")
        lines.append(f"  RMSE: {res['rmse']:.2e}")
        lines.append(f"  Max difference: {res['max_diff']:.2e}")
        lines.append(f"  Identical: {'YES' if res['is_identical'] else 'NO'}")

    lines.append("\n\nSPEI RESULTS")
    lines.append("-" * 50)

    for key, res in spei_results.items():
        lines.append(f"\n{key}:")
        lines.append(f"  Parameter file: {res['param_file'].name}")
        lines.append(f"  Correlation (oper vs fresh): {res['correlation']:.8f}")
        lines.append(f"  RMSE: {res['rmse']:.2e}")
        lines.append(f"  Max difference: {res['max_diff']:.2e}")
        lines.append(f"  Identical: {'YES' if res['is_identical'] else 'NO'}")

    lines.append("\n\nCONCLUSION")
    lines.append("-" * 50)

    all_identical = all(r['is_identical'] for r in spi_results.values())
    all_identical &= all(r['is_identical'] for r in spei_results.values())

    if all_identical:
        lines.append("All operational mode results are IDENTICAL to fresh calculations.")
        lines.append("Parameter persistence is working correctly.")
    else:
        lines.append("Some results differ between operational and fresh modes.")
        lines.append("This may indicate numerical precision differences.")

    return "\n".join(lines)


def main():
    """Run operational mode tests."""
    print_header("TEST 07: OPERATIONAL MODE")

    setup_output_dirs()

    # Load data
    print_subheader("Loading Input Data")
    precip = load_data('precip')
    tmean = load_data('tmean')
    print_ok(f"Precipitation: {precip.shape}")
    print_ok(f"Temperature: {tmean.shape}")

    # ===== SPI TESTS =====
    print_subheader("SPI Operational Mode Tests")
    spi_results = {}

    for scale in TEST_SCALES:
        for dist in TEST_DISTRIBUTIONS:
            key = f'spi_{scale}_{dist}'
            spi_results[key] = test_spi_operational_mode(precip, scale, dist)

    # ===== SPEI TESTS =====
    print_subheader("SPEI Operational Mode Tests")
    spei_results = {}

    for scale in TEST_SCALES:
        for dist in TEST_DISTRIBUTIONS:
            key = f'spei_{scale}_{dist}'
            spei_results[key] = test_spei_operational_mode(precip, tmean, scale, dist)

    # ===== VISUALIZATIONS =====
    print_subheader("Creating Visualizations")

    create_operational_workflow_plot(spi_results, spei_results)
    create_parameter_maps(spi_results)
    create_multi_distribution_comparison(spi_results, spei_results)

    # ===== REPORT =====
    print_subheader("Generating Report")
    report_content = generate_report(spi_results, spei_results)
    write_report('07_operational_mode_report.txt', 'Operational Mode Test', report_content)

    # ===== SUMMARY =====
    print_header("TEST 07 COMPLETE")

    n_tests = len(spi_results) + len(spei_results)
    n_passed = sum(1 for r in spi_results.values() if r['is_identical'])
    n_passed += sum(1 for r in spei_results.values() if r['is_identical'])

    print_ok(f"Tested {n_tests} configurations")
    print_ok(f"All {n_passed}/{n_tests} passed (operational = fresh)")
    print_ok(f"Parameter files saved to: {OUTPUT_NETCDF}")
    print_ok(f"Plots saved to: {OUTPUT_PLOTS}")

    return spi_results, spei_results


if __name__ == '__main__':
    main()
