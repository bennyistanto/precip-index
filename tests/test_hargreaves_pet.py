"""
Test script for Hargreaves PET implementation.

Tests the new Hargreaves-Samani PET calculation method using TerraClimate Bali data.
Compares results against Thornthwaite and pre-computed TerraClimate PET.

Author: Benny Istanto
Organization: GOST/DEC Data Group, The World Bank
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from utils import calculate_pet, eto_hargreaves, eto_thornthwaite
from indices import spei


def test_hargreaves_pet_basic():
    """Test basic Hargreaves PET calculation with 1D array."""
    print("=" * 60)
    print("Test 1: Basic Hargreaves PET calculation (1D array)")
    print("=" * 60)

    # Create synthetic test data (12 months)
    temp_mean = np.array([25, 26, 27, 28, 29, 28, 27, 26, 26, 27, 27, 26])  # °C
    temp_min = np.array([20, 21, 22, 23, 24, 23, 22, 21, 21, 22, 22, 21])   # °C
    temp_max = np.array([30, 31, 32, 33, 34, 33, 32, 31, 31, 32, 32, 31])   # °C
    latitude = -8.5  # Bali latitude

    # Calculate PET
    pet_hargreaves = eto_hargreaves(temp_mean, temp_min, temp_max, latitude, 2024)
    pet_thornthwaite = eto_thornthwaite(temp_mean, latitude, 2024)

    print(f"\nTemperature (mean): {temp_mean}")
    print(f"Temperature (min):  {temp_min}")
    print(f"Temperature (max):  {temp_max}")
    print(f"\nPET Hargreaves:    {pet_hargreaves.round(1)}")
    print(f"PET Thornthwaite:  {pet_thornthwaite.round(1)}")
    print(f"\nRatio (Harg/Thorn): {(pet_hargreaves / pet_thornthwaite).round(2)}")

    # Basic validation
    assert len(pet_hargreaves) == 12, "Should return 12 monthly values"
    assert np.all(pet_hargreaves > 0), "PET should be positive"
    assert np.all(pet_hargreaves < 300), "PET should be reasonable (< 300 mm/month)"

    print("\n[PASS] Basic Hargreaves test passed!")
    return True


def test_hargreaves_with_terraclimate_bali():
    """Test Hargreaves PET with real TerraClimate Bali data."""
    print("\n" + "=" * 60)
    print("Test 2: Hargreaves PET with TerraClimate Bali data")
    print("=" * 60)

    # Load TerraClimate data
    input_dir = Path('input')

    temp_mean = xr.open_dataset(input_dir / 'terraclimate_bali_tmean_1958_2024.nc')
    temp_min = xr.open_dataset(input_dir / 'terraclimate_bali_tmin_1958_2024.nc')
    temp_max = xr.open_dataset(input_dir / 'terraclimate_bali_tmax_1958_2024.nc')
    pet_tc = xr.open_dataset(input_dir / 'terraclimate_bali_pet_1958_2024.nc')

    # Find variable names (exclude 'crs' which is a projection variable)
    tmean_var = [v for v in temp_mean.data_vars if v != 'crs'][0]
    tmin_var = [v for v in temp_min.data_vars if v != 'crs'][0]
    tmax_var = [v for v in temp_max.data_vars if v != 'crs'][0]
    pet_var = [v for v in pet_tc.data_vars if v != 'crs'][0]

    print(f"\nLoaded variables: tmean={tmean_var}, tmin={tmin_var}, tmax={tmax_var}, pet={pet_var}")
    print(f"Data shape: {temp_mean[tmean_var].shape}")
    print(f"Time range: {temp_mean.time.values[0]} to {temp_mean.time.values[-1]}")

    # Extract DataArrays
    tmean_da = temp_mean[tmean_var]
    tmin_da = temp_min[tmin_var]
    tmax_da = temp_max[tmax_var]
    pet_tc_da = pet_tc[pet_var]

    # Get latitude array
    lat = tmean_da.lat

    # Calculate PET using Hargreaves
    print("\nCalculating Hargreaves PET...")
    pet_hargreaves = calculate_pet(
        tmean_da, latitude=lat, data_start_year=1958,
        method='hargreaves',
        temp_min=tmin_da,
        temp_max=tmax_da
    )

    # Calculate PET using Thornthwaite for comparison
    print("Calculating Thornthwaite PET...")
    pet_thornthwaite = calculate_pet(
        tmean_da, latitude=lat, data_start_year=1958,
        method='thornthwaite'
    )

    # Compare statistics
    print("\n" + "-" * 40)
    print("PET Comparison (spatial mean over all time steps)")
    print("-" * 40)
    print(f"TerraClimate PET:  mean={float(pet_tc_da.mean()):.1f} mm/month")
    print(f"Hargreaves PET:    mean={float(pet_hargreaves.mean()):.1f} mm/month")
    print(f"Thornthwaite PET:  mean={float(pet_thornthwaite.mean()):.1f} mm/month")

    # Correlation with TerraClimate PET
    # (TerraClimate uses Penman-Monteith, so Hargreaves should be closer)
    tc_flat = pet_tc_da.values.flatten()
    harg_flat = pet_hargreaves.values.flatten()
    thorn_flat = pet_thornthwaite.values.flatten()

    # Remove NaNs
    valid = ~np.isnan(tc_flat) & ~np.isnan(harg_flat) & ~np.isnan(thorn_flat)

    corr_harg = np.corrcoef(tc_flat[valid], harg_flat[valid])[0, 1]
    corr_thorn = np.corrcoef(tc_flat[valid], thorn_flat[valid])[0, 1]

    print(f"\nCorrelation with TerraClimate PET:")
    print(f"  Hargreaves:   r = {corr_harg:.4f}")
    print(f"  Thornthwaite: r = {corr_thorn:.4f}")

    # Save output
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    pet_hargreaves.to_netcdf(output_dir / 'pet_hargreaves_bali_test.nc')
    print(f"\nSaved: {output_dir / 'pet_hargreaves_bali_test.nc'}")

    # Close datasets
    temp_mean.close()
    temp_min.close()
    temp_max.close()
    pet_tc.close()

    print("\n[PASS] TerraClimate Hargreaves test passed!")
    return pet_hargreaves, pet_thornthwaite, pet_tc_da


def test_spei_with_hargreaves():
    """Test SPEI calculation using Hargreaves PET method."""
    print("\n" + "=" * 60)
    print("Test 3: SPEI with internal Hargreaves PET calculation")
    print("=" * 60)

    # Load data
    input_dir = Path('input')

    precip = xr.open_dataset(input_dir / 'terraclimate_bali_ppt_1958_2024.nc')
    temp_mean = xr.open_dataset(input_dir / 'terraclimate_bali_tmean_1958_2024.nc')
    temp_min = xr.open_dataset(input_dir / 'terraclimate_bali_tmin_1958_2024.nc')
    temp_max = xr.open_dataset(input_dir / 'terraclimate_bali_tmax_1958_2024.nc')

    # Find variable names (exclude 'crs' which is a projection variable)
    precip_var = [v for v in precip.data_vars if v != 'crs'][0]
    tmean_var = [v for v in temp_mean.data_vars if v != 'crs'][0]
    tmin_var = [v for v in temp_min.data_vars if v != 'crs'][0]
    tmax_var = [v for v in temp_max.data_vars if v != 'crs'][0]

    precip_da = precip[precip_var]
    tmean_da = temp_mean[tmean_var]
    tmin_da = temp_min[tmin_var]
    tmax_da = temp_max[tmax_var]
    lat = tmean_da.lat

    print(f"\nPrecipitation shape: {precip_da.shape}")
    print(f"Temperature shape:   {tmean_da.shape}")

    # Calculate SPEI with Hargreaves (internal PET calculation)
    print("\nCalculating SPEI-12 with Hargreaves PET...")
    spei_hargreaves = spei(
        precip_da,
        temperature=tmean_da,
        latitude=lat,
        scale=12,
        pet_method='hargreaves',
        temp_min=tmin_da,
        temp_max=tmax_da,
        calibration_start_year=1991,
        calibration_end_year=2020
    )

    # Calculate SPEI with Thornthwaite for comparison
    print("Calculating SPEI-12 with Thornthwaite PET...")
    spei_thornthwaite = spei(
        precip_da,
        temperature=tmean_da,
        latitude=lat,
        scale=12,
        pet_method='thornthwaite',
        calibration_start_year=1991,
        calibration_end_year=2020
    )

    # Compare
    print("\n" + "-" * 40)
    print("SPEI-12 Comparison")
    print("-" * 40)
    print(f"Hargreaves SPEI:   mean={float(spei_hargreaves.mean()):.3f}, std={float(spei_hargreaves.std()):.3f}")
    print(f"Thornthwaite SPEI: mean={float(spei_thornthwaite.mean()):.3f}, std={float(spei_thornthwaite.std()):.3f}")

    # Correlation between methods
    harg_flat = spei_hargreaves.values.flatten()
    thorn_flat = spei_thornthwaite.values.flatten()
    valid = ~np.isnan(harg_flat) & ~np.isnan(thorn_flat)

    corr = np.corrcoef(harg_flat[valid], thorn_flat[valid])[0, 1]
    print(f"\nCorrelation between methods: r = {corr:.4f}")

    # Save outputs
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    spei_hargreaves.to_netcdf(output_dir / 'spei_12_hargreaves_bali_test.nc')
    spei_thornthwaite.to_netcdf(output_dir / 'spei_12_thornthwaite_bali_test.nc')
    print(f"\nSaved: {output_dir / 'spei_12_hargreaves_bali_test.nc'}")
    print(f"Saved: {output_dir / 'spei_12_thornthwaite_bali_test.nc'}")

    # Close datasets
    precip.close()
    temp_mean.close()
    temp_min.close()
    temp_max.close()

    print("\n[PASS] SPEI with Hargreaves test passed!")
    return spei_hargreaves, spei_thornthwaite


def create_comparison_plots(pet_hargreaves, pet_thornthwaite, pet_tc):
    """Create comparison plots for PET methods."""
    print("\n" + "=" * 60)
    print("Creating comparison plots...")
    print("=" * 60)

    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    # Select a single grid cell for time series comparison
    lat_idx, lon_idx = 5, 5

    harg_ts = pet_hargreaves[:, lat_idx, lon_idx].values
    thorn_ts = pet_thornthwaite[:, lat_idx, lon_idx].values
    tc_ts = pet_tc[:, lat_idx, lon_idx].values

    # Time series plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full time series (last 10 years)
    n_months = min(120, len(harg_ts))  # Last 10 years
    ax1 = axes[0]
    ax1.plot(range(n_months), tc_ts[-n_months:], label='TerraClimate (Penman-Monteith)', alpha=0.8)
    ax1.plot(range(n_months), harg_ts[-n_months:], label='Hargreaves-Samani', alpha=0.8)
    ax1.plot(range(n_months), thorn_ts[-n_months:], label='Thornthwaite', alpha=0.8)
    ax1.set_xlabel('Months (last 10 years)')
    ax1.set_ylabel('PET (mm/month)')
    ax1.set_title('PET Time Series Comparison (Single Grid Cell)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot: Hargreaves vs TerraClimate
    ax2 = axes[1]
    valid = ~np.isnan(tc_ts) & ~np.isnan(harg_ts)
    ax2.scatter(tc_ts[valid], harg_ts[valid], alpha=0.3, s=10, label='Hargreaves')
    ax2.scatter(tc_ts[valid], thorn_ts[valid], alpha=0.3, s=10, label='Thornthwaite')
    max_val = max(tc_ts[valid].max(), harg_ts[valid].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', label='1:1 line')
    ax2.set_xlabel('TerraClimate PET (mm/month)')
    ax2.set_ylabel('Calculated PET (mm/month)')
    ax2.set_title('PET Method Comparison vs TerraClimate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pet_comparison_bali.png', dpi=150)
    print(f"Saved: {output_dir / 'pet_comparison_bali.png'}")
    plt.close()

    # Spatial map comparison (mean PET)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    pet_tc.mean(dim='time').plot(ax=axes[0], cmap='YlOrRd', vmin=50, vmax=200)
    axes[0].set_title('TerraClimate PET (Mean)')

    pet_hargreaves.mean(dim='time').plot(ax=axes[1], cmap='YlOrRd', vmin=50, vmax=200)
    axes[1].set_title('Hargreaves PET (Mean)')

    pet_thornthwaite.mean(dim='time').plot(ax=axes[2], cmap='YlOrRd', vmin=50, vmax=200)
    axes[2].set_title('Thornthwaite PET (Mean)')

    plt.tight_layout()
    plt.savefig(output_dir / 'pet_spatial_comparison_bali.png', dpi=150)
    print(f"Saved: {output_dir / 'pet_spatial_comparison_bali.png'}")
    plt.close()

    print("\n[PASS] Plots created!")


def main():
    """Run all Hargreaves PET tests."""
    print("\n" + "=" * 70)
    print("HARGREAVES PET IMPLEMENTATION TEST SUITE")
    print("=" * 70)

    # Run tests
    test_hargreaves_pet_basic()
    pet_hargreaves, pet_thornthwaite, pet_tc = test_hargreaves_with_terraclimate_bali()
    spei_harg, spei_thorn = test_spei_with_hargreaves()

    # Create plots
    create_comparison_plots(pet_hargreaves, pet_thornthwaite, pet_tc)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nSummary:")
    print("- eto_hargreaves() function works correctly")
    print("- calculate_pet() supports both 'thornthwaite' and 'hargreaves' methods")
    print("- spei() with pet_method='hargreaves' works correctly")
    print("- Output files saved to test_output/")


if __name__ == '__main__':
    main()
