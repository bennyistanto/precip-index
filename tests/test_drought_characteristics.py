"""
Test Run Theory Functions - Minimal Functionality Test

This script tests run theory functions with minimal outputs.
Only verifies that functions execute correctly without errors.

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

# Output directory
OUTPUT_DIR = 'test_output'

print("="*70)
print("RUN THEORY FUNCTIONS TEST")
print("="*70)
print()
print("Testing run theory functions with TerraClimate Bali data")
print("This is a minimal test - only verifies functions work correctly")
print()

# ============================================================================
# STEP 1: Load Test Data
# ============================================================================
print("1. Loading SPI-12 data...")

try:
    # Load SPI-12 from test output
    spi_path = os.path.join(OUTPUT_DIR, 'spi_multi_bali_test.nc')
    if os.path.exists(spi_path):
        ds = xr.open_dataset(spi_path)
        spi_data = ds['spi_gamma_12_month']
        print(f"   [OK] Loaded from test output")
    else:
        print(f"   [ERROR] SPI data not found at {spi_path}. Run test_spi.py first.")
        sys.exit(1)

    print(f"   Shape: {spi_data.shape}")
    print(f"   Time range: {spi_data.time[0].values} to {spi_data.time[-1].values}")
    print()

except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# Import functions
from runtheory import (
    identify_events,
    calculate_timeseries,
    summarize_events,
    get_event_state,
    calculate_period_statistics
)

# Extract single location for testing
spi_loc = spi_data.isel(lat=5, lon=5)

# ============================================================================
# STEP 2: Test Event Identification
# ============================================================================
print("2. Testing identify_events()...")

# Test with dry threshold
dry_events = identify_events(spi_loc, threshold=-1.2, min_duration=2)
print(f"   [OK] Dry events: Found {len(dry_events)} events")

# Test with wet threshold
wet_events = identify_events(spi_loc, threshold=+1.2, min_duration=2)
print(f"   [OK] Wet events: Found {len(wet_events)} events")
print()

# ============================================================================
# STEP 3: Test Time-Series Calculation
# ============================================================================
print("3. Testing calculate_timeseries()...")

dry_ts = calculate_timeseries(spi_loc, threshold=-1.2)
print(f"   [OK] Dry time-series: {len(dry_ts)} months")

wet_ts = calculate_timeseries(spi_loc, threshold=+1.2)
print(f"   [OK] Wet time-series: {len(wet_ts)} months")
print()

# ============================================================================
# STEP 4: Test Event Summary
# ============================================================================
print("4. Testing summarize_events()...")

if len(dry_events) > 0:
    dry_summary = summarize_events(dry_events)
    print(f"   [OK] Dry summary: {dry_summary['num_events']} events")

if len(wet_events) > 0:
    wet_summary = summarize_events(wet_events)
    print(f"   [OK] Wet summary: {wet_summary['num_events']} events")
print()

# ============================================================================
# STEP 5: Test Event State Detection
# ============================================================================
print("5. Testing get_event_state()...")

is_event, category, deviation = get_event_state(-1.5, threshold=-1.0)
print(f"   [OK] Dry state: is_event={is_event}, category='{category}'")

is_event, category, deviation = get_event_state(+1.5, threshold=+1.0)
print(f"   [OK] Wet state: is_event={is_event}, category='{category}'")
print()

# ============================================================================
# STEP 6: Test Period Statistics (minimal - single period)
# ============================================================================
print("6. Testing calculate_period_statistics()...")

# Test with a short recent period only
dry_stats = calculate_period_statistics(
    spi_data, threshold=-1.2, start_year=2020, end_year=2024, min_duration=2
)
print(f"   [OK] Dry statistics calculated")
print(f"   Variables: {list(dry_stats.data_vars)}")

wet_stats = calculate_period_statistics(
    spi_data, threshold=+1.2, start_year=2020, end_year=2024, min_duration=2
)
print(f"   [OK] Wet statistics calculated")
print()

# ============================================================================
# STEP 7: Summary
# ============================================================================
print("="*70)
print("TEST SUMMARY")
print("="*70)
print()
print("[OK] ALL TESTS PASSED!")
print()
print("Functions tested:")
print("  [OK] identify_events() - dry and wet")
print("  [OK] calculate_timeseries() - dry and wet")
print("  [OK] summarize_events() - dry and wet")
print("  [OK] get_event_state() - dry and wet")
print("  [OK] calculate_period_statistics() - dry and wet")
print()
print("No output files created - this is a minimal functionality test")
print()
print("="*70)
print()
