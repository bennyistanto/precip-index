#!/usr/bin/env python
"""
Run all integration tests for precip-index package.

This script executes all numbered test files in sequence and provides
a comprehensive summary of results.

Tests use TerraClimate Bali data (1950-2025) from input/ folder.

Author: Benny Istanto
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).parent.parent


def print_header(title: str, width: int = 70):
    """Print formatted header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_ok(msg: str):
    """Print success message."""
    print(f"  [OK] {msg}")


def print_fail(msg: str):
    """Print failure message."""
    print(f"  [FAIL] {msg}")


def print_info(msg: str):
    """Print info message."""
    print(f"  [INFO] {msg}")


def check_input_data():
    """Check that required input data exists."""
    print_header("CHECKING INPUT DATA")

    input_dir = REPO_ROOT / 'input'
    required_files = [
        ('terraclimate_bali_ppt_1950_2025.nc', 'Precipitation (required)'),
        ('terraclimate_bali_tmean_1950_2025.nc', 'Temperature mean (required)'),
        ('terraclimate_bali_pet_1950_2025.nc', 'PET (optional)'),
    ]
    optional_files = [
        ('terraclimate_bali_tmin_1950_2025.nc', 'Temperature min (for Hargreaves)'),
        ('terraclimate_bali_tmax_1950_2025.nc', 'Temperature max (for Hargreaves)'),
    ]

    all_required_exist = True

    for filename, description in required_files:
        filepath = input_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print_ok(f"{description}: {filename} ({size_mb:.1f} MB)")
        else:
            print_fail(f"{description}: {filename} NOT FOUND")
            all_required_exist = False

    for filename, description in optional_files:
        filepath = input_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print_ok(f"{description}: {filename} ({size_mb:.1f} MB)")
        else:
            print_info(f"{description}: {filename} (not found, will skip Hargreaves)")

    return all_required_exist


def run_test(test_file: Path) -> tuple:
    """
    Run a single test file.

    Returns:
        (success: bool, elapsed_time: float)
    """
    print(f"\n{'='*70}")
    print(f"Running: {test_file.name}")
    print("="*70)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(REPO_ROOT),
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[OK] {test_file.name} PASSED ({elapsed:.1f}s)")
            return True, elapsed
        else:
            print(f"\n[FAIL] {test_file.name} FAILED ({elapsed:.1f}s)")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] {test_file.name}: {e}")
        return False, elapsed


def main():
    """Run all tests."""
    print_header("PRECIP-INDEX TEST SUITE", width=70)
    print("Running integration tests with TerraClimate Bali data (1950-2025)")
    print("Source: input/ folder")

    # Check input data
    if not check_input_data():
        print("\n[FAIL] Required input data missing. Cannot proceed.")
        print("Please ensure TerraClimate Bali data files are in the input/ folder.")
        return 1

    # Find all numbered test files
    tests_dir = Path(__file__).parent
    test_files = sorted(tests_dir.glob('[0-9][0-9]_*.py'))

    if not test_files:
        print("\n[FAIL] No test files found!")
        return 1

    print_header("RUNNING TESTS")
    print(f"Found {len(test_files)} test files:")
    for tf in test_files:
        print(f"  - {tf.name}")

    # Run each test
    results = {}
    total_start = time.time()

    for test_file in test_files:
        success, elapsed = run_test(test_file)
        results[test_file.name] = {'success': success, 'time': elapsed}

    total_elapsed = time.time() - total_start

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for r in results.values() if r['success'])
    failed = len(results) - passed

    for name, result in results.items():
        status = "[OK]  " if result['success'] else "[FAIL]"
        print(f"  {status} {name} ({result['time']:.1f}s)")

    print("-" * 70)
    print(f"  Total: {passed}/{len(results)} tests passed")
    print(f"  Elapsed time: {total_elapsed:.1f}s")
    print("=" * 70)

    # Output summary
    output_dir = tests_dir / 'output'
    if output_dir.exists():
        netcdf_count = len(list((output_dir / 'netcdf').glob('*.nc'))) if (output_dir / 'netcdf').exists() else 0
        plots_count = len(list((output_dir / 'plots').glob('*.png'))) if (output_dir / 'plots').exists() else 0
        reports_count = len(list((output_dir / 'reports').glob('*.txt'))) if (output_dir / 'reports').exists() else 0

        print(f"\nOutputs created:")
        print(f"  - NetCDF files: {netcdf_count}")
        print(f"  - Plot files: {plots_count}")
        print(f"  - Report files: {reports_count}")
        print(f"  - Location: {output_dir}")

    # Exit status
    if failed > 0:
        print(f"\n[FAIL] {failed} test(s) failed")
        return 1
    else:
        print("\n[OK] All tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
