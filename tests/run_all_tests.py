#!/usr/bin/env python
"""
Run all integration tests for precip-index package.

This script executes all test files in sequence and provides a summary.

Tests use TerraClimate Bali data (1958-2024) from input/ folder.
"""

import sys
import os
import time
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_test(test_file):
    """Run a single test file and return success status."""
    print("\n" + "="*70)
    print(f"Running: {test_file}")
    print("="*70)

    start_time = time.time()

    try:
        # Run test file
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=os.path.join(os.path.dirname(__file__), '..'),
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n[OK] PASSED ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n[X] FAILED ({elapsed:.1f}s)")
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[X] ERROR ({elapsed:.1f}s): {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PRECIP-INDEX TEST SUITE")
    print("="*70)
    print("Running all integration tests...")
    print("Dataset: TerraClimate Bali (1958-2024)")
    print("Source: input/ folder\n")

    # Check input data exists
    required_files = [
        'input/terraclimate_bali_ppt_1958_2024.nc',
        'input/terraclimate_bali_tmean_1958_2024.nc',
        'input/terraclimate_bali_pet_1958_2024.nc'
    ]

    print("Checking input data files...")
    for file in required_files:
        file_path = os.path.join(os.path.dirname(__file__), '..', file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  [OK] {file} ({size_mb:.1f} MB)")
        else:
            print(f"  [X] {file} NOT FOUND!")
            print(f"\nERROR: Required input data missing.")
            print(f"Please ensure TerraClimate Bali data is in the input/ folder.")
            sys.exit(1)

    # Define test files
    tests = [
        'tests/test_spi.py',
        'tests/test_spei_with_pet.py',
        'tests/test_drought_characteristics.py',
        'tests/test_complete_analysis.py'
    ]

    # Track results
    results = {}
    total_start = time.time()

    # Run each test
    for test in tests:
        test_name = os.path.basename(test)
        success = run_test(test)
        results[test_name] = success

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(results.values())
    failed = len(results) - passed

    for test_name, success in results.items():
        status = "[OK] PASSED" if success else "[X] FAILED"
        print(f"{status:12s} - {test_name}")

    print("-"*70)
    print(f"Total: {passed}/{len(results)} tests passed")
    print(f"Elapsed time: {total_elapsed:.1f}s")
    print("="*70)

    # Exit with appropriate code
    if failed > 0:
        print(f"\n[X] {failed} test(s) failed")
        sys.exit(1)
    else:
        print("\n[OK] All tests passed!")
        print("\nAll test outputs created in:")
        print("  - test_output/ - NetCDF files, plots, and test results")
        sys.exit(0)

if __name__ == '__main__':
    main()
