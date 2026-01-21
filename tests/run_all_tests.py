#!/usr/bin/env python
"""
Run all integration tests for precip-index package.

This script executes all test files in sequence and provides a summary.
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
            print(f"\n✅ PASSED ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n❌ FAILED ({elapsed:.1f}s)")
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR ({elapsed:.1f}s): {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PRECIP-INDEX TEST SUITE")
    print("="*70)
    print("Running all integration tests...\n")

    # Define test files
    tests = [
        'tests/test_spi.py',
        'tests/test_spei_with_pet.py',
        'tests/test_drought_characteristics.py'
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
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12s} - {test_name}")

    print("-"*70)
    print(f"Total: {passed}/{len(results)} tests passed")
    print(f"Elapsed time: {total_elapsed:.1f}s")
    print("="*70)

    # Exit with appropriate code
    if failed > 0:
        print(f"\n❌ {failed} test(s) failed")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()
