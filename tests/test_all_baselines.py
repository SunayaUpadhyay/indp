"""
Master test runner for all baseline planners.

Runs all test suites and provides comprehensive summary.
"""

import subprocess
import sys
from pathlib import Path


def run_test_suite(test_file: str, name: str) -> bool:
    """
    Run a test suite and return success status.
    
    Args:
        test_file: Path to test file
        name: Name for display
        
    Returns:
        True if all tests passed, False otherwise
    """
    print("\n" + "="*80)
    print(f"RUNNING: {name}")
    print("="*80 + "\n")
    
    result = subprocess.run(
        [sys.executable, test_file],
        cwd=Path(__file__).parent.parent,
        capture_output=False
    )
    
    success = (result.returncode == 0)
    
    if success:
        print(f"\n✓ {name}: ALL TESTS PASSED")
    else:
        print(f"\n✗ {name}: SOME TESTS FAILED")
    
    return success


def main():
    """Run all baseline planner tests."""
    print("\n" + "="*80)
    print("BASELINE PLANNERS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    test_suites = [
        ("tests/test_lawnmower_planner.py", "Lawnmower Planner"),
        ("tests/test_sequential_greedy_planner.py", "Sequential Greedy IG Planner"),
        ("tests/test_independent_greedy_planner.py", "Independent Greedy IG Planner"),
        ("tests/test_auction_planner.py", "Auction Variance Planner"),
    ]
    
    results = {}
    
    for test_file, name in test_suites:
        success = run_test_suite(test_file, name)
        results[name] = success
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for success in results.values() if success)
    failed = total - passed
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {passed}/{total} test suites passed")
    print("="*80)
    
    if failed == 0:
        print("\n🎉 ALL BASELINE PLANNERS WORKING CORRECTLY! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {failed} test suite(s) failed. Please review errors above. ⚠️\n")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
