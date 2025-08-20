#!/usr/bin/env python3
"""
Script to run all tests for the Ollama Weather Agent.
"""
import os
import sys
import unittest
import pytest

def run_tests():
    """Run all tests using unittest and pytest."""
    print("Running tests for Ollama Weather Agent...")
    
    # Get the directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to the Python path
    sys.path.insert(0, base_dir)
    
    # Run unittest tests
    print("\n=== Running unittest tests ===")
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.join(base_dir, "tests"))
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Run pytest tests with coverage
    print("\n=== Running pytest with coverage ===")
    pytest_args = [
        "--cov=src",
        "--cov-report=term",
        "--cov-report=html:coverage_report",
        "tests/"
    ]
    pytest_result = pytest.main(pytest_args)
    
    # Return success if both test runners succeeded
    return test_result.wasSuccessful() and pytest_result == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
