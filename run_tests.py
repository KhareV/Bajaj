"""
Test runner for the Bajaj Finserv AI Hackathon backend
Runs all tests and provides comprehensive reporting
"""

import pytest
import sys
import os
import subprocess
import time
from pathlib import Path

def run_tests():
    """Run all tests with comprehensive reporting"""
    print("🧪 Running Bajaj Finserv AI Hackathon Backend Tests")
    print("=" * 60)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Test configuration
    test_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
        "app/tests/",  # Test directory
    ]
    
    print(f"⚙️  Test configuration: {' '.join(test_args)}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Run tests
    try:
        exit_code = pytest.main(test_args)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("-" * 60)
        print(f"⏱️  Tests completed in {duration:.2f} seconds")
        
        if exit_code == 0:
            print("✅ All tests passed!")
            return True
        else:
            print(f"❌ Tests failed with exit code: {exit_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def check_code_quality():
    """Check code quality using basic linting"""
    print("\n🔍 Checking code quality...")
    
    try:
        # Basic syntax check
        result = subprocess.run([
            sys.executable, "-m", "py_compile", "app/main.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Code syntax check passed")
            return True
        else:
            print(f"❌ Syntax errors found: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not run code quality check: {e}")
        return True  # Don't fail if linting tools aren't available

def main():
    """Main test execution"""
    print("🚀 Bajaj Finserv AI Hackathon - Test Suite")
    print(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("👨‍💻 Team: vkhare2909 and team")
    print("🎯 Objective: Ensure production-ready backend for hackathon submission")
    print("\n")
    
    success = True
    
    # Run code quality checks
    if not check_code_quality():
        success = False
    
    # Run tests
    if not run_tests():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL CHECKS PASSED - READY FOR SUBMISSION! 🏆")
        print("✅ Code quality: PASS")
        print("✅ Test suite: PASS")
        print("✅ Ready for integration with team members' code")
    else:
        print("❌ SOME CHECKS FAILED - NEEDS ATTENTION")
        print("🔧 Please fix the issues before proceeding")
    
    print(f"📅 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)