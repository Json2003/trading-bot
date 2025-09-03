#!/usr/bin/env python3
"""
Network connectivity test and pip installation helper

This script helps diagnose and fix pip installation issues related to
network connectivity, specifically for pandas and ccxt packages.
"""

import subprocess
import sys
import time
import socket
from urllib.parse import urlparse


def test_network_connectivity():
    """Test network connectivity to common PyPI endpoints."""
    print("Testing network connectivity...")
    
    endpoints = [
        "pypi.org",
        "pypi.python.org", 
        "files.pythonhosted.org"
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            socket.create_connection((endpoint, 443), timeout=10)
            results[endpoint] = "OK"
            print(f"✓ {endpoint} - reachable")
        except (socket.timeout, socket.error) as e:
            results[endpoint] = f"FAILED: {e}"
            print(f"✗ {endpoint} - {e}")
    
    return results


def install_package_with_fallback(package_specs):
    """
    Try to install packages with multiple fallback strategies.
    
    Args:
        package_specs: List of (package_name, version_list) tuples
    """
    
    pip_configs = [
        # Standard configuration
        {
            "timeout": 60,
            "retries": 3,
            "index_url": "https://pypi.org/simple/",
            "trusted_hosts": ["pypi.org", "pypi.python.org", "files.pythonhosted.org"]
        },
        # Alternative configuration with longer timeout
        {
            "timeout": 120,
            "retries": 5,
            "index_url": "https://pypi.python.org/simple/",
            "trusted_hosts": ["pypi.org", "pypi.python.org", "files.pythonhosted.org"]
        }
    ]
    
    for package_name, versions in package_specs:
        success = False
        
        for version in versions:
            package_spec = f"{package_name}=={version}"
            print(f"\nTrying to install {package_spec}...")
            
            for config_idx, config in enumerate(pip_configs):
                print(f"Using configuration {config_idx + 1}/{len(pip_configs)}")
                
                cmd = [
                    sys.executable, "-m", "pip", "install",
                    "--timeout", str(config["timeout"]),
                    "--retries", str(config["retries"]),
                    "--index-url", config["index_url"]
                ]
                
                for host in config["trusted_hosts"]:
                    cmd.extend(["--trusted-host", host])
                
                cmd.append(package_spec)
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=config["timeout"] + 30
                    )
                    
                    if result.returncode == 0:
                        print(f"✓ Successfully installed {package_spec}")
                        success = True
                        break
                    else:
                        print(f"✗ Failed with config {config_idx + 1}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"✗ Installation timed out with config {config_idx + 1}")
                except Exception as e:
                    print(f"✗ Error with config {config_idx + 1}: {e}")
            
            if success:
                break
        
        if not success:
            print(f"ERROR: Failed to install {package_name} with any version")
            return False
    
    return True


def verify_installations():
    """Verify that the required packages are properly installed."""
    print("\nVerifying installations...")
    
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import ccxt
        print(f"✓ ccxt {ccxt.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ ccxt import failed: {e}")
        return False
    
    return True


def main():
    """Main function to orchestrate the installation process."""
    print("=== Trading Bot Dependency Installation Helper ===\n")
    
    # Test network connectivity first
    connectivity_results = test_network_connectivity()
    
    # Check if any endpoints are unreachable
    failed_endpoints = [ep for ep, result in connectivity_results.items() if result != "OK"]
    if failed_endpoints:
        print(f"\nWarning: Some endpoints are unreachable: {failed_endpoints}")
        print("This may cause installation issues. Consider network configuration.\n")
    
    # Define package specifications with fallback versions
    package_specs = [
        ("pandas", ["2.2.2", "1.5.3"]),  # Current working version first, then requested
        ("ccxt", ["3.0.72", "4.3.35"])   # Current working version first, then requested
    ]
    
    print("Starting installation with fallback versions...")
    
    if install_package_with_fallback(package_specs):
        print("\n=== Installation Summary ===")
        if verify_installations():
            print("✓ All dependencies installed and verified successfully!")
            return 0
        else:
            print("✗ Installation completed but verification failed")
            return 1
    else:
        print("✗ Installation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())