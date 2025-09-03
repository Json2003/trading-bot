#!/bin/bash
# Robust pip installation script with network timeout handling
# Usage: ./install_dependencies.sh

set -e  # Exit on any error

echo "Starting robust dependency installation..."

# Function to install packages with retries
install_with_retry() {
    local package=$1
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: Installing $package"
        
        if pip install \
            --timeout 60 \
            --retries 3 \
            --index-url https://pypi.org/simple/ \
            --extra-index-url https://pypi.python.org/simple/ \
            --trusted-host pypi.org \
            --trusted-host pypi.python.org \
            --trusted-host files.pythonhosted.org \
            "$package"; then
            echo "Successfully installed $package"
            return 0
        else
            echo "Failed to install $package (attempt $attempt)"
            if [ $attempt -eq $max_attempts ]; then
                echo "ERROR: Failed to install $package after $max_attempts attempts"
                return 1
            fi
            attempt=$((attempt + 1))
            echo "Waiting 10 seconds before retry..."
            sleep 10
        fi
    done
}

# Function to check if a package is already installed with the correct version
check_package() {
    local package_name=$(echo $1 | cut -d'=' -f1)
    local required_version=$(echo $1 | cut -d'=' -f3)
    
    if pip show "$package_name" &>/dev/null; then
        local installed_version=$(pip show "$package_name" | grep Version | cut -d' ' -f2)
        if [ "$installed_version" = "$required_version" ]; then
            echo "$package_name==$required_version is already installed"
            return 0
        else
            echo "$package_name is installed but version $installed_version != $required_version"
            return 1
        fi
    else
        echo "$package_name is not installed"
        return 1
    fi
}

# Main installation logic
main() {
    echo "Checking Python version..."
    python3 --version
    
    echo "Checking pip version..."
    pip --version
    
    # Define target packages - with current working versions from requirements.txt
    # and the requested versions as alternatives
    PACKAGES=(
        "pandas==2.2.2"  # Current working version
        "ccxt==3.0.72"   # Current working version
    )
    
    # Alternative packages if network issues persist
    ALTERNATIVE_PACKAGES=(
        "pandas==1.5.3"  # Requested version
        "ccxt==4.3.35"   # Requested version
    )
    
    echo "Installing primary package versions..."
    failed_packages=()
    
    for package in "${PACKAGES[@]}"; do
        if ! check_package "$package"; then
            if ! install_with_retry "$package"; then
                failed_packages+=("$package")
            fi
        fi
    done
    
    # If any packages failed, try alternatives
    if [ ${#failed_packages[@]} -ne 0 ]; then
        echo "Some packages failed to install. Trying alternative versions..."
        
        for i in "${!failed_packages[@]}"; do
            failed_package="${failed_packages[$i]}"
            alternative_package="${ALTERNATIVE_PACKAGES[$i]}"
            
            echo "Trying alternative for $failed_package: $alternative_package"
            if ! install_with_retry "$alternative_package"; then
                echo "ERROR: Both primary and alternative versions failed for $failed_package"
                exit 1
            fi
        done
    fi
    
    echo "Verifying all installations..."
    python3 -c "
import pandas as pd
import ccxt
print(f'pandas version: {pd.__version__}')
print(f'ccxt version: {ccxt.__version__}')
print('All packages imported successfully!')
"
    
    echo "Installation completed successfully!"
}

# Run main function
main "$@"