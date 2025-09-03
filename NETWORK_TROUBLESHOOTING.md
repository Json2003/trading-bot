# Network Connectivity Issues - Troubleshooting Guide

## Problem Description

When installing dependencies with pip, you may encounter network connectivity issues such as:

- `tunnel connection blocked`
- `no matching distributions found`
- `ReadTimeoutError: HTTPSConnectionPool`
- `Connection timeout`

These issues commonly occur in:
- Corporate networks with proxies/firewalls
- CI/CD environments with restricted internet access
- Cloud environments with network policies
- Systems behind VPNs or security appliances

## Quick Fix Solutions

### 1. Use the Automated Fix Script

```bash
python3 fix_pip_install.py
```

This script automatically:
- Tests network connectivity to PyPI servers
- Tries multiple installation configurations
- Uses fallback package versions
- Provides detailed error reporting

### 2. Use the Robust Installation Script

```bash
./install_dependencies.sh
```

This bash script provides:
- Automatic retries with exponential backoff
- Multiple PyPI mirror fallbacks
- Network timeout handling
- Comprehensive error reporting

### 3. Manual Installation with Custom Config

Use the provided pip configuration file:

```bash
pip install --config pip.conf -r requirements-robust.txt
```

## Detailed Solutions

### Option A: Configure pip for better network handling

1. Copy the provided `pip.conf` to your pip configuration directory:
   - Linux/Mac: `~/.pip/pip.conf` or `/etc/pip.conf`
   - Windows: `%APPDATA%\pip\pip.ini`

2. Install with extended timeouts:
```bash
pip install --timeout 60 --retries 5 pandas==1.5.3 ccxt==4.3.35
```

### Option B: Use alternative package sources

```bash
pip install --index-url https://pypi.python.org/simple/ \
           --trusted-host pypi.python.org \
           --trusted-host files.pythonhosted.org \
           pandas==1.5.3 ccxt==4.3.35
```

### Option C: Use the robust requirements file

```bash
pip install -r requirements-robust.txt
```

This file includes:
- Current working versions as primary choices
- Requested versions as fallback options
- Platform-specific dependencies
- Detailed installation notes

## Version Compatibility Notes

### pandas==1.5.3 vs pandas==2.2.2
- **1.5.3**: Older version, may have compatibility issues with Python 3.12
- **2.2.2**: Current version in use, tested and working
- **Recommendation**: Use 2.2.2 unless specifically required otherwise

### ccxt==4.3.35 vs ccxt==3.0.72  
- **4.3.35**: Newer version with latest exchange support
- **3.0.72**: Current version in use, stable
- **Recommendation**: Test 4.3.35 in development before production use

## Network Troubleshooting Steps

1. **Test connectivity**:
```bash
python3 -c "
import socket
try:
    socket.create_connection(('pypi.org', 443), timeout=10)
    print('✓ PyPI reachable')
except Exception as e:
    print(f'✗ PyPI unreachable: {e}')
"
```

2. **Check proxy settings**:
```bash
echo $HTTP_PROXY
echo $HTTPS_PROXY
echo $NO_PROXY
```

3. **Test with curl**:
```bash
curl -I --connect-timeout 10 https://pypi.org/simple/
```

## Corporate Network Solutions

### If behind a corporate proxy:

1. Configure pip with proxy settings:
```bash
pip install --proxy http://user:password@proxy.company.com:8080 package_name
```

2. Add to pip.conf:
```ini
[global]
proxy = http://proxy.company.com:8080
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
```

### If using corporate certificates:

```bash
pip install --cert /path/to/certificate.pem package_name
```

## Air-Gapped Environment Solutions

For environments without internet access:

1. **Create a wheel archive**:
```bash
# On internet-connected machine
pip wheel pandas==1.5.3 ccxt==4.3.35 -w wheels/

# Transfer wheels/ directory to target machine
pip install --find-links wheels/ --no-index pandas ccxt
```

2. **Use pip-tools for dependency resolution**:
```bash
pip-compile requirements-robust.txt
pip-sync requirements-robust.txt
```

## Error-Specific Solutions

### "tunnel connection blocked"
- Check firewall/proxy settings
- Use `--trusted-host` flags
- Try alternative index URLs

### "no matching distributions found" 
- Check Python version compatibility
- Verify package name spelling
- Try alternative package versions

### "ReadTimeoutError"
- Increase timeout values
- Use `--retries` parameter
- Check network stability

## Testing the Fix

After applying any solution, verify with:

```bash
python3 -c "
import pandas as pd
import ccxt
print(f'pandas: {pd.__version__}')
print(f'ccxt: {ccxt.__version__}')
print('✓ All packages working!')
"
```

## Support

If these solutions don't resolve the issue:

1. Run the diagnostic script: `python3 fix_pip_install.py`
2. Check the full error logs
3. Consider using a different Python environment (conda, virtualenv)
4. Contact your network administrator for proxy/firewall configuration