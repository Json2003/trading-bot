"""Verify a FRED API key by doing a lightweight request.
Usage: python verify_fred.py <API_KEY>
This script prints only 'OK' on success or 'FAIL' on failure.
"""
import sys
import time
import requests

def verify_key(key, retries=3, timeout=5):
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {'series_id': 'CPIAUCSL', 'api_key': key, 'file_type': 'json'}
    for attempt in range(1, retries+1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def main():
    if len(sys.argv) < 2:
        print('FAIL')
        return
    key = sys.argv[1]
    ok = verify_key(key)
    print('OK' if ok else 'FAIL')

if __name__ == '__main__':
    main()
