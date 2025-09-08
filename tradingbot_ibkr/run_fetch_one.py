"""Enhanced FRED data fetcher with robust retry logic and comprehensive error handling.

Features:
- Robust retry logic with exponential backoff
- Comprehensive error handling and validation
- Data quality checks and cleaning
- Multiple output formats (CSV, JSON, Parquet)
- Detailed logging and progress tracking
- Rate limiting compliance
- Connection pooling and timeout management
- Data validation and integrity checks
"""
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
import pandas as pd

# Enhanced networking imports
try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.exceptions import (
        RequestException, ConnectionError, Timeout, 
        HTTPError, TooManyRedirects
    )
    from urllib3.util.retry import Retry
    requests_available = True
except ImportError:
    requests_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FREDDataFetcher:
    """Enhanced FRED data fetcher with robust error handling and retry logic."""
    
    def __init__(self, api_key: str, max_retries: int = 5, timeout: int = 30):
        """
        Initialize FRED data fetcher.
        
        Args:
            api_key: FRED API key
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        if not requests_available:
            raise ImportError("requests library is required for data fetching")
        
        self.api_key = api_key
        self.base_url = 'https://api.stlouisfed.org/fred'
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Configure session with retry strategy
        self.session = requests.Session()
        
        # Retry strategy with exponential backoff
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=2,  # Exponential backoff
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'FRED-Data-Fetcher/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        logger.info(f"FRED fetcher initialized with max_retries={max_retries}, timeout={timeout}s")
    
    def validate_api_key(self) -> bool:
        """Validate API key by making a test request."""
        logger.info("Validating API key...")
        
        try:
            response = self._make_request('/category', {'category_id': '0'})
            if response and response.get('categories'):
                logger.info("API key validation successful")
                return True
            else:
                logger.error("API key validation failed - invalid response")
                return False
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make HTTP request with retry logic and comprehensive error handling.
        
        Args:
            endpoint: API endpoint (e.g., '/series/observations')
            params: Request parameters
            
        Returns:
            JSON response data or None if failed
        """
        url = f"{self.base_url}{endpoint}"
        
        # Add API key and format
        request_params = {
            'api_key': self.api_key,
            'file_type': 'json',
            **params
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}: {endpoint} with params {request_params}")
                
                response = self.session.get(
                    url,
                    params=request_params,
                    timeout=self.timeout
                )
                
                # Check HTTP status
                response.raise_for_status()
                
                # Parse JSON
                data = response.json()
                
                # Check for API errors
                if 'error_code' in data:
                    error_msg = data.get('error_message', 'Unknown API error')
                    logger.error(f"FRED API error: {data['error_code']} - {error_msg}")
                    
                    # Don't retry on certain errors
                    if data['error_code'] in [400, 404]:  # Bad request, not found
                        return None
                    
                    raise HTTPError(f"API Error: {error_msg}")
                
                logger.debug(f"Request successful on attempt {attempt + 1}")
                return data
                
            except (ConnectionError, Timeout) as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries:
                    logger.error(f"Max retries reached for {endpoint}")
                    raise
                
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + (time.time() % 1)
                logger.info(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                
            except HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP error: {e}")
                    if attempt == self.max_retries:
                        raise
                    time.sleep(2 ** attempt)
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"JSON parsing error: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        
        return None
    
    def get_series_info(self, series_id: str) -> Optional[Dict]:
        """Get metadata for a FRED series."""
        logger.info(f"Fetching series info for {series_id}")
        
        data = self._make_request('/series', {'series_id': series_id})
        
        if data and 'seriess' in data and data['seriess']:
            series_info = data['seriess'][0]
            logger.info(f"Series info retrieved: {series_info.get('title', 'N/A')}")
            return series_info
        
        logger.warning(f"No series info found for {series_id}")
        return None
    
    def fetch_series_data(self, series_id: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch time series data with comprehensive validation.
        
        Args:
            series_id: FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with date and value columns
        """
        logger.info(f"Fetching data for series {series_id}")
        
        # Prepare parameters
        params = {'series_id': series_id}
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        
        # Make request
        data = self._make_request('/series/observations', params)
        
        if not data or 'observations' not in data:
            logger.error(f"No data returned for series {series_id}")
            return None
        
        observations = data['observations']
        
        if not observations:
            logger.warning(f"Empty dataset returned for series {series_id}")
            return None
        
        logger.info(f"Retrieved {len(observations)} observations")
        
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        
        if df.empty:
            logger.error("Empty DataFrame created")
            return None
        
        # Data validation and cleaning
        return self._clean_and_validate_data(df, series_id)
    
    def _clean_and_validate_data(self, df: pd.DataFrame, series_id: str) -> Optional[pd.DataFrame]:
        """Clean and validate the fetched data."""
        logger.info(f"Cleaning and validating data for {series_id}")
        
        original_count = len(df)
        
        try:
            # Rename columns for consistency
            df = df.rename(columns={'date': 'date', 'value': 'value'})
            
            # Ensure required columns exist
            if 'date' not in df.columns or 'value' not in df.columns:
                logger.error(f"Required columns missing. Available: {list(df.columns)}")
                return None
            
            # Keep only date and value columns
            df = df[['date', 'value']].copy()
            
            # Convert date column
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                logger.error(f"Date parsing failed: {e}")
                return None
            
            # Convert value column to numeric, handling missing/invalid values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Count missing values
            missing_count = df['value'].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing/invalid values")
                
                # Remove rows with missing values
                df = df.dropna(subset=['value'])
                
                if df.empty:
                    logger.error("No valid data remaining after cleaning")
                    return None
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Check for duplicates
            duplicate_count = df.duplicated(subset=['date']).sum()
            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} duplicate dates, keeping last values")
                df = df.drop_duplicates(subset=['date'], keep='last')
            
            # Validate date range
            date_range = df['date'].max() - df['date'].min()
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()} ({date_range.days} days)")
            
            # Validate value range
            value_stats = df['value'].describe()
            logger.info(f"Value statistics: min={value_stats['min']:.3f}, max={value_stats['max']:.3f}, mean={value_stats['mean']:.3f}")
            
            # Check for extreme outliers (values beyond 5 standard deviations)
            if len(df) > 10:  # Only if we have enough data
                mean_val = df['value'].mean()
                std_val = df['value'].std()
                
                if std_val > 0:  # Avoid division by zero
                    outliers = df[abs(df['value'] - mean_val) > 5 * std_val]
                    if len(outliers) > 0:
                        logger.warning(f"Found {len(outliers)} potential outliers")
                        logger.debug(f"Outlier dates: {outliers['date'].tolist()}")
            
            final_count = len(df)
            logger.info(f"Data cleaning completed: {original_count} -> {final_count} observations")
            
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, output_path: Path, format: str = 'csv') -> bool:
        """
        Save data in specified format with error handling.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            format: Output format ('csv', 'json', 'parquet')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', date_format='iso', indent=2)
            elif format.lower() == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Data saved to {output_path} ({format.upper()} format)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False

def fetch_fred_series(api_key: str, series_id: str = 'CPIAUCSL', 
                     start_date: Optional[str] = None, end_date: Optional[str] = None,
                     output_format: str = 'csv', max_retries: int = 5) -> bool:
    """
    Enhanced function to fetch FRED series data with comprehensive error handling.
    
    Args:
        api_key: FRED API key
        series_id: FRED series ID to fetch
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        output_format: Output format ('csv', 'json', 'parquet')
        max_retries: Maximum retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("="*60)
    logger.info("FRED DATA FETCHING STARTED")
    logger.info(f"Series: {series_id}")
    logger.info(f"Date range: {start_date or 'all'} to {end_date or 'all'}")
    logger.info(f"Output format: {output_format}")
    logger.info("="*60)
    
    try:
        # Initialize fetcher
        fetcher = FREDDataFetcher(api_key, max_retries=max_retries)
        
        # Validate API key
        if not fetcher.validate_api_key():
            logger.error("API key validation failed")
            return False
        
        # Get series metadata
        series_info = fetcher.get_series_info(series_id)
        if series_info:
            logger.info(f"Series title: {series_info.get('title', 'N/A')}")
            logger.info(f"Units: {series_info.get('units', 'N/A')}")
            logger.info(f"Frequency: {series_info.get('frequency', 'N/A')}")
        
        # Fetch data
        start_time = time.time()
        df = fetcher.fetch_series_data(series_id, start_date, end_date)
        fetch_time = time.time() - start_time
        
        if df is None or df.empty:
            logger.error("No data retrieved")
            return False
        
        logger.info(f"Data fetched successfully in {fetch_time:.1f}s")
        
        # Prepare output path
        output_dir = Path(__file__).resolve().parents[0] / 'datafiles' / 'econ'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{series_id}_fred_{timestamp}.{output_format}"
        output_path = output_dir / filename
        
        # Save data
        if fetcher.save_data(df, output_path, output_format):
            # Also save metadata
            metadata = {
                'series_id': series_id,
                'fetch_timestamp': datetime.now().isoformat(),
                'series_info': series_info,
                'data_points': len(df),
                'date_range': {
                    'start': df['date'].min().isoformat(),
                    'end': df['date'].max().isoformat()
                },
                'fetch_time_seconds': fetch_time,
                'output_file': filename
            }
            
            metadata_path = output_dir / f"{series_id}_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("="*60)
            logger.info("FETCH COMPLETED SUCCESSFULLY")
            logger.info(f"Data file: {output_path}")
            logger.info(f"Metadata file: {metadata_path}")
            logger.info(f"Records: {len(df)}")
            logger.info("="*60)
            
            # Print filename for backward compatibility
            print(filename)
            return True
        else:
            logger.error("Failed to save data")
            return False
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during fetch: {e}")
        return False

def fetch_one(api_key: str, series: str = 'CPIAUCSL'):
    """Legacy function for backward compatibility."""
    return fetch_fred_series(api_key, series)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced FRED data fetcher with robust error handling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_fetch_one.py YOUR_API_KEY
  python run_fetch_one.py YOUR_API_KEY --series GDPC1
  python run_fetch_one.py YOUR_API_KEY --series UNRATE --start-date 2020-01-01 --format json
  python run_fetch_one.py YOUR_API_KEY --series CPIAUCSL --start-date 2020-01-01 --end-date 2023-12-31
        """
    )
    
    parser.add_argument('api_key', help='FRED API key')
    parser.add_argument('--series', default='CPIAUCSL', 
                       help='FRED series ID (default: CPIAUCSL)')
    parser.add_argument('--start-date', 
                       help='Start date in YYYY-MM-DD format (optional)')
    parser.add_argument('--end-date', 
                       help='End date in YYYY-MM-DD format (optional)')
    parser.add_argument('--format', choices=['csv', 'json', 'parquet'], 
                       default='csv', help='Output format (default: csv)')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Maximum retry attempts (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        success = fetch_fred_series(
            api_key=args.api_key,
            series_id=args.series,
            start_date=args.start_date,
            end_date=args.end_date,
            output_format=args.format,
            max_retries=args.max_retries
        )
        
        if success:
            logger.info("Data fetch completed successfully")
            sys.exit(0)
        else:
            logger.error("Data fetch failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f'error: {str(e)}')
        sys.exit(1)
