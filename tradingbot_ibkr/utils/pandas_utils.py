"""Pandas utility module that handles both real pandas and custom pandas implementations.

This module provides a compatibility layer that:
1. Uses real pandas when available for production code
2. Falls back to custom pandas implementation for tests when real pandas is not available
3. Provides helper functions for common operations across both implementations
"""

import sys
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_pandas():
    """Get the best available pandas implementation."""
    # Try real pandas first - check if it's actually installed properly
    try:
        # Clear any cached imports to avoid conflicts
        if 'pandas' in sys.modules:
            pandas_module = sys.modules['pandas']
            # Verify it's real pandas
            if hasattr(pandas_module, '__version__') and hasattr(pandas_module, 'date_range'):
                logger.debug(f"Using real pandas version {pandas_module.__version__}")
                return pandas_module, True
        
        # Try fresh import
        import pandas as pd
        if hasattr(pd, '__version__') and hasattr(pd, 'date_range'):
            logger.debug(f"Using real pandas version {pd.__version__}")
            return pd, True
    except ImportError:
        pass
    
    # Fallback to custom implementation
    try:
        # Try different import paths for custom pandas
        custom_pd = None
        try:
            from ..pandas import DataFrame, Series, read_csv
            custom_pd = type('CustomPandas', (), {
                'DataFrame': DataFrame,
                'Series': Series, 
                'read_csv': read_csv,
                '__version__': 'custom-1.0'
            })()
        except ImportError:
            try:
                from tradingbot_ibkr.pandas import DataFrame, Series, read_csv
                custom_pd = type('CustomPandas', (), {
                    'DataFrame': DataFrame,
                    'Series': Series,
                    'read_csv': read_csv,
                    '__version__': 'custom-1.0'
                })()
            except ImportError:
                # Last resort - try direct path
                import os
                parent_dir = os.path.dirname(os.path.dirname(__file__))
                sys.path.insert(0, parent_dir)
                try:
                    from pandas import DataFrame, Series, read_csv
                    custom_pd = type('CustomPandas', (), {
                        'DataFrame': DataFrame,
                        'Series': Series,
                        'read_csv': read_csv,
                        '__version__': 'custom-1.0'
                    })()
                finally:
                    if parent_dir in sys.path:
                        sys.path.remove(parent_dir)
        
        if custom_pd:
            logger.warning("Using custom pandas implementation - limited functionality available")
            return custom_pd, False
            
    except Exception as e:
        logger.error(f"Failed to import custom pandas: {e}")
    
    raise ImportError("No pandas implementation available")


def is_using_real_pandas() -> bool:
    """Check if we're using real pandas or custom implementation."""
    try:
        pd, is_real = get_pandas()
        return is_real
    except:
        return False


def safe_date_range(*args, **kwargs):
    """Create a date range, with fallback for custom pandas."""
    try:
        pd, is_real = get_pandas()
        
        if is_real and hasattr(pd, 'date_range'):
            return pd.date_range(*args, **kwargs)
    except:
        pass
    
    # Fallback implementation for testing
    from datetime import datetime, timedelta
    
    start = kwargs.get('start') or (args[0] if args else datetime.now())
    periods = kwargs.get('periods') or (args[1] if len(args) > 1 else 10)
    freq = kwargs.get('freq', 'D')
    
    if isinstance(start, str):
        try:
            start = datetime.fromisoformat(start.replace('Z', '+00:00'))
        except ValueError:
            start = datetime.strptime(start, '%Y-%m-%d')
    
    # Simple implementation for common frequencies
    if freq in ['D', '1D']:
        delta = timedelta(days=1)
    elif freq in ['H', '1H', '1h']:
        delta = timedelta(hours=1)
    elif freq in ['T', '1T', '1min']:
        delta = timedelta(minutes=1)
    else:
        delta = timedelta(days=1)  # Default fallback
        
    dates = [start + i * delta for i in range(periods)]
    return dates


def safe_to_datetime(dates, **kwargs):
    """Convert dates to datetime, with fallback for custom pandas."""
    try:
        pd, is_real = get_pandas()
        
        if is_real and hasattr(pd, 'to_datetime'):
            return pd.to_datetime(dates, **kwargs)
    except:
        pass
    
    # Simple fallback implementation
    from datetime import datetime
    if isinstance(dates, (list, tuple)):
        result = []
        for d in dates:
            if isinstance(d, str):
                try:
                    result.append(datetime.fromisoformat(d.replace('Z', '+00:00')))
                except ValueError:
                    result.append(datetime.strptime(d, '%Y-%m-%d'))
            else:
                result.append(d)
        return result
    else:
        if isinstance(dates, str):
            try:
                return datetime.fromisoformat(dates.replace('Z', '+00:00'))
            except ValueError:
                return datetime.strptime(dates, '%Y-%m-%d')
        return dates