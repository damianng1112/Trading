import ccxt
import pandas as pd
import logging
import time
from datetime import datetime, timedelta

def fetch_historical_data(symbol="BTC/USDT", timeframe="15m", limit=500, exchange_name="binance"):
    """
    Fetch historical OHLCV data using ccxt from the specified exchange with improved error handling
    and sufficient data for calculating indicators.

    Args:
        symbol (str): The trading pair (e.g., "BTC/USDT").
        timeframe (str): The candle timeframe (e.g., "1h", "1d").
        limit (int): The number of historical candles to fetch.
        exchange_name (str): The exchange name (e.g., "binance").

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV data.
    """
    try:
        # Adjust limit to ensure enough data for indicators like SMA200
        adjusted_limit = max(limit, 500) + 300  # Get extra data for indicators
        
        # Initialize the exchange
        exchange = getattr(ccxt, exchange_name)()
        
        # Add rate limiting to avoid API restrictions
        exchange.enableRateLimit = True
        
        # Fetch historical OHLCV data with retry mechanism
        max_retries = 3
        for retry in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=adjusted_limit)
                
                if not ohlcv or len(ohlcv) < 50:
                    logging.warning(f"Received insufficient data: {len(ohlcv) if ohlcv else 0} candles")
                    if retry < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                    else:
                        logging.error(f"Failed to fetch sufficient data after {max_retries} attempts")
                        return None
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logging.warning(f"Retry {retry+1}/{max_retries}: Error fetching data: {e}")
                    time.sleep(2)
                else:
                    raise
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Check if we got enough data
        if len(df) < adjusted_limit * 0.9:  # Allow for some missing candles
            logging.warning(f"Received fewer candles than requested: {len(df)}/{adjusted_limit}")
            
            # For longer timeframes, we might need to fetch in batches
            if timeframe in ["1h", "4h", "1d"] and len(df) < 200:
                return fetch_in_batches(exchange, symbol, timeframe, adjusted_limit)
        
        logging.info(f"Successfully fetched {len(df)} candles for {symbol} ({timeframe})")
        return df
        
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return None

def fetch_in_batches(exchange, symbol, timeframe, total_limit):
    """
    Fetch historical data in batches for longer timeframes when a single request
    might not return enough data.
    
    Args:
        exchange: The ccxt exchange instance
        symbol (str): Trading pair
        timeframe (str): Candle timeframe
        total_limit (int): Total number of candles needed
        
    Returns:
        pd.DataFrame: Combined DataFrame with historical data
    """
    try:
        # Initial fetch
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)  # Most exchanges allow max 1000
        
        if not ohlcv or len(ohlcv) == 0:
            logging.error("Initial batch fetch returned no data")
            return None
            
        all_data = ohlcv.copy()
        
        # Continue fetching if we need more data
        while len(all_data) < total_limit:
            # Get timestamp for the oldest candle we have
            oldest_timestamp = all_data[0][0]
            
            # Wait to avoid rate limits
            time.sleep(exchange.rateLimit / 1000)
            
            # Fetch earlier data (before the oldest candle we have)
            earlier_ohlcv = exchange.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                limit=1000,
                since=oldest_timestamp - (get_timeframe_ms(timeframe) * 1000)
            )
            
            # If we got no new data or reached exchange limits, stop
            if not earlier_ohlcv or len(earlier_ohlcv) <= 1:
                break
                
            # Prepend new data and continue
            all_data = earlier_ohlcv + all_data
            
            logging.info(f"Batch fetch progress: {len(all_data)}/{total_limit} candles")
            
            # Stop if we're not getting new data anymore
            if earlier_ohlcv[0][0] >= oldest_timestamp:
                logging.warning("No earlier data available from exchange")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        return df
        
    except Exception as e:
        logging.error(f"Error in batch fetching: {e}")
        return None

def get_timeframe_ms(timeframe):
    """Convert timeframe string to milliseconds"""
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    
    if unit == 'm':
        return value * 60 * 1000
    elif unit == 'h':
        return value * 60 * 60 * 1000
    elif unit == 'd':
        return value * 24 * 60 * 60 * 1000
    elif unit == 'w':
        return value * 7 * 24 * 60 * 60 * 1000
    else:
        return 60 * 1000  # Default to 1m if unknown