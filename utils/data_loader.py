import ccxt
import pandas as pd

def fetch_historical_data(symbol="BTC/USDT", timeframe="15m", limit=500, exchange_name="binance"):
    """
    Fetch historical OHLCV data using ccxt from the specified exchange.

    Args:
        symbol (str): The trading pair (e.g., "BTC/USDT").
        timeframe (str): The candle timeframe (e.g., "1h", "1d").
        limit (int): The number of historical candles to fetch.
        exchange_name (str): The exchange name (e.g., "binance").

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV data.
    """
    # Initialize the exchange
    exchange = getattr(ccxt, exchange_name)()
    
    # Fetch historical OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def load_historical_data_from_csv(filepath):
    """
    Load historical OHLCV data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV data.
    """
    return pd.read_csv(filepath, parse_dates=["timestamp"])

def save_historical_data_to_csv(data, filepath):
    """
    Save OHLCV data to a CSV file.

    Args:
        data (pd.DataFrame): The data to save.
        filepath (str): Path to save the CSV file.
    """
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
