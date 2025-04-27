import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def indicator_wrapper(func):
    """
    Decorator to handle common error handling and data preparation for indicators.
    This ensures consistent error handling across all indicator functions.
    
    Args:
        func: The indicator function to wrap
        
    Returns:
        The wrapped function
    """
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        try:
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Check if the required column exists if specified in kwargs
            column = kwargs.get('column', 'close')
            if column not in df.columns:
                logging.error(f"Column '{column}' not found in dataframe for {func.__name__}")
                data[f'{func.__name__}_error'] = True
                return data  # Return original data in case of error
            
            # Call the indicator function
            result_df = func(df, *args, **kwargs)
            return result_df
        except Exception as e:
            import traceback
            logging.error(f"Error in {func.__name__}: {e}")
            logging.debug(traceback.format_exc())
            data[f'{func.__name__}_error'] = True
            return data
    return wrapper

@indicator_wrapper
def calculate_sma(data, period=20, column='close'):
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        period (int): Period for SMA calculation
        column (str): Column name to use for calculation
        
    Returns:
        pd.DataFrame: DataFrame with SMA column added
    """
    data[f'SMA_{period}'] = data[column].rolling(window=period).mean()
    return data

@indicator_wrapper
def calculate_ema(data, periods=None, column='close'):
    """
    Calculate Exponential Moving Average (EMA) for multiple periods.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        periods (list): List of periods for EMA calculation
        column (str): Column name to use for calculation
        
    Returns:
        pd.DataFrame: DataFrame with EMA columns added
    """
    if periods is None:
        periods = [9, 21, 50, 200]
    
    # Calculate EMAs for each period
    for period in periods:
        data[f'EMA_{period}'] = data[column].ewm(span=period, adjust=False).mean()
    
    # Add EMA crossover signals if we have at least two EMAs
    if len(periods) >= 2:
        # Sort periods to ensure consistent comparisons (shorter vs longer)
        sorted_periods = sorted(periods)
        
        # For the two shortest periods, calculate crossovers
        short_period = sorted_periods[0]
        medium_period = sorted_periods[1]
        
        data['EMA_Cross_Signal'] = 'neutral'
        
        # Bullish crossover (shorter EMA crosses above longer EMA)
        data.loc[(data[f'EMA_{short_period}'] > data[f'EMA_{medium_period}']) & 
              (data[f'EMA_{short_period}'].shift(1) <= data[f'EMA_{medium_period}'].shift(1)), 
              'EMA_Cross_Signal'] = 'bullish'
        
        # Bearish crossover (shorter EMA crosses below longer EMA)
        data.loc[(data[f'EMA_{short_period}'] < data[f'EMA_{medium_period}']) & 
              (data[f'EMA_{short_period}'].shift(1) >= data[f'EMA_{medium_period}'].shift(1)), 
              'EMA_Cross_Signal'] = 'bearish'
    
    return data

@indicator_wrapper
def calculate_rsi(data, period=14, column='close'):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        period (int): Period for RSI calculation
        column (str): Column name to use for calculation
        
    Returns:
        pd.DataFrame: DataFrame with RSI column added
    """
    # Calculate price changes
    delta = data[column].diff()
    
    # Create gain and loss series
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the specified period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    
    # Calculate RSI
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Add RSI oversold/overbought signals
    data['RSI_Signal'] = 'neutral'
    data.loc[data['RSI'] < 30, 'RSI_Signal'] = 'oversold'
    data.loc[data['RSI'] > 70, 'RSI_Signal'] = 'overbought'
    
    return data

@indicator_wrapper
def calculate_macd(data, column='close', fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        column (str): Column name to use for calculation
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        pd.DataFrame: DataFrame with MACD columns added
    """
    # Calculate EMAs
    fast_ema = data[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data[column].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    data['MACD'] = fast_ema - slow_ema
    
    # Calculate signal line
    data['Signal_Line'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    # Add MACD crossover signals
    data['MACD_Signal'] = 'neutral'
    
    # Bullish crossover (MACD crosses above signal line)
    data.loc[(data['MACD'] > data['Signal_Line']) & 
          (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)), 'MACD_Signal'] = 'bullish'
    
    # Bearish crossover (MACD crosses below signal line)
    data.loc[(data['MACD'] < data['Signal_Line']) & 
          (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)), 'MACD_Signal'] = 'bearish'
    
    return data

@indicator_wrapper
def calculate_bollinger_bands(data, column='close', window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        column (str): Column name to use for calculation
        window (int): Moving average window
        num_std (float): Number of standard deviations for bands
        
    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands columns added
    """
    # Calculate the simple moving average
    data['BB_Middle'] = data[column].rolling(window=window).mean()
    
    # Calculate the standard deviation
    rolling_std = data[column].rolling(window=window).std()
    
    # Calculate upper and lower bands
    data['BB_Upper'] = data['BB_Middle'] + (rolling_std * num_std)
    data['BB_Lower'] = data['BB_Middle'] - (rolling_std * num_std)
    
    # Calculate bandwidth and %B
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    data['BB_PercentB'] = (data[column] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Add Bollinger Band signals
    data['BB_Signal'] = 'neutral'
    data.loc[data[column] > data['BB_Upper'], 'BB_Signal'] = 'overbought'
    data.loc[data[column] < data['BB_Lower'], 'BB_Signal'] = 'oversold'
    
    return data

@indicator_wrapper
def calculate_stochastic_oscillator(data, k_period=14, d_period=3, smooth_k=3):
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        k_period (int): K period
        d_period (int): D period
        smooth_k (int): K smoothing period
        
    Returns:
        pd.DataFrame: DataFrame with Stochastic Oscillator columns added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for Stochastic Oscillator: {missing_columns}")
        return data
    
    # Calculate %K
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    data['Stoch_K_Raw'] = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    
    # Apply smoothing to %K if requested
    if smooth_k > 1:
        data['Stoch_K'] = data['Stoch_K_Raw'].rolling(window=smooth_k).mean()
    else:
        data['Stoch_K'] = data['Stoch_K_Raw']
    
    # Calculate %D (signal line)
    data['Stoch_D'] = data['Stoch_K'].rolling(window=d_period).mean()
    
    # Add Stochastic signals
    data['Stochastic_Signal'] = 'neutral'
    data.loc[data['Stoch_K'] < 20, 'Stochastic_Signal'] = 'oversold'
    data.loc[data['Stoch_K'] > 80, 'Stochastic_Signal'] = 'overbought'
    
    # Add crossover signals
    # Bullish crossover (K crosses above D)
    data.loc[(data['Stoch_K'] > data['Stoch_D']) & 
          (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1)), 'Stochastic_Signal'] = 'bullish_cross'
    
    # Bearish crossover (K crosses below D)
    data.loc[(data['Stoch_K'] < data['Stoch_D']) & 
          (data['Stoch_K'].shift(1) >= data['Stoch_D'].shift(1)), 'Stochastic_Signal'] = 'bearish_cross'
    
    return data

@indicator_wrapper
def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for ATR calculation
        
    Returns:
        pd.DataFrame: DataFrame with ATR column added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for ATR: {missing_columns}")
        return data
    
    # Calculate True Range
    data['TR1'] = abs(data['high'] - data['low'])
    data['TR2'] = abs(data['high'] - data['close'].shift(1))
    data['TR3'] = abs(data['low'] - data['close'].shift(1))
    data['TR'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    # Calculate ATR
    data['ATR'] = data['TR'].rolling(window=period).mean()
    
    # Calculate ATR percent (ATR as percentage of price)
    data['ATR_Percent'] = data['ATR'] / data['close'] * 100
    
    # Clean up temporary columns
    data = data.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1)
    
    return data

@indicator_wrapper
def calculate_vwap(data):
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with VWAP column added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for VWAP: {missing_columns}")
        return data
    
    # Reset VWAP calculation at the start of each day if timestamp exists
    if 'timestamp' in data.columns:
        # Extract date from timestamp to group by day
        data['date'] = pd.to_datetime(data['timestamp']).dt.date
        grouped = data.groupby('date')
        
        # Calculate VWAP for each day
        for _, group in grouped:
            # Calculate typical price
            typical_price = (group['high'] + group['low'] + group['close']) / 3
            
            # Calculate cumulative (price * volume) and cumulative volume
            cumulative_pv = (typical_price * group['volume']).cumsum()
            cumulative_volume = group['volume'].cumsum()
            
            # Calculate VWAP
            data.loc[group.index, 'VWAP'] = cumulative_pv / cumulative_volume
        
        # Remove temporary date column
        data = data.drop('date', axis=1)
    else:
        # If no timestamp, calculate VWAP for entire dataset
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate cumulative (price * volume) and cumulative volume
        cumulative_pv = (typical_price * data['volume']).cumsum()
        cumulative_volume = data['volume'].cumsum()
        
        # Calculate VWAP
        data['VWAP'] = cumulative_pv / cumulative_volume
    
    # Add VWAP signals
    data['VWAP_Signal'] = 'neutral'
    data.loc[data['close'] > data['VWAP'], 'VWAP_Signal'] = 'bullish'
    data.loc[data['close'] < data['VWAP'], 'VWAP_Signal'] = 'bearish'
    
    return data

@indicator_wrapper
def calculate_supertrend(data, period=10, multiplier=3.0):
    """
    Calculate SuperTrend indicator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): ATR period
        multiplier (float): ATR multiplier
        
    Returns:
        pd.DataFrame: DataFrame with SuperTrend columns added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for SuperTrend: {missing_columns}")
        return data
    
    # First calculate ATR
    data = calculate_atr(data, period=period)
    
    # Calculate basic upper and lower bands
    data['basic_upper_band'] = ((data['high'] + data['low']) / 2) + (multiplier * data['ATR'])
    data['basic_lower_band'] = ((data['high'] + data['low']) / 2) - (multiplier * data['ATR'])
    
    # Initialize SuperTrend columns
    data['SuperTrend_Upper'] = data['basic_upper_band']
    data['SuperTrend_Lower'] = data['basic_lower_band']
    data['SuperTrend'] = data['close'].copy()
    data['SuperTrend_Direction'] = 1  # 1 for uptrend, -1 for downtrend
    
    # Calculate SuperTrend using vectorized operations where possible
    for i in range(1, len(data)):
        # Update upper band
        if (data['basic_upper_band'].iloc[i] < data['SuperTrend_Upper'].iloc[i-1] or 
            data['close'].iloc[i-1] > data['SuperTrend_Upper'].iloc[i-1]):
            data.loc[data.index[i], 'SuperTrend_Upper'] = data['basic_upper_band'].iloc[i]
        else:
            data.loc[data.index[i], 'SuperTrend_Upper'] = data['SuperTrend_Upper'].iloc[i-1]
            
        # Update lower band
        if (data['basic_lower_band'].iloc[i] > data['SuperTrend_Lower'].iloc[i-1] or 
            data['close'].iloc[i-1] < data['SuperTrend_Lower'].iloc[i-1]):
            data.loc[data.index[i], 'SuperTrend_Lower'] = data['basic_lower_band'].iloc[i]
        else:
            data.loc[data.index[i], 'SuperTrend_Lower'] = data['SuperTrend_Lower'].iloc[i-1]
            
        # Determine trend direction
        if data['close'].iloc[i-1] <= data['SuperTrend'].iloc[i-1] and data['close'].iloc[i] > data['SuperTrend_Upper'].iloc[i]:
            # Trend changes to uptrend
            data.loc[data.index[i], 'SuperTrend_Direction'] = 1
            data.loc[data.index[i], 'SuperTrend'] = data['SuperTrend_Lower'].iloc[i]
        elif data['close'].iloc[i-1] >= data['SuperTrend'].iloc[i-1] and data['close'].iloc[i] < data['SuperTrend_Lower'].iloc[i]:
            # Trend changes to downtrend
            data.loc[data.index[i], 'SuperTrend_Direction'] = -1
            data.loc[data.index[i], 'SuperTrend'] = data['SuperTrend_Upper'].iloc[i]
        else:
            # Trend continues
            data.loc[data.index[i], 'SuperTrend_Direction'] = data['SuperTrend_Direction'].iloc[i-1]
            if data['SuperTrend_Direction'].iloc[i] == 1:
                data.loc[data.index[i], 'SuperTrend'] = data['SuperTrend_Lower'].iloc[i]
            else:
                data.loc[data.index[i], 'SuperTrend'] = data['SuperTrend_Upper'].iloc[i]
    
    # Add SuperTrend signals
    data['SuperTrend_Signal'] = 'neutral'
    data.loc[data['SuperTrend_Direction'] == 1, 'SuperTrend_Signal'] = 'bullish'
    data.loc[data['SuperTrend_Direction'] == -1, 'SuperTrend_Signal'] = 'bearish'
    
    # Clean up temporary columns
    data = data.drop(['basic_upper_band', 'basic_lower_band'], axis=1)
    
    return data

@indicator_wrapper
def calculate_ichimoku(data, tenkan_period=9, kijun_period=26, senkou_b_period=52, displacement=26):
    """
    Calculate Ichimoku Cloud indicator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        tenkan_period (int): Tenkan-sen (Conversion Line) period
        kijun_period (int): Kijun-sen (Base Line) period
        senkou_b_period (int): Senkou Span B period
        displacement (int): Displacement for Senkou Span A and B (Kumo/Cloud)
        
    Returns:
        pd.DataFrame: DataFrame with Ichimoku Cloud columns added
    """
    # Check if required columns exist
    required_columns = ['high', 'low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for Ichimoku Cloud: {missing_columns}")
        return data
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
    high_tenkan = data['high'].rolling(window=tenkan_period).max()
    low_tenkan = data['low'].rolling(window=tenkan_period).min()
    data['Tenkan_Sen'] = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
    high_kijun = data['high'].rolling(window=kijun_period).max()
    low_kijun = data['low'].rolling(window=kijun_period).min()
    data['Kijun_Sen'] = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 displaced forward 26 periods
    data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past 52 periods displaced forward 26 periods
    high_senkou = data['high'].rolling(window=senkou_b_period).max()
    low_senkou = data['low'].rolling(window=senkou_b_period).min()
    data['Senkou_Span_B'] = ((high_senkou + low_senkou) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span): Current closing price displaced backwards 26 periods
    data['Chikou_Span'] = data['close'].shift(-displacement)
    
    # Add Ichimoku signals
    data['Ichimoku_Signal'] = 'neutral'
    
    # Bullish signals
    bullish_conditions = (
        (data['close'] > data['Senkou_Span_A']) & 
        (data['close'] > data['Senkou_Span_B']) &
        (data['Tenkan_Sen'] > data['Kijun_Sen'])
    )
    data.loc[bullish_conditions, 'Ichimoku_Signal'] = 'bullish'
    
    # Bearish signals
    bearish_conditions = (
        (data['close'] < data['Senkou_Span_A']) & 
        (data['close'] < data['Senkou_Span_B']) &
        (data['Tenkan_Sen'] < data['Kijun_Sen'])
    )
    data.loc[bearish_conditions, 'Ichimoku_Signal'] = 'bearish'
    
    # TK Cross signals (Tenkan-sen crossing Kijun-sen)
    tk_cross_bullish = (
        (data['Tenkan_Sen'] > data['Kijun_Sen']) & 
        (data['Tenkan_Sen'].shift(1) <= data['Kijun_Sen'].shift(1))
    )
    data.loc[tk_cross_bullish, 'Ichimoku_Signal'] = 'tk_cross_bullish'
    
    tk_cross_bearish = (
        (data['Tenkan_Sen'] < data['Kijun_Sen']) & 
        (data['Tenkan_Sen'].shift(1) >= data['Kijun_Sen'].shift(1))
    )
    data.loc[tk_cross_bearish, 'Ichimoku_Signal'] = 'tk_cross_bearish'
    
    return data

@indicator_wrapper
def calculate_adx(data, period=14):
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for ADX calculation
        
    Returns:
        pd.DataFrame: DataFrame with ADX columns added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for ADX: {missing_columns}")
        return data
    
    # Calculate True Range
    data['TR'] = np.maximum(
        np.maximum(
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift(1))
        ),
        abs(data['low'] - data['close'].shift(1))
    )
    
    # Calculate Directional Movement
    data['DM_Plus'] = np.where(
        (data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
        np.maximum(data['high'] - data['high'].shift(1), 0),
        0
    )
    
    data['DM_Minus'] = np.where(
        (data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
        np.maximum(data['low'].shift(1) - data['low'], 0),
        0
    )
    
    # Calculate smoothed TR and DM
    smoothed_TR = data['TR'].rolling(window=period).sum()
    smoothed_DM_Plus = data['DM_Plus'].rolling(window=period).sum()
    smoothed_DM_Minus = data['DM_Minus'].rolling(window=period).sum()
    
    # Calculate Directional Indicators
    data['DI_Plus'] = 100 * smoothed_DM_Plus / smoothed_TR.replace(0, np.finfo(float).eps)
    data['DI_Minus'] = 100 * smoothed_DM_Minus / smoothed_TR.replace(0, np.finfo(float).eps)
    
    # Calculate Directional Index
    data['DX'] = 100 * abs(data['DI_Plus'] - data['DI_Minus']) / (data['DI_Plus'] + data['DI_Minus']).replace(0, np.finfo(float).eps)
    
    # Calculate ADX
    data['ADX'] = data['DX'].rolling(window=period).mean()
    
    # Add ADX signals
    data['ADX_Signal'] = 'neutral'
    data.loc[data['ADX'] > 25, 'ADX_Signal'] = 'trend'
    data.loc[data['ADX'] < 20, 'ADX_Signal'] = 'range'
    
    # Add trend direction based on DI+ and DI-
    data['ADX_Trend'] = 'neutral'
    data.loc[(data['ADX'] > 25) & (data['DI_Plus'] > data['DI_Minus']), 'ADX_Trend'] = 'bullish'
    data.loc[(data['ADX'] > 25) & (data['DI_Plus'] < data['DI_Minus']), 'ADX_Trend'] = 'bearish'
    
    # Clean up temporary columns
    data = data.drop(['TR', 'DM_Plus', 'DM_Minus', 'DX'], axis=1)
    
    return data

@indicator_wrapper
def calculate_fibonacci_levels(data, window=100):
    """
    Calculate Fibonacci retracement levels based on recent high and low.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        window (int): Lookback window to find recent high and low
        
    Returns:
        pd.DataFrame: DataFrame with Fibonacci levels added
    """
    # Find recent high and low
    for i in range(window, len(data)):
        # Get the lookback period
        window_data = data.iloc[i-window:i]
        
        # Find high and low in the window
        recent_high = window_data['high'].max()
        recent_low = window_data['low'].min()
        
        # Calculate the range
        price_range = recent_high - recent_low
        
        # Calculate Fibonacci retracement levels
        data.loc[data.index[i], 'Fib_0'] = recent_low
        data.loc[data.index[i], 'Fib_0.236'] = recent_low + 0.236 * price_range
        data.loc[data.index[i], 'Fib_0.382'] = recent_low + 0.382 * price_range
        data.loc[data.index[i], 'Fib_0.5'] = recent_low + 0.5 * price_range
        data.loc[data.index[i], 'Fib_0.618'] = recent_low + 0.618 * price_range
        data.loc[data.index[i], 'Fib_0.786'] = recent_low + 0.786 * price_range
        data.loc[data.index[i], 'Fib_1'] = recent_high
        
        # Calculate Fibonacci extension levels
        data.loc[data.index[i], 'Fib_1.272'] = recent_low + 1.272 * price_range
        data.loc[data.index[i], 'Fib_1.618'] = recent_low + 1.618 * price_range
        data.loc[data.index[i], 'Fib_2.618'] = recent_low + 2.618 * price_range
    
    return data

@indicator_wrapper
def calculate_pivot_points(data, method='standard'):
    """
    Calculate various types of pivot points.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        method (str): Pivot point calculation method ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
        
    Returns:
        pd.DataFrame: DataFrame with pivot points added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for Pivot Points: {missing_columns}")
        return data
    
    # For daily pivot points, we need date information
    if 'timestamp' in data.columns:
        data['date'] = pd.to_datetime(data['timestamp']).dt.date
        
        # Group by date and get previous day's data for each day
        grouped = data.groupby('date')
        
        for i, (date, group) in enumerate(grouped):
            if i == 0:  # Skip first day as we don't have previous data
                continue
                
            # Get previous day's data
            prev_date = list(grouped.groups.keys())[i-1]
            prev_day = data[data['date'] == prev_date]
            
            prev_high = prev_day['high'].max()
            prev_low = prev_day['low'].min()
            prev_close = prev_day['close'].iloc[-1]
            
            # Calculate pivot point based on method
            if method == 'standard':
                # Standard pivot points
                pivot = (prev_high + prev_low + prev_close) / 3
                r1 = 2 * pivot - prev_low
                s1 = 2 * pivot - prev_high
                r2 = pivot + (prev_high - prev_low)
                s2 = pivot - (prev_high - prev_low)
                r3 = pivot + 2 * (prev_high - prev_low)
                s3 = pivot - 2 * (prev_high - prev_low)
                
                # Assign to current day
                data.loc[group.index, 'PP'] = pivot
                data.loc[group.index, 'R1'] = r1
                data.loc[group.index, 'S1'] = s1
                data.loc[group.index, 'R2'] = r2
                data.loc[group.index, 'S2'] = s2
                data.loc[group.index, 'R3'] = r3
                data.loc[group.index, 'S3'] = s3
                
            elif method == 'fibonacci':
                # Fibonacci pivot points
                pivot = (prev_high + prev_low + prev_close) / 3
                r1 = pivot + 0.382 * (prev_high - prev_low)
                s1 = pivot - 0.382 * (prev_high - prev_low)
                r2 = pivot + 0.618 * (prev_high - prev_low)
                s2 = pivot - 0.618 * (prev_high - prev_low)
                r3 = pivot + 1.0 * (prev_high - prev_low)
                s3 = pivot - 1.0 * (prev_high - prev_low)
                
                # Assign to current day
                data.loc[group.index, 'PP'] = pivot
                data.loc[group.index, 'R1_Fib'] = r1
                data.loc[group.index, 'S1_Fib'] = s1
                data.loc[group.index, 'R2_Fib'] = r2
                data.loc[group.index, 'S2_Fib'] = s2
                data.loc[group.index, 'R3_Fib'] = r3
                data.loc[group.index, 'S3_Fib'] = s3
                
            elif method == 'woodie':
                # Woodie pivot points
                pivot = (prev_high + prev_low + 2 * prev_close) / 4
                r1 = 2 * pivot - prev_low
                s1 = 2 * pivot - prev_high
                r2 = pivot + (prev_high - prev_low)
                s2 = pivot - (prev_high - prev_low)
                
                # Assign to current day
                data.loc[group.index, 'PP_W'] = pivot
                data.loc[group.index, 'R1_W'] = r1
                data.loc[group.index, 'S1_W'] = s1
                data.loc[group.index, 'R2_W'] = r2
                data.loc[group.index, 'S2_W'] = s2
                
            elif method == 'camarilla':
                # Camarilla pivot points
                pivot = (prev_high + prev_low + prev_close) / 3
                r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
                s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
                r2 = prev_close + 1.1 * (prev_high - prev_low) / 6
                s2 = prev_close - 1.1 * (prev_high - prev_low) / 6
                r3 = prev_close + 1.1 * (prev_high - prev_low) / 4
                s3 = prev_close - 1.1 * (prev_high - prev_low) / 4
                r4 = prev_close + 1.1 * (prev_high - prev_low) / 2
                s4 = prev_close - 1.1 * (prev_high - prev_low) / 2
                
                # Assign to current day
                data.loc[group.index, 'PP_C'] = pivot
                data.loc[group.index, 'R1_C'] = r1
                data.loc[group.index, 'S1_C'] = s1
                data.loc[group.index, 'R2_C'] = r2
                data.loc[group.index, 'S2_C'] = s2
                data.loc[group.index, 'R3_C'] = r3
                data.loc[group.index, 'S3_C'] = s3
                data.loc[group.index, 'R4_C'] = r4
                data.loc[group.index, 'S4_C'] = s4
                
            elif method == 'demark':
                # Demark pivot points
                if 'open' not in prev_day.columns:
                    logging.warning("Open price required for Demark pivot points, using close price instead")
                    prev_open = prev_day['close'].iloc[0]
                else:
                    prev_open = prev_day['open'].iloc[0]
                if prev_close < prev_open:
                    x = prev_high + 2 * prev_low + prev_close
                elif prev_close > prev_open:
                    x = 2 * prev_high + prev_low + prev_close
                else:
                    x = prev_high + prev_low + 2 * prev_close
                
                pivot = x / 4
                r1 = x / 2 - prev_low
                s1 = x / 2 - prev_high
                
                # Assign to current day
                data.loc[group.index, 'PP_D'] = pivot
                data.loc[group.index, 'R1_D'] = r1
                data.loc[group.index, 'S1_D'] = s1
        
        # Remove temporary date column
        data = data.drop('date', axis=1)
    
    return data

@indicator_wrapper
def calculate_roc(data, period=12, column='close'):
    """
    Calculate Rate of Change (ROC).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        period (int): Period for ROC calculation
        column (str): Column name to use for calculation
        
    Returns:
        pd.DataFrame: DataFrame with ROC column added
    """
    # Calculate ROC
    data['ROC'] = ((data[column] - data[column].shift(period)) / data[column].shift(period)) * 100
    
    # Add ROC signals
    data['ROC_Signal'] = 'neutral'
    data.loc[data['ROC'] > 0, 'ROC_Signal'] = 'bullish'
    data.loc[data['ROC'] < 0, 'ROC_Signal'] = 'bearish'
    
    # Add overbought/oversold
    data.loc[data['ROC'] > 10, 'ROC_Signal'] = 'overbought'
    data.loc[data['ROC'] < -10, 'ROC_Signal'] = 'oversold'
    
    return data

@indicator_wrapper
def calculate_keltner_channels(data, ema_period=20, atr_period=10, multiplier=2, column='close'):
    """
    Calculate Keltner Channels.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        ema_period (int): Period for EMA calculation
        atr_period (int): Period for ATR calculation
        multiplier (float): Multiplier for ATR
        column (str): Column name to use for EMA calculation
        
    Returns:
        pd.DataFrame: DataFrame with Keltner Channels columns added
    """
    # Calculate EMA
    data[f'EMA_{ema_period}'] = data[column].ewm(span=ema_period, adjust=False).mean()
    
    # Calculate ATR if not already present
    if 'ATR' not in data.columns:
        data = calculate_atr(data, period=atr_period)
    
    # Calculate Keltner Channels
    data['KC_Middle'] = data[f'EMA_{ema_period}']
    data['KC_Upper'] = data['KC_Middle'] + multiplier * data['ATR']
    data['KC_Lower'] = data['KC_Middle'] - multiplier * data['ATR']
    
    # Add Keltner Channels signals
    data['KC_Signal'] = 'neutral'
    data.loc[data[column] > data['KC_Upper'], 'KC_Signal'] = 'overbought'
    data.loc[data[column] < data['KC_Lower'], 'KC_Signal'] = 'oversold'
    
    return data

@indicator_wrapper
def calculate_donchian_channels(data, period=20):
    """
    Calculate Donchian Channels.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for Donchian Channels calculation
        
    Returns:
        pd.DataFrame: DataFrame with Donchian Channels columns added
    """
    # Check if required columns exist
    required_columns = ['high', 'low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for Donchian Channels: {missing_columns}")
        return data
    
    # Calculate Donchian Channels
    data['DC_Upper'] = data['high'].rolling(window=period).max()
    data['DC_Lower'] = data['low'].rolling(window=period).min()
    data['DC_Middle'] = (data['DC_Upper'] + data['DC_Lower']) / 2
    
    # Add Donchian Channels signals
    data['DC_Signal'] = 'neutral'
    data.loc[data['close'] > data['DC_Upper'].shift(1), 'DC_Signal'] = 'breakout_up'
    data.loc[data['close'] < data['DC_Lower'].shift(1), 'DC_Signal'] = 'breakout_down'
    
    return data

@indicator_wrapper
def calculate_on_balance_volume(data):
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with OBV column added
    """
    # Check if required columns exist
    required_columns = ['close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for OBV: {missing_columns}")
        return data
    
    # Calculate price direction
    data['direction'] = np.where(data['close'] > data['close'].shift(1), 1, 
                               np.where(data['close'] < data['close'].shift(1), -1, 0))
    
    # Calculate OBV
    data['OBV'] = (data['direction'] * data['volume']).cumsum()
    
    # Calculate OBV EMA for signal line
    data['OBV_EMA'] = data['OBV'].ewm(span=20, adjust=False).mean()
    
    # Add OBV signals
    data['OBV_Signal'] = 'neutral'
    data.loc[data['OBV'] > data['OBV_EMA'], 'OBV_Signal'] = 'bullish'
    data.loc[data['OBV'] < data['OBV_EMA'], 'OBV_Signal'] = 'bearish'
    
    # Add divergence signals (simplified)
    for i in range(5, len(data)):
        # Check last 5 candles for divergence
        price_trend = data['close'].iloc[i] - data['close'].iloc[i-5]
        obv_trend = data['OBV'].iloc[i] - data['OBV'].iloc[i-5]
        
        # Bullish divergence (price down, OBV up)
        if price_trend < 0 and obv_trend > 0:
            data.loc[data.index[i], 'OBV_Signal'] = 'bullish_divergence'
            
        # Bearish divergence (price up, OBV down)
        elif price_trend > 0 and obv_trend < 0:
            data.loc[data.index[i], 'OBV_Signal'] = 'bearish_divergence'
    
    # Clean up temporary columns
    data = data.drop('direction', axis=1)
    
    return data

@indicator_wrapper
def calculate_cmf(data, period=20):
    """
    Calculate Chaikin Money Flow (CMF).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        period (int): Period for CMF calculation
        
    Returns:
        pd.DataFrame: DataFrame with CMF column added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for CMF: {missing_columns}")
        return data
    
    # Calculate Money Flow Multiplier
    data['MFM'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    
    # Replace infinity and NaN with 0
    data['MFM'] = data['MFM'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate Money Flow Volume
    data['MFV'] = data['MFM'] * data['volume']
    
    # Calculate Chaikin Money Flow
    data['CMF'] = data['MFV'].rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
    
    # Add CMF signals
    data['CMF_Signal'] = 'neutral'
    data.loc[data['CMF'] > 0.1, 'CMF_Signal'] = 'bullish'
    data.loc[data['CMF'] < -0.1, 'CMF_Signal'] = 'bearish'
    
    # Clean up temporary columns
    data = data.drop(['MFM', 'MFV'], axis=1)
    
    return data

@indicator_wrapper
def calculate_mfi(data, period=14):
    """
    Calculate Money Flow Index (MFI).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        period (int): Period for MFI calculation
        
    Returns:
        pd.DataFrame: DataFrame with MFI column added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for MFI: {missing_columns}")
        return data
    
    # Calculate typical price
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate raw money flow
    data['money_flow'] = data['typical_price'] * data['volume']
    
    # Positive and negative money flow
    data['price_change'] = data['typical_price'].diff()
    data['positive_money_flow'] = np.where(data['price_change'] > 0, data['money_flow'], 0)
    data['negative_money_flow'] = np.where(data['price_change'] < 0, data['money_flow'], 0)
    
    # Calculate money flow ratio
    data['positive_money_flow_sum'] = data['positive_money_flow'].rolling(window=period).sum()
    data['negative_money_flow_sum'] = data['negative_money_flow'].rolling(window=period).sum()
    data['money_flow_ratio'] = data['positive_money_flow_sum'] / data['negative_money_flow_sum'].replace(0, np.finfo(float).eps)
    
    # Calculate MFI
    data['MFI'] = 100 - (100 / (1 + data['money_flow_ratio']))
    
    # Add MFI signals
    data['MFI_Signal'] = 'neutral'
    data.loc[data['MFI'] < 20, 'MFI_Signal'] = 'oversold'
    data.loc[data['MFI'] > 80, 'MFI_Signal'] = 'overbought'
    
    # Clean up temporary columns
    data = data.drop(['typical_price', 'money_flow', 'price_change', 'positive_money_flow', 
                     'negative_money_flow', 'positive_money_flow_sum', 'negative_money_flow_sum', 
                     'money_flow_ratio'], axis=1)
    
    return data

@indicator_wrapper
def calculate_parabolic_sar(data, step=0.02, max_step=0.2):
    """
    Calculate Parabolic SAR.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        step (float): Acceleration factor step size
        max_step (float): Maximum acceleration factor
        
    Returns:
        pd.DataFrame: DataFrame with Parabolic SAR column added
    """
    # Check if required columns exist
    required_columns = ['high', 'low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for Parabolic SAR: {missing_columns}")
        return data
    
    # Initialize Parabolic SAR and trend columns
    data['PSAR'] = np.nan
    data['PSAR_Trend'] = np.nan
    
    # We need at least 2 candles
    if len(data) < 2:
        return data
    
    # Initialize variables
    trend = 1  # 1 for uptrend, -1 for downtrend
    extreme_point = data['high'].iloc[0]
    sar = data['low'].iloc[0]
    acceleration_factor = step
    
    # Calculate Parabolic SAR for each candle
    for i in range(1, len(data)):
        # Previous SAR
        prev_sar = sar
        
        # Current high and low
        current_high = data['high'].iloc[i]
        current_low = data['low'].iloc[i]
        
        # Calculate SAR
        if trend == 1:  # Uptrend
            sar = prev_sar + acceleration_factor * (extreme_point - prev_sar)
            
            # Ensure SAR does not exceed the low of the previous two candles
            sar = min(sar, data['low'].iloc[i-1], data['low'].iloc[max(0, i-2)])
            
            # Check for trend reversal
            if sar > current_low:
                trend = -1
                sar = extreme_point
                extreme_point = current_low
                acceleration_factor = step
            else:
                # Update extreme point and acceleration factor
                if current_high > extreme_point:
                    extreme_point = current_high
                    acceleration_factor = min(acceleration_factor + step, max_step)
        else:  # Downtrend
            sar = prev_sar - acceleration_factor * (prev_sar - extreme_point)
            
            # Ensure SAR does not exceed the high of the previous two candles
            sar = max(sar, data['high'].iloc[i-1], data['high'].iloc[max(0, i-2)])
            
            # Check for trend reversal
            if sar < current_high:
                trend = 1
                sar = extreme_point
                extreme_point = current_high
                acceleration_factor = step
            else:
                # Update extreme point and acceleration factor
                if current_low < extreme_point:
                    extreme_point = current_low
                    acceleration_factor = min(acceleration_factor + step, max_step)
        
        # Store SAR and trend
        data.loc[data.index[i], 'PSAR'] = sar
        data.loc[data.index[i], 'PSAR_Trend'] = 'uptrend' if trend == 1 else 'downtrend'
    
    # Add PSAR signals
    data['PSAR_Signal'] = 'neutral'
    
    # Bullish signal (trend changes from downtrend to uptrend)
    bullish_conditions = (
        (data['PSAR_Trend'] == 'uptrend') & 
        (data['PSAR_Trend'].shift(1) == 'downtrend')
    )
    data.loc[bullish_conditions, 'PSAR_Signal'] = 'bullish'
    
    # Bearish signal (trend changes from uptrend to downtrend)
    bearish_conditions = (
        (data['PSAR_Trend'] == 'downtrend') & 
        (data['PSAR_Trend'].shift(1) == 'uptrend')
    )
    data.loc[bearish_conditions, 'PSAR_Signal'] = 'bearish'
    
    return data

@indicator_wrapper
def calculate_trix(data, period=15, signal_period=9, column='close'):
    """
    Calculate TRIX indicator.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        period (int): Period for TRIX calculation
        signal_period (int): Period for signal line
        column (str): Column name to use for calculation
        
    Returns:
        pd.DataFrame: DataFrame with TRIX columns added
    """
    # Calculate triple EMA
    ema1 = data[column].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    # Calculate TRIX
    data['TRIX'] = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    
    # Calculate signal line
    data['TRIX_Signal'] = data['TRIX'].ewm(span=signal_period, adjust=False).mean()
    
    # Add TRIX crossover signals
    data['TRIX_Crossover'] = 'neutral'
    
    # Bullish crossover (TRIX crosses above signal line)
    bullish_conditions = (
        (data['TRIX'] > data['TRIX_Signal']) & 
        (data['TRIX'].shift(1) <= data['TRIX_Signal'].shift(1))
    )
    data.loc[bullish_conditions, 'TRIX_Crossover'] = 'bullish'
    
    # Bearish crossover (TRIX crosses below signal line)
    bearish_conditions = (
        (data['TRIX'] < data['TRIX_Signal']) & 
        (data['TRIX'].shift(1) >= data['TRIX_Signal'].shift(1))
    )
    data.loc[bearish_conditions, 'TRIX_Crossover'] = 'bearish'
    
    # Add zero-line crossover signals
    data['TRIX_ZeroLine'] = 'neutral'
    
    # Bullish zero-line crossover (TRIX crosses above zero)
    bullish_zero_conditions = (
        (data['TRIX'] > 0) & 
        (data['TRIX'].shift(1) <= 0)
    )
    data.loc[bullish_zero_conditions, 'TRIX_ZeroLine'] = 'bullish'
    
    # Bearish zero-line crossover (TRIX crosses below zero)
    bearish_zero_conditions = (
        (data['TRIX'] < 0) & 
        (data['TRIX'].shift(1) >= 0)
    )
    data.loc[bearish_zero_conditions, 'TRIX_ZeroLine'] = 'bearish'
    
    return data

@indicator_wrapper
def calculate_williams_r(data, period=14):
    """
    Calculate Williams %R indicator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for Williams %R calculation
        
    Returns:
        pd.DataFrame: DataFrame with Williams %R column added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for Williams %R: {missing_columns}")
        return data
    
    # Calculate highest high and lowest low over the period
    highest_high = data['high'].rolling(window=period).max()
    lowest_low = data['low'].rolling(window=period).min()
    
    # Calculate Williams %R
    data['Williams_R'] = -100 * (highest_high - data['close']) / (highest_high - lowest_low)
    
    # Add Williams %R signals
    data['Williams_R_Signal'] = 'neutral'
    data.loc[data['Williams_R'] < -80, 'Williams_R_Signal'] = 'oversold'
    data.loc[data['Williams_R'] > -20, 'Williams_R_Signal'] = 'overbought'
    
    return data

@indicator_wrapper
def calculate_awesome_oscillator(data, fast_period=5, slow_period=34):
    """
    Calculate Awesome Oscillator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        fast_period (int): Period for fast SMA
        slow_period (int): Period for slow SMA
        
    Returns:
        pd.DataFrame: DataFrame with Awesome Oscillator column added
    """
    # Check if required columns exist
    required_columns = ['high', 'low']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for Awesome Oscillator: {missing_columns}")
        return data
    
    # Calculate median price
    data['median_price'] = (data['high'] + data['low']) / 2
    
    # Calculate SMAs of median price
    fast_sma = data['median_price'].rolling(window=fast_period).mean()
    slow_sma = data['median_price'].rolling(window=slow_period).mean()
    
    # Calculate Awesome Oscillator
    data['AO'] = fast_sma - slow_sma
    
    # Add Awesome Oscillator signals
    data['AO_Signal'] = 'neutral'
    
    # Zero-line crossover
    data.loc[(data['AO'] > 0) & (data['AO'].shift(1) <= 0), 'AO_Signal'] = 'bullish_zero'
    data.loc[(data['AO'] < 0) & (data['AO'].shift(1) >= 0), 'AO_Signal'] = 'bearish_zero'
    
    # Saucer signal
    for i in range(3, len(data)):
        # Bullish saucer (3 consecutive bars below zero, all AO red, then green bar)
        if (data['AO'].iloc[i-3] < 0 and 
            data['AO'].iloc[i-2] < 0 and 
            data['AO'].iloc[i-1] < 0 and 
            data['AO'].iloc[i] < 0 and 
            data['AO'].iloc[i-2] < data['AO'].iloc[i-3] and 
            data['AO'].iloc[i-1] < data['AO'].iloc[i-2] and 
            data['AO'].iloc[i] > data['AO'].iloc[i-1]):
            data.loc[data.index[i], 'AO_Signal'] = 'bullish_saucer'
        
        # Bearish saucer (3 consecutive bars above zero, all AO green, then red bar)
        elif (data['AO'].iloc[i-3] > 0 and 
              data['AO'].iloc[i-2] > 0 and 
              data['AO'].iloc[i-1] > 0 and 
              data['AO'].iloc[i] > 0 and 
              data['AO'].iloc[i-2] > data['AO'].iloc[i-3] and 
              data['AO'].iloc[i-1] > data['AO'].iloc[i-2] and 
              data['AO'].iloc[i] < data['AO'].iloc[i-1]):
            data.loc[data.index[i], 'AO_Signal'] = 'bearish_saucer'
    
    # Twin peaks signal
    for i in range(5, len(data)):
        # Bullish twin peaks (two peaks below zero)
        if (data['AO'].iloc[i-5] < 0 and 
            data['AO'].iloc[i-4] < data['AO'].iloc[i-5] and 
            data['AO'].iloc[i-3] < data['AO'].iloc[i-4] and 
            data['AO'].iloc[i-2] > data['AO'].iloc[i-3] and 
            data['AO'].iloc[i-1] > data['AO'].iloc[i-2] and 
            data['AO'].iloc[i] < data['AO'].iloc[i-1] and 
            data['AO'].iloc[i-3] < data['AO'].iloc[i] and 
            data['AO'].iloc[i] < 0):
            data.loc[data.index[i], 'AO_Signal'] = 'bullish_twin_peaks'
        
        # Bearish twin peaks (two peaks above zero)
        elif (data['AO'].iloc[i-5] > 0 and 
              data['AO'].iloc[i-4] > data['AO'].iloc[i-5] and 
              data['AO'].iloc[i-3] > data['AO'].iloc[i-4] and 
              data['AO'].iloc[i-2] < data['AO'].iloc[i-3] and 
              data['AO'].iloc[i-1] < data['AO'].iloc[i-2] and 
              data['AO'].iloc[i] > data['AO'].iloc[i-1] and 
              data['AO'].iloc[i-3] > data['AO'].iloc[i] and 
              data['AO'].iloc[i] > 0):
            data.loc[data.index[i], 'AO_Signal'] = 'bearish_twin_peaks'
    
    # Clean up temporary column
    data = data.drop('median_price', axis=1)
    
    return data

@indicator_wrapper
def calculate_cci(data, period=20):
    """
    Calculate Commodity Channel Index (CCI).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for CCI calculation
        
    Returns:
        pd.DataFrame: DataFrame with CCI column added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for CCI: {missing_columns}")
        return data
    
    # Calculate typical price
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate simple moving average of typical price
    data['tp_sma'] = data['typical_price'].rolling(window=period).mean()
    
    # Calculate mean deviation
    data['tp_md'] = data['typical_price'].rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    
    # Calculate CCI
    data['CCI'] = (data['typical_price'] - data['tp_sma']) / (0.015 * data['tp_md'])
    
    # Add CCI signals
    data['CCI_Signal'] = 'neutral'
    data.loc[data['CCI'] > 100, 'CCI_Signal'] = 'overbought'
    data.loc[data['CCI'] < -100, 'CCI_Signal'] = 'oversold'
    
    # Zero-line crossover
    data.loc[(data['CCI'] > 0) & (data['CCI'].shift(1) <= 0), 'CCI_Signal'] = 'bullish_zero'
    data.loc[(data['CCI'] < 0) & (data['CCI'].shift(1) >= 0), 'CCI_Signal'] = 'bearish_zero'
    
    # Clean up temporary columns
    data = data.drop(['typical_price', 'tp_sma', 'tp_md'], axis=1)
    
    return data

@indicator_wrapper
def calculate_volatility_index(data, period=20, atr_period=14):
    """
    Calculate Volatility Index.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for volatility calculation
        atr_period (int): Period for ATR calculation
        
    Returns:
        pd.DataFrame: DataFrame with Volatility Index column added
    """
    # Calculate ATR if not already available
    if 'ATR' not in data.columns:
        data = calculate_atr(data, period=atr_period)
    
    # Calculate Close Price Change
    data['close_change'] = data['close'].pct_change() * 100
    
    # Calculate historical volatility (standard deviation of close price changes)
    data['HV'] = data['close_change'].rolling(window=period).std()
    
    # Calculate ATR Volatility (ATR as percentage of price)
    data['ATR_Volatility'] = data['ATR'] / data['close'] * 100
    
    # Calculate Volatility Index (average of HV and ATR Volatility)
    data['Volatility_Index'] = (data['HV'] + data['ATR_Volatility']) / 2
    
    # Add volatility regime signals
    data['Volatility_Regime'] = 'normal'
    
    # Calculate long-term average volatility
    long_period = period * 3  # 3x the regular period
    data['LT_Volatility'] = data['Volatility_Index'].rolling(window=long_period).mean()
    
    # Determine volatility regimes
    data.loc[data['Volatility_Index'] > data['LT_Volatility'] * 1.5, 'Volatility_Regime'] = 'high'
    data.loc[data['Volatility_Index'] < data['LT_Volatility'] * 0.5, 'Volatility_Regime'] = 'low'
    
    # Clean up temporary columns
    data = data.drop(['close_change', 'HV', 'LT_Volatility'], axis=1)
    
    return data

@indicator_wrapper
def calculate_volume_oscillator(data, fast_period=5, slow_period=10):
    """
    Calculate Volume Oscillator.
    
    Args:
        data (pd.DataFrame): DataFrame with volume data
        fast_period (int): Period for fast EMA
        slow_period (int): Period for slow EMA
        
    Returns:
        pd.DataFrame: DataFrame with Volume Oscillator column added
    """
    # Check if volume column exists
    if 'volume' not in data.columns:
        logging.error("Volume column not found in dataframe")
        return data
    
    # Calculate fast and slow EMAs of volume
    data['Volume_EMA_Fast'] = data['volume'].ewm(span=fast_period, adjust=False).mean()
    data['Volume_EMA_Slow'] = data['volume'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate Volume Oscillator (percentage difference)
    data['Volume_Oscillator'] = ((data['Volume_EMA_Fast'] - data['Volume_EMA_Slow']) / 
                               data['Volume_EMA_Slow'] * 100)
    
    # Add Volume Oscillator signals
    data['Volume_Oscillator_Signal'] = 'neutral'
    
    # Zero-line crossover
    data.loc[(data['Volume_Oscillator'] > 0) & 
           (data['Volume_Oscillator'].shift(1) <= 0), 'Volume_Oscillator_Signal'] = 'bullish'
    data.loc[(data['Volume_Oscillator'] < 0) & 
           (data['Volume_Oscillator'].shift(1) >= 0), 'Volume_Oscillator_Signal'] = 'bearish'
    
    # Clean up temporary columns
    data = data.drop(['Volume_EMA_Fast', 'Volume_EMA_Slow'], axis=1)
    
    return data

@indicator_wrapper
def calculate_elder_ray(data, period=13):
    """
    Calculate Elder Ray indicator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Period for EMA calculation
        
    Returns:
        pd.DataFrame: DataFrame with Elder Ray columns added
    """
    # Check if required columns exist
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns for Elder Ray: {missing_columns}")
        return data
    
    # Calculate EMA
    data['EMA'] = data['close'].ewm(span=period, adjust=False).mean()
    
    # Calculate Bull Power and Bear Power
    data['Bull_Power'] = data['high'] - data['EMA']
    data['Bear_Power'] = data['low'] - data['EMA']
    
    # Add Elder Ray signals
    data['Elder_Ray_Signal'] = 'neutral'
    
    # Bullish signal (Bull Power positive and increasing, Bear Power negative but rising)
    bullish_conditions = (
        (data['Bull_Power'] > 0) & 
        (data['Bull_Power'] > data['Bull_Power'].shift(1)) & 
        (data['Bear_Power'] < 0) & 
        (data['Bear_Power'] > data['Bear_Power'].shift(1))
    )
    data.loc[bullish_conditions, 'Elder_Ray_Signal'] = 'bullish'
    
    # Bearish signal (Bear Power negative and decreasing, Bull Power positive but falling)
    bearish_conditions = (
        (data['Bear_Power'] < 0) & 
        (data['Bear_Power'] < data['Bear_Power'].shift(1)) & 
        (data['Bull_Power'] > 0) & 
        (data['Bull_Power'] < data['Bull_Power'].shift(1))
    )
    data.loc[bearish_conditions, 'Elder_Ray_Signal'] = 'bearish'
    
    return data

@indicator_wrapper
def calculate_chande_momentum_oscillator(data, period=14, column='close'):
    """
    Calculate Chande Momentum Oscillator (CMO).
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        period (int): Period for CMO calculation
        column (str): Column name to use for calculation
        
    Returns:
        pd.DataFrame: DataFrame with CMO column added
    """
    # Calculate price changes
    data['price_change'] = data[column].diff()
    
    # Calculate positive and negative price changes
    data['up_sum'] = np.where(data['price_change'] > 0, data['price_change'], 0)
    data['down_sum'] = np.where(data['price_change'] < 0, -data['price_change'], 0)
    
    # Calculate sum of up and down changes over period
    data['up_sum_period'] = data['up_sum'].rolling(window=period).sum()
    data['down_sum_period'] = data['down_sum'].rolling(window=period).sum()
    
    # Calculate CMO
    data['CMO'] = 100 * ((data['up_sum_period'] - data['down_sum_period']) / 
                       (data['up_sum_period'] + data['down_sum_period']))
    
    # Add CMO signals
    data['CMO_Signal'] = 'neutral'
    data.loc[data['CMO'] > 50, 'CMO_Signal'] = 'overbought'
    data.loc[data['CMO'] < -50, 'CMO_Signal'] = 'oversold'
    
    # Zero-line crossover
    data.loc[(data['CMO'] > 0) & (data['CMO'].shift(1) <= 0), 'CMO_Signal'] = 'bullish_cross'
    data.loc[(data['CMO'] < 0) & (data['CMO'].shift(1) >= 0), 'CMO_Signal'] = 'bearish_cross'
    
    # Clean up temporary columns
    data = data.drop(['price_change', 'up_sum', 'down_sum', 'up_sum_period', 'down_sum_period'], axis=1)
    
    return data

def generate_combined_signals(data):
    """
    Generate combined trading signals based on multiple indicators.
    
    Args:
        data (pd.DataFrame): DataFrame with individual indicator signals.
        
    Returns:
        pd.DataFrame: DataFrame with combined signal column added.
    """
    try:
        # Make a copy of the data
        df = data.copy()
        
        # Initialize combined signal column
        df['Combined_Signal'] = 'neutral'
        
        # Create signal strength columns
        df['Bullish_Strength'] = 0
        df['Bearish_Strength'] = 0
        
        # Signal mapping for consistent processing
        signal_mapping = {
            'RSI_Signal': {
                'oversold': ('bullish', 1),
                'overbought': ('bearish', 1)
            },
            'MACD_Signal': {
                'bullish': ('bullish', 1),
                'bearish': ('bearish', 1)
            },
            'BB_Signal': {
                'oversold': ('bullish', 1),
                'overbought': ('bearish', 1)
            },
            'Stochastic_Signal': {
                'oversold': ('bullish', 1),
                'bullish_cross': ('bullish', 1),
                'overbought': ('bearish', 1),
                'bearish_cross': ('bearish', 1)
            },
            'EMA_Cross_Signal': {
                'bullish': ('bullish', 1),
                'bearish': ('bearish', 1)
            },
            'SuperTrend_Signal': {
                'bullish': ('bullish', 2),  # Higher weight
                'bearish': ('bearish', 2)
            },
            'PSAR_Signal': {
                'bullish': ('bullish', 1.5),
                'bearish': ('bearish', 1.5)
            },
            'VWAP_Signal': {
                'bullish': ('bullish', 1),
                'bearish': ('bearish', 1)
            },
            'Ichimoku_Signal': {
                'bullish': ('bullish', 2),
                'tk_cross_bullish': ('bullish', 1.5),
                'bearish': ('bearish', 2),
                'tk_cross_bearish': ('bearish', 1.5)
            },
            'ADX_Trend': {
                'bullish': ('bullish', 1.5),
                'bearish': ('bearish', 1.5)
            },
            'OBV_Signal': {
                'bullish': ('bullish', 1),
                'bullish_divergence': ('bullish', 2),
                'bearish': ('bearish', 1),
                'bearish_divergence': ('bearish', 2)
            },
            'MFI_Signal': {
                'oversold': ('bullish', 1),
                'overbought': ('bearish', 1)
            },
            'CCI_Signal': {
                'oversold': ('bullish', 1),
                'bullish_zero': ('bullish', 1),
                'overbought': ('bearish', 1),
                'bearish_zero': ('bearish', 1)
            },
            'AO_Signal': {
                'bullish_zero': ('bullish', 1),
                'bullish_saucer': ('bullish', 1.5),
                'bullish_twin_peaks': ('bullish', 2),
                'bearish_zero': ('bearish', 1),
                'bearish_saucer': ('bearish', 1.5),
                'bearish_twin_peaks': ('bearish', 2)
            },
            'Williams_R_Signal': {
                'oversold': ('bullish', 1),
                'overbought': ('bearish', 1)
            },
            'Elder_Ray_Signal': {
                'bullish': ('bullish', 1),
                'bearish': ('bearish', 1)
            },
            'CMO_Signal': {
                'oversold': ('bullish', 1),
                'bullish_cross': ('bullish', 1),
                'overbought': ('bearish', 1),
                'bearish_cross': ('bearish', 1)
            }
        }
        
        # Process each indicator signal using the mapping
        for indicator, signal_map in signal_mapping.items():
            if indicator in df.columns:
                for signal_value, (direction, weight) in signal_map.items():
                    if direction == 'bullish':
                        df.loc[df[indicator] == signal_value, 'Bullish_Strength'] += weight
                    else:  # bearish
                        df.loc[df[indicator] == signal_value, 'Bearish_Strength'] += weight
        
        # Set combined signal based on strength and thresholds
        # Strong bullish: At least 3 bullish points and more bullish than bearish points
        df.loc[(df['Bullish_Strength'] >= 3) & 
             (df['Bullish_Strength'] > df['Bearish_Strength']), 'Combined_Signal'] = 'strong_bullish'
        
        # Moderate bullish: At least 2 bullish points and more bullish than bearish points
        df.loc[(df['Bullish_Strength'] >= 2) & 
             (df['Bullish_Strength'] > df['Bearish_Strength']) & 
             (df['Combined_Signal'] == 'neutral'), 'Combined_Signal'] = 'moderate_bullish'
        
        # Strong bearish: At least 3 bearish points and more bearish than bullish points
        df.loc[(df['Bearish_Strength'] >= 3) & 
             (df['Bearish_Strength'] > df['Bullish_Strength']), 'Combined_Signal'] = 'strong_bearish'
        
        # Moderate bearish: At least 2 bearish points and more bearish than bullish points
        df.loc[(df['Bearish_Strength'] >= 2) & 
             (df['Bearish_Strength'] > df['Bullish_Strength']) & 
             (df['Combined_Signal'] == 'neutral'), 'Combined_Signal'] = 'moderate_bearish'
        
        # Clean up temporary columns
        df = df.drop(['Bullish_Strength', 'Bearish_Strength'], axis=1)
        
        return df
        
    except Exception as e:
        logging.error(f"Error generating combined signals: {e}")
        return data  # Return original data in case of error

def calculate_all_indicators(data, config=None):
    """
    Calculate all technical indicators using a pipeline approach.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data.
        config (dict): Configuration dictionary with indicator settings.
        
    Returns:
        pd.DataFrame: DataFrame with all indicators added.
    """
    # Check if data is not empty
    if data is None or len(data) == 0:
        logging.error("Empty data provided to calculate_all_indicators")
        return data
    
    # Make a copy of the data
    df = data.copy()
    
    # Load configuration
    if config is None:
        config = {}
    
    indicator_config = config.get('indicators', {})
    
    # Define indicator pipeline
    indicators_pipeline = [
        # (function, enabled_key, params_dict)
        (calculate_sma, 'sma', {
            'period': indicator_config.get('sma', {}).get('period', 20),
            'column': 'close'
        }),
        (calculate_ema, 'ema', {
            'periods': indicator_config.get('ema', {}).get('periods', [9, 21, 50, 200]),
            'column': 'close'
        }),
        (calculate_rsi, 'rsi', {
            'period': indicator_config.get('rsi', {}).get('period', 14),
            'column': 'close'
        }),
        (calculate_macd, 'macd', {
            'fast_period': indicator_config.get('macd', {}).get('fast_period', 12),
            'slow_period': indicator_config.get('macd', {}).get('slow_period', 26),
            'signal_period': indicator_config.get('macd', {}).get('signal_period', 9)
        }),
        (calculate_bollinger_bands, 'bollinger_bands', {
            'window': indicator_config.get('bollinger_bands', {}).get('period', 20),
            'num_std': indicator_config.get('bollinger_bands', {}).get('std_dev', 2)
        }),
        (calculate_stochastic_oscillator, 'stochastic', {
            'k_period': indicator_config.get('stochastic', {}).get('k_period', 14),
            'd_period': indicator_config.get('stochastic', {}).get('d_period', 3),
            'smooth_k': indicator_config.get('stochastic', {}).get('smooth_k', 3)
        }),
        (calculate_atr, 'atr', {
            'period': indicator_config.get('atr', {}).get('period', 14)
        }),
        (calculate_supertrend, 'supertrend', {
            'period': indicator_config.get('supertrend', {}).get('period', 10),
            'multiplier': indicator_config.get('supertrend', {}).get('multiplier', 3.0)
        }),
        (calculate_adx, 'adx', {
            'period': indicator_config.get('adx', {}).get('period', 14)
        }),
        (calculate_williams_r, 'williams_r', {
            'period': indicator_config.get('williams_r', {}).get('period', 14)
        }),
        (calculate_cci, 'cci', {
            'period': indicator_config.get('cci', {}).get('period', 20)
        }),
        (calculate_vwap, 'vwap', {}),
        (calculate_on_balance_volume, 'obv', {}),
        (calculate_mfi, 'mfi', {
            'period': indicator_config.get('mfi', {}).get('period', 14)
        }),
        (calculate_parabolic_sar, 'psar', {
            'step': indicator_config.get('psar', {}).get('step', 0.02),
            'max_step': indicator_config.get('psar', {}).get('max_step', 0.2)
        }),
        (calculate_chande_momentum_oscillator, 'cmo', {
            'period': indicator_config.get('cmo', {}).get('period', 14)
        }),
        (calculate_awesome_oscillator, 'ao', {
            'fast_period': indicator_config.get('ao', {}).get('fast_period', 5),
            'slow_period': indicator_config.get('ao', {}).get('slow_period', 34)
        }),
        (calculate_ichimoku, 'ichimoku', {
            'tenkan_period': indicator_config.get('ichimoku', {}).get('tenkan_period', 9),
            'kijun_period': indicator_config.get('ichimoku', {}).get('kijun_period', 26),
            'senkou_b_period': indicator_config.get('ichimoku', {}).get('senkou_b_period', 52),
            'displacement': indicator_config.get('ichimoku', {}).get('displacement', 26)
        }),
    ]

    cleanup_intermediate = config.get('cleanup_intermediate', True)
    
    # Apply each indicator function if enabled
    for func, key, params in indicators_pipeline:
        # Check if indicator is enabled (default to True if not specified)
        if indicator_config.get(key, {}).get('enabled', True):
            try:
                df = func(df, **params)
                logging.debug(f"Applied {func.__name__} to data")
            except Exception as e:
                logging.error(f"Error applying {func.__name__}: {e}")
        else:
            logging.debug(f"Skipped {func.__name__} (disabled in config)")
    
    # Generate combined signals
    df = generate_combined_signals(df)

    if cleanup_intermediate:
        intermediate_columns = [
            'median_price', 'tp_sma', 'tp_md', 'price_change', 
            'up_sum', 'down_sum', 'up_sum_period', 'down_sum_period',
            # Add other intermediate columns here
        ]
        for col in intermediate_columns:
            if col in df.columns:
                df = df.drop(col, axis=1)
    
    return df

def detect_market_regime(data, window=50):
    """
    Detect market regime (trending, ranging, volatile).
    
    Args:
        data (pd.DataFrame): DataFrame with price and indicator data
        window (int): Window for regime detection
        
    Returns:
        str: Market regime ('trending', 'ranging', 'volatile', or 'unknown')
    """
    try:
        # Ensure we have required indicators
        if 'ATR' not in data.columns:
            data = calculate_atr(data)
            
        if 'EMA_9' not in data.columns or 'EMA_50' not in data.columns:
            data = calculate_ema(data, periods=[9, 50])
        
        # Get the most recent data point
        last_row = data.iloc[-1]
        
        # Calculate volatility (ATR as percentage of price)
        volatility = last_row['ATR'] / last_row['close'] * 100
        
        # Calculate trend strength (distance between EMAs as percentage)
        if 'EMA_9' in last_row and 'EMA_50' in last_row:
            trend_strength = abs(last_row['EMA_9'] - last_row['EMA_50']) / last_row['EMA_50'] * 100
        else:
            trend_strength = 0
        
        # Calculate price range over window
        if len(data) >= window:
            window_data = data.iloc[-window:]
            price_range = (window_data['high'].max() - window_data['low'].min()) / last_row['close'] * 100
        else:
            price_range = 0
        
        # Determine regime
        if volatility > 5:  # High volatility threshold
            return 'volatile'
        elif trend_strength > 2:  # Strong trend threshold
            return 'trending'
        elif price_range < 10:  # Range-bound threshold
            return 'ranging'
        else:
            return 'ranging'  # Default to ranging if uncertain
            
    except Exception as e:
        logging.error(f"Error detecting market regime: {e}")
        return 'unknown'