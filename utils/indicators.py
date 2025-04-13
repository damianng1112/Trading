import pandas as pd
import numpy as np
import logging

def calculate_rsi(data, period=14, column="close"):
    """
    Calculate Relative Strength Index (RSI) with error handling.
    
    Args:
        data (pd.DataFrame): DataFrame with price data.
        period (int): Period for RSI calculation.
        column (str): Column name to use for calculation.
        
    Returns:
        pd.DataFrame: DataFrame with RSI column added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if the required column exists
        if column not in df.columns:
            logging.error(f"Column '{column}' not found in dataframe")
            return df
        
        # Calculate price changes
        delta = df[column].diff()
        
        # Create gain and loss series
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss over the specified period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        
        # Calculate RSI
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add RSI oversold/overbought signals
        df['RSI_Signal'] = 'neutral'
        df.loc[df['RSI'] < 30, 'RSI_Signal'] = 'oversold'
        df.loc[df['RSI'] > 70, 'RSI_Signal'] = 'overbought'
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating RSI: {e}")
        return data  # Return original data in case of error

def calculate_macd(data, column="close", fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) with error handling.
    
    Args:
        data (pd.DataFrame): DataFrame with price data.
        column (str): Column name to use for calculation.
        fast_period (int): Fast EMA period.
        slow_period (int): Slow EMA period.
        signal_period (int): Signal line period.
        
    Returns:
        pd.DataFrame: DataFrame with MACD columns added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if the required column exists
        if column not in df.columns:
            logging.error(f"Column '{column}' not found in dataframe")
            return df
        
        # Calculate EMAs
        fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['MACD'] = fast_ema - slow_ema
        
        # Calculate signal line
        df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Add MACD crossover signals
        df['MACD_Signal'] = 'neutral'
        # Bullish crossover (MACD crosses above signal line)
        df.loc[(df['MACD'] > df['Signal_Line']) & 
              (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)), 'MACD_Signal'] = 'bullish'
        # Bearish crossover (MACD crosses below signal line)
        df.loc[(df['MACD'] < df['Signal_Line']) & 
              (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)), 'MACD_Signal'] = 'bearish'
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating MACD: {e}")
        return data  # Return original data in case of error

def calculate_bollinger_bands(data, column="close", window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        data (pd.DataFrame): DataFrame with price data.
        column (str): Column name to use for calculation.
        window (int): Moving average window.
        num_std (float): Number of standard deviations for bands.
        
    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands columns added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if the required column exists
        if column not in df.columns:
            logging.error(f"Column '{column}' not found in dataframe")
            return df
        
        # Calculate the simple moving average
        df['BB_Middle'] = df[column].rolling(window=window).mean()
        
        # Calculate the standard deviation
        rolling_std = df[column].rolling(window=window).std()
        
        # Calculate upper and lower bands
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * num_std)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * num_std)
        
        # Calculate bandwidth and %B
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_PercentB'] = (df[column] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Add Bollinger Band signals
        df['BB_Signal'] = 'neutral'
        df.loc[df[column] > df['BB_Upper'], 'BB_Signal'] = 'overbought'
        df.loc[df[column] < df['BB_Lower'], 'BB_Signal'] = 'oversold'
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating Bollinger Bands: {e}")
        return data  # Return original data in case of error

def calculate_stochastic_oscillator(data, k_period=14, d_period=3, smooth_k=3):
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data.
        k_period (int): K period.
        d_period (int): D period.
        smooth_k (int): K smoothing period.
        
    Returns:
        pd.DataFrame: DataFrame with Stochastic Oscillator columns added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if required columns exist
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns for Stochastic Oscillator: {missing_columns}")
            return df
        
        # Calculate %K
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        df['Stoch_K_Raw'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Apply smoothing to %K if requested
        if smooth_k > 1:
            df['Stoch_K'] = df['Stoch_K_Raw'].rolling(window=smooth_k).mean()
        else:
            df['Stoch_K'] = df['Stoch_K_Raw']
        
        # Calculate %D (signal line)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        
        # Add Stochastic signals
        df['Stochastic_Signal'] = 'neutral'
        df.loc[df['Stoch_K'] < 20, 'Stochastic_Signal'] = 'oversold'
        df.loc[df['Stoch_K'] > 80, 'Stochastic_Signal'] = 'overbought'
        
        # Add crossover signals
        # Bullish crossover (K crosses above D)
        df.loc[(df['Stoch_K'] > df['Stoch_D']) & 
              (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)), 'Stochastic_Signal'] = 'bullish_cross'
        # Bearish crossover (K crosses below D)
        df.loc[(df['Stoch_K'] < df['Stoch_D']) & 
              (df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1)), 'Stochastic_Signal'] = 'bearish_cross'
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating Stochastic Oscillator: {e}")
        return data  # Return original data in case of error

def calculate_ema(data, periods=[9, 21, 50, 200], column='close'):
    """
    Calculate multiple Exponential Moving Averages.
    
    Args:
        data (pd.DataFrame): DataFrame with price data.
        periods (list): List of periods for EMA calculation.
        column (str): Column name to use for calculation.
        
    Returns:
        pd.DataFrame: DataFrame with EMA columns added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if the required column exists
        if column not in df.columns:
            logging.error(f"Column '{column}' not found in dataframe")
            return df
        
        # Calculate EMAs for each period
        for period in periods:
            df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        
        # Add EMA crossover signals if we have at least two EMAs
        if len(periods) >= 2:
            # Sort periods to ensure consistent comparisons (shorter vs longer)
            sorted_periods = sorted(periods)
            
            # For the two shortest periods, calculate crossovers
            short_period = sorted_periods[0]
            medium_period = sorted_periods[1]
            
            df['EMA_Cross_Signal'] = 'neutral'
            
            # Bullish crossover (shorter EMA crosses above longer EMA)
            df.loc[(df[f'EMA_{short_period}'] > df[f'EMA_{medium_period}']) & 
                  (df[f'EMA_{short_period}'].shift(1) <= df[f'EMA_{medium_period}'].shift(1)), 
                  'EMA_Cross_Signal'] = 'bullish'
            
            # Bearish crossover (shorter EMA crosses below longer EMA)
            df.loc[(df[f'EMA_{short_period}'] < df[f'EMA_{medium_period}']) & 
                  (df[f'EMA_{short_period}'].shift(1) >= df[f'EMA_{medium_period}'].shift(1)), 
                  'EMA_Cross_Signal'] = 'bearish'
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating EMAs: {e}")
        return data  # Return original data in case of error

def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data.
        period (int): Period for ATR calculation.
        
    Returns:
        pd.DataFrame: DataFrame with ATR column added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if required columns exist
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns for ATR: {missing_columns}")
            return df
        
        # Calculate True Range
        df['TR1'] = abs(df['high'] - df['low'])
        df['TR2'] = abs(df['high'] - df['close'].shift(1))
        df['TR3'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        
        # Calculate ATR
        df['ATR'] = df['TR'].rolling(window=period).mean()
        
        # Clean up temporary columns
        df = df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1)
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")
        return data  # Return original data in case of error

def calculate_vwap(data):
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data.
        
    Returns:
        pd.DataFrame: DataFrame with VWAP column added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if required columns exist
        required_columns = ['high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns for VWAP: {missing_columns}")
            return df
        
        # Calculate typical price
        df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP components
        df['VP'] = df['TypicalPrice'] * df['volume']
        
        # Calculate cumulative values
        df['CumulativeVP'] = df['VP'].cumsum()
        df['CumulativeVolume'] = df['volume'].cumsum()
        
        # Calculate VWAP
        df['VWAP'] = df['CumulativeVP'] / df['CumulativeVolume']
        
        # Clean up temporary columns
        df = df.drop(['TypicalPrice', 'VP', 'CumulativeVP', 'CumulativeVolume'], axis=1)
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating VWAP: {e}")
        return data  # Return original data in case of error

def calculate_supertrend(data, period=10, multiplier=3.0):
    """
    Calculate SuperTrend indicator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data.
        period (int): ATR period.
        multiplier (float): ATR multiplier.
        
    Returns:
        pd.DataFrame: DataFrame with SuperTrend columns added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if required columns exist
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns for SuperTrend: {missing_columns}")
            return df
        
        # Calculate ATR
        df = calculate_atr(df, period=period)
        
        # Calculate basic upper and lower bands
        df['basic_upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['ATR'])
        df['basic_lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['ATR'])
        
        # Initialize SuperTrend columns
        df['SuperTrend_Upper'] = df['basic_upper_band']
        df['SuperTrend_Lower'] = df['basic_lower_band']
        df['SuperTrend'] = df['close'].copy()
        df['SuperTrend_Direction'] = 1  # 1 for uptrend, -1 for downtrend
        
        # Calculate SuperTrend
        for i in range(1, len(df)):
            # Update upper band
            if df['basic_upper_band'].iloc[i] < df['SuperTrend_Upper'].iloc[i-1] or df['close'].iloc[i-1] > df['SuperTrend_Upper'].iloc[i-1]:
                df.loc[df.index[i], 'SuperTrend_Upper'] = df['basic_upper_band'].iloc[i]
            else:
                df.loc[df.index[i], 'SuperTrend_Upper'] = df['SuperTrend_Upper'].iloc[i-1]
                
            # Update lower band
            if df['basic_lower_band'].iloc[i] > df['SuperTrend_Lower'].iloc[i-1] or df['close'].iloc[i-1] < df['SuperTrend_Lower'].iloc[i-1]:
                df.loc[df.index[i], 'SuperTrend_Lower'] = df['basic_lower_band'].iloc[i]
            else:
                df.loc[df.index[i], 'SuperTrend_Lower'] = df['SuperTrend_Lower'].iloc[i-1]
                
            # Determine trend direction
            if df['close'].iloc[i-1] <= df['SuperTrend'].iloc[i-1] and df['close'].iloc[i] > df['SuperTrend_Upper'].iloc[i]:
                # Trend changes to uptrend
                df.loc[df.index[i], 'SuperTrend_Direction'] = 1
                df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend_Lower'].iloc[i]
            elif df['close'].iloc[i-1] >= df['SuperTrend'].iloc[i-1] and df['close'].iloc[i] < df['SuperTrend_Lower'].iloc[i]:
                # Trend changes to downtrend
                df.loc[df.index[i], 'SuperTrend_Direction'] = -1
                df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend_Upper'].iloc[i]
            else:
                # Trend continues
                df.loc[df.index[i], 'SuperTrend_Direction'] = df['SuperTrend_Direction'].iloc[i-1]
                if df['SuperTrend_Direction'].iloc[i] == 1:
                    df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend_Lower'].iloc[i]
                else:
                    df.loc[df.index[i], 'SuperTrend'] = df['SuperTrend_Upper'].iloc[i]
        
        # Add SuperTrend signals
        df['SuperTrend_Signal'] = 'neutral'
        df.loc[df['SuperTrend_Direction'] == 1, 'SuperTrend_Signal'] = 'bullish'
        df.loc[df['SuperTrend_Direction'] == -1, 'SuperTrend_Signal'] = 'bearish'
        
        # Clean up temporary columns
        df = df.drop(['basic_upper_band', 'basic_lower_band'], axis=1)
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating SuperTrend: {e}")
        return data  # Return original data in case of error

def calculate_all_indicators(data):
    """
    Calculate all technical indicators.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data.
        
    Returns:
        pd.DataFrame: DataFrame with all indicators added.
    """
    # Check if data is not empty
    if data is None or len(data) == 0:
        logging.error("Empty data provided to calculate_all_indicators")
        return data
    
    # Make a copy of the data
    df = data.copy()
    
    # Apply each indicator function
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_stochastic_oscillator(df)
    df = calculate_ema(df)
    df = calculate_atr(df)
    df = calculate_vwap(df)
    df = calculate_supertrend(df)
    
    # Generate combined signals
    df = generate_combined_signals(df)
    
    return df

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
        
        # Check RSI
        if 'RSI_Signal' in df.columns:
            df.loc[df['RSI_Signal'] == 'oversold', 'Bullish_Strength'] += 1
            df.loc[df['RSI_Signal'] == 'overbought', 'Bearish_Strength'] += 1
            
        # Check MACD
        if 'MACD_Signal' in df.columns:
            df.loc[df['MACD_Signal'] == 'bullish', 'Bullish_Strength'] += 1
            df.loc[df['MACD_Signal'] == 'bearish', 'Bearish_Strength'] += 1
            
        # Check Bollinger Bands
        if 'BB_Signal' in df.columns:
            df.loc[df['BB_Signal'] == 'oversold', 'Bullish_Strength'] += 1
            df.loc[df['BB_Signal'] == 'overbought', 'Bearish_Strength'] += 1
            
        # Check Stochastic
        if 'Stochastic_Signal' in df.columns:
            df.loc[df['Stochastic_Signal'].isin(['oversold', 'bullish_cross']), 'Bullish_Strength'] += 1
            df.loc[df['Stochastic_Signal'].isin(['overbought', 'bearish_cross']), 'Bearish_Strength'] += 1
            
        # Check EMA Cross
        if 'EMA_Cross_Signal' in df.columns:
            df.loc[df['EMA_Cross_Signal'] == 'bullish', 'Bullish_Strength'] += 1
            df.loc[df['EMA_Cross_Signal'] == 'bearish', 'Bearish_Strength'] += 1
            
        # Check SuperTrend
        if 'SuperTrend_Signal' in df.columns:
            df.loc[df['SuperTrend_Signal'] == 'bullish', 'Bullish_Strength'] += 2  # SuperTrend has higher weight
            df.loc[df['SuperTrend_Signal'] == 'bearish', 'Bearish_Strength'] += 2  # SuperTrend has higher weight
            
        # Set combined signal based on strength
        # Strong bullish: At least 3 bullish signals and more bullish than bearish signals
        df.loc[(df['Bullish_Strength'] >= 3) & (df['Bullish_Strength'] > df['Bearish_Strength']), 'Combined_Signal'] = 'strong_bullish'
        
        # Moderate bullish: At least 2 bullish signals and more bullish than bearish signals
        df.loc[(df['Bullish_Strength'] >= 2) & (df['Bullish_Strength'] > df['Bearish_Strength']) & 
              (df['Combined_Signal'] == 'neutral'), 'Combined_Signal'] = 'moderate_bullish'
        
        # Strong bearish: At least 3 bearish signals and more bearish than bullish signals
        df.loc[(df['Bearish_Strength'] >= 3) & (df['Bearish_Strength'] > df['Bullish_Strength']), 'Combined_Signal'] = 'strong_bearish'
        
        # Moderate bearish: At least 2 bearish signals and more bearish than bullish signals
        df.loc[(df['Bearish_Strength'] >= 2) & (df['Bearish_Strength'] > df['Bullish_Strength']) & 
              (df['Combined_Signal'] == 'neutral'), 'Combined_Signal'] = 'moderate_bearish'
        
        # Clean up temporary columns
        df = df.drop(['Bullish_Strength', 'Bearish_Strength'], axis=1)
        
        return df
        
    except Exception as e:
        logging.error(f"Error generating combined signals: {e}")
        return data  # Return original data in case of error