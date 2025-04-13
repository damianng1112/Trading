import pandas as pd
import numpy as np
import logging
import talib as ta  # Using TA-Lib for more efficient calculation
import math

class TechnicalIndicators:
    """Class for calculating various technical indicators"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration parameters
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def calculate_rsi(self, data, period=14, column="close"):
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
            
            # Try using TA-Lib for calculation
            try:
                df['RSI'] = ta.RSI(df[column].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add RSI divergence detection
            self._detect_rsi_divergence(df, column=column, period=period)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return data  # Return original data in case of error

    def _detect_rsi_divergence(self, df, column="close", period=14, window=5):
        """
        Detect RSI divergence patterns
        
        Args:
            df (pd.DataFrame): DataFrame with price and RSI data
            column (str): Price column to use
            period (int): RSI period
            window (int): Window to look for divergence
        """
        try:
            # Initialize divergence column
            df['RSI_Divergence'] = 'none'
            
            # Need at least 2*window data points
            if len(df) < 2*window:
                return
                
            # Find local price highs and lows
            for i in range(window, len(df) - window):
                price_window = df[column].iloc[i-window:i+window+1]
                rsi_window = df['RSI'].iloc[i-window:i+window+1]
                
                # Check if current point is a local high in price
                if df[column].iloc[i] == price_window.max():
                    # Look for lower high in RSI (bearish divergence)
                    prev_highs = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_highs) > 0:
                        prev_price_highs = prev_highs[prev_highs[column] == prev_highs[column].max()]
                        if len(prev_price_highs) > 0:
                            prev_idx = prev_price_highs.index[-1]
                            if df['RSI'].loc[prev_idx] > df['RSI'].iloc[i] and df[column].loc[prev_idx] < df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bearish'
                
                # Check if current point is a local low in price
                if df[column].iloc[i] == price_window.min():
                    # Look for higher low in RSI (bullish divergence)
                    prev_lows = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_lows) > 0:
                        prev_price_lows = prev_lows[prev_lows[column] == prev_lows[column].min()]
                        if len(prev_price_lows) > 0:
                            prev_idx = prev_price_lows.index[-1]
                            if df['RSI'].loc[prev_idx] < df['RSI'].iloc[i] and df[column].loc[prev_idx] > df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bullish'
                            
        except Exception as e:
            logging.error(f"Error detecting RSI divergence: {e}")

    def calculate_macd(self, data, column="close", fast_period=12, slow_period=26, signal_period=9):
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
            
            # Try using TA-Lib for calculation
            try:
                macd, signal, hist = ta.MACD(df[column].values, fastperiod=fast_period, 
                                           slowperiod=slow_period, signalperiod=signal_period)
                df['MACD'] = macd
                df['Signal_Line'] = signal
                df['MACD_Histogram'] = hist
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add MACD zero line crossover
            df.loc[(df['MACD'] > 0) & (df['MACD'].shift(1) <= 0), 'MACD_Signal'] = 'bullish_zero'
            df.loc[(df['MACD'] < 0) & (df['MACD'].shift(1) >= 0), 'MACD_Signal'] = 'bearish_zero'
            
            # Add histogram reversal signals (early reversals)
            df.loc[(df['MACD_Histogram'] > 0) & (df['MACD_Histogram'].shift(1) <= 0), 'MACD_Histogram_Signal'] = 'bullish'
            df.loc[(df['MACD_Histogram'] < 0) & (df['MACD_Histogram'].shift(1) >= 0), 'MACD_Histogram_Signal'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return data  # Return original data in case of error

    def calculate_bollinger_bands(self, data, column="close", window=20, num_std=2):
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
            
            # Try using TA-Lib for calculation
            try:
                upper, middle, lower = ta.BBANDS(df[column].values, timeperiod=window, 
                                              nbdevup=num_std, nbdevdn=num_std, matype=0)
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add Bollinger Band squeeze detection
            # A "squeeze" is when the bands narrow significantly, often preceding a breakout
            bb_avg_width = df['BB_Width'].rolling(window=50).mean()
            df['BB_Squeeze'] = False
            df.loc[df['BB_Width'] < bb_avg_width * 0.5, 'BB_Squeeze'] = True
            
            # Add Bollinger Band bounce signals
            # Price bounces off the lower band back inside
            df['BB_Bounce'] = 'none'
            df.loc[(df[column].shift(1) <= df['BB_Lower'].shift(1)) & 
                  (df[column] > df['BB_Lower']), 'BB_Bounce'] = 'bullish'
            # Price bounces off the upper band back inside
            df.loc[(df[column].shift(1) >= df['BB_Upper'].shift(1)) & 
                  (df[column] < df['BB_Upper']), 'BB_Bounce'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            return data  # Return original data in case of error

    def calculate_stochastic_oscillator(self, data, k_period=14, d_period=3, smooth_k=3):
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
            
            # Try using TA-Lib for calculation
            try:
                slowk, slowd = ta.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                      fastk_period=k_period, slowk_period=smooth_k, 
                                      slowk_matype=0, slowd_period=d_period, slowd_matype=0)
                df['Stoch_K'] = slowk
                df['Stoch_D'] = slowd
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add oversold/overbought crossing signals
            # Oversold exit (K crosses above 20 from below)
            df.loc[(df['Stoch_K'] > 20) & (df['Stoch_K'].shift(1) <= 20), 'Stochastic_Signal'] = 'oversold_exit'
            # Overbought exit (K crosses below 80 from above)
            df.loc[(df['Stoch_K'] < 80) & (df['Stoch_K'].shift(1) >= 80), 'Stochastic_Signal'] = 'overbought_exit'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator: {e}")
            return data  # Return original data in case of error

    def calculate_ema(self, data, periods=[9, 21, 50, 200], column='close'):
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
                # Try using TA-Lib for calculation
                try:
                    df[f'EMA_{period}'] = ta.EMA(df[column].values, timeperiod=period)
                except:
                    # Fall back to pandas EMA if TA-Lib fails
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
                
                # Golden cross (50 EMA crosses above 200 EMA)
                if 50 in periods and 200 in periods:
                    df.loc[(df[f'EMA_50'] > df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) <= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'golden_cross'
                    
                    # Death cross (50 EMA crosses below 200 EMA)
                    df.loc[(df[f'EMA_50'] < df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) >= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'death_cross'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating EMAs: {e}")
            return data  # Return original data in case of error

    def calculate_atr(self, data, period=14):
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
            
            # Try using TA-Lib for calculation
            try:
                df['ATR'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
                # Calculate True Range
                df['TR1'] = abs(df['high'] - df['low'])
                df['TR2'] = abs(df['high'] - df['close'].shift(1))
                df['TR3'] = abs(df['low'] - df['close'].shift(1))
                df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
                
                # Calculate ATR as exponential moving average of TR
                df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
                
                # Clean up temporary columns
                df = df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1)
            
            # Add volatility regime detection
            df['Volatility_Regime'] = 'normal'
            
            # Calculate average ATR as percentage of price
            df['ATR_Pct'] = df['ATR'] / df['close'] * 100
            avg_atr_pct = df['ATR_Pct'].rolling(window=50).mean()
            std_atr_pct = df['ATR_Pct'].rolling(window=50).std()
            
            # High volatility: ATR percentage is more than 1.5 std devs above average
            df.loc[df['ATR_Pct'] > avg_atr_pct + 1.5 * std_atr_pct, 'Volatility_Regime'] = 'high'
            
            # Low volatility: ATR percentage is more than 1.5 std devs below average
            df.loc[df['ATR_Pct'] < avg_atr_pct - 1.5 * std_atr_pct, 'Volatility_Regime'] = 'low'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return data  # Return original data in case of error

    def calculate_vwap(self, data, period=1):
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data.
            period (int): Number of days for VWAP calculation (intraday=1)
            
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
            
            # Add date column if timestamp is available
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp'].dt.date
            else:
                # If no timestamp, assume data is continuous and use arbitrary periods
                df['date'] = (np.arange(len(df)) / (24 * period)).astype(int)
            
            # Calculate typical price
            df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP components
            df['VP'] = df['TypicalPrice'] * df['volume']
            
            # Group by date and calculate cumulative sums within each group
            df['CumulativeVP'] = df.groupby('date')['VP'].cumsum()
            df['CumulativeVolume'] = df.groupby('date')['volume'].cumsum()
            
            # Calculate VWAP
            df['VWAP'] = df['CumulativeVP'] / df['CumulativeVolume']
            
            # Add VWAP signals
            df['VWAP_Signal'] = 'neutral'
            
            # Bullish when price crosses above VWAP
            df.loc[(df['close'] > df['VWAP']) & 
                  (df['close'].shift(1) <= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bullish'
            
            # Bearish when price crosses below VWAP
            df.loc[(df['close'] < df['VWAP']) & 
                  (df['close'].shift(1) >= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bearish'
            
            # Clean up temporary columns
            df = df.drop(['TypicalPrice', 'VP', 'CumulativeVP', 'CumulativeVolume', 'date'], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating VWAP: {e}")
            return data  # Return original data in case of error

    def calculate_ichimoku(self, data):
        """
        Calculate Ichimoku Cloud indicator.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data.
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku components added.
        """
        try:
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Check if required columns exist
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing columns for Ichimoku: {missing_columns}")
                return df
            
            # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
            
            # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['Ichimoku_Base'] = (high_26 + low_26) / 2
            
            # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
            
            # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
            
            # Calculate Chikou Span (Lagging Span): Close price shifted backwards by 26 periods
            df['Ichimoku_Lagging'] = df['close'].shift(-26)
            
            # Add cloud signals
            df['Ichimoku_Signal'] = 'neutral'
            
            # Bullish signal: Price above the cloud
            df.loc[(df['close'] > df['Ichimoku_SpanA']) & 
                  (df['close'] > df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bullish'
            
            # Bearish signal: Price below the cloud
            df.loc[(df['close'] < df['Ichimoku_SpanA']) & 
                  (df['close'] < df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bearish'
            
            # TK Cross (Conversion Line crosses above Base Line)
            df.loc[(df['Ichimoku_Conversion'] > df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) <= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bullish'
            
            # TK Cross (Conversion Line crosses below Base Line)
            df.loc[(df['Ichimoku_Conversion'] < df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) >= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bearish'
            
            # Add cloud thickness as a measure of trend strength
            df['Cloud_Thickness'] = abs(df['Ichimoku_SpanA'] - df['Ichimoku_SpanB'])
            df['Cloud_Thickness_Pct'] = df['Cloud_Thickness'] / df['close'] * 100
            
            # Color the cloud (useful for visualization)
            df['Cloud_Color'] = 'none'
            df.loc[df['Ichimoku_SpanA'] >= df['Ichimoku_SpanB'], 'Cloud_Color'] = 'green'
            df.loc[df['Ichimoku_SpanA'] < df['Ichimoku_SpanB'], 'Cloud_Color'] = 'red'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud: {e}")
            return data  # Return original data in case of error
    
    def calculate_supertrend(self, data, period=10, multiplier=3.0):
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
            missing_columns = [col for col in required_columns if col not in df.columnsimport pandas as pd
import numpy as np
import logging
import talib as ta  # Using TA-Lib for more efficient calculation
import math

class TechnicalIndicators:
    """Class for calculating various technical indicators"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration parameters
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def calculate_rsi(self, data, period=14, column="close"):
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
            
            # Try using TA-Lib for calculation
            try:
                df['RSI'] = ta.RSI(df[column].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add RSI divergence detection
            self._detect_rsi_divergence(df, column=column, period=period)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return data  # Return original data in case of error

    def _detect_rsi_divergence(self, df, column="close", period=14, window=5):
        """
        Detect RSI divergence patterns
        
        Args:
            df (pd.DataFrame): DataFrame with price and RSI data
            column (str): Price column to use
            period (int): RSI period
            window (int): Window to look for divergence
        """
        try:
            # Initialize divergence column
            df['RSI_Divergence'] = 'none'
            
            # Need at least 2*window data points
            if len(df) < 2*window:
                return
                
            # Find local price highs and lows
            for i in range(window, len(df) - window):
                price_window = df[column].iloc[i-window:i+window+1]
                rsi_window = df['RSI'].iloc[i-window:i+window+1]
                
                # Check if current point is a local high in price
                if df[column].iloc[i] == price_window.max():
                    # Look for lower high in RSI (bearish divergence)
                    prev_highs = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_highs) > 0:
                        prev_price_highs = prev_highs[prev_highs[column] == prev_highs[column].max()]
                        if len(prev_price_highs) > 0:
                            prev_idx = prev_price_highs.index[-1]
                            if df['RSI'].loc[prev_idx] > df['RSI'].iloc[i] and df[column].loc[prev_idx] < df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bearish'
                
                # Check if current point is a local low in price
                if df[column].iloc[i] == price_window.min():
                    # Look for higher low in RSI (bullish divergence)
                    prev_lows = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_lows) > 0:
                        prev_price_lows = prev_lows[prev_lows[column] == prev_lows[column].min()]
                        if len(prev_price_lows) > 0:
                            prev_idx = prev_price_lows.index[-1]
                            if df['RSI'].loc[prev_idx] < df['RSI'].iloc[i] and df[column].loc[prev_idx] > df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bullish'
                            
        except Exception as e:
            logging.error(f"Error detecting RSI divergence: {e}")

    def calculate_macd(self, data, column="close", fast_period=12, slow_period=26, signal_period=9):
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
            
            # Try using TA-Lib for calculation
            try:
                macd, signal, hist = ta.MACD(df[column].values, fastperiod=fast_period, 
                                           slowperiod=slow_period, signalperiod=signal_period)
                df['MACD'] = macd
                df['Signal_Line'] = signal
                df['MACD_Histogram'] = hist
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add MACD zero line crossover
            df.loc[(df['MACD'] > 0) & (df['MACD'].shift(1) <= 0), 'MACD_Signal'] = 'bullish_zero'
            df.loc[(df['MACD'] < 0) & (df['MACD'].shift(1) >= 0), 'MACD_Signal'] = 'bearish_zero'
            
            # Add histogram reversal signals (early reversals)
            df.loc[(df['MACD_Histogram'] > 0) & (df['MACD_Histogram'].shift(1) <= 0), 'MACD_Histogram_Signal'] = 'bullish'
            df.loc[(df['MACD_Histogram'] < 0) & (df['MACD_Histogram'].shift(1) >= 0), 'MACD_Histogram_Signal'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return data  # Return original data in case of error

    def calculate_bollinger_bands(self, data, column="close", window=20, num_std=2):
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
            
            # Try using TA-Lib for calculation
            try:
                upper, middle, lower = ta.BBANDS(df[column].values, timeperiod=window, 
                                              nbdevup=num_std, nbdevdn=num_std, matype=0)
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add Bollinger Band squeeze detection
            # A "squeeze" is when the bands narrow significantly, often preceding a breakout
            bb_avg_width = df['BB_Width'].rolling(window=50).mean()
            df['BB_Squeeze'] = False
            df.loc[df['BB_Width'] < bb_avg_width * 0.5, 'BB_Squeeze'] = True
            
            # Add Bollinger Band bounce signals
            # Price bounces off the lower band back inside
            df['BB_Bounce'] = 'none'
            df.loc[(df[column].shift(1) <= df['BB_Lower'].shift(1)) & 
                  (df[column] > df['BB_Lower']), 'BB_Bounce'] = 'bullish'
            # Price bounces off the upper band back inside
            df.loc[(df[column].shift(1) >= df['BB_Upper'].shift(1)) & 
                  (df[column] < df['BB_Upper']), 'BB_Bounce'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            return data  # Return original data in case of error

    def calculate_stochastic_oscillator(self, data, k_period=14, d_period=3, smooth_k=3):
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
            
            # Try using TA-Lib for calculation
            try:
                slowk, slowd = ta.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                      fastk_period=k_period, slowk_period=smooth_k, 
                                      slowk_matype=0, slowd_period=d_period, slowd_matype=0)
                df['Stoch_K'] = slowk
                df['Stoch_D'] = slowd
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add oversold/overbought crossing signals
            # Oversold exit (K crosses above 20 from below)
            df.loc[(df['Stoch_K'] > 20) & (df['Stoch_K'].shift(1) <= 20), 'Stochastic_Signal'] = 'oversold_exit'
            # Overbought exit (K crosses below 80 from above)
            df.loc[(df['Stoch_K'] < 80) & (df['Stoch_K'].shift(1) >= 80), 'Stochastic_Signal'] = 'overbought_exit'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator: {e}")
            return data  # Return original data in case of error

    def calculate_ema(self, data, periods=[9, 21, 50, 200], column='close'):
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
                # Try using TA-Lib for calculation
                try:
                    df[f'EMA_{period}'] = ta.EMA(df[column].values, timeperiod=period)
                except:
                    # Fall back to pandas EMA if TA-Lib fails
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
                
                # Golden cross (50 EMA crosses above 200 EMA)
                if 50 in periods and 200 in periods:
                    df.loc[(df[f'EMA_50'] > df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) <= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'golden_cross'
                    
                    # Death cross (50 EMA crosses below 200 EMA)
                    df.loc[(df[f'EMA_50'] < df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) >= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'death_cross'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating EMAs: {e}")
            return data  # Return original data in case of error

    def calculate_atr(self, data, period=14):
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
            
            # Try using TA-Lib for calculation
            try:
                df['ATR'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
                # Calculate True Range
                df['TR1'] = abs(df['high'] - df['low'])
                df['TR2'] = abs(df['high'] - df['close'].shift(1))
                df['TR3'] = abs(df['low'] - df['close'].shift(1))
                df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
                
                # Calculate ATR as exponential moving average of TR
                df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
                
                # Clean up temporary columns
                df = df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1)
            
            # Add volatility regime detection
            df['Volatility_Regime'] = 'normal'
            
            # Calculate average ATR as percentage of price
            df['ATR_Pct'] = df['ATR'] / df['close'] * 100
            avg_atr_pct = df['ATR_Pct'].rolling(window=50).mean()
            std_atr_pct = df['ATR_Pct'].rolling(window=50).std()
            
            # High volatility: ATR percentage is more than 1.5 std devs above average
            df.loc[df['ATR_Pct'] > avg_atr_pct + 1.5 * std_atr_pct, 'Volatility_Regime'] = 'high'
            
            # Low volatility: ATR percentage is more than 1.5 std devs below average
            df.loc[df['ATR_Pct'] < avg_atr_pct - 1.5 * std_atr_pct, 'Volatility_Regime'] = 'low'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return data  # Return original data in case of error

    def calculate_vwap(self, data, period=1):
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data.
            period (int): Number of days for VWAP calculation (intraday=1)
            
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
            
            # Add date column if timestamp is available
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp'].dt.date
            else:
                # If no timestamp, assume data is continuous and use arbitrary periods
                df['date'] = (np.arange(len(df)) / (24 * period)).astype(int)
            
            # Calculate typical price
            df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP components
            df['VP'] = df['TypicalPrice'] * df['volume']
            
            # Group by date and calculate cumulative sums within each group
            df['CumulativeVP'] = df.groupby('date')['VP'].cumsum()
            df['CumulativeVolume'] = df.groupby('date')['volume'].cumsum()
            
            # Calculate VWAP
            df['VWAP'] = df['CumulativeVP'] / df['CumulativeVolume']
            
            # Add VWAP signals
            df['VWAP_Signal'] = 'neutral'
            
            # Bullish when price crosses above VWAP
            df.loc[(df['close'] > df['VWAP']) & 
                  (df['close'].shift(1) <= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bullish'
            
            # Bearish when price crosses below VWAP
            df.loc[(df['close'] < df['VWAP']) & 
                  (df['close'].shift(1) >= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bearish'
            
            # Clean up temporary columns
            df = df.drop(['TypicalPrice', 'VP', 'CumulativeVP', 'CumulativeVolume', 'date'], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating VWAP: {e}")
            return data  # Return original data in case of error

    def calculate_ichimoku(self, data):
        """
        Calculate Ichimoku Cloud indicator.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data.
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku components added.
        """
        try:
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Check if required columns exist
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing columns for Ichimoku: {missing_columns}")
                return df
            
            # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
            
            # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['Ichimoku_Base'] = (high_26 + low_26) / 2
            
            # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
            
            # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
            
            # Calculate Chikou Span (Lagging Span): Close price shifted backwards by 26 periods
            df['Ichimoku_Lagging'] = df['close'].shift(-26)
            
            # Add cloud signals
            df['Ichimoku_Signal'] = 'neutral'
            
            # Bullish signal: Price above the cloud
            df.loc[(df['close'] > df['Ichimoku_SpanA']) & 
                  (df['close'] > df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bullish'
            
            # Bearish signal: Price below the cloud
            df.loc[(df['close'] < df['Ichimoku_SpanA']) & 
                  (df['close'] < df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bearish'
            
            # TK Cross (Conversion Line crosses above Base Line)
            df.loc[(df['Ichimoku_Conversion'] > df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) <= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bullish'
            
            # TK Cross (Conversion Line crosses below Base Line)
            df.loc[(df['Ichimoku_Conversion'] < df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) >= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bearish'
            
            # Add cloud thickness as a measure of trend strength
            df['Cloud_Thickness'] = abs(df['Ichimoku_SpanA'] - df['Ichimoku_SpanB'])
            df['Cloud_Thickness_Pct'] = df['Cloud_Thickness'] / df['close'] * 100
            
            # Color the cloud (useful for visualization)
            df['Cloud_Color'] = 'none'
            df.loc[df['Ichimoku_SpanA'] >= df['Ichimoku_SpanB'], 'Cloud_Color'] = 'green'
            df.loc[df['Ichimoku_SpanA'] < df['Ichimoku_SpanB'], 'Cloud_Color'] = 'red'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud: {e}")
            return data  # Return original data in case of error
    
    def calculate_supertrend(self, data, period=10, multiplier=3.0):
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
            missing_columns = [col for col in required_columns if colimport pandas as pd
import numpy as np
import logging
import talib as ta  # Using TA-Lib for more efficient calculation
import math

class TechnicalIndicators:
    """Class for calculating various technical indicators"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration parameters
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def calculate_rsi(self, data, period=14, column="close"):
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
            
            # Try using TA-Lib for calculation
            try:
                df['RSI'] = ta.RSI(df[column].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add RSI divergence detection
            self._detect_rsi_divergence(df, column=column, period=period)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return data  # Return original data in case of error

    def _detect_rsi_divergence(self, df, column="close", period=14, window=5):
        """
        Detect RSI divergence patterns
        
        Args:
            df (pd.DataFrame): DataFrame with price and RSI data
            column (str): Price column to use
            period (int): RSI period
            window (int): Window to look for divergence
        """
        try:
            # Initialize divergence column
            df['RSI_Divergence'] = 'none'
            
            # Need at least 2*window data points
            if len(df) < 2*window:
                return
                
            # Find local price highs and lows
            for i in range(window, len(df) - window):
                price_window = df[column].iloc[i-window:i+window+1]
                rsi_window = df['RSI'].iloc[i-window:i+window+1]
                
                # Check if current point is a local high in price
                if df[column].iloc[i] == price_window.max():
                    # Look for lower high in RSI (bearish divergence)
                    prev_highs = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_highs) > 0:
                        prev_price_highs = prev_highs[prev_highs[column] == prev_highs[column].max()]
                        if len(prev_price_highs) > 0:
                            prev_idx = prev_price_highs.index[-1]
                            if df['RSI'].loc[prev_idx] > df['RSI'].iloc[i] and df[column].loc[prev_idx] < df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bearish'
                
                # Check if current point is a local low in price
                if df[column].iloc[i] == price_window.min():
                    # Look for higher low in RSI (bullish divergence)
                    prev_lows = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_lows) > 0:
                        prev_price_lows = prev_lows[prev_lows[column] == prev_lows[column].min()]
                        if len(prev_price_lows) > 0:
                            prev_idx = prev_price_lows.index[-1]
                            if df['RSI'].loc[prev_idx] < df['RSI'].iloc[i] and df[column].loc[prev_idx] > df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bullish'
                            
        except Exception as e:
            logging.error(f"Error detecting RSI divergence: {e}")

    def calculate_macd(self, data, column="close", fast_period=12, slow_period=26, signal_period=9):
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
            
            # Try using TA-Lib for calculation
            try:
                macd, signal, hist = ta.MACD(df[column].values, fastperiod=fast_period, 
                                           slowperiod=slow_period, signalperiod=signal_period)
                df['MACD'] = macd
                df['Signal_Line'] = signal
                df['MACD_Histogram'] = hist
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add MACD zero line crossover
            df.loc[(df['MACD'] > 0) & (df['MACD'].shift(1) <= 0), 'MACD_Signal'] = 'bullish_zero'
            df.loc[(df['MACD'] < 0) & (df['MACD'].shift(1) >= 0), 'MACD_Signal'] = 'bearish_zero'
            
            # Add histogram reversal signals (early reversals)
            df.loc[(df['MACD_Histogram'] > 0) & (df['MACD_Histogram'].shift(1) <= 0), 'MACD_Histogram_Signal'] = 'bullish'
            df.loc[(df['MACD_Histogram'] < 0) & (df['MACD_Histogram'].shift(1) >= 0), 'MACD_Histogram_Signal'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return data  # Return original data in case of error

    def calculate_bollinger_bands(self, data, column="close", window=20, num_std=2):
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
            
            # Try using TA-Lib for calculation
            try:
                upper, middle, lower = ta.BBANDS(df[column].values, timeperiod=window, 
                                              nbdevup=num_std, nbdevdn=num_std, matype=0)
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add Bollinger Band squeeze detection
            # A "squeeze" is when the bands narrow significantly, often preceding a breakout
            bb_avg_width = df['BB_Width'].rolling(window=50).mean()
            df['BB_Squeeze'] = False
            df.loc[df['BB_Width'] < bb_avg_width * 0.5, 'BB_Squeeze'] = True
            
            # Add Bollinger Band bounce signals
            # Price bounces off the lower band back inside
            df['BB_Bounce'] = 'none'
            df.loc[(df[column].shift(1) <= df['BB_Lower'].shift(1)) & 
                  (df[column] > df['BB_Lower']), 'BB_Bounce'] = 'bullish'
            # Price bounces off the upper band back inside
            df.loc[(df[column].shift(1) >= df['BB_Upper'].shift(1)) & 
                  (df[column] < df['BB_Upper']), 'BB_Bounce'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            return data  # Return original data in case of error

    def calculate_stochastic_oscillator(self, data, k_period=14, d_period=3, smooth_k=3):
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
            
            # Try using TA-Lib for calculation
            try:
                slowk, slowd = ta.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                      fastk_period=k_period, slowk_period=smooth_k, 
                                      slowk_matype=0, slowd_period=d_period, slowd_matype=0)
                df['Stoch_K'] = slowk
                df['Stoch_D'] = slowd
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add oversold/overbought crossing signals
            # Oversold exit (K crosses above 20 from below)
            df.loc[(df['Stoch_K'] > 20) & (df['Stoch_K'].shift(1) <= 20), 'Stochastic_Signal'] = 'oversold_exit'
            # Overbought exit (K crosses below 80 from above)
            df.loc[(df['Stoch_K'] < 80) & (df['Stoch_K'].shift(1) >= 80), 'Stochastic_Signal'] = 'overbought_exit'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator: {e}")
            return data  # Return original data in case of error

    def calculate_ema(self, data, periods=[9, 21, 50, 200], column='close'):
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
                # Try using TA-Lib for calculation
                try:
                    df[f'EMA_{period}'] = ta.EMA(df[column].values, timeperiod=period)
                except:
                    # Fall back to pandas EMA if TA-Lib fails
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
                
                # Golden cross (50 EMA crosses above 200 EMA)
                if 50 in periods and 200 in periods:
                    df.loc[(df[f'EMA_50'] > df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) <= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'golden_cross'
                    
                    # Death cross (50 EMA crosses below 200 EMA)
                    df.loc[(df[f'EMA_50'] < df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) >= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'death_cross'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating EMAs: {e}")
            return data  # Return original data in case of error

    def calculate_atr(self, data, period=14):
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
            
            # Try using TA-Lib for calculation
            try:
                df['ATR'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
                # Calculate True Range
                df['TR1'] = abs(df['high'] - df['low'])
                df['TR2'] = abs(df['high'] - df['close'].shift(1))
                df['TR3'] = abs(df['low'] - df['close'].shift(1))
                df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
                
                # Calculate ATR as exponential moving average of TR
                df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
                
                # Clean up temporary columns
                df = df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1)
            
            # Add volatility regime detection
            df['Volatility_Regime'] = 'normal'
            
            # Calculate average ATR as percentage of price
            df['ATR_Pct'] = df['ATR'] / df['close'] * 100
            avg_atr_pct = df['ATR_Pct'].rolling(window=50).mean()
            std_atr_pct = df['ATR_Pct'].rolling(window=50).std()
            
            # High volatility: ATR percentage is more than 1.5 std devs above average
            df.loc[df['ATR_Pct'] > avg_atr_pct + 1.5 * std_atr_pct, 'Volatility_Regime'] = 'high'
            
            # Low volatility: ATR percentage is more than 1.5 std devs below average
            df.loc[df['ATR_Pct'] < avg_atr_pct - 1.5 * std_atr_pct, 'Volatility_Regime'] = 'low'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return data  # Return original data in case of error

    def calculate_vwap(self, data, period=1):
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data.
            period (int): Number of days for VWAP calculation (intraday=1)
            
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
            
            # Add date column if timestamp is available
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp'].dt.date
            else:
                # If no timestamp, assume data is continuous and use arbitrary periods
                df['date'] = (np.arange(len(df)) / (24 * period)).astype(int)
            
            # Calculate typical price
            df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP components
            df['VP'] = df['TypicalPrice'] * df['volume']
            
            # Group by date and calculate cumulative sums within each group
            df['CumulativeVP'] = df.groupby('date')['VP'].cumsum()
            df['CumulativeVolume'] = df.groupby('date')['volume'].cumsum()
            
            # Calculate VWAP
            df['VWAP'] = df['CumulativeVP'] / df['CumulativeVolume']
            
            # Add VWAP signals
            df['VWAP_Signal'] = 'neutral'
            
            # Bullish when price crosses above VWAP
            df.loc[(df['close'] > df['VWAP']) & 
                  (df['close'].shift(1) <= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bullish'
            
            # Bearish when price crosses below VWAP
            df.loc[(df['close'] < df['VWAP']) & 
                  (df['close'].shift(1) >= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bearish'
            
            # Clean up temporary columns
            df = df.drop(['TypicalPrice', 'VP', 'CumulativeVP', 'CumulativeVolume', 'date'], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating VWAP: {e}")
            return data  # Return original data in case of error

    def calculate_ichimoku(self, data):
        """
        Calculate Ichimoku Cloud indicator.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data.
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku components added.
        """
        try:
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Check if required columns exist
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing columns for Ichimoku: {missing_columns}")
                return df
            
            # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
            
            # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['Ichimoku_Base'] = (high_26 + low_26) / 2
            
            # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
            
            # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
            
            # Calculate Chikou Span (Lagging Span): Close price shifted backwards by 26 periods
            df['Ichimoku_Lagging'] = df['close'].shift(-26)
            
            # Add cloud signals
            df['Ichimoku_Signal'] = 'neutral'
            
            # Bullish signal: Price above the cloud
            df.loc[(df['close'] > df['Ichimoku_SpanA']) & 
                  (df['close'] > df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bullish'
            
            # Bearish signal: Price below the cloud
            df.loc[(df['close'] < df['Ichimoku_SpanA']) & 
                  (df['close'] < df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bearish'
            
            # TK Cross (Conversion Line crosses above Base Line)
            df.loc[(df['Ichimoku_Conversion'] > df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) <= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bullish'
            
            # TK Cross (Conversion Line crosses below Base Line)
            df.loc[(df['Ichimoku_Conversion'] < df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) >= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bearish'
            
            # Add cloud thickness as a measure of trend strength
            df['Cloud_Thickness'] = abs(df['Ichimoku_SpanA'] - df['Ichimoku_SpanB'])
            df['Cloud_Thickness_Pct'] = df['Cloud_Thickness'] / df['close'] * 100
            
            # Color the cloud (useful for visualization)
            df['Cloud_Color'] = 'none'
            df.loc[df['Ichimoku_SpanA'] >= df['Ichimoku_SpanB'], 'Cloud_Color'] = 'green'
            df.loc[df['Ichimoku_SpanA'] < df['Ichimoku_SpanB'], 'Cloud_Color'] = 'red'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud: {e}")
            return data  # Return original data in case of error
    
    def calculate_supertrend(self, data, period=10, multiplier=3.0):
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
            ifimport pandas as pd
import numpy as np
import logging
import talib as ta  # Using TA-Lib for more efficient calculation
import math

class TechnicalIndicators:
    """Class for calculating various technical indicators"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration parameters
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def calculate_rsi(self, data, period=14, column="close"):
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
            
            # Try using TA-Lib for calculation
            try:
                df['RSI'] = ta.RSI(df[column].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add RSI divergence detection
            self._detect_rsi_divergence(df, column=column, period=period)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return data  # Return original data in case of error

    def _detect_rsi_divergence(self, df, column="close", period=14, window=5):
        """
        Detect RSI divergence patterns
        
        Args:
            df (pd.DataFrame): DataFrame with price and RSI data
            column (str): Price column to use
            period (int): RSI period
            window (int): Window to look for divergence
        """
        try:
            # Initialize divergence column
            df['RSI_Divergence'] = 'none'
            
            # Need at least 2*window data points
            if len(df) < 2*window:
                return
                
            # Find local price highs and lows
            for i in range(window, len(df) - window):
                price_window = df[column].iloc[i-window:i+window+1]
                rsi_window = df['RSI'].iloc[i-window:i+window+1]
                
                # Check if current point is a local high in price
                if df[column].iloc[i] == price_window.max():
                    # Look for lower high in RSI (bearish divergence)
                    prev_highs = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_highs) > 0:
                        prev_price_highs = prev_highs[prev_highs[column] == prev_highs[column].max()]
                        if len(prev_price_highs) > 0:
                            prev_idx = prev_price_highs.index[-1]
                            if df['RSI'].loc[prev_idx] > df['RSI'].iloc[i] and df[column].loc[prev_idx] < df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bearish'
                
                # Check if current point is a local low in price
                if df[column].iloc[i] == price_window.min():
                    # Look for higher low in RSI (bullish divergence)
                    prev_lows = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_lows) > 0:
                        prev_price_lows = prev_lows[prev_lows[column] == prev_lows[column].min()]
                        if len(prev_price_lows) > 0:
                            prev_idx = prev_price_lows.index[-1]
                            if df['RSI'].loc[prev_idx] < df['RSI'].iloc[i] and df[column].loc[prev_idx] > df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bullish'
                            
        except Exception as e:
            logging.error(f"Error detecting RSI divergence: {e}")

    def calculate_macd(self, data, column="close", fast_period=12, slow_period=26, signal_period=9):
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
            
            # Try using TA-Lib for calculation
            try:
                macd, signal, hist = ta.MACD(df[column].values, fastperiod=fast_period, 
                                           slowperiod=slow_period, signalperiod=signal_period)
                df['MACD'] = macd
                df['Signal_Line'] = signal
                df['MACD_Histogram'] = hist
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add MACD zero line crossover
            df.loc[(df['MACD'] > 0) & (df['MACD'].shift(1) <= 0), 'MACD_Signal'] = 'bullish_zero'
            df.loc[(df['MACD'] < 0) & (df['MACD'].shift(1) >= 0), 'MACD_Signal'] = 'bearish_zero'
            
            # Add histogram reversal signals (early reversals)
            df.loc[(df['MACD_Histogram'] > 0) & (df['MACD_Histogram'].shift(1) <= 0), 'MACD_Histogram_Signal'] = 'bullish'
            df.loc[(df['MACD_Histogram'] < 0) & (df['MACD_Histogram'].shift(1) >= 0), 'MACD_Histogram_Signal'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return data  # Return original data in case of error

    def calculate_bollinger_bands(self, data, column="close", window=20, num_std=2):
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
            
            # Try using TA-Lib for calculation
            try:
                upper, middle, lower = ta.BBANDS(df[column].values, timeperiod=window, 
                                              nbdevup=num_std, nbdevdn=num_std, matype=0)
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add Bollinger Band squeeze detection
            # A "squeeze" is when the bands narrow significantly, often preceding a breakout
            bb_avg_width = df['BB_Width'].rolling(window=50).mean()
            df['BB_Squeeze'] = False
            df.loc[df['BB_Width'] < bb_avg_width * 0.5, 'BB_Squeeze'] = True
            
            # Add Bollinger Band bounce signals
            # Price bounces off the lower band back inside
            df['BB_Bounce'] = 'none'
            df.loc[(df[column].shift(1) <= df['BB_Lower'].shift(1)) & 
                  (df[column] > df['BB_Lower']), 'BB_Bounce'] = 'bullish'
            # Price bounces off the upper band back inside
            df.loc[(df[column].shift(1) >= df['BB_Upper'].shift(1)) & 
                  (df[column] < df['BB_Upper']), 'BB_Bounce'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            return data  # Return original data in case of error

    def calculate_stochastic_oscillator(self, data, k_period=14, d_period=3, smooth_k=3):
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
            
            # Try using TA-Lib for calculation
            try:
                slowk, slowd = ta.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                      fastk_period=k_period, slowk_period=smooth_k, 
                                      slowk_matype=0, slowd_period=d_period, slowd_matype=0)
                df['Stoch_K'] = slowk
                df['Stoch_D'] = slowd
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add oversold/overbought crossing signals
            # Oversold exit (K crosses above 20 from below)
            df.loc[(df['Stoch_K'] > 20) & (df['Stoch_K'].shift(1) <= 20), 'Stochastic_Signal'] = 'oversold_exit'
            # Overbought exit (K crosses below 80 from above)
            df.loc[(df['Stoch_K'] < 80) & (df['Stoch_K'].shift(1) >= 80), 'Stochastic_Signal'] = 'overbought_exit'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator: {e}")
            return data  # Return original data in case of error

    def calculate_ema(self, data, periods=[9, 21, 50, 200], column='close'):
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
                # Try using TA-Lib for calculation
                try:
                    df[f'EMA_{period}'] = ta.EMA(df[column].values, timeperiod=period)
                except:
                    # Fall back to pandas EMA if TA-Lib fails
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
                
                # Golden cross (50 EMA crosses above 200 EMA)
                if 50 in periods and 200 in periods:
                    df.loc[(df[f'EMA_50'] > df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) <= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'golden_cross'
                    
                    # Death cross (50 EMA crosses below 200 EMA)
                    df.loc[(df[f'EMA_50'] < df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) >= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'death_cross'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating EMAs: {e}")
            return data  # Return original data in case of error

    def calculate_atr(self, data, period=14):
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
            
            # Try using TA-Lib for calculation
            try:
                df['ATR'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
                # Calculate True Range
                df['TR1'] = abs(df['high'] - df['low'])
                df['TR2'] = abs(df['high'] - df['close'].shift(1))
                df['TR3'] = abs(df['low'] - df['close'].shift(1))
                df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
                
                # Calculate ATR as exponential moving average of TR
                df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
                
                # Clean up temporary columns
                df = df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1)
            
            # Add volatility regime detection
            df['Volatility_Regime'] = 'normal'
            
            # Calculate average ATR as percentage of price
            df['ATR_Pct'] = df['ATR'] / df['close'] * 100
            avg_atr_pct = df['ATR_Pct'].rolling(window=50).mean()
            std_atr_pct = df['ATR_Pct'].rolling(window=50).std()
            
            # High volatility: ATR percentage is more than 1.5 std devs above average
            df.loc[df['ATR_Pct'] > avg_atr_pct + 1.5 * std_atr_pct, 'Volatility_Regime'] = 'high'
            
            # Low volatility: ATR percentage is more than 1.5 std devs below average
            df.loc[df['ATR_Pct'] < avg_atr_pct - 1.5 * std_atr_pct, 'Volatility_Regime'] = 'low'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return data  # Return original data in case of error

    def calculate_vwap(self, data, period=1):
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data.
            period (int): Number of days for VWAP calculation (intraday=1)
            
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
            
            # Add date column if timestamp is available
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp'].dt.date
            else:
                # If no timestamp, assume data is continuous and use arbitrary periods
                df['date'] = (np.arange(len(df)) / (24 * period)).astype(int)
            
            # Calculate typical price
            df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP components
            df['VP'] = df['TypicalPrice'] * df['volume']
            
            # Group by date and calculate cumulative sums within each group
            df['CumulativeVP'] = df.groupby('date')['VP'].cumsum()
            df['CumulativeVolume'] = df.groupby('date')['volume'].cumsum()
            
            # Calculate VWAP
            df['VWAP'] = df['CumulativeVP'] / df['CumulativeVolume']
            
            # Add VWAP signals
            df['VWAP_Signal'] = 'neutral'
            
            # Bullish when price crosses above VWAP
            df.loc[(df['close'] > df['VWAP']) & 
                  (df['close'].shift(1) <= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bullish'
            
            # Bearish when price crosses below VWAP
            df.loc[(df['close'] < df['VWAP']) & 
                  (df['close'].shift(1) >= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bearish'
            
            # Clean up temporary columns
            df = df.drop(['TypicalPrice', 'VP', 'CumulativeVP', 'CumulativeVolume', 'date'], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating VWAP: {e}")
            return data  # Return original data in case of error

    def calculate_ichimoku(self, data):
        """
        Calculate Ichimoku Cloud indicator.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data.
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku components added.
        """
        try:
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Check if required columns exist
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing columns for Ichimoku: {missing_columns}")
                return df
            
            # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
            
            # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['Ichimoku_Base'] = (high_26 + low_26) / 2
            
            # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
            
            # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
            
            # Calculate Chikou Span (Lagging Span): Close price shifted backwards by 26 periods
            df['Ichimoku_Lagging'] = df['close'].shift(-26)
            
            # Add cloud signals
            df['Ichimoku_Signal'] = 'neutral'
            
            # Bullish signal: Price above the cloud
            df.loc[(df['close'] > df['Ichimoku_SpanA']) & 
                  (df['close'] > df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bullish'
            
            # Bearish signal: Price below the cloud
            df.loc[(df['close'] < df['Ichimoku_SpanA']) & 
                  (df['close'] < df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bearish'
            
            # TK Cross (Conversion Line crosses above Base Line)
            df.loc[(df['Ichimoku_Conversion'] > df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) <= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bullish'
            
            # TK Cross (Conversion Line crosses below Base Line)
            df.loc[(df['Ichimoku_Conversion'] < df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) >= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bearish'
            
            # Add cloud thickness as a measure of trend strength
            df['Cloud_Thickness'] = abs(df['Ichimoku_SpanA'] - df['Ichimoku_SpanB'])
            df['Cloud_Thickness_Pct'] = df['Cloud_Thickness'] / df['close'] * 100
            
            # Color the cloud (useful for visualization)
            df['Cloud_Color'] = 'none'
            df.loc[df['Ichimoku_SpanA'] >= df['Ichimoku_SpanB'], 'Cloud_Color'] = 'green'
            df.loc[df['Ichimoku_SpanA'] < df['Ichimoku_SpanB'], 'Cloud_Color'] = 'red'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud: {e}")
            return data  # Return original data in case of error
    
    def calculate_supertrend(self, data, period=10, multiplier=3.0):
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
            if missing_columnsimport pandas as pd
import numpy as np
import logging
import talib as ta  # Using TA-Lib for more efficient calculation
import math

class TechnicalIndicators:
    """Class for calculating various technical indicators"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration parameters
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def calculate_rsi(self, data, period=14, column="close"):
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
            
            # Try using TA-Lib for calculation
            try:
                df['RSI'] = ta.RSI(df[column].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add RSI divergence detection
            self._detect_rsi_divergence(df, column=column, period=period)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            return data  # Return original data in case of error

    def _detect_rsi_divergence(self, df, column="close", period=14, window=5):
        """
        Detect RSI divergence patterns
        
        Args:
            df (pd.DataFrame): DataFrame with price and RSI data
            column (str): Price column to use
            period (int): RSI period
            window (int): Window to look for divergence
        """
        try:
            # Initialize divergence column
            df['RSI_Divergence'] = 'none'
            
            # Need at least 2*window data points
            if len(df) < 2*window:
                return
                
            # Find local price highs and lows
            for i in range(window, len(df) - window):
                price_window = df[column].iloc[i-window:i+window+1]
                rsi_window = df['RSI'].iloc[i-window:i+window+1]
                
                # Check if current point is a local high in price
                if df[column].iloc[i] == price_window.max():
                    # Look for lower high in RSI (bearish divergence)
                    prev_highs = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_highs) > 0:
                        prev_price_highs = prev_highs[prev_highs[column] == prev_highs[column].max()]
                        if len(prev_price_highs) > 0:
                            prev_idx = prev_price_highs.index[-1]
                            if df['RSI'].loc[prev_idx] > df['RSI'].iloc[i] and df[column].loc[prev_idx] < df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bearish'
                
                # Check if current point is a local low in price
                if df[column].iloc[i] == price_window.min():
                    # Look for higher low in RSI (bullish divergence)
                    prev_lows = df.iloc[max(0, i-3*window):i-window]
                    if len(prev_lows) > 0:
                        prev_price_lows = prev_lows[prev_lows[column] == prev_lows[column].min()]
                        if len(prev_price_lows) > 0:
                            prev_idx = prev_price_lows.index[-1]
                            if df['RSI'].loc[prev_idx] < df['RSI'].iloc[i] and df[column].loc[prev_idx] > df[column].iloc[i]:
                                df.loc[i, 'RSI_Divergence'] = 'bullish'
                            
        except Exception as e:
            logging.error(f"Error detecting RSI divergence: {e}")

    def calculate_macd(self, data, column="close", fast_period=12, slow_period=26, signal_period=9):
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
            
            # Try using TA-Lib for calculation
            try:
                macd, signal, hist = ta.MACD(df[column].values, fastperiod=fast_period, 
                                           slowperiod=slow_period, signalperiod=signal_period)
                df['MACD'] = macd
                df['Signal_Line'] = signal
                df['MACD_Histogram'] = hist
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add MACD zero line crossover
            df.loc[(df['MACD'] > 0) & (df['MACD'].shift(1) <= 0), 'MACD_Signal'] = 'bullish_zero'
            df.loc[(df['MACD'] < 0) & (df['MACD'].shift(1) >= 0), 'MACD_Signal'] = 'bearish_zero'
            
            # Add histogram reversal signals (early reversals)
            df.loc[(df['MACD_Histogram'] > 0) & (df['MACD_Histogram'].shift(1) <= 0), 'MACD_Histogram_Signal'] = 'bullish'
            df.loc[(df['MACD_Histogram'] < 0) & (df['MACD_Histogram'].shift(1) >= 0), 'MACD_Histogram_Signal'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            return data  # Return original data in case of error

    def calculate_bollinger_bands(self, data, column="close", window=20, num_std=2):
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
            
            # Try using TA-Lib for calculation
            try:
                upper, middle, lower = ta.BBANDS(df[column].values, timeperiod=window, 
                                              nbdevup=num_std, nbdevdn=num_std, matype=0)
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add Bollinger Band squeeze detection
            # A "squeeze" is when the bands narrow significantly, often preceding a breakout
            bb_avg_width = df['BB_Width'].rolling(window=50).mean()
            df['BB_Squeeze'] = False
            df.loc[df['BB_Width'] < bb_avg_width * 0.5, 'BB_Squeeze'] = True
            
            # Add Bollinger Band bounce signals
            # Price bounces off the lower band back inside
            df['BB_Bounce'] = 'none'
            df.loc[(df[column].shift(1) <= df['BB_Lower'].shift(1)) & 
                  (df[column] > df['BB_Lower']), 'BB_Bounce'] = 'bullish'
            # Price bounces off the upper band back inside
            df.loc[(df[column].shift(1) >= df['BB_Upper'].shift(1)) & 
                  (df[column] < df['BB_Upper']), 'BB_Bounce'] = 'bearish'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            return data  # Return original data in case of error

    def calculate_stochastic_oscillator(self, data, k_period=14, d_period=3, smooth_k=3):
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
            
            # Try using TA-Lib for calculation
            try:
                slowk, slowd = ta.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                      fastk_period=k_period, slowk_period=smooth_k, 
                                      slowk_matype=0, slowd_period=d_period, slowd_matype=0)
                df['Stoch_K'] = slowk
                df['Stoch_D'] = slowd
            except:
                # Fall back to manual calculation if TA-Lib fails
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
            
            # Add oversold/overbought crossing signals
            # Oversold exit (K crosses above 20 from below)
            df.loc[(df['Stoch_K'] > 20) & (df['Stoch_K'].shift(1) <= 20), 'Stochastic_Signal'] = 'oversold_exit'
            # Overbought exit (K crosses below 80 from above)
            df.loc[(df['Stoch_K'] < 80) & (df['Stoch_K'].shift(1) >= 80), 'Stochastic_Signal'] = 'overbought_exit'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator: {e}")
            return data  # Return original data in case of error

    def calculate_ema(self, data, periods=[9, 21, 50, 200], column='close'):
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
                # Try using TA-Lib for calculation
                try:
                    df[f'EMA_{period}'] = ta.EMA(df[column].values, timeperiod=period)
                except:
                    # Fall back to pandas EMA if TA-Lib fails
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
                
                # Golden cross (50 EMA crosses above 200 EMA)
                if 50 in periods and 200 in periods:
                    df.loc[(df[f'EMA_50'] > df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) <= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'golden_cross'
                    
                    # Death cross (50 EMA crosses below 200 EMA)
                    df.loc[(df[f'EMA_50'] < df[f'EMA_200']) & 
                          (df[f'EMA_50'].shift(1) >= df[f'EMA_200'].shift(1)), 
                          'EMA_Cross_Signal'] = 'death_cross'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating EMAs: {e}")
            return data  # Return original data in case of error

    def calculate_atr(self, data, period=14):
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
            
            # Try using TA-Lib for calculation
            try:
                df['ATR'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            except:
                # Fall back to manual calculation if TA-Lib fails
                # Calculate True Range
                df['TR1'] = abs(df['high'] - df['low'])
                df['TR2'] = abs(df['high'] - df['close'].shift(1))
                df['TR3'] = abs(df['low'] - df['close'].shift(1))
                df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
                
                # Calculate ATR as exponential moving average of TR
                df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
                
                # Clean up temporary columns
                df = df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1)
            
            # Add volatility regime detection
            df['Volatility_Regime'] = 'normal'
            
            # Calculate average ATR as percentage of price
            df['ATR_Pct'] = df['ATR'] / df['close'] * 100
            avg_atr_pct = df['ATR_Pct'].rolling(window=50).mean()
            std_atr_pct = df['ATR_Pct'].rolling(window=50).std()
            
            # High volatility: ATR percentage is more than 1.5 std devs above average
            df.loc[df['ATR_Pct'] > avg_atr_pct + 1.5 * std_atr_pct, 'Volatility_Regime'] = 'high'
            
            # Low volatility: ATR percentage is more than 1.5 std devs below average
            df.loc[df['ATR_Pct'] < avg_atr_pct - 1.5 * std_atr_pct, 'Volatility_Regime'] = 'low'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            return data  # Return original data in case of error

    def calculate_vwap(self, data, period=1):
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data.
            period (int): Number of days for VWAP calculation (intraday=1)
            
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
            
            # Add date column if timestamp is available
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp'].dt.date
            else:
                # If no timestamp, assume data is continuous and use arbitrary periods
                df['date'] = (np.arange(len(df)) / (24 * period)).astype(int)
            
            # Calculate typical price
            df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP components
            df['VP'] = df['TypicalPrice'] * df['volume']
            
            # Group by date and calculate cumulative sums within each group
            df['CumulativeVP'] = df.groupby('date')['VP'].cumsum()
            df['CumulativeVolume'] = df.groupby('date')['volume'].cumsum()
            
            # Calculate VWAP
            df['VWAP'] = df['CumulativeVP'] / df['CumulativeVolume']
            
            # Add VWAP signals
            df['VWAP_Signal'] = 'neutral'
            
            # Bullish when price crosses above VWAP
            df.loc[(df['close'] > df['VWAP']) & 
                  (df['close'].shift(1) <= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bullish'
            
            # Bearish when price crosses below VWAP
            df.loc[(df['close'] < df['VWAP']) & 
                  (df['close'].shift(1) >= df['VWAP'].shift(1)), 'VWAP_Signal'] = 'bearish'
            
            # Clean up temporary columns
            df = df.drop(['TypicalPrice', 'VP', 'CumulativeVP', 'CumulativeVolume', 'date'], axis=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating VWAP: {e}")
            return data  # Return original data in case of error

    def calculate_ichimoku(self, data):
        """
        Calculate Ichimoku Cloud indicator.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data.
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku components added.
        """
        try:
            # Make a copy of the data to avoid modifying the original
            df = data.copy()
            
            # Check if required columns exist
            required_columns = ['high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing columns for Ichimoku: {missing_columns}")
                return df
            
            # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
            
            # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['Ichimoku_Base'] = (high_26 + low_26) / 2
            
            # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
            
            # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
            
            # Calculate Chikou Span (Lagging Span): Close price shifted backwards by 26 periods
            df['Ichimoku_Lagging'] = df['close'].shift(-26)
            
            # Add cloud signals
            df['Ichimoku_Signal'] = 'neutral'
            
            # Bullish signal: Price above the cloud
            df.loc[(df['close'] > df['Ichimoku_SpanA']) & 
                  (df['close'] > df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bullish'
            
            # Bearish signal: Price below the cloud
            df.loc[(df['close'] < df['Ichimoku_SpanA']) & 
                  (df['close'] < df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bearish'
            
            # TK Cross (Conversion Line crosses above Base Line)
            df.loc[(df['Ichimoku_Conversion'] > df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) <= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bullish'
            
            # TK Cross (Conversion Line crosses below Base Line)
            df.loc[(df['Ichimoku_Conversion'] < df['Ichimoku_Base']) & 
                  (df['Ichimoku_Conversion'].shift(1) >= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bearish'
            
            # Add cloud thickness as a measure of trend strength
            df['Cloud_Thickness'] = abs(df['Ichimoku_SpanA'] - df['Ichimoku_SpanB'])
            df['Cloud_Thickness_Pct'] = df['Cloud_Thickness'] / df['close'] * 100
            
            # Color the cloud (useful for visualization)
            df['Cloud_Color'] = 'none'
            df.loc[df['Ichimoku_SpanA'] >= df['Ichimoku_SpanB'], 'Cloud_Color'] = 'green'
            df.loc[df['Ichimoku_SpanA'] < df['Ichimoku_SpanB'], 'Cloud_Color'] = 'red'
            
            return df
            
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud: {e}")
            return data  # Return original data in case of error
    
    def calculate_supertrend(self, data, period=10, multiplier=3.0):
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
            missing_columns = [col for col in required_columns if col not in df.columns]import pandas as pd
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

def calculate_ichimoku(data):
    """
    Calculate Ichimoku Cloud indicator.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data.
        
    Returns:
        pd.DataFrame: DataFrame with Ichimoku components added.
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Check if required columns exist
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns for Ichimoku: {missing_columns}")
            return df
        
        # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
        
        # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['Ichimoku_Base'] = (high_26 + low_26) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
        
        # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
        
        # Calculate Chikou Span (Lagging Span): Close price shifted backwards by 26 periods
        df['Ichimoku_Lagging'] = df['close'].shift(-26)
        
        # Add cloud signals
        df['Ichimoku_Signal'] = 'neutral'
        
        # Bullish signal: Price above the cloud
        df.loc[(df['close'] > df['Ichimoku_SpanA']) & 
              (df['close'] > df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bullish'
        
        # Bearish signal: Price below the cloud
        df.loc[(df['close'] < df['Ichimoku_SpanA']) & 
              (df['close'] < df['Ichimoku_SpanB']), 'Ichimoku_Signal'] = 'bearish'
        
        # TK Cross (Conversion Line crosses above Base Line)
        df.loc[(df['Ichimoku_Conversion'] > df['Ichimoku_Base']) & 
              (df['Ichimoku_Conversion'].shift(1) <= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bullish'
        
        # TK Cross (Conversion Line crosses below Base Line)
        df.loc[(df['Ichimoku_Conversion'] < df['Ichimoku_Base']) & 
              (df['Ichimoku_Conversion'].shift(1) >= df['Ichimoku_Base'].shift(1)), 'Ichimoku_Signal'] = 'tk_cross_bearish'
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating Ichimoku Cloud: {e}")
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
    df = calculate_ichimoku(df)
    
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
        
        # Count bullish and bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Check RSI
        if 'RSI_Signal' in df.columns:
            bullish_signals += (df['RSI_Signal'] == 'oversold').astype(int)
            bearish_signals += (df['RSI_Signal'] == 'overbought').astype(int)
            
        # Check MACD
        if 'MACD_Signal' in df.columns:
            bullish_signals += (df['MACD_Signal'] == 'bullish').astype(int)
            bearish_signals += (df['MACD_Signal'] == 'bearish').astype(int)
            
        # Check Bollinger Bands
        if 'BB_Signal' in df.columns:
            bullish_signals += (df['BB_Signal'] == 'oversold').astype(int)
            bearish_signals += (df['BB_Signal'] == 'overbought').astype(int)
            
        # Check Stochastic
        if 'Stochastic_Signal' in df.columns:
            bullish_signals += ((df['Stochastic_Signal'] == 'oversold') | 
                                (df['Stochastic_Signal'] == 'bullish_cross')).astype(int)
            bearish_signals += ((df['Stochastic_Signal'] == 'overbought') | 
                                (df['Stochastic_Signal'] == 'bearish_cross')).astype(int)
            
        # Check EMA Cross
        if 'EMA_Cross_Signal' in df.columns:
            bullish_signals += (df['EMA_Cross_Signal'] == 'bullish').astype(int)
            bearish_signals += (df['EMA_Cross_Signal'] == 'bearish').astype(int)
            
        # Check Ichimoku
        if 'Ichimoku_Signal' in df.columns:
            bullish_signals += ((df['Ichimoku_Signal'] == 'bullish') | 
                               (df['Ichimoku_Signal'] == 'tk_cross_bullish')).astype(int)
            bearish_signals += ((df['Ichimoku_Signal'] == 'bearish') | 
                               (df['Ichimoku_Signal'] == 'tk_cross_bearish')).astype(int)
            
        # Set combined signal
        # Strong bullish: At least 3 bullish signals and more bullish than bearish signals
        df.loc[(bullish_signals >= 3) & (bullish_signals > bearish_signals), 'Combined_Signal'] = 'strong_bullish'
        
        # Moderate bullish: At least 2 bullish signals and more bullish than bearish signals
        df.loc[(bullish_signals >= 2) & (bullish_signals > bearish_signals) & 
              (df['Combined_Signal'] == 'neutral'), 'Combined_Signal'] = 'moderate_bullish'
        
        # Strong bearish: At least 3 bearish signals and more bearish than bullish signals
        df.loc[(bearish_signals >= 3) & (bearish_signals > bullish_signals), 'Combined_Signal'] = 'strong_bearish'
        
        # Moderate bearish: At least 2 bearish signals and more bearish than bullish signals
        df.loc[(bearish_signals >= 2) & (bearish_signals > bullish_signals) & 
              (df['Combined_Signal'] == 'neutral'), 'Combined_Signal'] = 'moderate_bearish'
        
        return df
        
    except Exception as e:
        logging.error(f"Error generating combined signals: {e}")
        return data  # Return original data in case of error
 > 1:
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
        # Make a