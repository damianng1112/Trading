import numpy as np
import pandas as pd
import logging

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.name = "BaseStrategy"
    
    def generate_signal(self, data):
        """
        Generate trading signal based on the strategy.
        
        Args:
            data (pd.DataFrame): Price and indicator data
            
        Returns:
            str or None: 'buy', 'sell', or None
        """
        raise NotImplementedError("Strategy must implement generate_signal method")

class RSIStrategy(TradingStrategy):
    """RSI-based trading strategy"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "RSI Strategy"
        
        # Get RSI parameters from config or use defaults
        rsi_config = self.config.get('indicators', {}).get('rsi', {})
        self.period = rsi_config.get('period', 14)
        self.oversold = rsi_config.get('oversold', 30)
        self.overbought = rsi_config.get('overbought', 70)
    
    def generate_signal(self, data):
        """Generate signals based on RSI"""
        if 'RSI' not in data.columns:
            logging.warning("RSI column not found in data")
            return None
            
        # Get the most recent RSI value
        current_rsi = data['RSI'].iloc[-1]
        prev_rsi = data['RSI'].iloc[-2] if len(data) > 1 else None
        
        # RSI crosses below oversold threshold -> buy
        if prev_rsi is not None and prev_rsi > self.oversold and current_rsi < self.oversold:
            return "buy"
            
        # RSI crosses above overbought threshold -> sell
        elif prev_rsi is not None and prev_rsi < self.overbought and current_rsi > self.overbought:
            return "sell"
            
        # RSI exits oversold zone from below -> buy
        elif prev_rsi is not None and prev_rsi < self.oversold and current_rsi > self.oversold:
            return "buy"
            
        # RSI exits overbought zone from above -> sell
        elif prev_rsi is not None and prev_rsi > self.overbought and current_rsi < self.overbought:
            return "sell"
            
        return None

class MACDStrategy(TradingStrategy):
    """MACD-based trading strategy"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "MACD Strategy"
    
    def generate_signal(self, data):
        """Generate signals based on MACD crossovers"""
        if 'MACD' not in data.columns or 'Signal_Line' not in data.columns:
            logging.warning("MACD or Signal_Line column not found in data")
            return None
            
        # Check for MACD crossover
        current_macd = data['MACD'].iloc[-1]
        current_signal = data['Signal_Line'].iloc[-1]
        
        prev_macd = data['MACD'].iloc[-2] if len(data) > 1 else None
        prev_signal = data['Signal_Line'].iloc[-2] if len(data) > 1 else None
        
        # MACD crosses above signal line -> buy
        if (prev_macd is not None and prev_signal is not None and 
            prev_macd < prev_signal and current_macd > current_signal):
            return "buy"
            
        # MACD crosses below signal line -> sell
        elif (prev_macd is not None and prev_signal is not None and 
              prev_macd > prev_signal and current_macd < current_signal):
            return "sell"
            
        return None

class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands-based trading strategy"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "Bollinger Bands Strategy"
    
    def generate_signal(self, data):
        """Generate signals based on Bollinger Bands"""
        required_columns = ['close', 'BB_Upper', 'BB_Middle', 'BB_Lower']
        if not all(col in data.columns for col in required_columns):
            logging.warning("Required Bollinger Bands columns not found in data")
            return None
            
        # Get current values
        current_price = data['close'].iloc[-1]
        current_upper = data['BB_Upper'].iloc[-1]
        current_lower = data['BB_Lower'].iloc[-1]
        current_middle = data['BB_Middle'].iloc[-1]
        
        # Get previous values
        prev_price = data['close'].iloc[-2] if len(data) > 1 else None
        prev_upper = data['BB_Upper'].iloc[-2] if len(data) > 1 else None
        prev_lower = data['BB_Lower'].iloc[-2] if len(data) > 1 else None
        
        # Price crosses below lower band and then moves back up -> buy
        if (prev_price is not None and prev_lower is not None and 
            prev_price < prev_lower and current_price > current_lower):
            return "buy"
            
        # Price crosses above upper band and then moves back down -> sell
        elif (prev_price is not None and prev_upper is not None and 
              prev_price > prev_upper and current_price < current_upper):
            return "sell"
              
        # Price closes below lower band -> buy (oversold)
        elif current_price < current_lower:
            return "buy"
            
        # Price closes above upper band -> sell (overbought)
        elif current_price > current_upper:
            return "sell"
            
        return None

class PriceActionStrategy(TradingStrategy):
    """Price action-based trading strategy"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "Price Action Strategy"
    
    def generate_signal(self, data):
        """Generate signals based on price action patterns"""
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            logging.warning("Required price action columns not found in data")
            return None
            
        # Check for bullish engulfing pattern
        if len(data) >= 2:
            curr_candle = data.iloc[-1]
            prev_candle = data.iloc[-2]
            
            # Bullish engulfing
            if (prev_candle['close'] < prev_candle['open'] and  # Previous candle is bearish
                curr_candle['close'] > curr_candle['open'] and  # Current candle is bullish
                curr_candle['open'] < prev_candle['close'] and  # Current open below previous close
                curr_candle['close'] > prev_candle['open']):    # Current close above previous open
                return "buy"
                
            # Bearish engulfing
            elif (prev_candle['close'] > prev_candle['open'] and  # Previous candle is bullish
                  curr_candle['close'] < curr_candle['open'] and  # Current candle is bearish
                  curr_candle['open'] > prev_candle['close'] and  # Current open above previous close
                  curr_candle['close'] < prev_candle['open']):    # Current close below previous open
                return "sell"
        
        # Check for doji pattern (indecision)
        if len(data) >= 1:
            curr_candle = data.iloc[-1]
            body_size = abs(curr_candle['close'] - curr_candle['open'])
            total_range = curr_candle['high'] - curr_candle['low']
            
            # If body is very small compared to total range (doji)
            if total_range > 0 and body_size / total_range < 0.1:
                # Doji at top of uptrend
                if data['close'].iloc[-5:].is_monotonic_increasing:
                    return "sell"
                # Doji at bottom of downtrend
                elif data['close'].iloc[-5:].is_monotonic_decreasing:
                    return "buy"
        
        return None

class CombinedStrategy(TradingStrategy):
    """Strategy that combines multiple sub-strategies with a voting mechanism"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "Combined Strategy"
        
        # Initialize sub-strategies
        self.strategies = [
            RSIStrategy(config),
            MACDStrategy(config),
            BollingerBandsStrategy(config),
            PriceActionStrategy(config)
        ]
        
        # Weight for each strategy (can be adjusted in config)
        strategy_weights = config.get('strategy_weights', {}) if config else {}
        self.weights = {
            "RSI Strategy": strategy_weights.get("rsi", 1.0),
            "MACD Strategy": strategy_weights.get("macd", 1.0),
            "Bollinger Bands Strategy": strategy_weights.get("bollinger", 1.0),
            "Price Action Strategy": strategy_weights.get("price_action", 1.0)
        }
    
    def generate_signal(self, data):
        """Generate signals by combining multiple strategies"""
        buy_score = 0
        sell_score = 0
        
        # Collect signals from all strategies
        for strategy in self.strategies:
            signal = strategy.generate_signal(data)
            weight = self.weights.get(strategy.name, 1.0)
            
            if signal == "buy":
                buy_score += weight
            elif signal == "sell":
                sell_score += weight
        
        # LSTM model signal if available
        if 'model_prediction' in data.columns:
            model_pred = data['model_prediction'].iloc[-1]
            model_weight = 2.0  # Higher weight for the ML model
            
            if model_pred > 0.65:  # Strong buy signal
                buy_score += model_weight
            elif model_pred < 0.35:  # Strong sell signal
                sell_score += model_weight
        
        # Decision thresholds (configurable)
        threshold = self.config.get('signal_threshold', 2.0) if self.config else 2.0
        
        # Generate final signal based on scores
        if buy_score >= threshold and buy_score > sell_score:
            return "buy"
        elif sell_score >= threshold and sell_score > buy_score:
            return "sell"
        
        return None

class MarketRegimeStrategy(TradingStrategy):
    """Strategy that adapts to different market regimes (trending, ranging, volatile)"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "Market Regime Strategy"
        
        # Initialize sub-strategies
        self.trend_strategy = MACDStrategy(config)
        self.range_strategy = RSIStrategy(config)
        self.volatile_strategy = BollingerBandsStrategy(config)
    
    def detect_market_regime(self, data):
        """
        Detect the current market regime.
        
        Returns:
            str: 'trending', 'ranging', or 'volatile'
        """
        if 'ATR' not in data.columns or 'close' not in data.columns:
            return "unknown"
            
        # Calculate volatility using ATR relative to price
        recent_data = data.tail(20)
        atr = recent_data['ATR'].iloc[-1]
        price = recent_data['close'].iloc[-1]
        volatility = atr / price * 100  # ATR as percentage of price
        
        # Calculate trend strength
        if 'EMA_9' in data.columns and 'EMA_50' in data.columns:
            ema_short = recent_data['EMA_9'].iloc[-1]
            ema_long = recent_data['EMA_50'].iloc[-1]
            ema_diff = abs(ema_short - ema_long) / ema_long * 100  # Difference as percentage
            
            # Calculate price range
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / price * 100
            
            # Determine regime
            if volatility > 5:  # High volatility threshold
                return "volatile"
            elif ema_diff > 2:  # Strong trend threshold
                return "trending"
            elif price_range < 10:  # Range-bound threshold
                return "ranging"
        
        # Default to ranging if we can't determine
        return "ranging"
    
    def generate_signal(self, data):
        """Generate signals based on detected market regime"""
        regime = self.detect_market_regime(data)
        
        if regime == "trending":
            return self.trend_strategy.generate_signal(data)
        elif regime == "ranging":
            return self.range_strategy.generate_signal(data)
        elif regime == "volatile":
            return self.volatile_strategy.generate_signal(data)
        
        return None

class StrategyFactory:
    """Factory class to create strategy instances"""
    
    @staticmethod
    def create_strategy(strategy_name, config=None):
        """
        Create a strategy instance by name.
        
        Args:
            strategy_name (str): Name of the strategy
            config (dict): Configuration dictionary
            
        Returns:
            TradingStrategy: Strategy instance
        """
        strategies = {
            "rsi": RSIStrategy,
            "macd": MACDStrategy,
            "bollinger": BollingerBandsStrategy,
            "price_action": PriceActionStrategy,
            "combined": CombinedStrategy,
            "market_regime": MarketRegimeStrategy
        }
        
        strategy_class = strategies.get(strategy_name.lower())
        if strategy_class:
            return strategy_class(config)
        else:
            logging.error(f"Strategy '{strategy_name}' not found")
            return None