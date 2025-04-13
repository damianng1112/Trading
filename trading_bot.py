import ccxt
import os
import pandas as pd
import numpy as np
import time
import logging
import tensorflow as tf
import keras
import json
from utils.indicators import calculate_rsi, calculate_macd
from utils.model_training import train_model
from utils.data_loader import fetch_historical_data
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Enhanced logging setup
logging.basicConfig(
    filename="logs/trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler()  # Also log to console
    ]
)

MODEL_PATH = "models/lstm_trading_model.keras"

class LSTMTradingBot:
    def __init__(self):
        # Load configuration with error handling
        try:
            with open("config.json") as config_file:
                self.config = json.load(config_file)
        except FileNotFoundError:
            logging.error("config.json not found. Creating default configuration.")
            self.config = {
                "symbol": "BTC/USDT",
                "timeframe": "15m",
                "risk_per_trade": 0.02,  # 2% risk per trade
                "stop_loss_pct": 0.03,   # 3% stop loss
                "take_profit_pct": 0.06,  # 6% take profit
                "max_drawdown": 0.15     # 15% max drawdown
            }
            with open("config.json", "w") as config_file:
                json.dump(self.config, config_file, indent=4)
        
        # Initialize exchange with better error handling
        try:
            api_key = os.getenv("API_KEY")
            secret_key = os.getenv("SECRET_KEY")
            
            if not api_key or not secret_key:
                logging.warning("API keys not found in environment variables. Running in simulation mode.")
                self.simulation_mode = True
            else:
                self.exchange = ccxt.binance({
                    "apiKey": api_key,
                    "secret": secret_key,
                })
                self.simulation_mode = False
                
                # Test connection
                self.exchange.fetch_balance()
                logging.info("Successfully connected to exchange")
        except Exception as e:
            logging.error(f"Failed to connect to exchange: {e}")
            logging.warning("Running in simulation mode")
            self.simulation_mode = True
            
        # Load other configurations
        self.symbol = self.config["symbol"]
        self.timeframe = self.config["timeframe"]
        self.risk_per_trade = self.config["risk_per_trade"]
        self.stop_loss_pct = self.config["stop_loss_pct"]
        self.take_profit_pct = self.config["take_profit_pct"]
        self.max_drawdown = self.config["max_drawdown"]
        
        # Initialize trading state
        self.trade_history = []
        self.active_positions = []
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.current_drawdown = 0
        self.initial_balance = 0
        self.current_balance = 0
        
        # Create required directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Load trading history if exists
        if os.path.exists("data/trade_history.csv"):
            self.trade_history = pd.read_csv("data/trade_history.csv").to_dict('records')
    
    def check_model_validity(self):
        """
        Ensures the model exists and is recent. Retrains if necessary.
        Returns True if model is valid, False otherwise.
        """
        try:
            if not os.path.exists(MODEL_PATH):
                logging.info("Model not found. Training a new model...")
                historical_data = self.fetch_historical_data(limit=1000)  # Get more data for training
                if historical_data is None or len(historical_data) < 200:
                    logging.error("Not enough historical data to train model")
                    return False
                    
                # Calculate indicators and prepare data
                prepared_data = self.prepare_data(historical_data)
                train_model(prepared_data)
            else:
                model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
                if model_age > timedelta(days=7):  # Retrain weekly instead of monthly
                    logging.info("Model is outdated. Retraining...")
                    historical_data = self.fetch_historical_data(limit=1000)
                    if historical_data is None or len(historical_data) < 200:
                        logging.error("Not enough historical data to train model")
                        return False
                        
                    # Calculate indicators and prepare data
                    prepared_data = self.prepare_data(historical_data)
                    train_model(prepared_data)

            # Load the model
            try:
                self.model = keras.models.load_model(MODEL_PATH)
                logging.info(f"Model loaded successfully: {MODEL_PATH}")
                return True
            except Exception as e:
                logging.error(f"Failed to load model: {e}")
                return False
                
        except Exception as e:
            logging.error(f"Error checking model validity: {e}")
            return False

    def fetch_historical_data(self, limit=200):
        """
        Fetches historical data with better error handling.
        """
        try:
            if self.simulation_mode:
                # In simulation mode, use the data loader utility
                return fetch_historical_data(self.symbol, self.timeframe, limit)
            else:
                # In live mode, use the exchange connection
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                return df
        except Exception as e:
            logging.error(f"Error fetching historical data: {e}")
            return None

    def prepare_data(self, data):
        """
        Prepares data for model input by calculating indicators and scaling.
        Now includes proper error handling and normalization.
        """
        try:
            if data is None or len(data) < 50:
                logging.error("Not enough data to prepare")
                return None
                
            # Make a copy to avoid modifications to original data
            df = data.copy()
            
            # Calculate technical indicators
            df = calculate_rsi(df, period=14)
            df = calculate_macd(df)
            
            # Add additional indicators
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["sma_200"] = df["close"].rolling(window=200).mean()
            
            # Add price momentum features
            df["price_change_1"] = df["close"].pct_change(1)
            df["price_change_5"] = df["close"].pct_change(5)
            df["price_change_10"] = df["close"].pct_change(10)
            
            # Add volatility indicators
            df["volatility"] = df["close"].rolling(window=14).std()
            
            # Create feature columns for model input
            feature_columns = [
                "close", "RSI", "MACD", "Signal_Line", 
                "sma_50", "sma_200", "volatility",
                "price_change_1", "price_change_5", "price_change_10"
            ]
            
            # Drop rows with NaN values
            df = df.dropna()
            
            if len(df) < 50:
                logging.error("Not enough data after calculating indicators")
                return None
                
            # Scale features using MinMaxScaler
            features = df[feature_columns].values
            self.feature_scaler = MinMaxScaler().fit(features)
            scaled_features = self.feature_scaler.transform(features)
            
            # Create a new DataFrame with scaled features
            for i, col in enumerate(feature_columns):
                df[f"{col}_scaled"] = scaled_features[:, i]
            
            return df
            
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            return None

    def check_signal_with_lstm(self, data, sequence_length=50, confidence_threshold=0.6):
        """
        Predicts trade signals using the LSTM model with confidence threshold.
        Also incorporates traditional indicator confirmation.
        """
        try:
            if data is None or len(data) < sequence_length:
                logging.warning(f"Not enough data for prediction. Need {sequence_length} rows, got {len(data) if data is not None else 0}")
                return None
                
            # Get feature columns for prediction
            feature_columns = [
                "close_scaled", "RSI_scaled", "MACD_scaled", "Signal_Line_scaled",
                "sma_50_scaled", "sma_200_scaled", "volatility_scaled",
                "price_change_1_scaled", "price_change_5_scaled", "price_change_10_scaled"
            ]
            
            # Check if all required columns exist
            missing_columns = [col for col in feature_columns if col not in data.columns]
            if missing_columns:
                logging.error(f"Missing columns for prediction: {missing_columns}")
                return None
                
            # Get sequence of recent data
            recent_data = data.tail(sequence_length)[feature_columns].values
            
            # Make prediction
            prediction = self.model.predict(recent_data[np.newaxis, :, :], verbose=0)
            pred_value = prediction[0][0]
            
            # Log raw prediction for debugging
            logging.info(f"Raw model prediction: {pred_value}")
            
            # Get confirmation from traditional indicators
            last_row = data.iloc[-1]
            rsi = last_row["RSI"]
            macd = last_row["MACD"]
            signal_line = last_row["Signal_Line"]
            sma_50 = last_row["sma_50"]
            sma_200 = last_row["sma_200"]
            
            # Define signal based on model prediction and confidence threshold
            signal = None
            
            if pred_value > confidence_threshold:
                # Model suggests BUY
                # Confirm with traditional indicators
                if (rsi < 70 and  # Not overbought
                    macd > signal_line and  # MACD bullish
                    sma_50 > sma_200):  # Golden cross (uptrend)
                    signal = "buy"
                    logging.info("Buy signal confirmed by traditional indicators")
                else:
                    logging.info("Buy signal from model but not confirmed by indicators")
                    
            elif pred_value < (1 - confidence_threshold):
                # Model suggests SELL
                # Confirm with traditional indicators
                if (rsi > 30 and  # Not oversold
                    macd < signal_line and  # MACD bearish
                    sma_50 < sma_200):  # Death cross (downtrend)
                    signal = "sell"
                    logging.info("Sell signal confirmed by traditional indicators")
                else:
                    logging.info("Sell signal from model but not confirmed by indicators")
            
            return signal
            
        except Exception as e:
            logging.error(f"Error checking signal: {e}")
            return None

    def calculate_position_size(self, price):
        """
        Calculate position size based on risk management rules.
        """
        try:
            if self.simulation_mode:
                # In simulation mode, assume we have $10,000
                balance = 10000
            else:
                # Get actual balance
                account_balance = self.exchange.fetch_balance()
                balance = account_balance["free"].get("USDT", 0)
            
            # Calculate position size based on risk percentage
            risk_amount = balance * self.risk_per_trade
            position_size = risk_amount / (price * self.stop_loss_pct)
            
            # Log the calculation
            logging.info(f"Balance: ${balance}, Risk amount: ${risk_amount}, Position size: {position_size}")
            
            return position_size
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0

    def execute_trade(self, signal, data):
        """
        Executes a trade based on the signal with risk management.
        """
        try:
            if data is None or len(data) == 0:
                logging.error("No data available for trade execution")
                return
                
            current_price = data["close"].iloc[-1]
            position_size = self.calculate_position_size(current_price)
            
            if position_size <= 0:
                logging.warning("Position size calculation returned zero or negative value")
                return
                
            # Calculate stop loss and take profit levels
            stop_loss = current_price * (1 - self.stop_loss_pct) if signal == "buy" else current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct) if signal == "buy" else current_price * (1 - self.take_profit_pct)
            
            # Record the trade
            trade = {
                "timestamp": pd.Timestamp.now(),
                "signal": signal,
                "price": current_price,
                "size": position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "status": "open"
            }
            
            if not self.simulation_mode:
                try:
                    # Execute the actual trade on the exchange
                    if signal == "buy":
                        order = self.exchange.create_market_buy_order(self.symbol, position_size)
                    else:  # sell
                        order = self.exchange.create_market_sell_order(self.symbol, position_size)
                        
                    trade["order_id"] = order["id"]
                    logging.info(f"Order executed: {order}")
                except Exception as e:
                    logging.error(f"Failed to execute order: {e}")
                    return
            
            # Add trade to history and active positions
            self.trade_history.append(trade)
            self.active_positions.append(trade)
            
            # Log trade execution
            logging.info(f"Executed {signal.capitalize()} Order at {current_price} | " +
                        f"Stop Loss: {stop_loss} | Take Profit: {take_profit}")
            
            # Save updated trade history
            self.save_trade_history()
            
        except Exception as e:
            logging.error(f"Error executing trade: {e}")

    def update_positions(self, current_price):
        """
        Updates active positions and checks for stop loss/take profit triggers.
        """
        if not self.active_positions:
            return
            
        new_active_positions = []
        
        for position in self.active_positions:
            # Check if position should be closed
            should_close = False
            close_reason = ""
            profit_loss = 0
            
            if position["signal"] == "buy":
                # For buy positions
                if current_price <= position["stop_loss"]:
                    should_close = True
                    close_reason = "stop_loss"
                    profit_loss = (position["stop_loss"] / position["price"] - 1) * 100
                elif current_price >= position["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"
                    profit_loss = (position["take_profit"] / position["price"] - 1) * 100
            else:
                # For sell positions
                if current_price >= position["stop_loss"]:
                    should_close = True
                    close_reason = "stop_loss"
                    profit_loss = (position["price"] / position["stop_loss"] - 1) * 100
                elif current_price <= position["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"
                    profit_loss = (position["price"] / position["take_profit"] - 1) * 100
            
            if should_close:
                # Close the position
                position["close_timestamp"] = pd.Timestamp.now()
                position["close_price"] = current_price
                position["close_reason"] = close_reason
                position["profit_loss_pct"] = profit_loss
                position["status"] = "closed"
                
                logging.info(f"Position closed: {close_reason.upper()} | " +
                            f"Profit/Loss: {profit_loss:.2f}% | " +
                            f"Entry: {position['price']} | Exit: {current_price}")
                
                if not self.simulation_mode:
                    try:
                        # Execute the closing order
                        if position["signal"] == "buy":
                            order = self.exchange.create_market_sell_order(self.symbol, position["size"])
                        else:
                            order = self.exchange.create_market_buy_order(self.symbol, position["size"])
                            
                        position["close_order_id"] = order["id"]
                    except Exception as e:
                        logging.error(f"Failed to close position: {e}")
            else:
                # Keep position active
                new_active_positions.append(position)
        
        # Update active positions
        self.active_positions = new_active_positions
        
        # Save updated trade history
        self.save_trade_history()

    def save_trade_history(self):
        """
        Saves trade history to CSV file.
        """
        try:
            pd.DataFrame(self.trade_history).to_csv("data/trade_history.csv", index=False)
        except Exception as e:
            logging.error(f"Failed to save trade history: {e}")

    def calculate_performance_metrics(self):
        """
        Calculates performance metrics.
        """
        if not self.trade_history:
            return {}
            
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(self.trade_history)
        
        # Filter closed trades
        closed_trades = trades_df[trades_df["status"] == "closed"]
        
        if len(closed_trades) == 0:
            return {"total_trades": 0}
            
        # Calculate metrics
        metrics = {
            "total_trades": len(closed_trades),
            "win_rate": len(closed_trades[closed_trades["profit_loss_pct"] > 0]) / len(closed_trades) * 100,
            "avg_profit": closed_trades["profit_loss_pct"].mean(),
            "max_profit": closed_trades["profit_loss_pct"].max(),
            "max_loss": closed_trades["profit_loss_pct"].min(),
            "profit_factor": abs(closed_trades[closed_trades["profit_loss_pct"] > 0]["profit_loss_pct"].sum() / 
                              closed_trades[closed_trades["profit_loss_pct"] < 0]["profit_loss_pct"].sum()) 
                              if closed_trades[closed_trades["profit_loss_pct"] < 0]["profit_loss_pct"].sum() != 0 else float('inf')
        }
        
        return metrics

    def generate_performance_report(self):
        """
        Generates and saves a performance report.
        """
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            logging.info("No trade data for performance report")
            return
            
        # Create a report
        report = f"""
        =========================================
        TRADING BOT PERFORMANCE REPORT
        =========================================
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Symbol: {self.symbol}
        Timeframe: {self.timeframe}
        
        METRICS:
        -----------------------------------------
        Total Trades: {metrics.get('total_trades', 0)}
        Win Rate: {metrics.get('win_rate', 0):.2f}%
        Average Profit/Loss: {metrics.get('avg_profit', 0):.2f}%
        Maximum Profit: {metrics.get('max_profit', 0):.2f}%
        Maximum Loss: {metrics.get('max_loss', 0):.2f}%
        Profit Factor: {metrics.get('profit_factor', 0):.2f}
        
        RISK PARAMETERS:
        -----------------------------------------
        Risk Per Trade: {self.risk_per_trade * 100}%
        Stop Loss: {self.stop_loss_pct * 100}%
        Take Profit: {self.take_profit_pct * 100}%
        Maximum Drawdown Limit: {self.max_drawdown * 100}%
        =========================================
        """
        
        logging.info(report)
        
        # Save report to file
        with open("data/performance_report.txt", "w") as f:
            f.write(report)
            
        # Generate performance chart if we have enough data
        if metrics.get('total_trades', 0) > 0:
            self.generate_performance_chart()

    def generate_performance_chart(self):
        """
        Generates a performance chart based on trade history.
        """
        try:
            trades_df = pd.DataFrame(self.trade_history)
            
            # Filter closed trades
            closed_trades = trades_df[trades_df["status"] == "closed"].copy()
            
            if len(closed_trades) == 0:
                return
                
            # Convert timestamps to datetime if they're strings
            if isinstance(closed_trades["timestamp"].iloc[0], str):
                closed_trades["timestamp"] = pd.to_datetime(closed_trades["timestamp"])
                
            if "close_timestamp" in closed_trades.columns and isinstance(closed_trades["close_timestamp"].iloc[0], str):
                closed_trades["close_timestamp"] = pd.to_datetime(closed_trades["close_timestamp"])
            
            # Sort by timestamp
            closed_trades = closed_trades.sort_values("timestamp")
            
            # Calculate cumulative returns
            closed_trades["cumulative_return"] = (1 + closed_trades["profit_loss_pct"] / 100).cumprod() - 1
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot cumulative returns
            ax1.plot(closed_trades["close_timestamp"], closed_trades["cumulative_return"] * 100, 'b-')
            ax1.set_title(f"Trading Performance - {self.symbol} ({self.timeframe})")
            ax1.set_ylabel("Cumulative Return (%)")
            ax1.grid(True)
            
            # Plot individual trade returns
            colors = ['g' if x > 0 else 'r' for x in closed_trades["profit_loss_pct"]]
            ax2.bar(range(len(closed_trades)), closed_trades["profit_loss_pct"], color=colors)
            ax2.set_title("Individual Trade Returns")
            ax2.set_xlabel("Trade Number")
            ax2.set_ylabel("Profit/Loss (%)")
            ax2.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig("data/performance_chart.png")
            plt.close()
            
            logging.info("Performance chart saved to data/performance_chart.png")
            
        except Exception as e:
            logging.error(f"Failed to generate performance chart: {e}")

    def run_backtesting(self, start_date=None, end_date=None):
        """
        Runs backtesting on historical data.
        """
        logging.info("Starting backtesting mode...")
        
        # Fetch historical data
        historical_data = self.fetch_historical_data(limit=1000)
        
        if historical_data is None or len(historical_data) < 200:
            logging.error("Not enough historical data for backtesting")
            return
        
        # Filter by date range if specified
        if start_date:
            historical_data = historical_data[historical_data["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            historical_data = historical_data[historical_data["timestamp"] <= pd.to_datetime(end_date)]
        
        # Prepare data
        prepared_data = self.prepare_data(historical_data)
        
        if prepared_data is None or len(prepared_data) < 100:
            logging.error("Not enough prepared data for backtesting")
            return
        
        # Reset trade history and positions for backtesting
        self.trade_history = []
        self.active_positions = []
        
        # Ensure model is loaded
        if not self.check_model_validity():
            logging.error("Failed to load model for backtesting")
            return
        
        # Set initial balance for backtesting
        self.initial_balance = 10000  # $10,000 starting capital
        self.current_balance = self.initial_balance
        
        # Run backtesting simulation
        logging.info(f"Running backtesting on {len(prepared_data)} data points")
        
        # Use a sliding window approach for backtesting
        sequence_length = 50  # For LSTM input
        
        for i in range(sequence_length, len(prepared_data)):
            # Get data up to current point
            current_data = prepared_data.iloc[:i+1]
            current_price = current_data["close"].iloc[-1]
            
            # Update existing positions
            self.update_positions(current_price)
            
            # Check for new trading signal
            signal = self.check_signal_with_lstm(current_data, sequence_length=sequence_length)
            
            if signal:
                self.execute_trade(signal, current_data)
        
        # Close any remaining open positions at the end
        final_price = prepared_data["close"].iloc[-1]
        for position in self.active_positions:
            position["close_timestamp"] = prepared_data["timestamp"].iloc[-1]
            position["close_price"] = final_price
            position["close_reason"] = "backtest_end"
            
            # Calculate profit/loss
            if position["signal"] == "buy":
                profit_loss = (final_price / position["price"] - 1) * 100
            else:
                profit_loss = (position["price"] / final_price - 1) * 100
                
            position["profit_loss_pct"] = profit_loss
            position["status"] = "closed"
        
        self.active_positions = []
        
        # Generate performance report
        self.generate_performance_report()
        
        logging.info("Backtesting completed")

    def run(self, interval=60):
        """
        Main loop of the trading bot with improved error handling and monitoring.
        """
        logging.info(f"Starting trading bot for {self.symbol} on {self.timeframe} timeframe")
        
        # First, check if the model is valid
        if not self.check_model_validity():
            logging.error("Failed to validate or train model. Cannot start trading bot.")
            return
        
        # Main loop
        while True:
            try:
                # Fetch and prepare data
                data = self.fetch_historical_data()
                
                if data is None:
                    logging.error("Failed to fetch data. Retrying in 60 seconds.")
                    time.sleep(60)
                    continue
                
                prepared_data = self.prepare_data(data)
                
                if prepared_data is None:
                    logging.error("Failed to prepare data. Retrying in 60 seconds.")
                    time.sleep(60)
                    continue
                
                # Get current price
                current_price = prepared_data["close"].iloc[-1]
                
                # Update existing positions
                self.update_positions(current_price)
                
                # Check for new trading signal
                signal = self.check_signal_with_lstm(prepared_data)
                logging.info(f"Signal: {signal}")
                
                # Execute trade if signal exists
                if signal:
                    self.execute_trade(signal, prepared_data)
                
                # Generate performance report every 24 hours (24 * 60 * 60 seconds)
                if time.time() % (24 * 60 * 60) < interval:
                    self.generate_performance_report()
                
                # Sleep until next interval
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logging.info("Trading bot stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(interval)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LSTM Trading Bot")
    parser.add_argument("--backtest", action="store_true", help="Run in backtesting mode")
    parser.add_argument("--start", type=str, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds between checks")
    
    args = parser.parse_args()
    
    bot = LSTMTradingBot()
    
    if args.backtest:
        bot.run_backtesting(start_date=args.start, end_date=args.end)
    else:
        bot.run(interval=args.interval)