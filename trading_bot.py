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

# Load configuration
with open("config.json") as config_file:
    config = json.load(config_file)

# Logging setup
logging.basicConfig(
    filename="logs/trading_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_PATH = "models/lstm_trading_model.keras"    # Load trained LSTM model

class LSTMTradingBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("SECRET_KEY"),
        })
        self.symbol = config["symbol"]
        self.timeframe = config["timeframe"]
        self.risk_per_trade = config["risk_per_trade"]
        self.trade_history = []
        self.model = None
    
    def check_model_validity(self):
        """
        Ensures the model exists and is recent. Retrains if necessary.
        """
        if not os.path.exists(MODEL_PATH):
            logging.info("Model not found. Training a new model...")
            historical_data = fetch_historical_data(self.symbol, self.timeframe)
            # Drop non-numeric columns and NaN values
            historical_data = historical_data.select_dtypes(include=[np.number]).dropna()
            train_model(historical_data)
        else:
            model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
            if model_age > timedelta(days=30):  # Retrain if older than 30 days
                logging.info("Model is outdated. Retraining...")
                historical_data = fetch_historical_data(self.symbol, self.timeframe)
                # Drop non-numeric columns and NaN values
                historical_data = historical_data.select_dtypes(include=[np.number]).dropna()
                train_model(historical_data)

        # Load the model
        self.model = keras.models.load_model(MODEL_PATH)

    def fetch_data(self, limit=200):
        """
        Fetches live data from the exchange.
        """
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def prepare_data(self, data):
        """
        Prepares data for model input by calculating indicators and scaling.
        """
        # Calculate existing indicators
        data = calculate_rsi(data, period=14)
        data = calculate_macd(data)

        # Add additional indicators (e.g., Bollinger Bands, Moving Averages)
        data["sma_50"] = data["close"].rolling(window=50).mean()  # Simple Moving Average (50 periods)
        data["sma_200"] = data["close"].rolling(window=200).mean()  # Simple Moving Average (200 periods)

        # Scale features
        data["close_scaled"] = (data["close"] - data["close"].min()) / (data["close"].max() - data["close"].min())
        data["RSI_scaled"] = (data["RSI"] - data["RSI"].min()) / (data["RSI"].max() - data["RSI"].min())
        data["MACD_scaled"] = (data["MACD"] - data["MACD"].min()) / (data["MACD"].max() - data["MACD"].min())
        data["sma_50_scaled"] = (data["sma_50"] - data["sma_50"].min()) / (data["sma_50"].max() - data["sma_50"].min())
        data["sma_200_scaled"] = (data["sma_200"] - data["sma_200"].min()) / (data["sma_200"].max() - data["sma_200"].min())

        # Drop NaN values and return
        return data.dropna()

    def check_signal_with_lstm(self, data, sequence_length=50):
        """
        Predicts trade signals using the LSTM model.
        """
        recent_data = data.tail(sequence_length)[["close_scaled", "RSI_scaled", "MACD_scaled"]].values
        if len(recent_data) < sequence_length:
            return None

        prediction = self.model.predict(recent_data[np.newaxis, :, :])
        if prediction > 0.5:
            return "buy"
        else:
            return "sell"

    def execute_trade(self, signal, data):
        """
        Executes a trade based on the signal.
        """
        balance = self.exchange.fetch_balance()
        usdt_balance = balance["free"]["USDT"]
        current_price = data["close"].iloc[-1]
        position_size = (usdt_balance * self.risk_per_trade) / current_price

        trade = {
            "timestamp": pd.Timestamp.now(),
            "signal": signal,
            "price": current_price,
            "size": position_size
        }
        self.trade_history.append(trade)
        logging.info(f"Executed {signal.capitalize()} Order at {current_price}")

    def run(self, interval=60):
        """
        Main loop of the trading bot.
        """
        self.check_model_validity()
        while True:
            try:
                data = self.fetch_data()
                data = self.prepare_data(data)
                signal = self.check_signal_with_lstm(data)
                logging.info(f"Signal: {signal}")

                if signal:
                    self.execute_trade(signal, data)

                pd.DataFrame(self.trade_history).to_csv("data/trade_history.csv", index=False)
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Error: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    bot = LSTMTradingBot()
    bot.run()
