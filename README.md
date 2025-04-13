# Advanced Cryptocurrency Trading Bot

This bot automates cryptocurrency trading using deep learning, multiple technical indicators, and sophisticated risk management strategies.

## Features

- **LSTM-based Price Prediction**: Uses deep learning to forecast cryptocurrency price movements
- **Technical Analysis**: Implements multiple indicators including RSI, MACD, Bollinger Bands, and more
- **Advanced Risk Management**: Includes position sizing, stop-loss, take-profit, and maximum drawdown protection
- **Multiple Trading Strategies**: Choose from several strategy implementations or combine them
- **Market Regime Detection**: Adapts trading approach based on market conditions (trending, ranging, volatile)
- **Backtesting Module**: Test your strategies on historical data before deploying
- **Performance Visualization**: Interactive dashboard to monitor bot performance
- **Configurability**: Extensive configuration options via JSON

## System Architecture

```
crypto_trading_bot/
│
├── trading_bot.py            # Main bot implementation
├── config.json               # Configuration file
│
├── data/                     # Data directory
│   ├── historical_data.csv   # Historical price data
│   └── trade_history.csv     # Record of executed trades
│
├── models/                   # Model directory
│   ├── lstm_trading_model.keras  # Saved model
│   ├── model_stats.pkl       # Model statistics
│   └── report/               # Model reports directory
│
├── logs/                     # Log files
│   └── trading_bot.log       # Bot logs
│
├── utils/                    # Utility modules
│   ├── data_loader.py        # Data loading utilities
│   ├── indicators.py         # Technical indicators
│   ├── strategies.py         # Trading strategies
│   ├── model_training.py     # Model training functions
│   └── model_visualization.py # Visualization utilities
│
└── dashboard/                # Dashboard for visualizing performance
    ├── trading_bot_dashboard.py  # Dashboard implementation
    └── reports/              # Dashboard reports
```

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crypto_trading_bot.git
   cd crypto_trading_bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your trading parameters in `config.json`:
   ```json
   {
     "symbol": "BTC/USDT",
     "timeframe": "15m",
     "risk_per_trade": 0.02,
     "stop_loss_pct": 0.03,
     "take_profit_pct": 0.06,
     "max_drawdown": 0.15
   }
   ```

4. Set up your exchange API keys as environment variables:

   **Option 1: Using a .env file (recommended for development)**
   
   Create a `.env` file in the project root:
   ```
   API_KEY=your_api_key_here
   SECRET_KEY=your_secret_key_here
   ```
   
   Then install python-dotenv:
   ```bash
   pip install python-dotenv
   ```
   
   **Option 2: Set environment variables directly**
   
   For Linux/macOS:
   ```bash
   export API_KEY="your_api_key_here"
   export SECRET_KEY="your_secret_key_here"
   ```
   
   For Windows (Command Prompt):
   ```
   set API_KEY=your_api_key_here
   set SECRET_KEY=your_secret_key_here
   ```
   
   For Windows (PowerShell):
   ```
   $env:API_KEY="your_api_key_here"
   $env:SECRET_KEY="your_secret_key_here"
   ```

5. Create necessary directories if they don't exist:
   ```bash
   mkdir -p data models logs models/report dashboard/reports
   ```

## Usage

### Training a Model

```bash
# Train a new model with default parameters
python trading_bot.py --train

# Train with specific parameters
python trading_bot.py --train --sequence-length 50 --epochs 100
```

### Running the Trading Bot

```bash
# Run in live trading mode
python trading_bot.py

# Run with a specific check interval (in seconds)
python trading_bot.py --interval 300  # Check every 5 minutes

# Run in simulation mode (no real trades)
python trading_bot.py --simulation
```

### Backtesting

```bash
# Run backtesting with default parameters
python trading_bot.py --backtest

# Specify date range for backtesting
python trading_bot.py --backtest --start 2023-01-01 --end 2023-06-30
```

### Viewing Performance Dashboard

```bash
# Generate and view the dashboard
python dashboard/trading_bot_dashboard.py

# Generate with specific parameters
python dashboard/trading_bot_dashboard.py --symbol "ETH/USDT" --timeframe "1h"
```

## Configuration Options

Example configuration (see `config.json` for a complete example):

```json
{
  "symbol": "BTC/USDT",       # Trading pair
  "timeframe": "15m",         # Candle timeframe
  "risk_per_trade": 0.02,     # 2% risk per trade
  "stop_loss_pct": 0.03,      # 3% stop loss
  "take_profit_pct": 0.06,    # 6% take profit
  "max_drawdown": 0.15,       # 15% max drawdown
  "strategy": "combined",     # Strategy to use
  "indicators": {
    "rsi": {
      "period": 14,
      "oversold": 30,
      "overbought": 70
    },
    "macd": {
      "fast_period": 12,
      "slow_period": 26,
      "signal_period": 9
    }
  },
  "strategy_weights": {
    "rsi": 1.0,
    "macd": 1.0,
    "bollinger": 1.0,
    "price_action": 1.0
  },
  "model": {
    "sequence_length": 50,
    "retrain_days": 7,
    "confidence_threshold": 0.6
  }
}
```

## Advanced Features

### Multi-Strategy Trading

The bot supports multiple trading strategies that can be used individually or combined:

- RSI Strategy
- MACD Strategy
- Bollinger Bands Strategy
- Price Action Strategy
- Market Regime Strategy
- Combined Strategy (weighted voting)

### Market Regime Detection

The bot automatically detects the current market regime and applies the most appropriate strategy:

- Trending markets: Uses momentum-based strategies
- Ranging markets: Uses oscillator-based strategies
- Volatile markets: Uses mean-reversion strategies

### LSTM-based Prediction

The deep learning model is periodically retrained on recent data to adapt to changing market conditions. Model parameters and training frequency are configurable.

## Performance Metrics

The dashboard provides detailed performance metrics:

- Win/Loss Ratio
- Average Profit per Trade
- Maximum Drawdown
- Sharpe Ratio
- Profit Factor
- Total Return

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Matplotlib
- ccxt (for exchange connectivity)
- scikit-learn
- Other dependencies in requirements.txt

## Security Considerations

- Never commit your API keys to the code repository
- Use environment variables or a .env file (with .env in .gitignore)
- Set appropriate permissions on files containing sensitive information
- Review exchange API permissions and limit to only what's necessary

## License

MIT