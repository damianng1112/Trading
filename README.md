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
├── trading_bot.py             # Main bot implementation
├── dashboard/                 # Performance visualization dashboard
├── models/                    # Saved LSTM models
├── data/                      # Historical data and trade logs
├── logs/                      # Bot logging output
├── utils/
│   ├── data_loader.py         # Data fetching utilities
│   ├── indicators.py          # Technical indicators implementation
│   ├── model_training.py      # LSTM model training
│   └── strategies.py          # Trading strategy implementations
├── config.json                # Bot configuration
└── README.md                  # Documentation
```

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your trading parameters in `config.json`
4. Set your API keys as environment variables:
   ```bash
   export API_KEY="your_api_key"
   export SECRET_KEY="your_secret_key"
   ```

## Usage

### Running the Trading Bot

```bash
# Run in live trading mode
python trading_bot.py

# Run with a specific configuration
python trading_bot.py --config custom_config.json

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

### Viewing Performance

```bash
# Generate and view the dashboard
python trading_bot_dashboard.py

# Generate with specific parameters
python trading_bot_dashboard.py --symbol "ETH/USDT" --timeframe "1h"
```

## Configuration Options

Example configuration (see `config.json` for a complete example):

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "15m",
  "risk_per_trade": 0.02,
  "stop_loss_pct": 0.03,
  "take_profit_pct": 0.06,
  "max_drawdown": 0.15
  // ...additional configuration options
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
- ccxt

## License

MIT