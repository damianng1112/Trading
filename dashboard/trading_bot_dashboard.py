import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import datetime
import os
import json
import logging
from utils.indicators import calculate_all_indicators
from utils.data_loader import fetch_historical_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class TradingBotDashboard:
    """
    Dashboard to visualize trading bot performance, analyze signals and display indicators.
    """
    
    def __init__(self, symbol="BTC/USDT", timeframe="15m"):
        """
        Initialize the dashboard.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Chart timeframe
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
        self.trade_history = None
        
        # Set style for plots
        plt.style.use('dark_background')
        sns.set(font_scale=1.2)
        
        # Create directories if they don't exist
        os.makedirs("dashboard", exist_ok=True)
        os.makedirs("dashboard/reports", exist_ok=True)
        
    def fetch_data(self, limit=500):
        """
        Fetch and prepare data for display.
        
        Args:
            limit (int): Number of candles to fetch
        """
        logging.info(f"Fetching {limit} candles for {self.symbol} on {self.timeframe} timeframe")
        
        try:
            # Fetch data from exchange
            raw_data = fetch_historical_data(self.symbol, self.timeframe, limit)
            
            if raw_data is None or len(raw_data) < 100:
                logging.error("Failed to fetch sufficient data")
                return False
                
            # Calculate indicators
            self.data = calculate_all_indicators(raw_data)
            logging.info(f"Successfully fetched and prepared {len(self.data)} data points")
            return True
            
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return False
    
    def load_trade_history(self):
        """
        Load trade history from CSV file.
        
        Returns:
            bool: Success status
        """
        try:
            if os.path.exists("data/trade_history.csv"):
                self.trade_history = pd.read_csv("data/trade_history.csv")
                
                # Convert timestamp columns to datetime
                timestamp_cols = [col for col in self.trade_history.columns if 'timestamp' in col.lower()]
                for col in timestamp_cols:
                    self.trade_history[col] = pd.to_datetime(self.trade_history[col])
                
                logging.info(f"Loaded {len(self.trade_history)} trades from history")
                return True
            else:
                logging.warning("No trade history file found")
                return False
                
        except Exception as e:
            logging.error(f"Error loading trade history: {e}")
            return False
    
    def generate_price_chart(self, days=7):
        """
        Generate price chart with indicators and signals.
        
        Args:
            days (int): Number of days to display
        """
        if self.data is None or len(self.data) == 0:
            logging.error("No data available for price chart")
            return
            
        try:
            # Filter data for the specified time period
            if 'timestamp' in self.data.columns:
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                chart_data = self.data[self.data['timestamp'] >= cutoff_date].copy()
            else:
                # If no timestamp column, just take the last N rows
                chart_data = self.data.tail(days * 24 * 4).copy()  # Assuming 15m candles
                
            if len(chart_data) == 0:
                logging.error("No data available for the specified time period")
                return
                
            # Create figure and grid for subplots
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
            
            # Price chart with indicators
            ax1 = fig.add_subplot(gs[0])
            ax1.set_title(f"{self.symbol} Price Chart ({self.timeframe})", fontsize=16)
            
            # Plot price candles
            ax1.plot(chart_data.index, chart_data['close'], color='white', linewidth=1.5, label='Close Price')
            
            # Plot EMAs if available
            for period in [9, 21, 50, 200]:
                ema_col = f'EMA_{period}'
                if ema_col in chart_data.columns:
                    ax1.plot(chart_data.index, chart_data[ema_col], 
                             label=f'{period} EMA', 
                             linewidth=1, 
                             alpha=0.8)
            
            # Plot Bollinger Bands if available
            if 'BB_Upper' in chart_data.columns and 'BB_Lower' in chart_data.columns:
                ax1.plot(chart_data.index, chart_data['BB_Upper'], 'r--', alpha=0.6, label='BB Upper')
                ax1.plot(chart_data.index, chart_data['BB_Middle'], 'g--', alpha=0.6, label='BB Middle')
                ax1.plot(chart_data.index, chart_data['BB_Lower'], 'r--', alpha=0.6, label='BB Lower')
                
            # Add buy/sell signals
            if 'Combined_Signal' in chart_data.columns:
                # Plot buy signals
                buy_signals = chart_data[chart_data['Combined_Signal'].isin(['strong_bullish', 'moderate_bullish'])]
                ax1.scatter(buy_signals.index, buy_signals['close'] * 0.99, 
                           marker='^', color='lime', s=100, label='Buy Signal')
                
                # Plot sell signals
                sell_signals = chart_data[chart_data['Combined_Signal'].isin(['strong_bearish', 'moderate_bearish'])]
                ax1.scatter(sell_signals.index, sell_signals['close'] * 1.01, 
                           marker='v', color='red', s=100, label='Sell Signal')
            
            # Plot trades if available
            if self.trade_history is not None and len(self.trade_history) > 0:
                # Filter trades for the time period
                if 'timestamp' in self.trade_history.columns and 'timestamp' in chart_data.columns:
                    start_time = chart_data['timestamp'].min()
                    end_time = chart_data['timestamp'].max()
                    period_trades = self.trade_history[
                        (self.trade_history['timestamp'] >= start_time) & 
                        (self.trade_history['timestamp'] <= end_time)
                    ]
                    
                    # Plot buy trades
                    buy_trades = period_trades[period_trades['signal'] == 'buy']
                    if len(buy_trades) > 0:
                        # Find the index in chart_data closest to each trade timestamp
                        for _, trade in buy_trades.iterrows():
                            closest_idx = np.abs(chart_data['timestamp'] - trade['timestamp']).idxmin()
                            trade_price = trade['price']
                            ax1.scatter(closest_idx, trade_price * 0.995, 
                                       marker='$B$', color='white', s=150)
                    
                    # Plot sell trades
                    sell_trades = period_trades[period_trades['signal'] == 'sell']
                    if len(sell_trades) > 0:
                        for _, trade in sell_trades.iterrows():
                            closest_idx = np.abs(chart_data['timestamp'] - trade['timestamp']).idxmin()
                            trade_price = trade['price']
                            ax1.scatter(closest_idx, trade_price * 1.005, 
                                       marker='$S$', color='white', s=150)
            
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=10)
            
            # RSI subplot
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            if 'RSI' in chart_data.columns:
                ax2.plot(chart_data.index, chart_data['RSI'], color='purple', label='RSI')
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('RSI')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper left')
            
            # MACD subplot
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            if all(col in chart_data.columns for col in ['MACD', 'Signal_Line']):
                ax3.plot(chart_data.index, chart_data['MACD'], label='MACD', color='cyan')
                ax3.plot(chart_data.index, chart_data['Signal_Line'], label='Signal', color='magenta')
                
                # Plot MACD histogram
                if 'MACD_Histogram' in chart_data.columns:
                    for i in range(len(chart_data) - 1):
                        value = chart_data['MACD_Histogram'].iloc[i]
                        color = 'g' if value >= 0 else 'r'
                        ax3.bar(chart_data.index[i], value, color=color, alpha=0.5, width=0.7)
                
                ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
                ax3.set_ylabel('MACD')
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper left')
            
            # Volume subplot
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            if 'volume' in chart_data.columns:
                # Color volume bars based on price change
                for i in range(len(chart_data)):
                    if i > 0:
                        color = 'g' if chart_data['close'].iloc[i] >= chart_data['close'].iloc[i-1] else 'r'
                        ax4.bar(chart_data.index[i], chart_data['volume'].iloc[i], color=color, alpha=0.7, width=0.7)
                
                ax4.set_ylabel('Volume')
                ax4.grid(True, alpha=0.3)
            
            # Format x-axis
            if 'timestamp' in chart_data.columns:
                date_format = '%m/%d' if days > 3 else '%m/%d %H:%M'
                date_rotation = 45
                plt.xticks(rotation=date_rotation)
                
            plt.tight_layout()
            plt.savefig("dashboard/price_chart.png", dpi=120, bbox_inches='tight')
            logging.info("Price chart generated successfully")
            
        except Exception as e:
            logging.error(f"Error generating price chart: {e}")
    
    def generate_performance_metrics(self):
        """
        Generate performance metrics from trade history.
        
        Returns:
            dict: Performance metrics
        """
        if self.trade_history is None or len(self.trade_history) == 0:
            logging.warning("No trade history available for performance metrics")
            return None
            
        try:
            # Create a copy of trade history
            trades = self.trade_history.copy()
            
            # Initialize metrics dictionary
            metrics = {
                "total_trades": len(trades),
                "buy_trades": len(trades[trades['signal'] == 'buy']),
                "sell_trades": len(trades[trades['signal'] == 'sell']),
                "win_rate": 0,
                "avg_profit": 0,
                "max_profit": 0,
                "max_loss": 0,
                "profit_factor": 0,
                "total_profit": 0
            }
            
            # Calculate profit metrics if we have closed trades
            if 'profit_loss_pct' in trades.columns:
                profitable_trades = trades[trades['profit_loss_pct'] > 0]
                losing_trades = trades[trades['profit_loss_pct'] < 0]
                
                metrics["win_rate"] = len(profitable_trades) / max(len(trades), 1) * 100
                metrics["avg_profit"] = trades['profit_loss_pct'].mean() if len(trades) > 0 else 0
                metrics["max_profit"] = trades['profit_loss_pct'].max() if len(trades) > 0 else 0
                metrics["max_loss"] = trades['profit_loss_pct'].min() if len(trades) > 0 else 0
                
                total_profit = profitable_trades['profit_loss_pct'].sum() if len(profitable_trades) > 0 else 0
                total_loss = abs(losing_trades['profit_loss_pct'].sum()) if len(losing_trades) > 0 else 0
                
                metrics["profit_factor"] = total_profit / max(total_loss, 0.0001)
                metrics["total_profit"] = trades['profit_loss_pct'].sum() if len(trades) > 0 else 0
            
            # Calculate time-based metrics
            if 'timestamp' in trades.columns:
                start_date = trades['timestamp'].min()
                end_date = trades['timestamp'].max()
                trading_days = (end_date - start_date).days + 1
                
                metrics["trading_period_days"] = trading_days
                metrics["trades_per_day"] = metrics["total_trades"] / max(trading_days, 1)
                metrics["start_date"] = start_date.strftime('%Y-%m-%d')
                metrics["end_date"] = end_date.strftime('%Y-%m-%d')
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating performance metrics: {e}")
            return None
    
    def generate_performance_charts(self):
        """
        Generate charts showing trading performance.
        """
        if self.trade_history is None or len(self.trade_history) == 0:
            logging.warning("No trade history available for performance charts")
            return
            
        try:
            # Create profit distribution chart
            if 'profit_loss_pct' in self.trade_history.columns:
                plt.figure(figsize=(10, 6))
                
                # Create histogram of profit/loss percentages
                sns.histplot(self.trade_history['profit_loss_pct'], bins=20, kde=True)
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title('Trade Profit/Loss Distribution')
                plt.xlabel('Profit/Loss (%)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("dashboard/profit_distribution.png", dpi=120, bbox_inches='tight')
                
                # Create equity curve if we have cumulative profit data
                if 'timestamp' in self.trade_history.columns and 'profit_loss_pct' in self.trade_history.columns:
                    # Sort trades by timestamp
                    trades_sorted = self.trade_history.sort_values('timestamp')
                    
                    # Calculate cumulative returns
                    initial_capital = 1000  # Hypothetical starting capital
                    trades_sorted['cumulative_return'] = (1 + trades_sorted['profit_loss_pct'] / 100).cumprod() * initial_capital
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(trades_sorted['timestamp'], trades_sorted['cumulative_return'], 'b-')
                    plt.title('Equity Curve')
                    plt.xlabel('Date')
                    plt.ylabel('Account Value ($)')
                    plt.grid(True, alpha=0.3)
                    
                    # Add drawdown shading
                    running_max = trades_sorted['cumulative_return'].cummax()
                    drawdown = (trades_sorted['cumulative_return'] / running_max - 1) * 100
                    
                    # Calculate and annotate maximum drawdown
                    max_drawdown = drawdown.min()
                    max_dd_idx = drawdown.idxmin()
                    max_dd_date = trades_sorted.loc[max_dd_idx, 'timestamp']
                    
                    plt.annotate(f'Max Drawdown: {max_drawdown:.2f}%', 
                                xy=(max_dd_date, trades_sorted.loc[max_dd_idx, 'cumulative_return']),
                                xytext=(30, -30),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', color='red'),
                                color='red')
                    
                    plt.tight_layout()
                    plt.savefig("dashboard/equity_curve.png", dpi=120, bbox_inches='tight')
                    
                # Create win/loss chart
                plt.figure(figsize=(8, 8))
                trades_result = ['Win' if p > 0 else 'Loss' for p in self.trade_history['profit_loss_pct']]
                result_counts = pd.Series(trades_result).value_counts()
                
                plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', 
                       colors=['green', 'red'], startangle=90, explode=[0.05, 0])
                plt.title('Win/Loss Ratio')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig("dashboard/win_loss_ratio.png", dpi=120, bbox_inches='tight')
                
            logging.info("Performance charts generated successfully")
            
        except Exception as e:
            logging.error(f"Error generating performance charts: {e}")
    
    def generate_indicator_analysis(self):
        """
        Generate analysis of indicator effectiveness.
        """
        if self.data is None or self.trade_history is None:
            logging.warning("Data or trade history missing for indicator analysis")
            return
            
        try:
            # Create indicator signal analysis
            signal_columns = [col for col in self.data.columns if '_Signal' in col]
            
            if not signal_columns:
                logging.warning("No signal columns found for analysis")
                return
                
            # Initialize DataFrame for results
            results = pd.DataFrame(columns=['Indicator', 'Signal', 'Count', 'Success_Rate'])
            
            # Analyze each indicator
            for indicator in signal_columns:
                indicator_name = indicator.split('_')[0]
                
                # Get all unique signals for this indicator
                signal_values = self.data[indicator].dropna().unique()
                
                for signal in signal_values:
                    if signal == 'neutral':
                        continue
                        
                    # Count occurrences of this signal
                    signal_count = len(self.data[self.data[indicator] == signal])
                    
                    # Determine if signal is bullish/bearish
                    is_bullish = any(bullish_term in signal.lower() for bullish_term in ['bullish', 'oversold'])
                    expected_direction = 1 if is_bullish else -1  # 1 for price up, -1 for price down
                    
                    # Check how many times the signal was correct
                    correct_count = 0
                    for i in range(len(self.data) - 1):
                        if self.data[indicator].iloc[i] == signal:
                            # Check if price moved in the expected direction
                            actual_direction = 1 if self.data['close'].iloc[i+1] > self.data['close'].iloc[i] else -1
                            if actual_direction == expected_direction:
                                correct_count += 1
                    
                    # Calculate success rate
                    success_rate = correct_count / max(signal_count, 1) * 100
                    
                    # Add to results
                    results = pd.concat([results, pd.DataFrame({
                        'Indicator': [indicator_name],
                        'Signal': [signal],
                        'Count': [signal_count],
                        'Success_Rate': [success_rate]
                    })])
            
            # Generate indicator effectiveness chart
            if len(results) > 0:
                # Sort by success rate
                results = results.sort_values('Success_Rate', ascending=False)
                
                plt.figure(figsize=(12, 8))
                
                # Create bar chart
                plt.barh(results['Indicator'] + ': ' + results['Signal'], results['Success_Rate'], color='skyblue')
                plt.axvline(x=50, color='r', linestyle='--', label='Random Chance')
                
                plt.title('Indicator Signal Effectiveness')
                plt.xlabel('Success Rate (%)')
                plt.xlim(0, 100)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                plt.savefig("dashboard/indicator_effectiveness.png", dpi=120, bbox_inches='tight')
                
                # Save results to CSV
                results.to_csv("dashboard/reports/indicator_effectiveness.csv", index=False)
                
            logging.info("Indicator analysis completed successfully")
            
        except Exception as e:
            logging.error(f"Error generating indicator analysis: {e}")
    
    def generate_html_report(self):
        """
        Generate an HTML report with all charts and metrics.
        """
        try:
            # Calculate performance metrics
            metrics = self.generate_performance_metrics()
            
            if metrics is None:
                metrics = {
                    "total_trades": 0,
                    "win_rate": 0,
                    "avg_profit": 0,
                    "total_profit": 0
                }
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Bot Dashboard</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #121212;
                        color: #eee;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    header {{
                        text-align: center;
                        padding: 20px;
                        background-color: #1e1e1e;
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .metrics {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                        margin-bottom: 20px;
                    }}
                    .metric-card {{
                        background-color: #1e1e1e;
                        border-radius: 5px;
                        padding: 15px;
                        width: 22%;
                        margin-bottom: 15px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                    }}
                    .metric-value {{
                        font-size: 24px;
                        font-weight: bold;
                        margin-top: 10px;
                        color: #4CAF50;
                    }}
                    .negative {{
                        color: #f44336;
                    }}
                    .chart-container {{
                        background-color: #1e1e1e;
                        border-radius: 5px;
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                    }}
                    .chart-container h2 {{
                        margin-top: 0;
                    }}
                    .chart-img {{
                        max-width: 100%;
                        height: auto;
                        display: block;
                        margin: 0 auto;
                    }}
                    footer {{
                        text-align: center;
                        padding: 20px;
                        background-color: #1e1e1e;
                        border-radius: 5px;
                        margin-top: 20px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <header>
                        <h1>Trading Bot Dashboard</h1>
                        <p>Symbol: {self.symbol} | Timeframe: {self.timeframe} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                    </header>
                    
                    <div class="metrics">
                        <div class="metric-card">
                            <p>Total Trades</p>
                            <div class="metric-value">{metrics.get('total_trades', 0)}</div>
                        </div>
                        <div class="metric-card">
                            <p>Win Rate</p>
                            <div class="metric-value">{metrics.get('win_rate', 0):.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <p>Average Profit</p>
                            <div class="metric-value {'' if metrics.get('avg_profit', 0) >= 0 else 'negative'}">{metrics.get('avg_profit', 0):.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <p>Total Profit</p>
                            <div class="metric-value {'' if metrics.get('total_profit', 0) >= 0 else 'negative'}">{metrics.get('total_profit', 0):.2f}%</div>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <h2>Price Chart</h2>
                        <img src="price_chart.png" class="chart-img" alt="Price Chart">
                    </div>
                    
                    <div class="chart-container">
                        <h2>Equity Curve</h2>
                        <img src="equity_curve.png" class="chart-img" alt="Equity Curve">
                    </div>
                    
                    <div class="chart-container">
                        <h2>Profit Distribution</h2>
                        <img src="profit_distribution.png" class="chart-img" alt="Profit Distribution">
                    </div>
                    
                    <div class="chart-container">
                        <h2>Win/Loss Ratio</h2>
                        <img src="win_loss_ratio.png" class="chart-img" alt="Win/Loss Ratio">
                    </div>
                    
                    <div class="chart-container">
                        <h2>Indicator Effectiveness</h2>
                        <img src="indicator_effectiveness.png" class="chart-img" alt="Indicator Effectiveness">
                    </div>
                    
                    <footer>
                        <p>&copy; {datetime.datetime.now().year} Trading Bot Dashboard | All rights reserved</p>
                    </footer>
                </div>
            </body>
            </html>
            """
            
            # Write HTML to file
            with open("dashboard/index.html", "w") as f:
                f.write(html_content)
            
            logging.info("HTML report generated successfully at dashboard/index.html")
            
        except Exception as e:
            logging.error(f"Error generating HTML report: {e}")
    
    def run(self):
        """
        Run the dashboard generation process.
        """
        logging.info("Starting dashboard generation")
        
        # Fetch data
        if not self.fetch_data(limit=1000):
            logging.error("Failed to fetch data. Aborting dashboard generation.")
            return
        
        # Load trade history
        self.load_trade_history()
        
        # Generate charts and analysis
        self.generate_price_chart(days=30)
        self.generate_performance_charts()
        self.generate_indicator_analysis()
        
        # Create HTML report
        self.generate_html_report()
        
        logging.info("Dashboard generation completed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Bot Dashboard Generator")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="15m", help="Chart timeframe")
    
    args = parser.parse_args()
    
    dashboard = TradingBotDashboard(symbol=args.symbol, timeframe=args.timeframe)
    dashboard.run()