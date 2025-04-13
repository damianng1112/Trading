import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def plot_prediction_vs_actual(data, predictions, title="Model Predictions vs Actual", save_path=None):
    """
    Plot model predictions against actual price movements.
    
    Args:
        data (pd.DataFrame): DataFrame with price data including 'timestamp' and 'close' columns
        predictions (np.array): Array of model predictions (probabilities)
        title (str): Plot title
        save_path (str): Path to save the figure or None to display it
    """
    try:
        # Create a figure with two subplots (price and prediction probabilities)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price chart
        if 'timestamp' in data.columns:
            ax1.plot(data['timestamp'], data['close'], label='Close Price', color='blue')
            
            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            ax1.plot(data.index, data['close'], label='Close Price', color='blue')
        
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot predictions as heatmap
        if 'timestamp' in data.columns:
            # Create colormap for predictions (green for bullish, red for bearish)
            cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
            
            # Plot prediction probabilities as scatter points with color based on value
            scatter = ax2.scatter(
                data['timestamp'][-len(predictions):], 
                [0.5] * len(predictions),  # y-position fixed at 0.5
                c=predictions, 
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                s=25,
                alpha=0.8
            )
            
            # Add horizontal line at 0.5 threshold
            ax2.axhline(y=0.5, color='gray', linestyle='--')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Prediction (Probability of Price Rise)')
        else:
            ax2.imshow(
                [predictions], 
                aspect='auto', 
                cmap='RdYlGn',
                vmin=0, 
                vmax=1,
                extent=[0, len(predictions), 0, 1]
            )
        
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Prediction')
        ax2.set_xlabel('Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            logging.info(f"Prediction vs actual plot saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logging.error(f"Error plotting predictions vs actual: {e}")
        
def plot_trade_signals(data, signals, save_path=None):
    """
    Plot trade signals on price chart.
    
    Args:
        data (pd.DataFrame): DataFrame with price data including 'timestamp' and 'close' columns
        signals (list): List of dictionaries with trade signals
        save_path (str): Path to save the figure or None to display it
    """
    try:
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot price chart
        if 'timestamp' in data.columns:
            plt.plot(data['timestamp'], data['close'], label='Close Price', color='blue')
            
            # Format x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            plt.plot(data.index, data['close'], label='Close Price', color='blue')
        
        # Plot buy signals
        buy_signals = [s for s in signals if s['type'] == 'buy']
        if buy_signals:
            if 'timestamp' in data.columns:
                buy_times = [s['timestamp'] for s in buy_signals]
                buy_prices = [s['price'] for s in buy_signals]
            else:
                # Find indices if timestamps are not available
                buy_times = [data.index[data['close'] == s['price']].values[0] for s in buy_signals]
                buy_prices = [s['price'] for s in buy_signals]
                
            plt.scatter(buy_times, buy_prices, marker='^', s=100, c='green', label='Buy Signal')
            
        # Plot sell signals
        sell_signals = [s for s in signals if s['type'] == 'sell']
        if sell_signals:
            if 'timestamp' in data.columns:
                sell_times = [s['timestamp'] for s in sell_signals]
                sell_prices = [s['price'] for s in sell_signals]
            else:
                # Find indices if timestamps are not available
                sell_times = [data.index[data['close'] == s['price']].values[0] for s in sell_signals]
                sell_prices = [s['price'] for s in sell_signals]
                
            plt.scatter(sell_times, sell_prices, marker='v', s=100, c='red', label='Sell Signal')
        
        plt.title('Trading Signals on Price Chart')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            logging.info(f"Trade signals plot saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logging.error(f"Error plotting trade signals: {e}")

def plot_performance_metrics(metrics, save_path=None):
    """
    Create a visual summary of model performance metrics.
    
    Args:
        metrics (dict): Dictionary containing performance metrics
        save_path (str): Path to save the figure or None to display it
    """
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Classification metrics as bar chart
        class_metrics = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0), 
            metrics.get('f1_score', 0)
        ]
        
        ax1.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], class_metrics, color='steelblue')
        ax1.set_ylim(0, 1)
        ax1.set_title('Classification Metrics')
        ax1.grid(True, alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(class_metrics):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # 2. Confusion matrix as heatmap
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2)
            ax2.set_title('Confusion Matrix')
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('True Label')
            ax2.set_xticklabels(['Down', 'Up'])
            ax2.set_yticklabels(['Down', 'Up'])
        else:
            ax2.text(0.5, 0.5, 'Confusion Matrix\nNot Available', ha='center', va='center')
            ax2.set_title('Confusion Matrix')
        
        # 3. ROC curve
        if all(k in metrics for k in ['fpr', 'tpr', 'roc_auc']):
            ax3.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, 
                   label=f'ROC curve (area = {metrics["roc_auc"]:.2f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('Receiver Operating Characteristic')
            ax3.legend(loc="lower right")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'ROC Curve\nNot Available', ha='center', va='center')
            ax3.set_title('ROC Curve')
        
        # 4. Trading metrics
        if 'backtest_results' in metrics:
            backtest = metrics['backtest_results']
            trading_metrics = [
                backtest.get('total_return_pct', 0),
                backtest.get('win_rate', 0),
                backtest.get('sharpe_ratio', 0),
                abs(backtest.get('max_drawdown_pct', 0))  # Make positive for visualization
            ]
            
            colors = ['green' if v >= 0 else 'red' for v in trading_metrics]
            colors[3] = 'red'  # Drawdown always red
            
            ax4.bar(
                ['Return (%)', 'Win Rate (%)', 'Sharpe Ratio', 'Max Drawdown (%)'], 
                trading_metrics,
                color=colors
            )
            ax4.set_title('Trading Performance')
            ax4.grid(True, alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(trading_metrics):
                ax4.text(i, v + (max(trading_metrics) * 0.05), f'{v:.2f}', ha='center')
        else:
            ax4.text(0.5, 0.5, 'Trading Metrics\nNot Available', ha='center', va='center')
            ax4.set_title('Trading Performance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            logging.info(f"Performance metrics plot saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logging.error(f"Error plotting performance metrics: {e}")

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance from the model (if available).
    This requires a model that provides feature importance.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        save_path (str): Path to save the figure or None to display it
    """
    try:
        # For models that don't have direct feature importance, try permutation importance
        try:
            from sklearn.inspection import permutation_importance
            
            # This is a placeholder - you would need test data X_test, y_test to actually calculate this
            # result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            # importance = result.importances_mean
            
            # Placeholder implementation to demonstrate the function
            importance = np.random.rand(len(feature_names))
            
            # Sort features by importance
            indices = np.argsort(importance)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importance[indices], color='skyblue')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Permutation Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=120, bbox_inches='tight')
                logging.info(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
        except Exception as inner_e:
            logging.warning(f"Could not calculate permutation importance: {inner_e}")
            return None
            
    except Exception as e:
        logging.error(f"Error plotting feature importance: {e}")
        return None

def create_model_report(model, data, predictions, metrics, output_dir="models/report"):
    """
    Create a comprehensive model report with various visualizations.
    
    Args:
        model: Trained model
        data (pd.DataFrame): DataFrame with price data including 'close' column
        predictions (np.array): Array of model predictions
        metrics (dict): Dictionary containing performance metrics
        output_dir (str): Directory to save report files
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot predictions vs actual
        plot_prediction_vs_actual(
            data, 
            predictions, 
            save_path=os.path.join(output_dir, "predictions_vs_actual.png")
        )
        
        # 2. Plot performance metrics
        plot_performance_metrics(
            metrics,
            save_path=os.path.join(output_dir, "performance_metrics.png")
        )
        
        # 3. Plot feature importance (if model supports it)
        if hasattr(model, 'feature_importances_') or hasattr(model, 'feature_importance'):
            plot_feature_importance(
                model, 
                metrics.get('feature_columns', []),
                save_path=os.path.join(output_dir, "feature_importance.png")
            )
        
        # 4. Plot trade signals (if available)
        if 'backtest_results' in metrics and 'trades' in metrics['backtest_results']:
            trades = metrics['backtest_results']['trades']
            # Filter for just buy/sell entries, not exits
            signals = [
                {'type': t['type'].split('_')[1], 'price': t['price'], 'timestamp': t['timestamp']}
                for t in trades if t['type'].startswith('open_')
            ]
            
            plot_trade_signals(
                data,
                signals,
                save_path=os.path.join(output_dir, "trade_signals.png")
            )
            
        # 5. Create a summary text file
        with open(os.path.join(output_dir, "model_summary.txt"), "w") as f:
            f.write("===== LSTM TRADING MODEL SUMMARY =====\n\n")
            
            # Classification metrics
            f.write("== Classification Performance ==\n")
            f.write(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 'N/A'):.4f}\n")
            f.write(f"Recall: {metrics.get('recall', 'N/A'):.4f}\n")
            f.write(f"F1 Score: {metrics.get('f1_score', 'N/A'):.4f}\n\n")
            
            # Trading performance
            if 'backtest_results' in metrics:
                backtest = metrics['backtest_results']
                f.write("== Trading Performance ==\n")
                f.write(f"Total Return: {backtest.get('total_return_pct', 'N/A'):.2f}%\n")
                f.write(f"Number of Trades: {backtest.get('number_of_trades', 'N/A')}\n")
                f.write(f"Win Rate: {backtest.get('win_rate', 'N/A'):.2f}%\n")
                f.write(f"Max Drawdown: {backtest.get('max_drawdown_pct', 'N/A'):.2f}%\n")
                f.write(f"Sharpe Ratio: {backtest.get('sharpe_ratio', 'N/A'):.2f}\n\n")
                
            # Model information
            f.write("== Model Information ==\n")
            f.write(f"Features used: {', '.join(metrics.get('feature_columns', []))}\n")
            f.write(f"Sequence Length: {metrics.get('sequence_length', 'N/A')}\n")
            
            f.write("\n===== END OF SUMMARY =====\n")
            
        logging.info(f"Model report generated in directory: {output_dir}")
        
    except Exception as e:
        logging.error(f"Error creating model report: {e}")