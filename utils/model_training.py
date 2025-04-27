import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def build_lstm_model(input_shape, dropout_rate=0.3):
    """
    Builds an improved LSTM model for binary classification with 
    additional regularization and architecture improvements.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        keras.models.Sequential: Compiled LSTM model.
    """
    model = keras.Sequential([
        # First LSTM layer with batch normalization
        layers.LSTM(64, return_sequences=True, input_shape=input_shape, 
                  recurrent_dropout=0.1, 
                  kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Second LSTM layer with attention mechanism
        layers.LSTM(32, return_sequences=False,
                  recurrent_dropout=0.1,
                  kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Dense layers for classification
        layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate/2),
        
        # Output layer
        layers.Dense(1, activation="sigmoid")
    ])
    
    # Use Adam optimizer with learning rate scheduler
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile model with binary crossentropy loss and class weights
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", 
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    
    return model

def create_sequences(data, feature_columns, target_column, sequence_length=50):
    """
    Generates sequences of features and targets for LSTM training with improved error handling.

    Args:
        data (pd.DataFrame): The input dataframe with feature columns and a target column.
        feature_columns (list): List of feature column names to use.
        target_column (str): The name of the target column.
        sequence_length (int): The number of timesteps per sequence.

    Returns:
        tuple: (np.array of sequences, np.array of targets, dict of data statistics)
    """
    logging.info(f"Creating sequences with {len(feature_columns)} features and length {sequence_length}")
    
    # Verify all required columns exist
    missing_columns = [col for col in feature_columns + [target_column] if col not in data.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return None, None, None
    
    if len(data) <= sequence_length:
        logging.error(f"Not enough data for sequence creation. Need more than {sequence_length} rows.")
        return None, None, None
    
    sequences = []
    targets = []
    stats = {}
    
    try:
        # Select only needed columns and drop rows with any NaN values
        data_subset = data[feature_columns + [target_column]].dropna()
        logging.info(f"Data shape after selecting columns and dropping NaNs: {data_subset.shape}")
        
        if len(data_subset) <= sequence_length:
            logging.error(f"Not enough data after removing NaNs. Only {len(data_subset)} rows left.")
            return None, None, None
        
        # Store data statistics for scaling during prediction
        stats["feature_mins"] = data_subset[feature_columns].min().to_dict()
        stats["feature_maxs"] = data_subset[feature_columns].max().to_dict()
        
        # Normalize data
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(data_subset[feature_columns])
        
        # Create sequences
        for i in range(len(data_subset) - sequence_length):
            # Extract sequence of features
            sequence = scaled_features[i:i+sequence_length]
            # Extract target value
            target = data_subset.iloc[i+sequence_length][target_column]
            
            sequences.append(sequence)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Balance classes if needed
        unique, counts = np.unique(targets, return_counts=True)
        logging.info(f"Target class distribution: {dict(zip(unique, counts))}")
        
        stats["class_weights"] = {
            0: len(targets) / (2 * (len(targets) - sum(targets))),
            1: len(targets) / (2 * sum(targets))
        }
        
        return sequences, targets, stats
    
    except Exception as e:
        logging.error(f"Error creating sequences: {e}")
        return None, None, None

def evaluate_model_predictions(model, test_data, feature_columns, target_column="target", sequence_length=50):
    """
    Evaluates model predictions and generates performance visualizations.
    
    Args:
        model: The trained LSTM model.
        test_data (pd.DataFrame): Test data containing features and target.
        feature_columns (list): Feature columns used for prediction.
        target_column (str): Target column name.
        sequence_length (int): Length of input sequences.
        
    Returns:
        dict: Dictionary with evaluation metrics.
    """
    try:
        # Create sequences for testing
        X_test, y_test, _ = create_sequences(
            test_data,
            feature_columns=feature_columns,
            target_column=target_column,
            sequence_length=sequence_length
        )
        
        if X_test is None:
            return None
            
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        acc = accuracy_score(y_test, y_pred)
        
        metrics = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
        disp.plot(cmap="Blues")
        plt.title("LSTM Model Prediction Confusion Matrix")
        plt.savefig("models/confusion_matrix.png")
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("models/roc_curve.png")
        
        # Add backtesting simulation on test data
        backtest = backtest_model_on_test_data(model, test_data, feature_columns, sequence_length)
        metrics["backtest_results"] = backtest
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return None

def backtest_model_on_test_data(model, test_data, feature_columns, sequence_length=50):
    """
    Performs a simple backtesting of the model on test data.
    
    Args:
        model: Trained LSTM model.
        test_data (pd.DataFrame): Test data with price information.
        feature_columns (list): Feature columns used for prediction.
        sequence_length (int): Sequence length for prediction.
        
    Returns:
        dict: Backtesting performance metrics.
    """
    try:
        # Create a copy of the test data
        data = test_data.copy()
        
        # Make sure we have a timestamp column
        if "timestamp" not in data.columns:
            data["timestamp"] = pd.date_range(start="2023-01-01", periods=len(data), freq="15min")
            
        # Prepare data
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(data[feature_columns])
        
        # Initialize variables for tracking performance
        initial_balance = 1000.0  # Starting with $1000
        balance = initial_balance
        position = None
        entry_price = 0
        trades = []
        equity_curve = [initial_balance]
        timestamps = [data["timestamp"].iloc[sequence_length]]
        
        # Iterate through the data points
        for i in range(sequence_length, len(data) - 1):
            # Get the sequence for this prediction
            sequence = scaled_features[i-sequence_length:i]
            current_price = data["close"].iloc[i]
            next_price = data["close"].iloc[i+1]
            timestamp = data["timestamp"].iloc[i]
            
            # Make prediction
            prediction = model.predict(np.array([sequence]), verbose=0)[0][0]
            
            # Trading logic - simple version
            if prediction > 0.6 and position != "long":
                # Close any existing position
                if position == "short":
                    profit = entry_price - current_price
                    balance += profit * 10  # Simple leverage
                    trades.append({
                        "type": "close_short",
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit": profit * 10,
                        "timestamp": timestamp
                    })
                
                # Open long position
                position = "long"
                entry_price = current_price
                trades.append({
                    "type": "open_long",
                    "price": entry_price,
                    "timestamp": timestamp
                })
                
            elif prediction < 0.4 and position != "short":
                # Close any existing position
                if position == "long":
                    profit = current_price - entry_price
                    balance += profit * 10  # Simple leverage
                    trades.append({
                        "type": "close_long",
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "profit": profit * 10,
                        "timestamp": timestamp
                    })
                
                # Open short position
                position = "short"
                entry_price = current_price
                trades.append({
                    "type": "open_short",
                    "price": entry_price,
                    "timestamp": timestamp
                })
            
            # Update equity curve
            unrealized_profit = 0
            if position == "long":
                unrealized_profit = (current_price - entry_price) * 10
            elif position == "short":
                unrealized_profit = (entry_price - current_price) * 10
                
            equity_curve.append(balance + unrealized_profit)
            timestamps.append(timestamp)
        
        # Close final position
        if position == "long":
            final_price = data["close"].iloc[-1]
            profit = final_price - entry_price
            balance += profit * 10
            trades.append({
                "type": "close_long",
                "entry_price": entry_price,
                "exit_price": final_price,
                "profit": profit * 10,
                "timestamp": data["timestamp"].iloc[-1]
            })
        elif position == "short":
            final_price = data["close"].iloc[-1]
            profit = entry_price - final_price
            balance += profit * 10
            trades.append({
                "type": "close_short",
                "entry_price": entry_price,
                "exit_price": final_price,
                "profit": profit * 10,
                "timestamp": data["timestamp"].iloc[-1]
            })
            
        # Calculate performance metrics
        final_balance = balance
        total_return = (final_balance / initial_balance - 1) * 100
        num_trades = len([t for t in trades if t["type"].startswith("open")])
        winning_trades = len([t for t in trades if t["type"].startswith("close") and t.get("profit", 0) > 0])
        win_rate = winning_trades / max(1, len([t for t in trades if t["type"].startswith("close")])) * 100
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity_curve)
        plt.title("Backtesting Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Account Balance ($)")
        plt.grid(True)
        plt.savefig("models/equity_curve.png")
        
        # Calculate drawdown
        rolling_max = pd.Series(equity_curve).cummax()
        drawdown = (pd.Series(equity_curve) / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Sharpe ratio calculation (assuming risk-free rate of 0)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)  # Annualized for 15-min data
        
        results = {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return_pct": total_return,
            "number_of_trades": num_trades,
            "win_rate": win_rate,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades
        }
        
        return results
        
    except Exception as e:
        logging.error(f"Error in backtesting: {e}")
        return None

def perform_walk_forward_optimization(data, feature_columns, sequence_length=50, n_splits=5):
    """
    Performs walk-forward optimization to validate model stability over time.
    
    Args:
        data (pd.DataFrame): Historical price data.
        feature_columns (list): Feature columns to use.
        sequence_length (int): Sequence length for the model.
        n_splits (int): Number of time splits for walk-forward analysis.
        
    Returns:
        dict: Performance results for each time period.
    """
    try:
        # Create target column
        data["target"] = np.where(data["close"].shift(-1) > data["close"], 1, 0)
        data = data.dropna()
        
        # Sort data by timestamp if present
        if "timestamp" in data.columns:
            data = data.sort_values("timestamp")
        
        # Split data into chunks for walk-forward testing
        chunk_size = len(data) // n_splits
        results = []
        
        for i in range(n_splits - 1):
            # Define training and test sets
            train_start = 0
            train_end = (i + 1) * chunk_size
            test_start = train_end
            test_end = test_start + chunk_size
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Train model
            logging.info(f"Walk-forward optimization: Training on period {i+1}/{n_splits-1}")
            model, _, stats = train_model(
                train_data, 
                sequence_length=sequence_length,
                epochs=10,  # Reduced epochs for faster optimization
                batch_size=32,
                validation_split=0.2
            )
            
            if model is None:
                logging.error(f"Failed to train model for period {i+1}")
                continue
                
            # Evaluate on test data
            logging.info(f"Evaluating on test period {i+1}")
            test_metrics = evaluate_model_predictions(
                model,
                test_data,
                feature_columns=feature_columns,
                sequence_length=sequence_length
            )
            
            if test_metrics is None:
                logging.error(f"Failed to evaluate model for period {i+1}")
                continue
                
            # Store results
            period_result = {
                "period": i+1,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "train_start": train_data["timestamp"].iloc[0] if "timestamp" in train_data.columns else None,
                "train_end": train_data["timestamp"].iloc[-1] if "timestamp" in train_data.columns else None,
                "test_start": test_data["timestamp"].iloc[0] if "timestamp" in test_data.columns else None,
                "test_end": test_data["timestamp"].iloc[-1] if "timestamp" in test_data.columns else None,
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1_score": test_metrics["f1_score"],
                "backtest_return": test_metrics["backtest_results"]["total_return_pct"],
                "backtest_win_rate": test_metrics["backtest_results"]["win_rate"],
                "backtest_max_drawdown": test_metrics["backtest_results"]["max_drawdown_pct"]
            }
            
            results.append(period_result)
            
        # Plot walk-forward results
        if results:
            periods = [r["period"] for r in results]
            returns = [r["backtest_return"] for r in results]
            accuracies = [r["accuracy"] * 100 for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.bar(periods, returns, alpha=0.7, label="Return (%)")
            plt.plot(periods, accuracies, 'ro-', label="Accuracy (%)")
            plt.xlabel("Time Period")
            plt.ylabel("Performance (%)")
            plt.title("Walk-Forward Optimization Results")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("models/walk_forward_results.png")
            
        return results
            
    except Exception as e:
        logging.error(f"Error in walk-forward optimization: {e}")
        return None

def train_model(data, sequence_length=50, epochs=100, batch_size=32, validation_split=0.2):
    """
    Trains an LSTM model on historical price data.
    
    Args:
        data (pd.DataFrame): Historical price data.
        sequence_length (int): Sequence length for LSTM input.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of data to use for validation.
        
    Returns:
        tuple: (trained model, training history, model statistics)
    """
    try:
        logging.info("Starting model training...")
        
        # Create target column: 1 if price goes up, 0 if it goes down or stays the same
        data["target"] = np.where(data["close"].shift(-1) > data["close"], 1, 0)
        
        # Drop rows with NaN
        data = data.dropna()
        
        # Define feature columns - using both raw and technical indicators
        feature_columns = [
            "close", "high", "low", "open", "volume",  # OHLCV data
            "RSI", "MACD", "Signal_Line",              # Technical indicators
            "sma_50", "sma_200"                        # If calculated
        ]
        
        # Filter to only include columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in data.columns]
        logging.info(f"Using features: {available_features}")
        
        # Create sequences for LSTM
        X, y, stats = create_sequences(
            data, 
            feature_columns=available_features,
            target_column="target",
            sequence_length=sequence_length
        )
        
        if X is None or y is None:
            logging.error("Failed to create sequences")
            return None, None, None
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Validation data shape: {X_val.shape}")
        
        # Build the LSTM model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)
        
        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath="models/lstm_best_model.keras",
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model with class weights
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=stats["class_weights"]
        )
        
        # Evaluate model
        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        
        model_stats = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "feature_columns": available_features,
            "sequence_length": sequence_length,
            **stats
        }
        
        logging.info(f"Model training completed with validation accuracy: {model_stats['accuracy']:.4f}")
        
        # Save the model
        os.makedirs("models", exist_ok=True)
        model.save("models/lstm_trading_model.keras")
        
        # Save model statistics
        import pickle
        with open("models/model_stats.pkl", "wb") as f:
            pickle.dump(model_stats, f)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("models/training_history.png")
        
        return model, history, model_stats
        
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None, None