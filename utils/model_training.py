import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import tensorflow as tf
import keras
from keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def build_lstm_model(input_shape):
    """
    Builds an LSTM model for binary classification.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
        keras.models.Sequential: Compiled LSTM model.
    """
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(50),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def create_sequences(data, target_column, sequence_length=50):
    """
    Generates sequences of features and targets for LSTM training.

    Args:
        data (pd.DataFrame): The input dataframe with feature columns and a target column.
        target_column (str): The name of the target column.
        sequence_length (int): The number of timesteps per sequence.

    Returns:
        np.array: Array of feature sequences.
        np.array: Array of corresponding target values.
    """
    sequences = []
    targets = []

    # Drop non-numeric columns, such as timestamps
    data = data.select_dtypes(include=[np.number])

    for i in range(len(data) - sequence_length):
        # Extract sequence of features
        sequence = data.iloc[i:i+sequence_length].drop(columns=[target_column]).values
        # Extract target value
        target = data.iloc[i+sequence_length][target_column]
        sequences.append(sequence)
        targets.append(target)

    print(data.columns)  # Ensure all expected features are present
    print(data.shape)    # Verify the number of columns matches the model's input requirements

    return np.array(sequences), np.array(targets)

def train_model(data, sequence_length=50):
    """
    Trains an LSTM model on the given data.

    Args:
        data (pd.DataFrame): The input dataframe with features and target.
        sequence_length (int): The number of timesteps per sequence.

    Saves:
        The trained model in the 'models/' directory as 'lstm_trading_model.keras'.
    """
    # Step 1: Generate sequences and targets
    data["target"] = np.where(data["close"].shift(-1) > data["close"], 1, 0)  # Binary target: 1 if price goes up
    sequences, targets = create_sequences(data, target_column="target", sequence_length=sequence_length)

    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)

    # Step 3: Build the LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Step 4: Train the model
    print("Training the LSTM model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Step 5: Save the trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_trading_model.keras")
    print("Model saved to models/lstm_trading_model.keras")
