import pandas as pd
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_basic_lstm_model(input_shape):
    """
    Builds a basic LSTM model with a single LSTM layer.
    
    Parameters:
    - input_shape: A tuple indicating the shape of the input data, e.g., (30, 1) for 30 time steps with 1 feature per step.
    
    Returns:
    - Compiled Keras model ready for training.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_stacked_lstm_model(input_shape):
    """
    Builds a stacked LSTM model with dropout for improved generalization. 
    It includes two LSTM layers and two Dropout layers.
    
    Parameters:
    - input_shape: A tuple indicating the shape of the input data.
    
    Returns:
    - Compiled Keras model ready for training.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Trains the given LSTM model with the provided training data and evaluates it on the test data.
    
    Parameters:
    - model: The LSTM model to be trained.
    - X_train, y_train: Training data (features and target).
    - X_test, y_test: Test data (features and target).
    - epochs: Number of epochs to train the model.
    - batch_size: Batch size for the training process.
    
    Returns:
    - History object containing recorded training/validation loss over epochs.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)
    return history
