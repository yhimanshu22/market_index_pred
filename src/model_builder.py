from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow import keras

def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    Builds the multi-layer LSTM neural network for multivariate time-series prediction.
    """
    model = Sequential()
    # First LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model: Sequential, X_train, y_train, epochs: int = 100, batch_size: int = 32):
    """
    Trains the compiled LSTM model on the provided data sequences.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
