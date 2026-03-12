import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw continuous time series index data from CSV and handles NaNs.
    """
    df = pd.read_csv(filepath)
    # Fill null values in 'Close' column with the last valid observation
    if 'Close' in df.columns:
        df['Close'] = df['Close'].ffill()
    return df

def preprocess_training_data(data: pd.DataFrame, sequence_length: int = 10):
    """
    Normalizes the closing price and creates a supervised learning sequence dataset.
    Returns the X (sequences), y (targets), and the fitted scaler object.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_data = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler
