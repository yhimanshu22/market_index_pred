import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators: SMA, EMA, and RSI.
    """
    # Create indicators on the fly
    df = df.copy()
    
    # 20-period Simple Moving Average
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    # 20-period Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Handle division by zero for RSI
    loss = loss.replace(0, 1e-9)
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Fill NaN values created by rolling windows with forward then backward fill
    df = df.ffill().bfill()
    return df

def load_and_clean_data(filepath: str, include_indicators: bool = False) -> pd.DataFrame:
    """
    Loads raw continuous time series index data from CSV and handles NaNs.
    Optionally adds technical indicators.
    """
    df = pd.read_csv(filepath)
    # Fill null values in 'Close' column with the last valid observation
    if 'Close' in df.columns:
        df['Close'] = df['Close'].ffill()
    
    if include_indicators:
        df = add_technical_indicators(df)
        
    return df

def preprocess_training_data(data: pd.DataFrame, sequence_length: int = 10, features: list = None):
    """
    Normalizes the selected features and creates a supervised learning sequence dataset.
    Returns the X (sequences), y (targets), and the fitted scaler object.
    """
    if features is None:
        features = ['Close']
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Select features and target (Close is always the target at index 0 of features)
    feature_data = data[features].values
    scaled_data = scaler.fit_transform(feature_data)
    
    # Target index (assuming 'Close' is in features)
    target_idx = features.index('Close')

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i, target_idx])

    return np.array(X), np.array(y), scaler
