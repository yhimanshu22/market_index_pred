import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

def predict_next_steps(model: Sequential, recent_data: np.ndarray, scaler: MinMaxScaler, sequence_length: int = 10, steps: int = 2) -> list:
    """
    Generates autoregressive predictions for the given time steps iteratively.
    recent_data should be the unscaled most recent `sequence_length` data points.
    """
    # Scale recent data using the same scaler fitted on training data
    scaled_recent = scaler.transform(recent_data)
    
    last_sequence = scaled_recent[-sequence_length:]
    next_sequence = []

    for _ in range(steps):
        # Predict the next point
        prediction = model.predict(last_sequence.reshape(1, sequence_length, recent_data.shape[1]), verbose=0)
        next_sequence.append(prediction[0, 0])
        # Append predicted point and slide sequence window
        last_sequence = np.concatenate((last_sequence[1:], prediction.reshape(1, -1)), axis=0)

    # Inverse transform to get human-readable price values
    predictions = scaler.inverse_transform(np.array(next_sequence).reshape(-1, 1))
    return predictions.flatten().tolist()
