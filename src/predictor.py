import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

def predict_next_steps(model: Sequential, recent_data: np.ndarray, scaler: MinMaxScaler, sequence_length: int = 10, steps: int = 2) -> list:
    """
    Generates autoregressive predictions for the given time steps iteratively.
    recent_data should be the unscaled most recent `sequence_length` data points (multivariate).
    """
    # Scale recent data using the same scaler fitted on training data
    scaled_recent = scaler.transform(recent_data)
    
    last_sequence = scaled_recent[-sequence_length:]
    next_sequence = []
    
    # Identify target index (assumed to be 0 for 'Close')
    target_idx = 0 

    for _ in range(steps):
        # Predict the next point
        input_data = last_sequence.reshape(1, sequence_length, recent_data.shape[1])
        prediction_vec = model.predict(input_data, verbose=0)
        
        # Extract predicted Close price
        predicted_close = prediction_vec[0, 0]
        next_sequence.append(predicted_close)
        
        # Prepare next feature vector for autoregression
        # We take the last known feature vector and replace 'Close' with our prediction
        new_feature_vec = last_sequence[-1].copy()
        new_feature_vec[target_idx] = predicted_close
        
        # Slide sequence window
        last_sequence = np.concatenate((last_sequence[1:], new_feature_vec.reshape(1, -1)), axis=0)

    # Inverse transform to get human-readable price values
    # We need to create a dummy array for inverse_transform because scaler expects all features
    dummy_output = np.zeros((len(next_sequence), recent_data.shape[1]))
    dummy_output[:, target_idx] = next_sequence
    predictions = scaler.inverse_transform(dummy_output)
    
    return predictions[:, target_idx].flatten().tolist()
