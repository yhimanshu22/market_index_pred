import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.data_loader import load_and_clean_data
from src.predictor import predict_next_steps
from src.evaluate import calculate_metrics

def evaluate_predictions():
    """
    Loads saved model, processes sample test data, and runs prediction 
    evaluations against ground-truth targets.
    """
    print("Loading model and scaler from models/...")
    try:
        model = load_model('models/saved_model.keras')
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"Error loading model or scaler. Ensure you have run train.py first: {e}")
        return

    # Load sample inputs to test
    print("Loading test input sequences with technical indicators...")
    df_sample = load_and_clean_data('data/evaluation_index_data.csv', include_indicators=True)
    actual_close_values = np.loadtxt('data/ground_truth_close_prices.txt')

    # Define features to use (must match train.py)
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_20', 'RSI']
    sequence_length = 20
    
    # Get recent sequences for predicting
    recent_data = df_sample[features].values[-sequence_length:]

    print("Generating predictions...")
    predictions = predict_next_steps(
        model=model, 
        recent_data=recent_data, 
        scaler=scaler, 
        sequence_length=sequence_length, 
        steps=2
    )

    print("Evaluating metrics against ground truth...")
    actual_prev_list = [df_sample['Close'].iloc[-1]]
    mse, direction_acc = calculate_metrics(
        actual_current=actual_close_values, 
        pred_current=predictions, 
        actual_previous_list=actual_prev_list
    )

    print("\n--- Evaluation Results ---")
    print(f"Mean Square Error:      {mse:.6f}")
    print(f"Directional Accuracy:   {direction_acc:.1f}%")


if __name__ == "__main__":
    evaluate_predictions()
