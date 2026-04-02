import os
from src.data_loader import load_and_clean_data, preprocess_training_data
from src.model_builder import build_lstm_model, train_model
from src.evaluate import cross_validate_model
import pickle

def main():
    print("Loading datasets and generating technical indicators...")
    # Load training data with indicators
    df = load_and_clean_data('data/training_index_data.csv', include_indicators=True)
    
    # Define features to use
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_20', 'EMA_20', 'RSI']
    sequence_length = 20
    
    # Preprocess
    print(f"Preprocessing data and creating sequences with length {sequence_length}...")
    X_train, y_train, scaler = preprocess_training_data(
        df, 
        sequence_length=sequence_length, 
        features=features
    )
    
    # Build model
    print(f"Building Multivariate LSTM Neural Network (Features: {len(features)})...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Optional: Perform Cross-Validation (Health Check)
    # We use a 3-fold CV here for a balance between data segments and speed
    cross_validate_model(
        X=X_train, 
        y=y_train, 
        build_fn=build_lstm_model, 
        train_fn=train_model, 
        n_splits=3
    )

    # Final Train
    print("\nTraining the final production model (100 Epochs)...")
    model = train_model(model, X_train, y_train, epochs=100, batch_size=32)
    
    # Save Model and Scaler for predictor
    print("Saving the model and scaler to models/...")
    model.save('models/saved_model.keras')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training Complete! Scaler and model have been saved in models/")

if __name__ == "__main__":
    main()
