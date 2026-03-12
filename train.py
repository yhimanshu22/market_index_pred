import os
from src.data_loader import load_and_clean_data, preprocess_training_data
from src.model_builder import build_lstm_model, train_model
import pickle

def main():
    print("Loading datasets...")
    # Load training data
    df = load_and_clean_data('data/pred.csv')
    
    # Preprocess
    print("Preprocessing data and creating sequences...")
    X_train, y_train, scaler = preprocess_training_data(df, sequence_length=10)
    
    # Build model
    print("Building LSTM Neural Network...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train
    print("Training the model (100 Epochs)...")
    model = train_model(model, X_train, y_train, epochs=100, batch_size=32)
    
    # Save Model and Scaler for predictor
    print("Saving the model and scaler to models/...")
    model.save('models/saved_model.keras')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training Complete! Scaler and model have been saved in models/")

if __name__ == "__main__":
    main()
