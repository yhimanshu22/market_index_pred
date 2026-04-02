import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def calculate_metrics(actual_current: np.ndarray, pred_current: list, actual_previous_list: list) -> tuple:
    """
    Computes regression and classification metrics assessing the model.
    Returns (Mean Squared Error, Directional Accuracy %).
    """
    pred_current = np.array(pred_current)
    actual_current = np.array(actual_current)

    # Calculation of Mean Squared Error
    mean_square_error = np.mean(np.square(actual_current - pred_current))

    # Preparation for directional accuracy
    # Requires the prior step to determine "up" or "down" movement
    pred_prev = actual_previous_list.copy()
    pred_prev.append(pred_current[0])
    
    actual_prev = actual_previous_list.copy()
    actual_prev.append(actual_current[0])
    
    # Calculation of directional_accuracy
    pred_dir = np.array(pred_current) - np.array(pred_prev)
    actual_dir = np.array(actual_current) - np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir * actual_dir) > 0) * 100

    return mean_square_error, dir_accuracy

def cross_validate_model(X, y, build_fn, train_fn, n_splits=5, epochs=30):
    """
    Performs Time Series Cross-Validation (Walk-Forward Validation).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    print(f"\nStarting {n_splits}-Fold Time Series Cross-Validation ({epochs} Epochs per fold)...")
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Build a fresh model for each fold
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_fn(input_shape)
        
        # Train on the training fold
        model = train_fn(model, X_train, y_train, epochs=epochs)
        
        # Evaluate on the test fold
        predictions = model.predict(X_test, verbose=0)
        mse = np.mean(np.square(y_test - predictions.flatten()))
        cv_scores.append(mse)
        
        print(f"Fold {fold} MSE: {mse:.6f}")
        fold += 1
        
    avg_mse = np.mean(cv_scores)
    print(f"Average CV Mean Squared Error: {avg_mse:.6f}")
    return avg_mse
