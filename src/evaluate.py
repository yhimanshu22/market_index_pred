import numpy as np

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
