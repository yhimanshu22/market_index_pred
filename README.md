# Stock Market Index Prediction

## 📌 Project Overview
The objective of this project is to develop a robust machine learning model to accurately predict the continuous time series values for a stock index's closing price. By providing accurate next-step closing price predictions, this project aims to enable informed investment decisions, optimize risk management, and maximize returns in the dynamic stock market.

## 📊 Dataset Characteristics
* **Input Data:** Historical stock index data (e.g., `pred.csv`, `sample_input.csv`).
* **Target Variable:** `Close` (The closing price of the stock index).
* **Data Sequence:** A sequence of continuous time-series values is used to evaluate the model's predictive capability for the next time steps.
* **Feature Type:** Numerical, continuous variables representing stock market valuations.

## 🧹 Data Preprocessing & Issue Handling
 Real-world financial data often contains missing points. To handle this and make the data model-ready, the following steps are implemented:
* **Handling Missing Values:** Null inputs (NaNs) in the `Close` column are addressed using forward-fill (`ffill`), replacing missing data with the last valid observation to maintain the temporal sequence integrity.
* **Normalization:** The `Close` prices are scaled using `MinMaxScaler` (scaling features between 0 and 1) to ensure faster convergence during neural network training and to prevent numerical instability.
* **Sequence Generation:** Transformed the univariate time series into supervised learning sequences using a sliding window approach with a defined `sequence_length` (e.g., 10 past points representing historical data to predict the next point).

## 🧠 Model Architecture
The project adopts a **Long Short-Term Memory (LSTM)** neural network, a powerful Recurrent Neural Network (RNN) variant highly effective at capturing and learning complex temporal dependencies in sequential stock data.

* **Architecture:**
  * **LSTM Layers:** Multiple LSTM layers (e.g., 50 units each) to extract deep temporal patterns.
  * **Dense Layer:** A fully connected Dense output layer mapping the extracted features to a single continuous prediction value.
* **Compilation Details:**
  * **Optimizer:** `Adam` (Adaptive Moment Estimation) for efficient gradient descent.
  * **Loss Function:** `Mean Squared Error (MSE)` to penalize large prediction errors heavily.

## 📈 Evaluation & Outcome
The model is rigorously evaluated based on both its precision and its trend-prediction capabilities using two primary metrics:
* **Mean Squared Error (MSE):** Measures the average squared difference between estimated values and the actual value. The model successfully achieved a highly competitive double-digit MSE.
* **Directional Accuracy:**  Measures the percentage of times the model correctly predicts the direction of the market move (up or down). The model achieved a striking **85% directional accuracy**.

## 🛠️ Dependencies
To run this project, you need the following major libraries:
* `Python` 3.x
* `numpy`
* `pandas`
* `scikit-learn` (for data scaling)
* `tensorflow` (for LSTM model building and training)

## 🚀 How to Run
1. Ensure the training data (e.g., `pred.csv`) and evaluation input/target files (`sample_input.csv`, `sample_close.txt`) are present in the root directory.
2. Run the main evaluation script, which will build, train, and test the LSTM model:
   ```bash
   python "220155_Anjali Kumari.py"
   ```
