# Stock Market Index Prediction

## 📌 Project Overview
The objective of this project is to develop a robust machine learning model to accurately predict the continuous time series values for a stock index's closing price. By providing accurate next-step closing price predictions, this project aims to enable informed investment decisions, optimize risk management, and maximize returns in the dynamic stock market.

## 🏗️ Architecture
The codebase has been refactored into a scalable, modular architecture with Single Responsibility Principles:

```text
Market-Index-Prediction/
├── data/                      # Dataset directory
│   ├── pred.csv               # Training sequences
│   ├── sample_input.csv       # Evaluation sequences
│   └── sample_close.txt       # Ground truth targets
├── models/                    # Saved artifacts
│   ├── saved_model.keras      # Trained LSTM network
│   └── scaler.pkl             # Fitted MinMaxScaler
├── src/                       # Core Library
│   ├── data_loader.py         # Parsing and normalization logic
│   ├── evaluate.py            # Statistical metrics
│   ├── model_builder.py       # Keras model generation
│   └── predictor.py           # Autoregressive inference logic
├── predict.py                 # Evaluation entrypoint script
├── train.py                   # Training entrypoint script
└── README.md
```

* `src/data_loader.py` - Responsible for parsing real-world incomplete data, applying forward-fills (`ffill`), and scaling.
* `src/model_builder.py` - Responsible for compiling the multi-layer Long Short-Term Memory (LSTM) recurrent neural network architecture.
* `src/predictor.py` - Responsible for autoregressively predicting sequences.
* `src/evaluate.py` - Responsible for scoring statistical metrics against raw targets.
* `train.py` - The primary entrypoint to generate the learning model safely.
* `predict.py` - The primary entrypoint to measure predictions against hold-out samples.

## 📊 Dataset Characteristics
* **Input Data:** Located in `data/`, structured as `pred.csv` and `sample_input.csv`.
* **Target Variable:** `Close` (The closing price of the stock index).
* **Data Sequence:** A sequence of continuous time-series values is used to evaluate the model's predictive capability for the next time steps.
* **Feature Type:** Numerical, continuous variables representing stock market valuations.

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
* **Mean Squared Error (MSE):** Measures the average squared difference between estimated values and the actual value.
* **Directional Accuracy:**  Measures the percentage of times the model correctly predicts the direction of the market move (up or down). 

## 🛠️ Dependencies
To run this project, you need the following major libraries. We manage our environment with `uv`:
* `Python` 3.x
* `numpy`
* `pandas`
* `scikit-learn` (for data scaling)
* `tensorflow` (for LSTM model building and training)

## 🚀 How to Run
Ensure `uv` is installed, then set up your environment:
```bash
uv venv
source .venv/Scripts/activate
uv pip install numpy pandas scikit-learn tensorflow
```

**1. Train the model from scratch:**
```bash
uv run python train.py
```
*This will read `data/pred.csv` and output standard serialization objects into `models/saved_model.keras` and `models/scaler.pkl`.*

**2. Evaluate the model:**
```bash
uv run python predict.py
```
*This script will load the evaluation set (`data/sample_input.csv`), compute tests against actuals (`data/sample_close.txt`), and print evaluation metrics directly.*
