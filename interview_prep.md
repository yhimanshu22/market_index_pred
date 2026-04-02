# Interview Preparation: Stock Market Index Prediction

This document contains a curated list of interview questions and answers tailored to the **Market Index Prediction** project. It covers general overview, deep learning, time series specific, and software engineering aspects.

---

## 🏗️ Project Overview & Behavioral

### 1. Can you explain your project in simple terms?
**Answer:** This project is a machine learning pipeline that predicts the future closing prices of a stock market index. It uses historical sequential data and processes it through a type of neural network called an **LSTM (Long Short-Term Memory)**, which is particularly good at remembering patterns over time. The goal is to provide a forecasted trend rather than just a single point estimate.

### 2. What was your motivation for choosing an LSTM model for this task?
**Answer:** Stock market data is traditionally a **time series**, where the value at any given time $t$ is dependent on its previous values $t-1, t-2$, etc. Standard neural networks treat each input independently (stationary data). LSTMs handle this sequential dependency by maintaining a "cell state" that acts as a memory, allowing the model to capture long-term trends and seasonality.

### 3. What were the most significant challenges you faced?
**Answer:** One major challenge was **data cleaning** and dealing with missing values without breaking the temporal order. Stock markets are closed on weekends and holidays, which can create gaps in the data. Another challenge was implementing **autoregressive prediction**, where the model's own output is fed back as an input for the next step, as errors can propagate and compound.

---

## 🧠 Machine Learning & Deep Learning

### 4. Why did you use Mean Squared Error (MSE) as your loss function?
**Answer:** MSE penalizes larger errors more heavily because it squares the difference between the predicted and actual values. In stock price prediction, a large deviation is much more risky than multiple small ones, making MSE a standard choice for regression-based forecasting.

### 5. What is the role of the 'Dense' layer at the end of your LSTM network?
**Answer:** The LSTM layers extract temporal features and patterns from the data. The **Dense layer** (multi-layer perceptron) acts as the final regression head, mapping those high-dimensional features down to a single continuous value: the predicted closing price.

### 6. Why did you use 50 units in each LSTM layer?
**Answer:** The number of units (hidden dimensions) represents the model's capacity to learn complex patterns. I chose 50 as a balance between learning capacity and the risk of **overfitting**. Since market data is noisy, too many units might lead the model to "memorize" the noise rather than the trend.

---

## 📈 Time Series Analysis

### 7. What is a "Sliding Window" and how did you use it here?
**Answer:** A sliding window transforms a raw time series into a supervised learning problem. For a window of size 10 (as used in this project), we take the prices from day 1 to 10 $(X)$ to predict the price on day 11 $(y)$. Then we slide one step: use day 2 to 11 to predict day 12. This creates the sequences $X$ and targets $y$ used for training.

### 8. Explain the difference between 'fit' and 'transform' in your `MinMaxScaler`?
**Answer:** 
- `fit`: Calculates the minimum and maximum values of the training data.
- `transform`: Scales the data using those calculated values to the specified range (0 to 1).
**Critical Point:** We must only `fit` on the training data and then `transform` both training and test data using those same parameters to prevent **data leakage**.

### 9. What is "Autoregressive Prediction" and why is it used in `predict_next_steps`?
**Answer:** Autoregressive prediction is when the model predicts the next value, appends it to the input sequence, and uses this *new* sequence to predict the *following* value. It's used because, in real-world scenarios, we don't have future prices. This allows us to forecast multiple steps into the future using only historical data.

---

## 🏗️ Data & Model Inputs

### 10. What are the specific inputs to your model?
**Answer:** The project handles inputs at different stages:
1.  **Raw Input:** A CSV file (like `data/pred.csv`) containing historical stock prices. The primary feature used is the `Close` price.
2.  **Training Input (X):** A 3D tensor of shape `(samples, time_steps, features)`. In this project, `time_steps` is 10 and `features` is 1 (univariate), so the input shape for a single batch is `(batch_size, 10, 1)`.
3.  **Prediction Input:** The most recent 10 closing prices. These are scaled using the `MinMaxScaler` before being fed into the model.

### 11. How did you handle missing data in `data_loader.py`?
**Answer:** I used the **Forward Fill (`ffill`)** method. In financial data, if a price is missing (e.g., due to a data collection error during market hours), the most logical estimate is the last known price. This maintains the continuity of the time series better than simple abandonment or mean imputation.

### 11. Your code is very modular. Why separate `data_loader`, `model_builder`, and `predictor`?
**Answer:** This follows the **Single Responsibility Principle (SRP)**. It makes the system easier to test (unit testing each part independently), enables reusability (the same model builder can be used for different indices), and improves maintainability for future developers.

### 12. Why save the `scaler.pkl` along with the model?
**Answer:** The model was trained on normalized data (0 to 1). To generate human-readable predictions, we must **inverse transform** the model's output. Saving the exact scaler used during training ensures that the reverse mapping is mathematically consistent with the training distribution.

---

## 📊 Evaluation Metrics

### 13. What is "Directional Accuracy"?
**Answer:** While MSE measures how *close* we are to the actual price, **Directional Accuracy** measures if we correctly predicted the *movement* (Up or Down). In trading, knowing if the price will go up is often more important than the exact value.

### 14. What are some limitations of your current model?
**Answer:** The model currently only uses **univariate** data (historical prices). It doesn't account for external factors like news sentiment, interest rate changes, or volume. Additionally, stock markets are highly stochastic and non-stationary, meaning past patterns don't always guarantee future performance.
