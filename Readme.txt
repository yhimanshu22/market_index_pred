220155 Anjali Kumari

Code For Training The Model:

train_size = int(len(data) * 0.85)  # 75% for training, 20% for testing
train_data = data[:train_size]
test_data = data[train_size:]

def create_sequences(data, sequence_length):
    x = []
    y = []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(x), np.array(y)

sequence_length = 10  # Adjust as needed
x_train, y_train = create_sequences(train_data, sequence_length)
x_test, y_test = create_sequences(test_data, sequence_length)

model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=100, batch_size=32)

train_loss = model.evaluate(x_train, y_train, verbose=0)
test_loss = model.evaluate(x_test, y_test, verbose=0)

print("Training loss:", train_loss)
print("Testing loss:", test_loss)

model.save('saved_model.h5') # Save the model to use it in pred_func