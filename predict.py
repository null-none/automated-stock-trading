import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler



class PredictStockPrice:

    def __init__(self, name):
        self.df = pd.read_csv("./data/{}.csv".format(name))

    def stock_price(self):
        data = self.df.filter(["Close"]).values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        training_data_len = int(len(data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]
        x_train = []
        y_train = []

        for i in range(10, len(train_data)):
            x_train.append(train_data[i - 10 : i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.LSTM(
                100, return_sequences=True, input_shape=(x_train.shape[1], 1)
            )
        )
        model.add(tf.keras.layers.LSTM(100, return_sequences=False))
        model.add(tf.keras.layers.Dense(25))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=32, epochs=100)


        test_data = scaled_data[training_data_len - 10 :, :]
        x_test = []
        y_test = data[training_data_len:, :]

        for i in range(10, len(test_data)):
            x_test.append(test_data[i - 10 : i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)


        future_data = self.df.filter(["Close"])
        last_60_days = future_data[-10:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        future_prediction = model.predict(X_test)
        future_prediction = scaler.inverse_transform(future_prediction)
        print("Predicted price for the next day:", future_prediction[0][0])

        future_data = self.df.filter(["Close"])
        last_X_days = future_data[
            -800:
        ] 
        last_X_days_scaled = scaler.transform(last_X_days)
        X_test = []
        X_test.append(last_X_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_prices = []
        for i in range(10):
            predicted_price = model.predict(X_test)
            predicted_prices.append(predicted_price[0][0])
            X_test = np.append(X_test, [predicted_price], axis=1)

        predicted_prices = np.array(predicted_prices).reshape(-1, 1)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        return predicted_prices