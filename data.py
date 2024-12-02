import yfinance as yf
import pandas as pd
import numpy as np


class Data:
    def __init__(self):
        pass

    def download_data(self):
        tickers = [
            "MMM",
            "AXP",
            "AAPL",
            "BA",
            "CAT",
            "CVX",
            "CSCO",
            "KO",
            "DIS",
            "DOW",
            "GS",
            "HD",
            "IBM",
            "INTC",
            "JNJ",
            "JPM",
            "MCD",
            "MRK",
            "MSFT",
            "NKE",
            "PFE",
            "PG",
            "TRV",
            "UNH",
            "UTX",
            "VZ",
            "V",
            "WBA",
            "WMT",
            "XOM",
        ]

        def get_data(tickers):
            stock_data = {}
            for ticker in tickers:
                df = yf.download(ticker, start="2009-01-01", end="2024-12-01")
                stock_data[ticker] = df
            return stock_data

        stock_data = get_data(tickers)

        for ticker, df in stock_data.items():
            df.to_csv(f"data/{ticker}.csv")

    def load_data(self):

        tickers = [
            "MMM",
            "AXP",
            "AAPL",
            "BA",
            "CAT",
            "CVX",
            "CSCO",
            "KO",
            "DIS",
            "DOW",
            "GS",
            "HD",
            "IBM",
            "INTC",
            "JNJ",
            "JPM",
            "MCD",
            "MRK",
            "MSFT",
            "NKE",
            "PFE",
            "PG",
            "TRV",
            "UNH",
            "UTX",
            "VZ",
            "V",
            "WBA",
            "WMT",
            "XOM",
        ]
        benchmark = "^DJI"

        stock_data = {}
        for ticker in tickers:
            df = pd.read_csv(f"data/{ticker}.csv", index_col="Date", parse_dates=True)
            stock_data[ticker] = df

        training_data_time_range = ("2009-01-01", "2015-12-31")
        validation_data_time_range = ("2016-01-01", "2016-12-31")
        test_data_time_range = ("2017-01-01", "2020-05-08")

        self.training_data = {}
        self.validation_data = {}
        self.test_data = {}

        for ticker, df in stock_data.items():
            self.training_data[ticker] = df.loc[
                training_data_time_range[0] : training_data_time_range[1]
            ]
            self.validation_data[ticker] = df.loc[
                validation_data_time_range[0] : validation_data_time_range[1]
            ]
            self.test_data[ticker] = df.loc[
                test_data_time_range[0] : test_data_time_range[1]
            ]

        ticker = "AAPL"
        print(f"Training data shape for {ticker}: {self.training_data[ticker].shape}")
        print(
            f"Validation data shape for {ticker}: {self.validation_data[ticker].shape}"
        )
        print(f"Test data shape for {ticker}: {self.test_data[ticker].shape}")

        stock_data["AAPL"].head()

    def add_technical_indicators(self, df):

        delta = df["Close"].diff()
        up = delta.where(delta > 0, 0)
        down = -delta.where(delta < 0, 0)
        rs = up.rolling(window=14).mean() / down.rolling(window=14).mean()
        df["RSI"] = 100 - (100 / (1 + rs))

        df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        sma_tp = tp.rolling(window=20).mean()
        mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df["CCI"] = (tp - sma_tp) / (0.015 * mean_dev)

        high_diff = df["High"].diff()
        low_diff = df["Low"].diff()
        df["+DM"] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        df["-DM"] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                np.abs(df["High"] - df["Close"].shift(1)),
                np.abs(df["Low"] - df["Close"].shift(1)),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        df["+DI"] = 100 * (df["+DM"].ewm(span=14, adjust=False).mean() / atr)
        df["-DI"] = 100 * (df["-DM"].ewm(span=14, adjust=False).mean() / atr)
        dx = 100 * np.abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])
        df["ADX"] = dx.ewm(span=14, adjust=False).mean()

        df.dropna(inplace=True)

        df = df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "MACD",
                "Signal",
                "RSI",
                "CCI",
                "ADX",
            ]
        ]

        return df

    def training(self):
        for ticker, df in self.training_data.items():
            self.training_data[ticker] = self.add_technical_indicators(df)

        for ticker, df in self.validation_data.items():
            self.validation_data[ticker] = self.add_technical_indicators(df)

        for ticker, df in self.test_data.items():
            self.test_data[ticker] = self.add_technical_indicators(df)

        print("Shape of training data for AAPL:", self.training_data["AAPL"].shape)
        print("Shape of validation data for AAPL:", self.validation_data["AAPL"].shape)
        print("Shape of test data for AAPL:", self.test_data["AAPL"].shape)
        return self.training_data