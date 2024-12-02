import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, stock_data):
        super(StockTradingEnv, self).__init__()

        self.stock_data = {
            ticker: df for ticker, df in stock_data.items() if not df.empty
        }
        self.tickers = list(self.stock_data.keys())

        if not self.tickers:
            raise ValueError("All provided stock data is empty")

        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32
        )

        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

        self.initial_balance = 1000
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}

        self.current_step = 0

        self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        return self._next_observation(), {}

    def _next_observation(self):
        frame = np.zeros(self.obs_shape)

        idx = 0
        for ticker in self.tickers:
            df = self.stock_data[ticker]
            if self.current_step < len(df):
                frame[idx : idx + self.n_features] = df.iloc[self.current_step].values
            elif len(df) > 0:
                frame[idx : idx + self.n_features] = df.iloc[-1].values
            idx += self.n_features

        frame[-4 - len(self.tickers)] = self.balance
        frame[-3 - len(self.tickers) : -3] = [
            self.shares_held[ticker] for ticker in self.tickers
        ]
        frame[-3] = self.net_worth
        frame[-2] = self.max_net_worth
        frame[-1] = self.current_step

        return frame

    def step(self, actions):
        self.current_step += 1

        if self.current_step > self.max_steps:
            return self._next_observation(), 0, True, False, {}

        current_prices = {}
        for i, ticker in enumerate(self.tickers):
            current_prices[ticker] = self.stock_data[ticker].iloc[self.current_step][
                "Close"
            ]
            action = actions[i]

            if action > 0:  # Buy
                shares_to_buy = int(self.balance * action / current_prices[ticker])
                cost = shares_to_buy * current_prices[ticker]
                self.balance -= cost
                self.shares_held[ticker] += shares_to_buy
            elif action < 0:  # Sell
                shares_to_sell = int(self.shares_held[ticker] * abs(action))
                sale = shares_to_sell * current_prices[ticker]
                self.balance += sale
                self.shares_held[ticker] -= shares_to_sell
                self.total_shares_sold[ticker] += shares_to_sell
                self.total_sales_value[ticker] += sale

        self.net_worth = self.balance + sum(
            self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers
        )
        self.max_net_worth = max(self.net_worth, self.max_net_worth)

        reward = self.net_worth - self.initial_balance
        done = self.net_worth <= 0 or self.current_step >= self.max_steps

        obs = self._next_observation()
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        for ticker in self.tickers:
            print(f"{ticker} Shares held: {self.shares_held[ticker]}")
        print(f"Net worth: {self.net_worth:.2f}")
        print(f"Profit: {profit:.2f}")

    def close(self):
        pass

    def update_stock_data(self, new_stock_data):
        """
        Update the environment with the new stocks dataset.

        Parameters:
        new_stock_data (dict): Dictionary containing new stock data,
                                with the key being the stock code and the value being the DataFrame.
        """

        self.stock_data = {
            ticker: df for ticker, df in new_stock_data.items() if not df.empty
        }
        self.tickers = list(self.stock_data.keys())

        if not self.tickers:
            raise ValueError("All new stock data is blank")

        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)

        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

        self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)

        self.reset()

        print(f"The environment has been updated with new {len(self.tickers)} shares.")
