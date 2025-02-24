#!/usr/bin/env python3
"""
Reinforcement Learning Trading Agent with Extended Features

This script creates a custom Gym environment that uses multiple features from historical data,
trains a PPO agent on this data, and can run in test (simulation) or live trading mode via Binance.
"""

import argparse
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from binance.client import Client

# -----------------------------------------------------------------------------
# Configuration and Global Parameters
# -----------------------------------------------------------------------------

# Environment parameters
WINDOW: int = 24 * 7  # one week (in steps)
RSI_PERIOD: int = 14
TRAINING_STEPS: int = 1_000_000

# Trading and risk parameters
TRANSACTION_FEE_RATE: float = 0.001  # 0.1% fee on trades
SLIPPAGE_RATE: float = 0.001         # 0.1% simulated slippage
MIN_PORTFOLIO_THRESHOLD: float = 0.5  # 50% of the initial portfolio value

# API Credentials (ensure these are set as environment variables for security)
API_KEY: str = os.getenv("BINANCE_API_KEY", "YOUR_API_KEY")
API_SECRET: str = os.getenv("BINANCE_API_SECRET", "YOUR_API_SECRET")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# -----------------------------------------------------------------------------
# Technical Indicator Functions
# -----------------------------------------------------------------------------

def compute_rsi(prices: np.ndarray, period: int = RSI_PERIOD) -> float:
    """
    Compute the Relative Strength Index (RSI) for a given price series.
    Returns a neutral value (50.0) if there is insufficient data.
    """
    if len(prices) < period + 1:
        return 50.0  # neutral if not enough data
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -----------------------------------------------------------------------------
# Custom Trading Environment
# -----------------------------------------------------------------------------

class TradingEnv(gym.Env):
    """
    A trading environment that uses multiple technical indicators and additional
    features extracted from the dataset.
    
    Observation (20-dimensional vector):
      1. Normalized current price (Close / initial price)
      2. Normalized SMA (SMA / initial price)
      3. Normalized momentum (price change / initial price)
      4. RSI (raw, 0-100)
      5. Normalized volatility (std. dev. of Close / initial price)
      6. Normalized cash balance (balance / initial balance)
      7. Crypto held (raw number)
      8. Portfolio crypto ratio (crypto value / total portfolio)
      9. Normalized volume (Volume / avg_volume)
     10. Normalized high-low spread ((High - Low) / current price)
     11. Normalized Hour (Hour / 24)
     12-18. DayOfWeek one-hot (7 values; assuming DayOfWeek in [1,7])
     19. Normalized Trades (Trades / avg_trades)
     20. Normalized Taker Buy Base Volume (Taker Buy Base Volume / avg_taker_buy)
    
    Action (continuous in [-1, 1]):
      - action > 0: Buy a fraction of available cash (amplified).
      - action < 0: Sell a fraction of crypto held.
      - action == 0: Hold.
    
    Reward:
      Change in portfolio value adjusted for transaction fees, slippage,
      and risk penalties.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000, window: int = WINDOW) -> None:
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance: float = initial_balance
        self.balance: float = initial_balance
        self.crypto_held: float = 0.0
        self.current_step: int = 0
        self.window: int = window
        self.max_steps: int = len(self.data) - 1
        self.initial_price: float = self.data.loc[0, 'Close']  # For price normalization

        # Compute normalization constants from the dataset
        self.avg_volume: float = self.data['Volume'].mean()
        self.avg_trades: float = self.data['Trades'].mean()
        self.avg_taker_buy: float = self.data['Taker Buy Base Volume'].mean()

        # Define the observation space (20 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        # Action is a continuous value in [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        Compute and return the current observation vector (20 features).
        """
        row = self.data.loc[self.current_step]
        current_price = row['Close']
        start = max(0, self.current_step - self.window + 1)
        sma = self.data.loc[start:self.current_step, 'Close'].mean()
        momentum = current_price - (self.data.loc[self.current_step - 1, 'Close'] if self.current_step > 0 else current_price)
        rsi_window_start = max(0, self.current_step - RSI_PERIOD)
        prices_for_rsi = self.data.loc[rsi_window_start:self.current_step, 'Close'].values
        rsi = compute_rsi(prices_for_rsi, period=RSI_PERIOD)
        volatility = self.data.loc[start:self.current_step, 'Close'].std() if self.current_step > 0 else 0.0

        # Normalize price-related features
        current_price_norm = current_price / self.initial_price
        sma_norm = sma / self.initial_price
        momentum_norm = momentum / self.initial_price
        volatility_norm = volatility / self.initial_price
        balance_norm = self.balance / self.initial_balance

        portfolio_value = self.balance + self.crypto_held * current_price
        crypto_ratio = (self.crypto_held * current_price) / portfolio_value if portfolio_value > 0 else 0.0

        # Additional features from the current row
        volume_norm = row['Volume'] / self.avg_volume if self.avg_volume != 0 else 0.0
        high_low_spread = (row['High'] - row['Low']) / current_price
        hour_norm = row['Hour'] / 24.0

        # One-hot encoding for DayOfWeek (assumed values 1-7)
        day_of_week = int(row['DayOfWeek'])
        day_of_week_one_hot = np.zeros(7)
        if 1 <= day_of_week <= 7:
            day_of_week_one_hot[day_of_week - 1] = 1.0

        trades_norm = row['Trades'] / self.avg_trades if self.avg_trades != 0 else 0.0
        taker_buy_norm = row['Taker Buy Base Volume'] / self.avg_taker_buy if self.avg_taker_buy != 0 else 0.0

        # Construct the observation vector (20 features)
        obs_list = [
            current_price_norm,   # 1
            sma_norm,             # 2
            momentum_norm,        # 3
            rsi,                  # 4
            volatility_norm,      # 5
            balance_norm,         # 6
            self.crypto_held,     # 7
            crypto_ratio,         # 8
            volume_norm,          # 9
            high_low_spread,      # 10
            hour_norm             # 11
        ]
        obs_list.extend(day_of_week_one_hot.tolist())  # 12-18 (7 features)
        obs_list.extend([trades_norm, taker_buy_norm])     # 19-20

        return np.array(obs_list, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        """
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.current_step = 0
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment, applying fees, slippage, and risk penalties.
        """
        action_val: float = float(np.clip(action[0], -1, 1))
        row = self.data.loc[self.current_step]
        current_price: float = row['Close']
        effective_price: float = current_price * (1 + random.uniform(-SLIPPAGE_RATE, SLIPPAGE_RATE))
        prev_portfolio: float = self.balance + self.crypto_held * current_price

        if action_val > 0:  # Buy
            fraction: float = min(action_val * 1.5, 1)  # amplification factor
            amount_to_spend: float = self.balance * fraction
            fee: float = amount_to_spend * TRANSACTION_FEE_RATE
            crypto_bought: float = (amount_to_spend / effective_price) * (1 - TRANSACTION_FEE_RATE)
            self.crypto_held += crypto_bought
            self.balance -= amount_to_spend
            logging.info(f"Buy: Spent {amount_to_spend:.2f} (fee: {fee:.2f}), acquired {crypto_bought:.4f} units.")
        elif action_val < 0:  # Sell
            fraction: float = min(abs(action_val), 1)
            crypto_to_sell: float = self.crypto_held * fraction
            fee: float = crypto_to_sell * effective_price * TRANSACTION_FEE_RATE
            proceeds: float = crypto_to_sell * effective_price * (1 - TRANSACTION_FEE_RATE)
            self.balance += proceeds
            self.crypto_held -= crypto_to_sell
            logging.info(f"Sell: Sold {crypto_to_sell:.4f} units, received {proceeds:.2f} (fee: {fee:.2f}).")
        else:
            logging.info("Hold action executed.")

        self.current_step += 1
        done: bool = self.current_step >= self.max_steps
        new_price: float = current_price if done else self.data.loc[self.current_step, 'Close']
        new_portfolio: float = self.balance + self.crypto_held * new_price

        reward: float = new_portfolio - prev_portfolio

        # Encourage investment when price is rising
        if new_price > current_price and self.crypto_held < 1e-6:
            reward -= 0.1 * (new_price - current_price) / current_price

        # Risk management penalty if portfolio falls below threshold
        if new_portfolio < self.initial_balance * MIN_PORTFOLIO_THRESHOLD:
            penalty = 0.05 * (self.initial_balance * MIN_PORTFOLIO_THRESHOLD - new_portfolio)
            reward -= penalty
            logging.warning(f"Risk penalty applied: {penalty:.2f}")

        observation: np.ndarray = self._get_observation()
        return observation, reward, done, {}

    def render(self, mode: str = 'human', close: bool = False) -> None:
        """
        Render the current environment state.
        """
        current_price: float = self.data.loc[self.current_step, 'Close']
        portfolio_value: float = self.balance + self.crypto_held * current_price
        logging.info(f"Step: {self.current_step}, Price: {current_price:.4f}, "
                     f"Balance: {self.balance:.2f}, Crypto: {self.crypto_held:.4f}, "
                     f"Portfolio: {portfolio_value:.2f}")

# -----------------------------------------------------------------------------
# Binance API Helper Functions with Retry Logic
# -----------------------------------------------------------------------------

def get_live_price(client: Client, symbol: str = "AUCTIONUSDT", retries: int = 3, delay: int = 5) -> float:
    """
    Retrieve the live price from Binance with retry logic.
    """
    for attempt in range(retries):
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            return price
        except Exception as e:
            logging.error(f"Error getting live price (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    raise ConnectionError("Failed to get live price after multiple attempts.")

def place_order(client: Client, symbol: str, side: str, quantity: float, retries: int = 3, delay: int = 5) -> Dict:
    """
    Place a market order on Binance with retry logic.
    """
    for attempt in range(retries):
        try:
            order = client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            return order
        except Exception as e:
            logging.error(f"Error placing order (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    raise ConnectionError("Failed to place order after multiple attempts.")

# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------

def train_model() -> None:
    """
    Train the PPO agent on historical data and save the model.
    """
    logging.info("Loading historical data for training...")
    data_df = pd.read_csv('auction_data.csv')
    env = TradingEnv(data_df, initial_balance=100000, window=WINDOW)
    
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path='./checkpoints/', name_prefix='ppo_model')
    
    logging.info("Starting training...")
    model.learn(total_timesteps=TRAINING_STEPS, callback=checkpoint_callback)
    model.save("ppo_auction_trader")
    logging.info("Training complete. Model saved as 'ppo_auction_trader'.")

# -----------------------------------------------------------------------------
# Test Mode: Simulation on Historical Data with Real-Time Plotting
# -----------------------------------------------------------------------------

def test_mode() -> None:
    """
    Run a simulation on historical data and display a live-updating plot.
    """
    logging.info("Loading historical data for test mode...")
    data_df = pd.read_csv('auction_data.csv')
    env = TradingEnv(data_df, initial_balance=1000, window=WINDOW)
    model = PPO.load("ppo_auction_trader")
    
    steps, price_history, portfolio_history = [], [], []
    plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    obs = env.reset()
    logging.info("Starting test simulation...")
    step_idx = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        current_price = data_df.loc[env.current_step, 'Close']
        portfolio_value = env.balance + env.crypto_held * current_price

        steps.append(step_idx)
        price_history.append(current_price)
        portfolio_history.append(portfolio_value)
        step_idx += 1

        ax1.clear()
        ax2.clear()
        ax1.plot(steps, price_history, color='blue', label='AUCTION Price')
        ax2.plot(steps, portfolio_history, color='green', label='Portfolio')
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Price", color='blue')
        ax2.set_ylabel("Portfolio (USDT)", color='green')
        plt.title("Test Mode: Price and Portfolio Evolution")
        plt.pause(0.01)
    plt.ioff()
    plt.show()

# -----------------------------------------------------------------------------
# Live Trading Mode
# -----------------------------------------------------------------------------

def live_trading() -> None:
    """
    Execute live trading using the Binance API and the trained PPO agent.
    The observation vector is constructed from live data, matching the 20 features
    used during training.
    """
    client = Client(API_KEY, API_SECRET)
    model = PPO.load("ppo_auction_trader")
    
    # Load training data for normalization constants
    try:
        data_df = pd.read_csv('auction_data.csv')
        initial_price = data_df.loc[0, 'Close']
        avg_volume = data_df['Volume'].mean()
        avg_trades = data_df['Trades'].mean()
        avg_taker_buy = data_df['Taker Buy Base Volume'].mean()
    except Exception as e:
        logging.error("Error loading training data for normalization: " + str(e))
        initial_price = None
        avg_volume = 1.0
        avg_trades = 1.0
        avg_taker_buy = 1.0

    balance: float = 1000.0  # Initial USDT balance for live trading simulation
    crypto_held: float = 0.0
    symbol: str = "AUCTIONUSDT"
    window: int = WINDOW
    live_prices: list[float] = []
    steps, price_history, portfolio_history = [], [], []
    idx: int = 0

    plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    logging.info("Starting live trading...")
    while True:
        try:
            # Get the latest candle for additional fields
            kline = client.get_klines(symbol=symbol, interval='1m', limit=1)[0]
            # kline structure: [Open time, Open, High, Low, Close, Volume, ... , Trades, Taker buy base asset volume, ...]
            live_price = float(kline[4])
            high = float(kline[2])
            low = float(kline[3])
            candle_volume = float(kline[5])
            candle_trades = float(kline[8])
            candle_taker_buy = float(kline[9])
        except Exception as e:
            logging.error("Error fetching live candle: " + str(e))
            time.sleep(60)
            continue

        live_prices.append(live_price)
        if len(live_prices) > window:
            sma = np.mean(live_prices[-window:])
        else:
            sma = np.mean(live_prices)
        momentum = live_price - (live_prices[-2] if len(live_prices) > 1 else live_price)
        volatility = np.std(live_prices[-window:]) if len(live_prices) > 1 else 0.0
        rsi = compute_rsi(np.array(live_prices[-(RSI_PERIOD+1):]), period=RSI_PERIOD) if len(live_prices) >= RSI_PERIOD+1 else 50.0

        portfolio_value = balance + crypto_held * live_price
        crypto_ratio = (crypto_held * live_price) / portfolio_value if portfolio_value > 0 else 0.0

        # Use current time for hour and day-of-week features
        now = datetime.now()
        hour_norm = now.hour / 24.0
        day_of_week_one_hot = np.zeros(7)
        # Using Python's weekday() where Monday=0 and Sunday=6
        day_of_week_one_hot[now.weekday()] = 1.0

        # Set normalization constants if not available
        if initial_price is None:
            initial_price = live_prices[0]
        current_price_norm = live_price / initial_price
        sma_norm = sma / initial_price
        momentum_norm = momentum / initial_price
        volatility_norm = volatility / initial_price
        balance_norm = balance / 1000.0

        volume_norm = candle_volume / avg_volume if avg_volume != 0 else 0.0
        high_low_spread = (high - low) / live_price
        trades_norm = candle_trades / avg_trades if avg_trades != 0 else 0.0
        taker_buy_norm = candle_taker_buy / avg_taker_buy if avg_taker_buy != 0 else 0.0

        # Construct the 20-dimensional observation vector
        obs_list = [
            current_price_norm,   # 1
            sma_norm,             # 2
            momentum_norm,        # 3
            rsi,                  # 4
            volatility_norm,      # 5
            balance_norm,         # 6
            crypto_held,          # 7
            crypto_ratio,         # 8
            volume_norm,          # 9
            high_low_spread,      # 10
            hour_norm             # 11
        ]
        obs_list.extend(day_of_week_one_hot.tolist())  # 12-18
        obs_list.extend([trades_norm, taker_buy_norm])     # 19-20
        observation = np.array(obs_list, dtype=np.float32)

        action, _ = model.predict(observation, deterministic=True)
        logging.info(f"Step {idx} | Live Price: {live_price:.4f} | Action: {action[0]:.4f}")

        effective_price = live_price * (1 + random.uniform(-SLIPPAGE_RATE, SLIPPAGE_RATE))
        if action[0] > 0 and balance > 0:
            fraction = min(action[0], 1)
            amount_to_spend = balance * fraction
            fee = amount_to_spend * TRANSACTION_FEE_RATE
            crypto_bought = (amount_to_spend / effective_price) * (1 - TRANSACTION_FEE_RATE)
            try:
                order = place_order(client, symbol, "BUY", crypto_bought)
                crypto_held += crypto_bought
                balance -= amount_to_spend
                logging.info(f"Order executed: BUY {crypto_bought:.4f} units at {live_price:.4f}")
            except ConnectionError as e:
                logging.error(e)
        elif action[0] < 0 and crypto_held > 0:
            fraction = min(abs(action[0]), 1)
            crypto_to_sell = crypto_held * fraction
            fee = crypto_to_sell * effective_price * TRANSACTION_FEE_RATE
            proceeds = crypto_to_sell * effective_price * (1 - TRANSACTION_FEE_RATE)
            try:
                order = place_order(client, symbol, "SELL", crypto_to_sell)
                balance += proceeds
                crypto_held -= crypto_to_sell
                logging.info(f"Order executed: SELL {crypto_to_sell:.4f} units at {live_price:.4f}")
            except ConnectionError as e:
                logging.error(e)
        else:
            logging.info("No trade executed (Hold).")

        portfolio_value = balance + crypto_held * live_price
        steps.append(idx)
        price_history.append(live_price)
        portfolio_history.append(portfolio_value)
        idx += 1

        ax1.clear()
        ax2.clear()
        ax1.plot(steps, price_history, color='blue', label="AUCTION Price")
        ax2.plot(steps, portfolio_history, color="green", label="Portfolio")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Price", color='blue')
        ax2.set_ylabel("Portfolio (USDT)", color='green')
        plt.title("Live Trading: Price and Portfolio Evolution")
        plt.pause(1)

        time.sleep(60)  # Pause before next iteration

# -----------------------------------------------------------------------------
# Main Entry Point with CLI Argument Parsing
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Parse command-line arguments and execute the corresponding mode.
    """
    parser = argparse.ArgumentParser(description="RL Trading Agent: Train, Test, or Live Trade.")
    parser.add_argument("--mode", choices=["train", "test", "live"], required=True,
                        help="Select mode: 'train' for training, 'test' for simulation, 'live' for live trading.")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        test_mode()
    elif args.mode == "live":
        live_trading()
    else:
        logging.error("Invalid mode selected.")

if __name__ == '__main__':
    main()
