# Reinforcement Learning Trading Agent

This project implements a reinforcement learning (RL) trading agent using a custom Gym environment. The agent is trained using the Proximal Policy Optimization (PPO) algorithm and incorporates multiple technical indicators and market features for decision-making.

## Features

- **Custom Gym Environment**: Simulates cryptocurrency trading using historical market data.
- **Advanced Feature Set**:
  - Relative Strength Index (RSI)
  - Moving Averages (SMA)
  - Momentum and volatility measures
  - Volume, trades, and high-low spread
  - Time-based features (Hour, Day of the Week)
- **Transaction Cost & Slippage Modeling**: Includes realistic trading constraints.
- **Live Trading Mode**: Fetches real-time price data from Binance API.
- **Risk Management**: Implements portfolio thresholds and risk penalties.

## Installation

### Requirements

- Python 3.8+
- Dependencies (install via pip):

```bash
pip install numpy pandas gym matplotlib stable-baselines3 binance
```

## Usage

### Training the Agent

To train the agent using historical data:

```bash
python train.py --data historical_data.csv --steps 1000000
```

### Running in Live Mode

To run the agent in live trading mode with Binance:

```bash
python trade.py --live
```

### Testing the Agent

To test the trained agent on unseen data:

```bash
python trade.py --test --data test_data.csv
```

## Binance API Configuration

To use live trading, set up your Binance API keys as environment variables:

```bash
export BINANCE_API_KEY='your_api_key'
export BINANCE_API_SECRET='your_api_secret'
```

## Observations & Actions

- **Observation Space (20 Features):**
  - Normalized price, SMA, momentum, RSI, volatility
  - Balance and portfolio composition
  - Market volume, high-low spread, trades, and taker buy volume
  - Time-based features (Hour, Day of the Week - one-hot encoded)
- **Action Space:**
  - Continuous action in range [-1, 1]
  - Positive: Buy, Negative: Sell, Zero: Hold

## Reward Function

- Portfolio value changes adjusted for transaction fees and slippage
- Penalties for excessive risk and missed opportunities

## Logging & Visualization

- Logging trade actions, portfolio changes, and risk penalties
- Portfolio performance visualization using Matplotlib

## License

This project is licensed under the MIT License.

## Contributors

- Me Ayvem
- I was alone
- Contributions and PRs are welcome!

## Future Improvements

- Implement deep Q-learning (DQN) for comparison
- Explore recurrent architectures for sequential decision-making
- Optimize hyperparameters using Bayesian optimization
- Im going to manualy tweak certains parameters to see if the results gets better 

---

For issues or feature requests, please open an issue on [GitHub](https://github.com/Ayvem/trading-IA).
