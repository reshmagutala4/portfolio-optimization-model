# Portfolio Optimization Model

Modern Portfolio Theory applied to 5 US stocks (AAPL, MSFT, NVDA, AMZN, TSLA). Weights are chosen in Excel Solver to maximize the Sharpe ratio, then re-implemented in Python.

## Result

Optimizing under a 40% cap per stock lifted the daily Sharpe ratio by about 20% over an equal-weight portfolio.

| | Equal weight | Optimized (max 40% per stock) |
|---|---|---|
| Expected daily return | 0.154% | 0.194% |
| Daily risk (std dev) | 1.76% | 1.84% |
| Sharpe ratio (daily) | 0.087 | **0.105** |
| Sharpe ratio (annualized) | 1.39 | 1.67 |

Optimized weights: AAPL 36.9%, NVDA 40.0%, AMZN 23.1%, MSFT 0%, TSLA 0%.

Return rose by more than risk did, which is where the Sharpe gain comes from. Without the 40% cap, the optimizer puts about 71% in NVDA (daily Sharpe 0.109), which is why the cap is used.

## Data and method

- 750 daily returns, Feb 2023 to Feb 2026 (3 years of daily prices)
- Risk-free rate = 0
- Portfolio risk uses the full covariance matrix, so correlations between stocks are included
- Excel Solver maximizes the Sharpe ratio, with weights summing to 100% and each weight between 0% and 40%
- Python version (Pandas, NumPy, SciPy, Matplotlib) re-runs the optimization with SciPy SLSQP (25 starting points) and gets the same Sharpe ratio and weights as Excel, within 0.1 point
- Charts: optimal allocation pie, and an efficient frontier scatter of 5,000 random portfolios (max 40% per stock) with the equal-weight and max-Sharpe portfolios marked

## Limitations

- In-sample: the same 3 years are used to choose the weights, so results do not predict future returns
- Past average returns are a noisy estimate of future returns
- No transaction costs or rebalancing

## Files

- `Portfolio_Optimizer(3)(1).xlsx`: prices, returns, covariance matrix, Solver model
- `Portfolio_Optimizer.py`: Python version (keep it in the same folder as the .xlsx file and run it)
- `optimal_allocation.png`, `efficient_frontier.png`: Python charts
