import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

folder = os.path.dirname(os.path.abspath(__file__))
excel_files = [f for f in os.listdir(folder) if f.endswith(".xlsx")]
FILE = os.path.join(folder, excel_files[0])
print(f"Using file: {FILE}")

CAP = 0.40
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]

raw = pd.read_excel(FILE, sheet_name=0)
raw.columns = [str(c).strip() for c in raw.columns]
prices = raw[["Date"] + TICKERS].dropna().set_index("Date")
returns = prices.pct_change().dropna()
print(f"{len(prices)} daily prices, {len(returns)} daily returns ({prices.index[0]} to {prices.index[-1]})")

mu = returns.mean().values
cov = returns.cov().values

def portfolio_stats(w):
    ret = float(w @ mu)
    risk = float(np.sqrt(w @ cov @ w))
    return ret, risk, ret / risk

def maximize_sharpe(cap):
    best = None
    for seed in range(25):
        start = np.repeat(1/5, 5) if seed==0 else np.random.default_rng(seed).dirichlet(np.ones(5))
        out = minimize(lambda w: -portfolio_stats(w)[2], start, method="SLSQP", bounds=[(0, cap)]*5, constraints=({"type": "eq", "fun": lambda w: w.sum()-1},))
        if out.success and (best is None or out.fun < best.fun):
            best = out
    return best.x

w_equal = np.repeat(1/5, 5)
w_opt = maximize_sharpe(CAP)

# --- Print in table format like expected ---
print(f"\n{'':<22}{'Equal weight':>14}{'Optimized (40% cap)':>22}")
for name, we, wo in zip(TICKERS, w_equal, w_opt):
    print(f"{name:<22}{we*100:>13.1f}%{wo*100:>21.1f}%")

re, ke, se = portfolio_stats(w_equal)
ro, ko, so = portfolio_stats(w_opt)

print(f"{'Daily return':<22}{re*100:>13.3f}%{ro*100:>21.3f}%")
print(f"{'Daily risk':<22}{ke*100:>13.3f}%{ko*100:>21.3f}%")
print(f"{'Daily Sharpe':<22}{se:>14.4f}{so:>22.4f}")
print(f"{'Annualized Sharpe':<22}{se*np.sqrt(252):>14.3f}{so*np.sqrt(252):>22.3f}")
print(f"\nSharpe improvement vs equal weight: {(so/se-1)*100:.1f}%")

# --- Charts ---
keep = w_opt > 0.001
plt.figure(figsize=(6,6))
plt.pie(w_opt[keep], labels=np.array(TICKERS)[keep], autopct="%1.1f%%", startangle=90)
plt.title(f"Optimal allocation (max {int(CAP*100)}% per stock)")
plt.tight_layout()
plt.savefig("optimal_allocation.png", dpi=200)
plt.show()

rng = np.random.default_rng(42)
pts = []
while len(pts) < 5000:
    w = rng.dirichlet(np.ones(5))
    if w.max() <= CAP:
        r,k,s = portfolio_stats(w)
        pts.append((k,r,s))
pts = np.array(pts)
plt.figure(figsize=(8,5.5))
plt.scatter(pts[:,0]*100, pts[:,1]*100, c=pts[:,2], cmap="viridis", s=8, alpha=0.7)
plt.scatter(ke*100, re*100, c="grey", s=120, marker="D", label="Equal weight", edgecolor="white")
plt.scatter(ko*100, ro*100, c="red", s=220, marker="*", label="Max Sharpe (40% cap)", edgecolor="white")
plt.xlabel("Daily risk (%)"); plt.ylabel("Daily expected return (%)")
plt.title("Efficient frontier (portfolios with max 40% per stock)")
plt.legend(); plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=200)
plt.show()
print("\nSaved: optimal_allocation.png, efficient_frontier.png")
