import numpy as np
import matplotlib.pyplot as plt

# Asset names
assets = ["Stock A", "Stock B", "Stock C"]

# Expected returns
returns = np.array([0.12, 0.10, 0.08])

# Covariance matrix
cov_matrix = np.array([
    [0.005, -0.010, 0.004],
    [-0.010, 0.040, -0.002],
    [0.004, -0.002, 0.023]
])

num_portfolios = 5000  # many portfolios

# Lists to store all portfolio results
all_returns = []
all_risks = []

best_sharpe = -1
best_weights = None
best_return = 0
best_risk = 0

# Generate portfolios
for _ in range(num_portfolios):
    weights = np.random.random(len(assets))
    weights /= np.sum(weights)

    portfolio_return = np.sum(weights * returns)
    portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)

    all_returns.append(portfolio_return)
    all_risks.append(portfolio_risk)

    sharpe_ratio = portfolio_return / portfolio_risk

    if sharpe_ratio > best_sharpe:
        best_sharpe = sharpe_ratio
        best_weights = weights
        best_return = portfolio_return
        best_risk = portfolio_risk

# Print results
print("\nOptimal Portfolio Allocation")
for i, w in enumerate(best_weights):
    print(f"{assets[i]}: {round(w*100, 2)}%")

print("\nPortfolio Performance")
print(f"Expected Return: {round(best_return*100,2)}%")
print(f"Risk (Volatility): {round(best_risk*100,2)}%")
print(f"Sharpe Ratio: {round(best_sharpe,2)}")

# Pie chart
plt.figure()
plt.pie(best_weights, labels=assets, autopct='%1.1f%%')
plt.title("Optimal Portfolio Allocation")
plt.show()

# Efficient frontier
plt.figure()
plt.scatter(all_risks, all_returns, alpha=0.3, color='blue')  # small dots
plt.scatter(best_risk, best_return, color='red', marker='*', s=300)  # best portfolio
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.show()
