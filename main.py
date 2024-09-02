import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from data_fetch import get_data
from monte_carlo import MonteCarlo

def main():
    # PORTFOLIO PARAMETERS
    stock_list = ['SPY', 'QQQ', 'SMH', 'GLD', 'TLT']  # list of stocks
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # portfolio weights
    end_date = dt.datetime.now().strftime('%Y-%m-%d')  # current date
    start_date = (dt.datetime.now() - dt.timedelta(days=252*30)).strftime('%Y-%m-%d')  # date 30 years ago
    portfolio_value = 10000  # initial portfolio value

    # SIMULATION PARAMETERS
    days = 252*10  # number of days to simulate
    simulations = 1000  # number of simulations to run
    confidence_interval = 0.99  # 99% confidence level

    # fetch data w yfinance
    returns, mean_returns, cov_matrix = get_data(stock_list, start=start_date, end=end_date)
    
    # run Monte Carlo simulation
    portfolio_sims = MonteCarlo.run_simulation(weights, mean_returns, cov_matrix, portfolio_value, days, simulations)
    final_values = portfolio_sims[-1, :]  # final portfolio values at the last simulation day

    # calculate VaR for confidence intervals
    VaR = MonteCarlo.calculate_var(portfolio_sims, confidence_interval, portfolio_value)
    
    # plot all simulation paths
    plt.figure(figsize=(10, 6))
    
    for i in range(simulations):
        plt.plot(portfolio_sims[:, i], color=np.random.rand(3,), alpha=0.3, linewidth=0.7)

    # plot mean simulation line
    mean_simulation = np.mean(portfolio_sims, axis=1)
    plt.plot(mean_simulation, color='red', linewidth=2, label='Mean Simulation')  # Highlight mean with a thicker line

    # annotate VaR line for confidence interval
    plt.axhline(y=(portfolio_value - VaR), color='green', linestyle='-', linewidth=1, label=f'VaR ({confidence_interval*100}%): ${VaR:.2f}')
    
    # add legend and labels
    plt.legend(loc='upper left')
    plt.title(f'Simulation of Portfolio Value over {days} days. ({simulations} trials)')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(False)
    plt.show()

    # plot histogram of final portfolio values
    plt.figure(figsize=(8, 6))
    plt.hist(final_values, bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel('Final Portfolio Value ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Portfolio Values')
    plt.grid(False)
    plt.show()

    # Print statements for additional context
    print(f"Initial Portfolio Value: ${portfolio_value:.2f}")
    print(f"Simulation Time Horizon: {days} days")
    print(f"There is a {(1-confidence_interval)*100:.0f}% chance that your portfolio value will fall below ${portfolio_value - VaR:.2f} in {days} days, based on a {100 - confidence_interval*100}% risk level ({confidence_interval*100}% confidence level).")

if __name__ == "__main__":
    main()
