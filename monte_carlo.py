import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MonteCarlo:
    @staticmethod
    def run_simulation(weights: np.ndarray, 
                       mean_returns: pd.Series, 
                       cov_matrix: pd.DataFrame, 
                       portfolio_value: float, days: int, 
                       simulations: int) -> np.ndarray:
        """
        Perform Monte Carlo simulations to generate portfolio values over time. returns a np array with columns of each trial and rows of the value at each time
        """                                                 
        #create days x simulations matrix filled with mean returns
        meanM = np.full(shape=(days, len(weights)), fill_value=mean_returns).T  # mean returns matrix
        #create days
        portfolio_sims = np.full(shape=(days, simulations), fill_value=0.0)  # matrix to hold the results of each simulation

        for m in range(simulations):  # loop through each simulation
            Z = np.random.normal(size=(days, len(weights)))  # generate random normal values
            L = np.linalg.cholesky(cov_matrix)  # perform Cholesky decomposition on covariance matrix
            daily_returns = meanM + np.inner(L, Z)  # simulate daily returns
            portfolio_sims[:, m] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * portfolio_value  # cumulative product for portfolio value

        return portfolio_sims  # return the results of the simulations as a 

    @staticmethod
    def calculate_var(portfolio_sims: np.ndarray, confidence_interval: float, initial_portfolio: float) -> float:
        """
        Calculate the Value at Risk (VaR) from the Monte Carlo simulation results.
        """

        port_results = portfolio_sims[-1, :]  # take the last day of simulations
        VaR = initial_portfolio - np.percentile(port_results, (1 - confidence_interval) * 100)  # calculate VaR at the given confidence level
        return VaR  # return the Value at Risk
