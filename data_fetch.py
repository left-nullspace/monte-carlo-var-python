import pandas as pd
import numpy as np
import yfinance as yf

def get_data(stocks: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Fetch historical data and calculate daily returns, mean returns, and covariance matrix.
    """
    
    stock_data = yf.download(stocks, start=start, end=end)['Adj Close']  # download adjusted close prices
    returns = stock_data.pct_change().dropna()  # calculate daily returns
    mean_returns = returns.mean()  # calculate mean returns
    cov_matrix = returns.cov()  # calculate covariance matrix
    return returns, mean_returns, cov_matrix  # return the results
