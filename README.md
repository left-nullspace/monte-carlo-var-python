# Value at Risk (VaR) Calculator

## Overview
This application uses Monte Carlo simulations to estimate the Value at Risk (VaR) for a portfolio of stocks. The VaR is a statistical technique used to measure the risk of loss on a specific portfolio. The app allows users to input stock tickers, define portfolio weights, and adjust various simulation settings to calculate potential risks over a selected period.

## Features
- Fetch historical data for a list of stock tickers.
- Simulate portfolio values over time using Monte Carlo methods.
- Calculate Value at Risk (VaR) at a specified confidence interval.
- Visualize simulation results with interactive plots.

## Requirements
- Python 3.8 or higher (I used 3.9.17)
- The required Python packages listed in `requirements.txt`.

## Usage
Input the stock tickers, portfolio weights, and simulation parameters in the Streamlit sidebar, then click **Run Simulation** to visualize the risk analysis of your portfolio.

