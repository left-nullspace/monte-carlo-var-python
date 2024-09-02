import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st
from data_fetch import get_data
from monte_carlo import MonteCarlo

#THIS FILE IS STRICTLY FOR STREAMLIT DISPLAY. OTHERWISE USE MAIN.PY
def main():
    # PAGE SETTINGS
    st.set_page_config(
        page_title="Value at Risk Calculator", 
        page_icon="📈", 
        layout='wide',  # Set the layout to wide
        initial_sidebar_state="expanded"
    )
    
    st.title("Monte Carlo VaR Simulator")
    st.write("##### Take advantage of math to understand the risks of your investment portfolio")
    st.write("Click **Run Simulation** to begin.")
    
    # SIDEBAR SETTINGS
    st.sidebar.title("⚙️ Settings")

    # Expander for Data Settings
    with st.sidebar.expander("📊 Data Settings", expanded=True):
        # Input: Stock tickers
        stock_list = st.text_input(
            "Stock Tickers", "SPY, QQQ, SMH, GLD, TLT",
            help="Enter stock tickers separated by commas (e.g., SPY, QQQ, SMH, GLD, TLT)"
        ).split(", ")

        # Input: Weights
        weights_input = st.text_input(
            "Portfolio Weights", "0.2, 0.2, 0.2, 0.2, 0.2",
            help="Enter portfolio weights separated by commas (must sum to 1). These are the weights of each of the tickers in your investment portfolio."
        )
        weights = np.array([float(w) for w in weights_input.split(",")])

        # Input: Number of years for historical data
        years_of_data = st.number_input(
            "Years of Historical Data", min_value=1, max_value=30, value=10,
            help="Select number of years to include in historical data (most recent n years)"
        )
        end_date = dt.datetime.now().strftime('%Y-%m-%d')
        start_date = (dt.datetime.now() - dt.timedelta(days=years_of_data * 365)).strftime('%Y-%m-%d')

        # Input: Portfolio initial value
        portfolio_value = st.number_input(
            "Initial Portfolio Value ($):", value=10000, help="Account balance"
        )

    # Expander for Simulation Settings
    with st.sidebar.expander("🎲 Simulation Settings", expanded=True):
        # Create two columns in the sidebar
        col1, col2 = st.columns(2)
        
        # Input: Days to Simulate in the first column
        with col1:
            days = st.number_input(
                "Days to Simulate", min_value=2, max_value=1260, value=252, step=1,
                help="Enter the number of days to simulate (from 2 to 1260). Reminder that 252 trading days = 1 year"
            )

        # Input: Simulations to Run in the second column
        with col2:
            simulations = st.number_input(
                "Simulations to Run", min_value=100, max_value=10000, value=1000, step=100,
                help="Enter the number of simulations to run (from 100 to 10000)"
            )

        # Input: Confidence Interval
        confidence_interval = st.number_input(
            "Confidence Interval", value=0.95, min_value=0.9, max_value=0.99,
            help="Select confidence interval for Value at Risk (from 0.9 to 0.99)"
        )
        
    # Calculate Risk Level (1 - Confidence Interval)
    risk_level = 1 - confidence_interval

    # RUN SIMULATION
    if st.sidebar.button("Run Simulation"):
        # Fetch data
        returns, mean_returns, cov_matrix = get_data(stock_list, start=start_date, end=end_date)
        
        # Run Monte Carlo simulation
        portfolio_sims = MonteCarlo.run_simulation(weights, mean_returns, cov_matrix, portfolio_value, days, simulations)

        # Calculate VaR
        VaR = MonteCarlo.calculate_var(portfolio_sims, confidence_interval, portfolio_value)

        # Create a dataframe to store simulation and data settings
        settings_data = {
            "Parameter": [
                "Stock Tickers",
                "Portfolio Weights",
                "Years of Data",
                "Start Date",
                "End Date (Date ran)",
                "Initial Portfolio Value ($)",
                "Days to Simulate",
                "Number of Simulations",
                "Confidence Interval",
                "Risk Level",
                "Value at Risk (VaR)"
            ],
            "Value": [
                ", ".join(stock_list),
                ", ".join(map(str, weights)),
                years_of_data,
                start_date,
                end_date,
                portfolio_value,
                days,
                simulations,
                f"{confidence_interval * 100}%",
                f"{risk_level * 100:.0f}%",
                f"${VaR:.2f}"
            ]
        }
        settings_df = pd.DataFrame(settings_data)
        settings_df = settings_df.set_index('Parameter')

        # Create two columns for plots
        plot_col1, plot_col2 = st.columns(2)

        # Plot simulation results in the first column
        with plot_col1:
            st.write("### Simulation Results")
            fig, ax = plt.subplots(figsize=(6, 4))
            for i in range(simulations):
                ax.plot(portfolio_sims[:, i], color=np.random.rand(3,), alpha=0.3, linewidth=0.7)

            mean_simulation = np.mean(portfolio_sims, axis=1)
            ax.plot(mean_simulation, color='red', linewidth=2, label='Mean Simulation')
            ax.axhline(y=(portfolio_value - VaR), color='green', linestyle='-', linewidth=1, label=f'VaR ({confidence_interval*100}%): ${VaR:.2f}')
            
            ax.set_title(f'Simulation of Portfolio Value over {days} days ({simulations} trials)')
            ax.set_xlabel('Days')
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend(loc='upper left')
            st.pyplot(fig)

        # Plot histogram of final portfolio values in the second column
        with plot_col2:
            final_values = portfolio_sims[-1, :]  # Final portfolio values at the last day of all simulations
            st.write("### Distribution of Final Portfolio Values")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.hist(final_values, bins=30, edgecolor='k', alpha=0.7)
            ax2.set_xlabel('Final Portfolio Value ($)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Final Portfolio Values')
            st.pyplot(fig2)

        # Display the settings dataframe
        st.write("#### Simulation and Data Settings")
        st.write("Hover over this to download the settings for future reference")
        st.dataframe(settings_df)

        # Display VaR results
        st.write(f"#### Results Interpretation")
        st.write(f"There is a {(1-confidence_interval)*100:.0f}% chance that your portfolio value will fall below ${portfolio_value - VaR:.2f} in {days} days, based on a {100 - confidence_interval*100}% risk level ({confidence_interval*100}% confidence level).")

        # Additional explanation for VaR
        st.write(f"In other words, with {confidence_interval*100:.0f}% confidence, the potential loss for your portfolio over the next {days} days will not exceed ${VaR:.2f}. \n")

        st.write(" \n \n Tool created by Alan")
        
if __name__ == "__main__":
    main()
