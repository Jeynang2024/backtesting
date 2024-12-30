import streamlit as st
import backtrader as bt
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime 
from datetime import date
import pandas as pd
from portfolio import *
from strategy import MovingAverageCrossover, MomentumStrategy, MeanReversion
from real import *
load_dotenv()
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_API_SECRET")




@st.cache_data
def fetch_data(stock_symbol, start, end):
    return yf.download(stock_symbol, start=start, end=end)




def run_portfolio_backtest(strategy_class, stock_symbols, weights, start_date, end_date, initial_cash,strategy_params):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class,**strategy_params)

    # Normalize weights
    weights = [w / sum(weights) for w in weights]

    portfolio_values = []
    highest_prices = {}

    for stock_symbol, weight in zip(stock_symbols, weights):
        # Download data
        data = fetch_data(stock_symbol, start_date, end_date)
        highest_prices[stock_symbol] = data['High'].max()
        data.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in data.columns]
        data.to_csv(f'{stock_symbol}_data.csv')
        data_feed = bt.feeds.GenericCSVData(
            dataname=f'{stock_symbol}_data.csv',
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            dtformat=('%Y-%m-%d')
        )
        cerebro.adddata(data_feed, name=stock_symbol)
        allocated_value = initial_cash * weight
        size = allocated_value // data[f'Close_{stock_symbol}'].iloc[0]  # Use first close price to calculate initial position size
        cerebro.broker.set_cash(initial_cash)

        # Use `size` to allocate the amount to buy each stock
        cerebro.addsizer(bt.sizers.FixedSize, stake=size)
    spy_dataa = fetch_data('SPY', start_date, end_date)
    spy_dataa.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in spy_dataa.columns]
    spy_dataa.to_csv(f'spydata.csv')
    spy_data_feed = bt.feeds.GenericCSVData(
            dataname=f'spydata.csv',
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            dtformat=('%Y-%m-%d')
        )
    
    cerebro.adddata(spy_data_feed, name="SPY")
    # Set broker parameters
    cerebro.broker.set_cash(initial_cash)

    # Add portfolio observer
    class PortfolioObserver(bt.Observer):
        lines = ('value',)
        plotinfo = {"plot": False}

        def next(self):
            portfolio_values.append(self._owner.broker.get_value())

    cerebro.addobserver(PortfolioObserver)



    # Run backtest
    cerebro.run()

    # Calculate portfolio metrics
    final_value = cerebro.broker.get_value()
    total_profit = final_value - initial_cash
    max_drawdown = calculate_max_drawdown(portfolio_values)
    total_return = calculate_total_return(portfolio_values)
    volatility = calculate_volatility(portfolio_values)
    cagr = calculate_cagr(portfolio_values, start_date, end_date)

    spy_total_return = calculate_total_return(spy_dataa['Close_SPY'])
    spy_volatility = calculate_volatility(spy_dataa['Close_SPY'])
    spy_max_drawdown = calculate_max_drawdown(spy_dataa['Close_SPY'])



    return total_profit, max_drawdown, total_return, volatility, cagr, cerebro, highest_prices, spy_total_return, spy_volatility, spy_max_drawdown
def main():
    st.title("📊 Portfolio Backtesting Dashboard")
    # Multi-Asset Support
    stock_symbols = st.text_area("Enter Stock Symbols (comma-separated)", value="AAPL,MSFT,GOOG").split(",")
    weights = st.text_area("Enter Portfolio Weights (comma-separated)", value="0.5,0.3,0.2").split(",")
    weights = [float(w) for w in weights]

    start_date = st.date_input("Select Start Date", value=date(2010, 1, 1))
    end_date = st.date_input("Select End Date", value=date(2020, 1, 1))
    initial_cash = st.number_input("Initial Cash", min_value=1000, value=10000)

    strategy_choice = st.selectbox("Choose Strategy", ("Mean Reversion", "Moving Average Crossover", "Momentum Strategy"))

    
    


    if strategy_choice == "Mean Reversion":
        bollinger_period = st.number_input("Bollinger Period", value=20)
        bollinger_dev = st.number_input("Bollinger Dev", value=2)
        rsi_period = st.number_input("RSI Period", value=14)
        rsi_oversold = st.number_input("RSI Oversold", value=30)
        rsi_overbought = st.number_input("RSI Overbought", value=70)
        
        strategy_params = {
            'bollinger_period': bollinger_period,
            'bollinger_dev': bollinger_dev,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought
        }
        strategy_class = MeanReversion

    elif strategy_choice == "Moving Average Crossover":
        short_period = st.number_input("Short SMA Period", value=20)
        long_period = st.number_input("Long SMA Period", value=50)
        rsi_period = st.number_input("RSI Period", value=14)
        rsi_overbought = st.number_input("RSI Overbought", value=70)
        rsi_oversold = st.number_input("RSI Oversold", value=30)
        
        strategy_params = {
            'short_period': short_period,
            'long_period': long_period,
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold
        }
        strategy_class = MovingAverageCrossover

    elif strategy_choice == "Momentum Strategy":
        momentum_period = st.number_input("Momentum Period", value=100)
        roc_threshold = st.number_input("ROC Threshold", value=0)
        
        strategy_params = {
            'momentum_period': momentum_period,
            'roc_threshold': roc_threshold
        }
        strategy_class = MomentumStrategy

    with st.sidebar:
        st.write("Check with Real Time Data")
        user_stock = st.text_input("Enter the stock name")
        if st.button("Run"):
            st.write("Running portfolio backtest... Please wait.")

            total_real_profit = backtest_real_data(user_stock,strategy_choice,strategy_params,initial_cash)
            st.write("Total Profit:{total_real_profit}")








    if st.button("Run Backtest"):
        st.write("Running portfolio backtest... Please wait.")

        total_profit, max_drawdown, total_return, volatility, cagr, cerebro, highest_prices, spy_total_return, spy_volatility, spy_max_drawdown = run_portfolio_backtest(
            strategy_class, stock_symbols, weights, str(start_date), str(end_date), initial_cash,strategy_params
        )

        st.subheader("Portfolio Performance Metrics")
        st.write(f"Total Profit:{total_profit:.2f}")
        st.write(f"Maximum Drawdown: {max_drawdown:.2f}%")
        st.write(f"Total Return: {total_return:.2f}%")
        st.write(f"Portfolio Volatility: {volatility:.2f}%")
        st.write(f"Compound Annual Growth Rate (CAGR): {cagr:.2f}%")
        
        st.subheader("SPY Benchmark Metrics")
        st.write(f"SPY Total Return: {spy_total_return:.2f}%")
        st.write(f"SPY Volatility: {spy_volatility:.2f}%")
        st.write(f"SPY Maximum Drawdown: {spy_max_drawdown:.2f}%")

        


        st.subheader("Highest Stock Prices During Selected Period")
        highest_prices_df = pd.DataFrame(list(highest_prices.items()), columns=["Stock Symbol", "Highest Price"])
        st.bar_chart(highest_prices_df.set_index("Stock Symbol"))

        # Allow downloading results
        st.download_button(
            label="Download Results as CSV",
            data=pd.DataFrame({'Metric': ['Max Drawdown', 'Total Return', 'Volatility', 'CAGR'], 
                               'Value': [max_drawdown, total_return, volatility, cagr]}).to_csv(),
            file_name="backtest_results.csv",
            mime="text/csv"
        )
        cerebro.plot(style='candlestick', iplot=True)
if __name__ == "__main__":
    main()
