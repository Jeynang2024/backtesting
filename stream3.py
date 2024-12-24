import streamlit as st
import backtrader as bt
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd

from stream import MovingAverageCrossover ,MomentumStrategy,MeanReversion

def run_portfolio_backtest(strategy_class, stock_symbols, weights, start_date, end_date, initial_cash):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)

    
    weights = [w / sum(weights) for w in weights]

    for stock_symbol, weight in zip(stock_symbols, weights):
        data = yf.download(stock_symbol, start=start_date, end=end_date)
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
    
    # Set broker parameters
    cerebro.broker.set_cash(initial_cash)

    # Add portfolio observer
    portfolio_values = []
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
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    avg_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0

    return final_value, total_profit, sharpe_ratio, cerebro


"""def run_backtest(strategy_class, stock_symbol, start_date, end_date, initial_cash):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)

    
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in data.columns]
    data.rename(columns={
    'Adj Close_AAPL': 'Adj Close',
    'Open_AAPL': 'Open',
    'High_AAPL': 'High',
    'Low_AAPL': 'Low',
    'Close_AAPL': 'Close',
    'Volume_AAPL': 'Volume'
    }, inplace=True)

    
    data.to_csv('historical_data.csv')
    data = bt.feeds.YahooFinanceData(dataname='historical_data.csv')
    data_feed = bt.feeds.GenericCSVData(
    dataname='historical_data.csv',
    datetime=0,  
    open=1,      
    high=2,      
    low=3,       
    close=4,     
    volume=5,    
    openinterest=-1,  
    dtformat=('%Y-%m-%d')  
    )



    
    
    cerebro.adddata(data_feed)

    
    cerebro.broker.set_cash(initial_cash)
    

    
    
    portfolio_values = []  

    
    class PortfolioObserver(bt.Observer):
        lines = ('value',)  
        plotinfo = {"plot": False}  

        def __init__(self):
            self.addminperiod(1)

        def next(self):
            portfolio_values.append(self._owner.broker.get_value())  
    cerebro.addobserver(PortfolioObserver)

    cerebro.run()

    final_value = cerebro.broker.get_value()
    total_profit = final_value - initial_cash

    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    avg_return = np.mean(returns)
    std_dev = np.std(returns)
    sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0

    return final_value, total_profit, sharpe_ratio, cerebro
"""
def main():
    st.title("Portfolio Backtesting with Streamlit")

    # Multi-Asset Support
    stock_symbols = st.text_area("Enter Stock Symbols (comma-separated)", value="AAPL,MSFT,GOOG").split(",")
    weights = st.text_area("Enter Portfolio Weights (comma-separated)", value="0.5,0.3,0.2").split(",")
    weights = [float(w) for w in weights]

    start_date = st.date_input("Select Start Date", value=datetime.date(2010, 1, 1))
    end_date = st.date_input("Select End Date", value=datetime.date(2020, 1, 1))
    initial_cash = st.number_input("Initial Cash", min_value=1000, value=10000)

    strategy_choice = st.selectbox("Choose Strategy", ("Mean Reversion", "Moving Average Crossover", "Momentum Strategy"))

    if strategy_choice == "Mean Reversion":
        strategy_class = MeanReversion
    elif strategy_choice == "Moving Average Crossover":
        strategy_class = MovingAverageCrossover
    elif strategy_choice == "Momentum Strategy":
        strategy_class = MomentumStrategy

    if st.button("Run Backtest"):
        st.write("Running portfolio backtest... Please wait.")

        final_value, total_profit, sharpe_ratio, cerebro = run_portfolio_backtest(
            strategy_class, stock_symbols, weights, str(start_date), str(end_date), initial_cash
        )

        st.subheader("Portfolio Performance Metrics")
        st.write(f"Final Portfolio Value: ${final_value:.2f}")
        st.write(f"Total Profit/Loss: ${total_profit:.2f}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        st.subheader("Portfolio Equity Curve")
        cerebro.plot(iplot=False)

if __name__ == "__main__":
    main()
