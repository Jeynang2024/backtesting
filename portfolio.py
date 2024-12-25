import numpy as np
import pandas as pd
from datetime import datetime

def sharpe_ratio(portfolio_values, risk_free_rate=0.2):
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess_returns = returns - risk_free_rate
    
    avg_return = np.mean(excess_returns)
    std_dev = np.std(excess_returns)
    
    sharpe_ratio = avg_return / std_dev if std_dev != 0 else 0
    
    return sharpe_ratio


def calculate_max_drawdown(portfolio_values):
    portfolio_values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = drawdowns.min()
    return max_drawdown

def calculate_total_return(portfolio_values):
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    return total_return

def calculate_volatility(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    volatility = np.std(returns)
    return volatility

def calculate_cagr(portfolio_values, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    start_value = portfolio_values[0]
    end_value = portfolio_values[-1]
    print("start",start_value)
    print("end",end_value)
    years = (end_date - start_date).days / 365.25
    cagr = (end_value / start_value) ** (1 / years) - 1
    return cagr
