
import backtrader as bt



class MovingAverageCrossover(bt.Strategy):
    params =(
        ('short_period',20),
        ('long_period',50),
        ('rsi_period',14),
        ('rsi_overbought',70),
        ('rsi_oversold',30),
    )

    def __init__(self):
        self.short_period = self.params.short_period
        self.long_period = self.params.long_period
        self.rsi_period = self.params.rsi_period
        self.rsi_overbought = self.params.rsi_overbought
        self.rsi_oversold = self.params.rsi_oversold
        self.short_sma = bt.indicators.MovingAverageSimple(self.data.close,period = self.short_period)
        self.long_sma = bt.indicators.MovingAverageSimple(self.data.close,period = self.long_period)
    
    def next(self):
        if self.short_sma > self.long_sma and not self.position:
            self.buy()
        elif self.short_sma < self.long_sma and not self.position:
            self.sell()

# momentum strategy

class MomentumStrategy(bt.Strategy):
    params = (
        ('momentum_period',100),
        ('roc_threshold', 0),
    )

    def __init__(self):
        self.roc = bt.indicators.RateOfChange(self.data.close,period = self.params.momentum_period)
    def next(self):
        if self.roc[0] > self.params.roc_threshold and not self.position:
            self.buy()
        elif self.roc[0] < self.params.roc_threshold and self.position:
            self.sell()



# Mean Reversion Strategy 
class MeanReversion(bt.Strategy):
    params = (
        ('bollinger_period', 20),
        ('bollinger_dev', 2),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70)
    )

    def __init__(self):
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.bollinger_period, devfactor=self.params.bollinger_dev)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

    def next(self):
        if self.data.close[0] < self.bollinger.bot[0] and self.rsi[0] < self.params.rsi_oversold:
            if not self.position:
                self.buy()
        elif self.data.close[0] > self.bollinger.top[0] and self.rsi[0] > self.params.rsi_overbought:
            if self.position:
                self.sell()


