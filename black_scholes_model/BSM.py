import numpy as np
import yfinance as yf
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import mplfinance as mpf

# modified to handle vectorized inputs
class BlackScholesModel:
    def __init__(self, S, K, t, r, sigma, q=1): #implement dividends later
        self.S = np.array(S)          # underlying stock
        self.K = np.array(K)          # strike price
        self.t = np.array(t)          # time till expiry in days/365
        self.r = np.array(r)          # risk-free interest rate
        self.sigma = np.array(sigma)  # implied volatility (maybe can use other volatility as well?)
        self.q = np.array(q)          # annual dividends

    def d1(self):
        d1 = (np.log(self.S/self.K) + (self.r + 0.5*(self.sigma**2))*self.t) / (self.sigma * np.sqrt(self.t))
        return d1

    def d2(self):
        d2 = self.d1() - (self.sigma * np.sqrt(self.t))
        return d2

    def call_price(self):
        call_price = self.S*stats.norm.cdf(self.d1()) - (self.K*np.e**(-self.r*self.t))*stats.norm.cdf(self.d2())
        return call_price

    def put_price(self):
        put_price = (self.K*np.e**(-self.r*self.t))*stats.norm.cdf(-self.d2()) - self.S*stats.norm.cdf(-self.d1())
        return put_price

    ## addition of greeks
    # def delta_call(self):
# bsm = BlackScholesModel(S = 437, K =435, r=0.035, sigma = 0.19, t=0.0384)
# print(bsm.call_price())

## TODO
### compartmentalize code sections (create utils.py)
### this BSM should only have the bsm class i think
### separate Y values (call prices) from X df
### feature engineer, add greeks, logs of ratios, etc
### create a BSM_main.py that runs through all data parsed
### have a function that calculates error from real y values (bsm model vs asking option price)
### create graphs for err vs different parameters in seaborn, matplotlib, mplfinance








