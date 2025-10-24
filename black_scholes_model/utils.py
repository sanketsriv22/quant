import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
import seaborn as sb
from matplotlib import pyplot as plt


def create_ticker(stock_symbol):
    ticker = yf.Ticker(stock_symbol)  # creates a Ticker class with stock symbol
    return ticker


def get_interest_rate():
    try:  # fetch data for 10yr treasury yield
        treasury_yield = yf.download("^TNX", period="1d", progress=False)

        # check to see if data exists, else raise flag
        if treasury_yield.empty:
            raise ValueError("No data available for 10yr Treasury Yield")
        return treasury_yield['Close'].iloc[0] / 100
    except Exception as e:
        print(f"Error fetching data: {e}")
        return 0.035  # return default interest rate


def get_time_to_expiry(ticker, calls, date, exp_date):
    # parse last trade dates
    trade_dates = calls['lastTradeDate'].apply(lambda x: x.to_pydatetime().replace(tzinfo=None))

    # calculate t in BSM
    time_to_expiry = trade_dates.apply(lambda x: (abs((x - exp_date).days)) / 365)

    # convert to string
    str_trade_dates = trade_dates.apply(lambda x: x.strftime('%Y-%m-%d'))  # for string lookup in underlying df

    return time_to_expiry, str_trade_dates


def gather_options_data(ticker):
    # features + last column y [stock price, strike price, expiry date, volatility, risk-free rate, dividends, call option price]
    X = pd.DataFrame(columns=['strike', 'ask', 'impliedVolatility'])
    options_dates = ticker.options

    # only checking prices in current year
    start_date = "2024-01-01"
    end_date = "2024-09-20"

    # Get historical data
    stock_data = ticker.history(start=start_date, end=end_date)['Close']

    for date in options_dates:
        # treasury yield interest rate
        interest_rate = get_interest_rate()

        # main call option data
        options_data = ticker.option_chain(date)
        calls = options_data.calls
        calls = calls[calls['inTheMoney'] == True] # deep OTM options are weird
        data = calls[['ask', 'strike', 'impliedVolatility']]

        # create string object of exp date
        expiration_date = datetime.strptime(date, '%Y-%m-%d')

        # grab t for BSM
        time_to_expiry, str_trade_dates = get_time_to_expiry(ticker, calls, date, expiration_date)

        # create df column of interest for concat
        interest = pd.DataFrame(np.ones(len(data)) * interest_rate, columns=['interest'])

        # parse closing stock prices
        stock_prices = str_trade_dates.apply(lambda x: stock_data[x] if x in stock_data else stock_data.iloc[-1])

        # Concatenate only if there's data ## ALSO CHECK FOR 0's LATER
        data = pd.concat([data, time_to_expiry.rename('time_to_expiry')], axis=1)
        data = pd.concat([data, interest], axis=1)
        data = pd.concat([data, stock_prices.rename('stock')], axis=1)

        # final merge with entire df
        X = pd.concat([X, data], ignore_index=True)

    return X


def features_and_label(combined_df):
    X = combined_df

    y = X['ask'].to_frame(name='option')

    X.drop('ask', axis=1, inplace=True)

    return X, y


def normalize_data(X, y):
    X['stock'] = X['stock'] / X['strike']
    y['option'] = y['option'] / X['strike']
    X['strike'] = 1
    return X, y


def error_calculation(y_true, y_hat):
    # simple RMSE
    mse = np.mean((y_true - y_hat)**2)
    rmse = np.sqrt(mse)
    return rmse


def df_to_np(X):
    S = X['stock'].to_numpy()
    K = X['strike'].to_numpy()
    r = X['interest'].to_numpy()
    t = X['time_to_expiry'].to_numpy()
    sigma = X['impliedVolatility'].to_numpy()

    return S, K, r, t, sigma


def plot_correlation_matrix(S, K, r, t, sigma):
    # Create a DataFrame for the BSM parameters
    bsm_params = pd.DataFrame({
        'Stock Price (S)': S,
        'Strike Price (K)': K,
        'Interest Rate (r)': r,
        'Time to Maturity (t)': t,
        'Volatility (σ)': sigma
    })

    # Compute the correlation matrix
    corr_matrix = bsm_params.corr()

    # Plot the correlation matrix using Seaborn
    plt.figure(figsize=(10, 10))
    sb.heatmap(corr_matrix, annot=True, cmap='magma', vmin=-1, vmax=1, annot_kws={'size': 10})

    # Rotate the x and y labels
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Tilt x-axis labels
    plt.yticks(rotation=45, fontsize=10)  # Tilt y-axis labels

    plt.title("Correlation Matrix of BSM Parameters")
    plt.show()


def line_plot_strike_price_vs_prices(options_data, y_true, y_pred):
    strike_prices = options_data['strike'].to_numpy()
    plt.figure(figsize=(10, 6))
    sb.lineplot(x=strike_prices, y=y_true, label='Actual Prices', color='blue')
    sb.lineplot(x=strike_prices, y=y_pred, label='Predicted Prices', color='red')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title('Actual vs Predicted Option Prices over Strike Price')
    plt.legend()
    plt.show()

def error_distribution_plot(y_true, y_pred):
    errors = y_true - y_pred
    sb.histplot(errors, bins=30, kde=True)
    plt.xlabel('Prediction Error')
    plt.title('Distribution of Prediction Errors')
    plt.show()

def residual_plot(y_true, y_pred):
    residuals = y_true - y_pred
    sb.residplot(x=y_pred, y=residuals, lowess=True, color="blue")
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Prices')
    plt.show()

def scatter_plot_actual_vs_predicted(y_true, y_pred):
    sb.scatterplot(x=y_true, y=y_pred)
    plt.xlabel('Actual Option Prices')
    plt.ylabel('Predicted Option Prices')
    plt.title('Actual vs Predicted Option Prices')
    plt.show()


