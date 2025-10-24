from utils import *
from BSM import BlackScholesModel


def main():
    ticker = create_ticker('msft')  # this can be changed to user input later via GUI
    options_data = gather_options_data(ticker)
    X, y_true = features_and_label(options_data)  # DataFrames X, y_true

    # Convert DataFrame to numpy arrays
    S, K, r, t, sigma = df_to_np(X)
    y_t = y_true['option'].to_numpy()

    bsm = BlackScholesModel(S, K, r, t, sigma)
    y_hat = bsm.call_price()

    print(error_calculation(y_t, y_hat))

    # Visualize results
    scatter_plot_actual_vs_predicted(y_t, y_hat)
    residual_plot(y_t, y_hat)
    error_distribution_plot(y_t, y_hat)
    line_plot_strike_price_vs_prices(options_data, y_t, y_hat)
    plot_correlation_matrix(S, K, r, t, sigma)


if __name__ == "__main__":
    main()