import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import warnings
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
import seaborn as sns
import sys
import openpyxl
import scipy.stats as stats
from scipy.stats import anderson
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
import statistics
from scipy.optimize import minimize

#   To do:
#   Daten in Excel speichern. Datenlänge und möglicherweise test train size anpassen


# Reversing the data

#   data = pd.read_excel(r"C:\Users\Paul\Uni\Daten Paper Graf_Tickers.xlsx", engine='openpyxl') # Reads the data

#   data_reversed = data.iloc[::-1].reset_index(drop=True) # Reverses the data

#   data_reversed.to_excel(r"C:\Users\Paul\Uni\Daten Paper Graf_reversed.xlsx", index=False, engine='openpyxl')


#   Filling empty Values

#   data = pd.read_excel(r"C:\Users\Paul\Uni\Daten Paper Graf_reversed.xlsx", engine='openpyxl')

#   data.ffill(inplace=True) # has to be to the same data, as otherwise we generate a None

#   data.to_excel(r"C:\Users\Paul\Uni\Daten Paper Graf_reversed_filled.xlsx", index=False, engine='openpyxl')


#   Loading the data

data = pd.read_excel(r"C:\Users\Paul\Uni\Daten Paper Graf_reversed_filled.xlsx", engine='openpyxl')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.set_index("Date")  # Reassigns the index and drops the Date Column, so that pct_change and so on can work
data = data.astype(float)


#   Get the tickers

tickers = data.columns.tolist()

#   Calculating simple returns

simple_returns = data.pct_change()
simple_returns.fillna(0, inplace=True)
simple_returns = simple_returns


#   Calculating log returns

log_returns = np.log(data / data.shift(1))
log_returns.fillna(0, inplace=True)
log_returns = log_returns


#   Calculating Volatility

hourly_volatility = log_returns.pow(2)


#   Creating 60/40 train test splits

split_point = int(len(data) * 0.6)
train = data.iloc[:split_point]
test = data.iloc[split_point:]

train_log_returns = log_returns.iloc[:split_point]
test_log_returns = log_returns.iloc[split_point:]

train_simple_returns = simple_returns.iloc[:split_point]
test_simple_returns = simple_returns.iloc[split_point:]

train_hourly_volatility = hourly_volatility.iloc[:split_point]
test_hourly_volatility = hourly_volatility.iloc[split_point:]


#   Plotting Stylized Facts

stock = "AAPL"   # Stock of interest


#   Histogram and KDE plot

plt.figure(figsize=(10, 6))
log_returns_plot = log_returns*100   # Has to be adjusted, as this fixes the scaling issue
sns.histplot(log_returns_plot[stock], kde=True, color='skyblue', stat='density', bins=250)
mu, std = 0, 1
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=0.5)
plt.title(f"Distribution of Log Returns with Normal Distribution for {stock}")
plt.xlabel('Log Returns (%)')
plt.ylabel('Density')
plt.grid(False)
plt.show()


#   Q-Q plot comparing the log returns with a normal distribution

plt.figure(figsize=(10, 6))
stats.probplot(log_returns_plot[stock], dist="norm", plot=plt)
plt.title(f"Q-Q Plot against Normal Distribution for {stock}")
plt.ylabel('Sample Quantiles')
plt.xlabel('Theoretical Quantiles')
plt.grid(False)
plt.show()


#   Autocorrelation plot of log returns

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
plot_acf(log_returns[stock].dropna(), lags=50, fft=True, ax=ax)
ax.set_title(f"Autocorrelation of Log Returns for {stock}")
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_ylim(bottom=-0.5, top=1.3)
plt.grid(False)
plt.show()


#   Autocorrelation plot of absolute log returns

absolute_returns = log_returns[stock].abs()
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
plot_acf(absolute_returns.dropna(), lags=50, fft=True, ax=ax)
ax.set_title(f"Autocorrelation of Absolute Log Returns for {stock}")
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_ylim(bottom=-0.5, top=1.3)
plt.grid(False)
plt.show()


#   Autocorrelation of prices

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
plot_acf(data[stock].dropna(), lags=50, fft=True, ax=ax)
ax.set_title(f"Autocorrelation of Stock Prices for {stock}")
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_ylim(bottom=-0.5, top=1.3)
plt.grid(False)
plt.show()


#   Volatility Clustering
plt.figure(figsize=(10, 6))
log_returns_plot[stock].plot()
plt.title(f"Log Returns Over Time for {stock}")
plt.ylabel('Log Returns (%)')
plt.xlabel('Date')
plt.grid(False)
plt.show()


#   Correlation
correlation = log_returns.corr()
print(correlation)
plt.figure(figsize=(14, 12))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title(f'Correlation Heatmap')
plt.show()



#   Non Normality Table and Gain/Loss Assymetry

non_normality_stats = pd.DataFrame(index=log_returns.columns, columns=['ticker', 'Mean', 'Standard Deviation', 'Skewness',
                                                                       'Kurtosis', 'Jarque-Bera test',
                                                                       'Jarque-Bera p-Value',
                                                                       'p-Value Stationarity log',
                                                                       'p-Value Stationarity Volatility'])

#for stock in log_returns.columns:
#    non_normality_stats.loc[stock, 'ticker'] = stock

#    mu = log_returns[stock].mean()
#    non_normality_stats.loc[stock, 'Mean'] = mu

#    std = log_returns[stock].std()
#    non_normality_stats.loc[stock, 'Standard Deviation'] = std

#    skew = log_returns[stock].skew()
#    non_normality_stats.loc[stock, 'Skewness'] = skew

#    kurtosis = log_returns[stock].kurtosis()
#    non_normality_stats.loc[stock, 'Kurtosis'] = kurtosis

#    jb_test_stat, jb_p_value = stats.jarque_bera(log_returns[stock])
#    non_normality_stats.loc[stock, 'Jarque-Bera test'] = jb_test_stat
#    non_normality_stats.loc[stock, 'Jarque-Bera p-Value'] = jb_p_value

#    adf_test_stat, adf_p_value, usedlag_, nobs_, critical_values_, icbest_ = sm.tsa.adfuller(log_returns[stock])
#    non_normality_stats.loc[stock, 'p-Value Stationarity log'] = adf_p_value

#    adf_test_stat, adf_p_value, usedlag_, nobs_, critical_values_, icbest_ = sm.tsa.adfuller(hourly_volatility[stock])
#    non_normality_stats.loc[stock, 'p-Value Stationarity Volatility'] = adf_p_value

#non_normality_stats.to_excel(r"C:\Users\Paul\Programming\Python\PyCharm Community Edition 2023.2.1\PycharmProjects\Graf Paper\Daten Tabellen.xlsx", index=False, engine='openpyxl')

#   Other Stylized Facts

summary_stats = pd.DataFrame(index=log_returns.columns, columns=['Volatility Clustering', 'Leverage Effect'])

for stock in log_returns.columns:
    # Volatility Clustering (autocorrelation of squared returns as a proxy)
    volatility_clustering = log_returns[stock].dropna().pow(2).autocorr()
    summary_stats.loc[stock, 'Volatility Clustering'] = volatility_clustering

    # Leverage Effect (correlation between returns and volatility)
    future_volatility = log_returns[stock].rolling(window=7).std().shift(-7)
    leverage_effect = log_returns[stock].corr(future_volatility)
    summary_stats.loc[stock, 'Leverage Effect'] = leverage_effect

summary_stats = summary_stats.apply(pd.to_numeric)
print(summary_stats)


#   Portfolios and forecasting

#   These Values are probably gonna be the same starting point for every strategy
total_investment = 100
n_stocks = len(tickers)
investment_per_stock = total_investment / n_stocks

initial_prices = test.iloc[0]   # data before, now test as we test all strategies
weights_static = ((investment_per_stock / initial_prices)
                  / np.sum(investment_per_stock / initial_prices))


#   Creating the basic equal weighted portfolio and backtesting it

portfolio_simple_returns = (test_simple_returns * weights_static).sum(axis=1)

portfolio_log_returns = np.log1p(portfolio_simple_returns)

cumulative_log_returns = portfolio_log_returns.cumsum()

portfolio_value_over_time = total_investment * np.exp(cumulative_log_returns)

plt.figure(figsize=(14, 7))
plt.plot(test_simple_returns.index, portfolio_value_over_time, label='Equal weighted Portfolio', color='blue')
plt.title('Equal weighted Portfolio')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.legend()
plt.plot()
plt.show()


#   ERC optimization function

def calculate_erc_weights_scaled(covariance_matrix, scale_factor=1e4):
    """
    Calculate ERC weights with a scaled covariance matrix to handle very small values.

    Args:
    covariance_matrix (np.ndarray): The covariance matrix of asset returns.
    scale_factor (float): Factor by which to scale the covariance matrix to improve optimization stability.

    Returns:
    np.ndarray: The optimized asset weights.
    """
    n_assets = len(covariance_matrix)

    if covariance_matrix.isna().any().any():
        print("NaN values detected in the correlation matrix. Replacing with pseudo-correlation matrix.")
        pseudo_correlation_matrix = np.full((n_assets, n_assets), 0.5)
        np.fill_diagonal(pseudo_correlation_matrix, 1.0)

        scaled_covariance_matrix = pseudo_correlation_matrix * scale_factor
    else:
        scaled_covariance_matrix = covariance_matrix * scale_factor
    # Calculate portfolio risk
    def portfolio_risk(weights):
        return np.sqrt(np.dot(weights.T, np.dot(scaled_covariance_matrix, weights)))

    # Calculate the marginal risk contributions of each asset
    def risk_contributions(weights):
        portfolio_volatility_func = portfolio_risk(weights)
        marginal_contributions = np.dot(scaled_covariance_matrix, weights)
        risk_contributions = np.divide(np.multiply(weights, marginal_contributions), portfolio_volatility_func)
        return risk_contributions

    # Objective function: minimize the sum of squared differences between risk contributions
    def objective(weights):
        contributions = risk_contributions(weights)
        mean_contribution = np.mean(contributions)
        dispersion = np.sum((contributions - mean_contribution) ** 2)
        return dispersion

    # Constraints: sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    # Bounds for weights (no short selling, no leverage)
    bounds = tuple((0.001, 1) for asset in range(n_assets))

    # Initial guess (equal weights)
    initial_weights = np.ones(n_assets) / n_assets

    # Optimization
    opt_result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'disp': True, 'maxiter': 5000})

    try:
        opt_result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': True, 'maxiter': 5000})
        if not opt_result.success:
            print(f"Optimization warning: {opt_result.message}")
            return None
        return opt_result.x
    except Exception as e:
        print(f"Optimization failed due to an exception: {e}")
        return None


#   ERC portfolio based off of the last 7 quotes

weights_hist_inverse = pd.DataFrame(index=test_simple_returns.index, columns=tickers)
weights_hist_inverse.iloc[0] = weights_static

portfolio_log_returns_hist_inverse = pd.Series(index=test_simple_returns.index, dtype=float)

rebalance_period = 7  # same variable is being used everywhere in the code

for hour in range(0, len(test_simple_returns)):
    if hour % rebalance_period == 0 and hour >= 7:

        lookback = min(hour, rebalance_period)
        recent_returns = test_log_returns.iloc[hour - lookback:hour]
        covariance_matrix = recent_returns.cov()

        new_weights = calculate_erc_weights_scaled(covariance_matrix)

        weights_hist_inverse.iloc[hour] = new_weights

    else:
        # On non-rebalancing periods, carry forward the last set of weights
        weights_hist_inverse.iloc[hour] = weights_hist_inverse.iloc[hour - 1]

    # Calculate portfolio return for the current hour using new weights
    current_simple_returns = test_simple_returns.iloc[hour]
    weighted_simple_returns = current_simple_returns.multiply(weights_hist_inverse.iloc[hour])
    portfolio_simple_return = weighted_simple_returns.sum()

    # Convert to log return and store
    portfolio_log_returns_hist_inverse.iloc[hour] = np.log1p(portfolio_simple_return)
    print(f"step {hour + 1} of {len(test_simple_returns)}")
# Cumulate log returns to track portfolio performance over time
cumulative_log_returns = portfolio_log_returns_hist_inverse.cumsum()

# Calculate the total portfolio value over time based on the initial investment
portfolio_value_over_time_hist_inverse = total_investment * np.exp(cumulative_log_returns)

plt.figure(figsize=(14, 7))
plt.plot(test_simple_returns.index, portfolio_value_over_time_hist_inverse,
         label='Historical based ERC Portfolio', color='blue')
plt.title('Historical based ERC Portfolio')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.legend()
plt.plot()
plt.show()


#   VAR Model fitting

model = VAR(train_log_returns)
bic_values = {}
for p in range(7, 25):
    results = model.fit(p)
    bic_values[p] = results.bic

optimal_lag = min(bic_values, key=bic_values.get)
print(optimal_lag)
minimum_hour_VAR = optimal_lag*2

weights_VAR_inverse = pd.DataFrame(index=test_simple_returns.index, columns=test_simple_returns.columns)
portfolio_log_returns_VAR_inverse = pd.Series(index=test_simple_returns.index)


#   VAR Model backtest

for hour in range(0, len(test_log_returns)):
    if hour % rebalance_period == 0 and hour >= minimum_hour_VAR:
        lookback_VAR = min(hour, 750)
        # Calculate the correlation matrix from the last 7 quotes
        lookback = min(hour, rebalance_period)
        recent_returns = test_log_returns.iloc[hour - lookback:hour-1]
        correlation_matrix = recent_returns.corr()

        # Prepare data for VAR model to forecast volatility
        data_for_VAR = test_log_returns.iloc[:hour-1]
        model = VAR(data_for_VAR)
        results = model.fit(optimal_lag)

        # Forecast the next 7 periods
        forecast = results.forecast(data_for_VAR.values[-optimal_lag:], 7)
        forecasted_volatility = {}
        for i, ticker in enumerate(tickers):
            period_volatilities = [np.std(forecast[:, i]) for period in range(forecast.shape[0])]
            forecasted_volatility[ticker] = np.mean(period_volatilities)

        forecasted_volatility = np.array([forecasted_volatility[key] for key in sorted(forecasted_volatility.keys())])

        # Calculate covariance matrix using forecasted volatility and historical correlation
        forecasted_covariance_matrix = np.outer(forecasted_volatility, forecasted_volatility) * correlation_matrix

        # Calculate ERC weights
        optimized_weights = calculate_erc_weights_scaled(forecasted_covariance_matrix)

        # Save new weights
        weights_VAR_inverse.iloc[hour] = optimized_weights
    elif hour == 0:
        weights_VAR_inverse.iloc[hour] = weights_static
    else:
        # On non-rebalancing periods, carry forward the last set of weights
        weights_VAR_inverse.iloc[hour] = weights_VAR_inverse.iloc[hour - 1]

    # Calculate portfolio return for the current hour using new weights
    current_simple_returns = test_simple_returns.iloc[hour]
    current_weights = weights_VAR_inverse.iloc[hour]
    weighted_simple_returns = current_simple_returns.multiply(current_weights)
    portfolio_simple_return = weighted_simple_returns.sum()

    # Convert to log return and store
    portfolio_log_returns_VAR_inverse.iloc[hour] = np.log1p(portfolio_simple_return)
    print(f"step {hour + 1} of {len(test_simple_returns)}")

# Cumulate log returns to track portfolio performance over time
cumulative_log_returns_VAR_inverse = portfolio_log_returns_VAR_inverse.cumsum()

# Calculate the total portfolio value over time based on the initial investment
portfolio_value_over_time_VAR_inverse = total_investment * np.exp(cumulative_log_returns_VAR_inverse)

plt.figure(figsize=(14, 7))
plt.plot(test_simple_returns.index, portfolio_value_over_time_VAR_inverse,
         label='VAR forecast based ERC Portfolio', color='blue')
plt.title('VAR forecast based ERC Portfolio')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.legend()
plt.plot()
plt.show()


#   Garch Model
#   GARCH Models have problems with values near 0. One can try to scale all of them by 100
warnings.filterwarnings("ignore")


def optimize_garch(returns):
    best_bic = np.inf
    best_p_bic = 0
    best_q_bic = 0
    best_model_bic = None

    for p in range(7, 15):  # Minimum 7, due to noticed saisonality in abs log returns
        for q in range(7, 15):  #
            try:
                model = arch_model(returns, vol='Garch', p=p, q=q)
                model_fit = model.fit(disp='off')

                # Check for the best model based on BIC
                if model_fit.bic < best_bic:
                    best_bic = model_fit.bic
                    best_p_bic = p
                    best_q_bic = q
                    best_model_bic = model_fit

            except Exception as e:
                print(f"An error occurred: {e}")
                continue

    print(f'Best BIC: {best_bic}, with p={best_p_bic} and q={best_q_bic}')
    return best_model_bic, best_p_bic, best_q_bic


def forecast_volatility(returns, p, q):
    model = arch_model(returns, vol="Garch", p=p, q=q)
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=7)
    forecasted_volatility = forecast.variance.iloc[-1] ** 0.5
    return forecasted_volatility.to_numpy()

#p_values_GARCH = {}
#q_values_GARCH = {}
#for ticker in tickers:
#    best_model_bic, p, q = optimize_garch(train_log_returns[ticker])
#    p_values_GARCH[ticker] = p
#    q_values_GARCH[ticker] = q

#p_value_GARCH = p_values_GARCH.values()
#optimal_p_value_GARCH = round(statistics.mean(p_value_GARCH))

#q_value_GARCH = q_values_GARCH.values()
#optimal_q_value_GARCH = round(statistics.mean(q_value_GARCH))

#print(optimal_p_value_GARCH, optimal_q_value_GARCH)
#minimum_hour_GARCH = max(optimal_p_value_GARCH, optimal_q_value_GARCH)*2


weights_GARCH_inverse = pd.DataFrame(index=test_simple_returns.index, columns=tickers)

portfolio_log_returns_GARCH_inverse = pd.Series(index=test_simple_returns.index, dtype=float)

for hour in range(0, len(test_simple_returns)):
    if hour % rebalance_period == 0 and hour >= 15:
        lookback_GARCH = min(hour, 750)
        forecasted_volatility = {}
        for ticker in tickers:
            returns_up_to_now = test_log_returns[ticker].iloc[hour - lookback_GARCH:hour-1]
            forecasted_volatility[ticker] = forecast_volatility(returns_up_to_now, 7, 7)

        # Calculate the correlation matrix from the last 7 quotes
        lookback = min(hour, rebalance_period)
        recent_returns = test_log_returns.iloc[hour - lookback:hour-1]
        correlation_matrix = recent_returns.corr()

        forecasted_volatility = np.array(list(forecasted_volatility.values()))
        forecasted_volatility = forecasted_volatility.mean(axis=1)
        # Construct the forecasted covariance matrix
        forecasted_covariance_matrix = np.outer(forecasted_volatility, forecasted_volatility) * correlation_matrix

        # Calculate ERC weights
        optimized_weights = calculate_erc_weights_scaled(forecasted_covariance_matrix)
        if optimized_weights is None:
            weights_GARCH_inverse.iloc[hour] = weights_GARCH_inverse.iloc[hour - 1]
        else:
            weights_GARCH_inverse.iloc[hour] = optimized_weights
    elif hour == 0:
        weights_GARCH_inverse.iloc[hour] = weights_static
    else:
        # On non-rebalancing periods, carry forward the last set of weights
        weights_GARCH_inverse.iloc[hour] = weights_GARCH_inverse.iloc[hour - 1]

    # Calculate portfolio return for the current hour using new weights
    current_simple_returns_GARCH = test_simple_returns.iloc[hour]
    current_weights_GARCH = weights_GARCH_inverse.iloc[hour]
    weighted_simple_returns_GARCH = current_simple_returns_GARCH.multiply(current_weights_GARCH)
    portfolio_simple_return_GARCH = weighted_simple_returns_GARCH.sum()

    # Convert to log return and store
    portfolio_log_returns_GARCH_inverse.iloc[hour] = np.log1p(portfolio_simple_return_GARCH)
    print(f"step {hour + 1} of {len(test_simple_returns)}")

# Cumulate log returns to track portfolio performance over time
cumulative_log_returns_GARCH_inverse = portfolio_log_returns_GARCH_inverse.cumsum()

# Calculate the total portfolio value over time based on the initial investment
portfolio_value_over_time_GARCH_inverse = total_investment * np.exp(cumulative_log_returns_GARCH_inverse)

plt.figure(figsize=(14, 7))
plt.plot(test_simple_returns.index, portfolio_value_over_time_GARCH_inverse,
         label='GARCH based inverse Volatility Portfolio', color='blue')
plt.title('GARCH based ERC Portfolio')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.legend()
plt.plot()
plt.show()


#   Comparing the strategies

plt.figure(figsize=(14, 10))
plt.plot(portfolio_value_over_time, label='Equal weighted Portfolio')
plt.plot(portfolio_value_over_time_hist_inverse, label='Historical based Inverse Volatility Portfolio')
plt.plot(portfolio_value_over_time_VAR_inverse, label='VAR forecast based inverse Volatility Portfolio')
plt.plot(portfolio_value_over_time_GARCH_inverse, label='GARCH based inverse Volatility Portfolio')

plt.title('Cumulative Returns Over Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()


#   Comparing the weights
#   Have to change it. Legend looks bad and we have bars instead of lines

fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

weights_hist_inverse.plot(ax=axs[0])
axs[0].set_title('Historical based Inverse Volatility Portfolio')
axs[0].set_ylabel('Weights')
axs[0].legend(loc='upper right')
axs[0].grid(True)

weights_VAR_inverse.plot(ax=axs[1])
axs[1].set_title('VAR forecast based inverse Volatility Portfolio')
axs[1].set_ylabel('Weights')
#axs[1].legend(loc='upper right')
axs[1].grid(True)

weights_GARCH_inverse.plot(ax=axs[2])
axs[2].set_title('GARCH based inverse Volatility Portfolio')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Weights')
#axs[2].legend(loc='upper right')
axs[2].grid(True)

plt.show()


#   Comparing the strategies table
def total_return(returns):
    return (returns.iloc[-1] / returns.iloc[0] - 1) * 100  # Return percentage


def max_drawdown(returns):
    peak = returns.expanding(min_periods=1).max()
    drawdown = (returns/peak) - 1
    return drawdown.min() * 100  # Return drawdown in percentage


def portfolio_volatility(returns):
    return returns.std() * np.sqrt(1764) * 100  # Annualizing for hourly data by 252*7

#   Funktioniert noch nicht ganz
#   Converting to Pandas df
portfolio_value_over_time = np.array([portfolio_value_over_time])
portfolio_value_over_time = pd.DataFrame(portfolio_value_over_time)

portfolio_value_over_time_hist_inverse = np.array([portfolio_value_over_time_hist_inverse])
portfolio_value_over_time_hist_inverse = pd.DataFrame(portfolio_value_over_time_hist_inverse)

portfolio_value_over_time_VAR_inverse = np.array([portfolio_value_over_time_VAR_inverse])
portfolio_value_over_time_VAR_inverse = pd.DataFrame(portfolio_value_over_time_VAR_inverse)

portfolio_value_over_time_GARCH_inverse = np.array([portfolio_value_over_time_GARCH_inverse])
portfolio_value_over_time_GARCH_inverse = pd.DataFrame(portfolio_value_over_time_GARCH_inverse)

strategy_returns = {
    'Equal weighted Portfolio': portfolio_value_over_time,
    'Historical based Inverse Volatility Portfolio': portfolio_value_over_time_hist_inverse,
    'VAR forecast based inverse Volatility Portfolio': portfolio_value_over_time_VAR_inverse,
    'GARCH based inverse Volatility Portfolio': portfolio_value_over_time_GARCH_inverse,
}

performance_metrics = pd.DataFrame(columns=['Sharpe Ratio', 'Max Drawdown', 'Total Return', 'Volatility'])

for strategy, returns in strategy_returns.items():
    tr = total_return(returns)
    md = max_drawdown(returns)
    volatility = portfolio_volatility(returns)
    sharpe_ratio = total_return / volatility  # Simplified Sharpe Ratio calculation with r = 0

    performance_metrics = performance_metrics.append({
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': md,
        'Total Return': tr,
        'Volatility': volatility
    }, ignore_index=True)

performance_metrics = performance_metrics.apply(pd.to_numeric)
print(performance_metrics)
print(optimal_lag)
