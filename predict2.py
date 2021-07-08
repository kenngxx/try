import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
from yahooquery import Ticker
import datetime as dt

asset = st.text_input("Enter here: ")

ticker_input_ti = Ticker(asset)
history_args = {
    "period": "1y",
    "interval": "1d",
    "start": dt.datetime.now() - dt.timedelta(days=365),
    "end": None,
}
option_1 = st.selectbox("Select Period or Start / End Dates", ["Period", "Dates"], 0)
if option_1 == "Period":
    history_args["period"] = st.selectbox(
        "Select Period", options=Ticker.PERIODS, index=5  # pylint: disable=protected-access
    )

    history_args["start"] = None
    history_args["end"] = None
else:
    history_args["start"] = st.date_input("Select Start Date", value=history_args["start"])
    history_args["end"] = st.date_input("Select End Date")
    history_args["period"] = None

st.markdown("**THEN**")
history_args["interval"] = st.selectbox(
    "Select Interval", options=Ticker.INTERVALS, index=8  # pylint: disable=protected-access
)
args_string = [str(k) + "='" + str(v) + "'" for k, v in history_args.items() if v is not None]
st.write("Dataframe")
dataframe_ti = ticker_input_ti.history(**history_args)
if isinstance(dataframe_ti, dict):
    st.write(dataframe_ti)
else:
    st.dataframe(dataframe_ti)


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv('data//AAPL.csv',sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)

#plot close price
fig_1 = plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Close Prices')
plt.plot(data['Close'])
plt.title('Apple Inc. (AAPL) closing price')

st.subheader("Figure 1")
st.pyplot(fig_1)


df_close = data['Close']
df_close.plot(style='k.')
plt.title('Scatter plot of closing price')
st.subheader("Figure 2")
st.pyplot(fig_1)

# Test for staionarity
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    adft = adfuller(timeseries, autolag='AIC')
    # output for dft will give us without defining what the values are.
    # hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],
                       index=['Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values




result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
fig_3 = plt.figure()
fig_3 = result.plot()
fig_3.set_size_inches(16, 9)

rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
st.subheader("Figure 3")
st.write(fig_3)

#split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
fig_4 = plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()
st.write(fig_4)

model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)



fig_5 = model_autoARIMA.plot_diagnostics(figsize=(15,8))

st.write(fig_5)

model = ARIMA(train_data, order=(3, 1, 2))
fitted = model.fit(disp=-1)


# Forecast
fc, se, conf = fitted.forecast(544, alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
fig_6 = plt.figure(figsize=(12,5), dpi=100)
plt.plot(train_data, label='training')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.10)
plt.title('Apple Inc. Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Actual Stock Price')
plt.legend(loc='upper left', fontsize=8)

st.write(fig_6)




