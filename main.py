# pip install streamlit prophet yfinance plotly
# streamlit run main.py

import csv
import requests
import streamlit as st
import pystan as ps
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

stocks = {}


CSV_URL = 'https://www.alphavantage.co/query?function=LISTING_STATUS&state=active&apikey=GG67ANDCMVCCRKPJ'

with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        if row[0] != "symbol" and "-" not in row[0]:
            stocks[row[0] + " - " + row[1]] = row[0]



START = "2013-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

selected_stock = st.selectbox('Select dataset for prediction', stocks.keys())

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


#replace with Alpha Vantage API 
@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(stocks[selected_stock])
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Prophet requires columns ds (Date) and y (value) 
df_train['ds'] = df_train['ds'].astype('datetime64[ns]')

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)