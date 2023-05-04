from flask import Flask, render_template, request

import pandas as pd
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
import numpy as np
import folium
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import requests
import plotly.graph_objs as go
from textblob import TextBlob
import redis

app = Flask(__name__)


# create a Redis instance
redis_instance = redis.Redis(host='localhost', port=6379, db=0)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the ticker symbol from the form data
    ticker = request.form['ticker']

    # Fetch the historical prices for this ticker
    tickerData = yf.Ticker(ticker)
    tickerDf = tickerData.history(period='1d', start='2010-1-1', end='2023-05-01')

    # Preprocess the data by scaling and creating the target variable
    df = pd.DataFrame(tickerDf['Close'])
    df['Prediction'] = df['Close'].shift(-1)
    imputer = SimpleImputer()
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    training_data = scaled_data[:-1]
    testing_data = scaled_data[-1:]

    # Fit a linear regression model on the training data
    lr_model = LinearRegression()
    lr_model.fit(training_data[:, :-1], training_data[:, -1])

    # Make a prediction for the next day's closing price
    last_close_price = tickerDf['Close'][-1]
    last_prediction = tickerDf['Close'][-2]  # Get the last known value of the 'Prediction' column
    input_data = testing_data[:, :-1]  # use last known value as input
    next_day_prediction = lr_model.predict(input_data)
    next_day_prediction = scaler.inverse_transform(np.concatenate((input_data, next_day_prediction.reshape(-1, 1)),
                                                                   axis=1))[-1, -1]


    #map data

    api_key = "362d4eca-6e5f-4f13-b7a2-3a9a7c76efb8"
    # Get the location of the company associated with the stock ticker

    api_key = "362d4eca-6e5f-4f13-b7a2-3a9a7c76efb8"
    geolocator = Nominatim(user_agent='somesh2')
    location = geolocator.geocode("New brunswick New Jersey")


    # Get the location of EV charging stations near the company location
    if location is not None:
        lat = location.latitude
        lon = location.longitude
        url = f'https://api.openchargemap.io/v3/poi/?output=json&countrycode=US&maxresults=50&compact=true&verbose=false&latitude={lat}&longitude={lon}&distance=10&distanceunit=Miles&key={api_key}'
        r = requests.get(url)
        charging_stations = r.json()

        # Create a map
        m = folium.Map(location=[lat, lon], zoom_start=12)

        # Add markers for each charging station
        for station in charging_stations:
            lat = station['AddressInfo']['Latitude']
            lon = station['AddressInfo']['Longitude']
            name = station['AddressInfo']['Title']
            folium.Marker(location=[lat, lon], popup=name).add_to(m)

        # Render the map as an HTML div string
        map_div = m._repr_html_()

    else:
        map_div = "<p>Sorry, we couldn't find the location of this company.</p>"

    #end map data

    # Create a list of dates for the historical data and the predicted data
    history_dates = tickerDf.index
    prediction_dates = pd.date_range(start=history_dates[-1], periods=2, freq='D')[1:]

    # Create a trace for the historical data
    history_trace = go.Scatter(x=history_dates, y=tickerDf['Close'], mode='lines', name='Historical Prices')

    # Create a trace for the predicted data
    prediction_trace = go.Scatter(x=prediction_dates, y=[last_close_price, next_day_prediction], mode='lines',
                                  name='Predicted Prices')

    # Combine the traces into a data list and create a layout
    data = [history_trace, prediction_trace]
    layout = go.Layout(title=f"{ticker.upper()} Price Prediction", xaxis_title="Date", yaxis_title="Closing Price")

    # Create a figure and render it as an HTML div string
    fig = go.Figure(data=data, layout=layout)
    plot_div = fig.to_html(full_html=False)

    news = get_news(ticker)
    return render_template('predict.html', ticker=ticker.upper(), last_close_price=last_close_price,
                           next_day_prediction=next_day_prediction, plot_div=plot_div, news=news, map_div=map_div)

def get_news(ticker):
    # create a list of articles
    news_articles = []
    # get the news data from Yahoo Finance
    yahoo_news = yf.Ticker(ticker).news[:4]

    for item in yahoo_news:
        # format the timestamp as a string
        try:
            timestamp_str = datetime.utcfromtimestamp(item['date']).strftime('%Y-%m-%d %H:%M:%S')
        except KeyError:
            timestamp_str = ''
        # create a dictionary for the article
        article = {
            'title': item['title'],
            'link': item['link'],
            'publisher':item['publisher'],
            'source': item.get('source', ''),
            'timestamp': timestamp_str,
            'summary': item.get('summary', '')
        }
        news_articles.append(article)
    return news_articles


@app.route('/compare_stocks', methods=['GET','POST'])
def compare_stocks():
    # Get the tickers from the form input
    ticker1 = request.form['ticker1']
    ticker2 = request.form['ticker2']

    # Get the historical price data for the two stocks
    stock1 = yf.Ticker(ticker1)
    stock2 = yf.Ticker(ticker2)
    history1 = stock1.history(period='max')
    history2 = stock2.history(period='max')

    # Plot the stock price data using matplotlib
    fig, ax = plt.subplots()
    ax.plot(history1.index, history1['Close'], label=ticker1)
    ax.plot(history2.index, history2['Close'], label=ticker2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title('Stock Comparison')
    ax.legend()

    # Save the plot image to a file
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    filename = 'images/comparison_{}.png'.format(timestamp)
    fig.savefig('static/' + filename)

    # Calculate the historical volatility of the two stocks
    vol1 = history1['Close'].pct_change().std()
    vol2 = history2['Close'].pct_change().std()

    # Create a bar chart of the volatilities
    fig2, ax2 = plt.subplots()
    ax2.bar([ticker1, ticker2], [vol1, vol2])
    ax2.set_title('Historical Volatility Comparison')
    ax2.set_xlabel('Stock Ticker')
    ax2.set_ylabel('Volatility')

    timestamp2 = datetime.timestamp(datetime.now())
    vol_filename = 'images/vol_comparison_{}.png'.format(timestamp2)
    fig2.savefig('static/' + vol_filename)

    # Get the latest news articles for the two companies
    api_key = '249e57e5c3bf44dbab560b3e7ebc3ed2'
    url1 = f'https://newsapi.org/v2/everything?q={ticker1}&apiKey={api_key}'
    url2 = f'https://newsapi.org/v2/everything?q={ticker2}&apiKey={api_key}'
    response1 = requests.get(url1).json()
    response2 = requests.get(url2).json()

    # Analyze the sentiment of the news articles
    sentiment1 = 0
    sentiment2 = 0
    count1 = 0
    count2 = 0
    for article in response1['articles']:
        text = article['title'] + ' ' + article['description']
        blob = TextBlob(text)
        sentiment1 += blob.sentiment.polarity
        count1 += 1
    for article in response2['articles']:
        text = article['title'] + ' ' + article['description']
        blob = TextBlob(text)
        sentiment2 += blob.sentiment.polarity
        count2 += 1

    # Calculate the average sentiment score for each company
    if count1 > 0:
        sentiment1 /= count1
    if count2 > 0:
        sentiment2 /= count2

    # Save the sentiment scores to a dictionary
    sentiment = {}
    sentiment[ticker1] = sentiment1
    sentiment[ticker2] = sentiment2

    # Plot the sentiment scores using matplotlib
    fig2, ax2 = plt.subplots()
    ax2.bar(sentiment.keys(), sentiment.values())
    ax2.set_xlabel('Stock Ticker')
    ax2.set_ylabel('Sentiment Score')
    ax2.set_title('Comparison of News Sentiment')

    # Save the plot image to a file
    sent_timestamp = datetime.timestamp(datetime.now())
    sent_filename = 'images/sentiment_{}.png'.format(sent_timestamp)
    fig2.savefig('static/' + sent_filename)

    # Render the HTML template with the comparison and sentiment plots
    return render_template('compare.html', filename=filename, vol_filename=vol_filename, sent_filename=sent_filename)


if __name__ == '__main__':
    app.run(debug=True)


