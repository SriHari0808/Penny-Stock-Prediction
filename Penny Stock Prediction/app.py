from flask import Flask, request, render_template
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib

# Use the 'Agg' backend for matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load the pre-trained RNN model
model = tf.keras.models.load_model('stock_prediction_rnn_model.h5')

def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def preprocess_data(df):
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
    return np.array(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = 'ZOM'
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    df = fetch_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    scaled_data, scaler = preprocess_data(df)
    seq_length = 60
    X = create_sequences(scaled_data, seq_length)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Predict next 30 trading days
    predictions = []
    current_sequence = X[-1]
    next_date = end_date
    while len(predictions) < 30:
        next_date += timedelta(days=1)
        if next_date.weekday() < 5:  # Monday to Friday are 0-4
            prediction = model.predict(np.expand_dims(current_sequence, axis=0))
            predictions.append(prediction[0][0])
            current_sequence = np.append(current_sequence[1:], prediction)
            current_sequence = np.reshape(current_sequence, (seq_length, 1))
    
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    prediction_dates = pd.date_range(start=end_date + timedelta(days=1), periods=30, freq='B')
    prediction_results = {prediction_dates[i].strftime('%Y-%m-%d'): predictions[i][0] for i in range(30)}
    
    # Plot the predictions
    plt.figure(figsize=(10, 5))
    plt.plot(prediction_dates, predictions, label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Predicted Stock Price')
    plt.title(f'Predicted Stock Prices for {ticker} for the Next 30 Trading Days')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/prediction_plot.png')
    plt.close()

    return render_template('index.html', prediction_results=prediction_results, ticker=ticker, plot_url='static/prediction_plot.png')

if __name__ == "__main__":
    app.run(debug=True)
