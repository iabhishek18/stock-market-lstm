# Stock Market Prediction using LSTM

> Deep learning-based stock price prediction using 3-layer LSTM neural networks with RSI, Moving Averages, and multi-step forecasting.

## 🚀 Overview

This project implements a Long Short-Term Memory (LSTM) neural network for time-series stock price prediction. It fetches historical data via yfinance, engineers technical indicators (RSI, MA7, MA21), trains a 3-layer LSTM with dropout regularization, and provides multi-day price forecasts through a Flask REST API.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📈 3-Layer LSTM | 128→64→32 neuron architecture with dropout |
| 📊 Technical Indicators | RSI(14), MA(7), MA(21) feature engineering |
| 🔄 Multi-Step Prediction | Predict 1-30 days ahead |
| 📉 Real-Time Data | Live stock data via yfinance API |
| 🌐 REST API | Flask API serving predictions |
| 📓 Jupyter Notebook | EDA and visualization |
| ⚡ Early Stopping | Training with patience=10, best weights |
| 📱 Frontend Ready | React + Chart.js visualization |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | TensorFlow/Keras LSTM |
| Data | yfinance, Pandas, NumPy |
| Processing | scikit-learn (MinMaxScaler) |
| API | Flask, Flask-CORS |
| Frontend | React, Chart.js |

## ⚡ Quick Start

```bash
# Install Python dependencies
cd model
pip install -r requirements.txt

# Train the model (default: AAPL, 5 years)
python train.py

# Start prediction API
cd ../api
python app.py
```

API at `http://localhost:5000`

### API Endpoints

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | `/api/predict` | `{ticker: "AAPL", days: 7}` | Array of {day, price} |
| GET | `/api/history/AAPL` | - | Last 90 days OHLCV |

## 📄 License

MIT
