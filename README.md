# Stock Market Prediction using LSTM

Deep learning-based stock price prediction using LSTM neural networks with technical indicators.

## Features
- 📈 LSTM model with 3 layers for time-series prediction
- 📊 Technical indicators: RSI, Moving Averages (7, 21 day)
- 🔄 Multi-step prediction (1-30 days ahead)
- 📉 Real-time stock data via yfinance
- 🌐 Flask REST API for predictions
- 📱 React frontend with Chart.js visualizations
- 📓 Jupyter notebook for EDA

## Tech Stack
- **Model**: TensorFlow/Keras, LSTM, scikit-learn
- **Data**: yfinance, Pandas, NumPy
- **API**: Flask, Flask-CORS
- **Frontend**: React, Chart.js

## Getting Started
```bash
cd model && pip install -r requirements.txt
python train.py  # Train on AAPL
cd ../api && python app.py  # Start API
```

## License
MIT
