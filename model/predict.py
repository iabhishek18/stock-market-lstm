import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from train import fetch_stock_data

def predict_next_days(ticker: str, days: int = 7) -> list:
    df = fetch_stock_data(ticker, period="2y")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    model = load_model(f'models/{ticker}_best.keras')
    
    lookback = 60
    last_sequence = scaled[-lookback:]
    predictions = []
    
    for _ in range(days):
        input_seq = last_sequence.reshape(1, lookback, -1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        new_row = np.zeros(scaled.shape[1])
        new_row[0] = pred
        last_sequence = np.vstack([last_sequence[1:], new_row])
    
    dummy = np.zeros((len(predictions), df.shape[1]))
    dummy[:, 0] = predictions
    actual_prices = scaler.inverse_transform(dummy)[:, 0]
    
    return [{"day": i + 1, "price": round(float(p), 2)} for i, p in enumerate(actual_prices)]
