import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close', 'Volume', 'High', 'Low', 'Open']]
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df.dropna(inplace=True)
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_sequences(data: np.ndarray, lookback: int = 60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train(ticker: str = "AAPL", lookback: int = 60, epochs: int = 100):
    print(f"Fetching data for {ticker}...")
    df = fetch_stock_data(ticker)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    X, y = prepare_sequences(scaled_data, lookback)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    model = build_model((lookback, X.shape[2]))
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(f'models/{ticker}_best.keras', save_best_only=True)
    ]
    
    os.makedirs('models', exist_ok=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
    
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss[0]:.6f}, MAE: {loss[1]:.6f}")
    return model, scaler, history

if __name__ == "__main__":
    train("AAPL")
