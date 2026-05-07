from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
sys.path.insert(0, '../model')
from predict import predict_next_days
from train import fetch_stock_data

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker', 'AAPL')
    days = data.get('days', 7)
    predictions = predict_next_days(ticker, days)
    return jsonify({"success": True, "ticker": ticker, "predictions": predictions})

@app.route('/api/history/<ticker>')
def history(ticker):
    df = fetch_stock_data(ticker, period="1y")
    data = [{"date": str(idx.date()), "close": round(row['Close'], 2), "volume": int(row['Volume'])} for idx, row in df.iterrows()]
    return jsonify({"success": True, "ticker": ticker, "data": data[-90:]})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
