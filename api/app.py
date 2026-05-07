from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

app = Flask(__name__)
CORS(app)


@app.route('/health')
def health():
    return jsonify({"status": "ok", "version": "2.0.0", "model_loaded": False})


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker', 'AAPL')
    days = data.get('days', 7)

    try:
        from predict import predict_next_days
        predictions = predict_next_days(ticker, days)
        return jsonify({"success": True, "ticker": ticker, "predictions": predictions})
    except FileNotFoundError:
        return jsonify({"success": False, "error": {"code": "MODEL_NOT_FOUND", "message": f"Model for {ticker} not trained yet. Run: python model/train.py"}}), 404
    except ImportError as e:
        return jsonify({"success": False, "error": {"code": "DEPENDENCY_MISSING", "message": str(e)}}), 503


@app.route('/api/history/<ticker>')
def history(ticker):
    days = request.args.get('days', 90, type=int)
    try:
        from train import fetch_stock_data
        df = fetch_stock_data(ticker, period=f"{days}d")
        data = [{"date": str(idx.date()), "open": round(row['Open'], 2), "high": round(row['High'], 2), "low": round(row['Low'], 2), "close": round(row['Close'], 2), "volume": int(row['Volume'])} for idx, row in df.iterrows()]
        return jsonify({"success": True, "ticker": ticker, "data": data, "meta": {"points": len(data), "period": f"{days}d"}})
    except Exception as e:
        return jsonify({"success": False, "error": {"code": "FETCH_ERROR", "message": str(e)}}), 502


@app.route('/api/indicators/<ticker>')
def indicators(ticker):
    try:
        from train import fetch_stock_data
        df = fetch_stock_data(ticker, period="90d")
        latest = df.iloc[-1]
        return jsonify({"success": True, "ticker": ticker, "indicators": {
            "close": round(latest['Close'], 2),
            "ma_7": round(latest['MA_7'], 2),
            "ma_21": round(latest['MA_21'], 2),
            "rsi": round(latest['RSI'], 2),
            "signal": "buy" if latest['RSI'] < 30 else "sell" if latest['RSI'] > 70 else "hold",
            "trend": "bullish" if latest['MA_7'] > latest['MA_21'] else "bearish",
        }})
    except Exception as e:
        return jsonify({"success": False, "error": {"code": "ERROR", "message": str(e)}}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5009))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'false') == 'true')
