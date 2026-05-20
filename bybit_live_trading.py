import os
import time
import hmac
import hashlib
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO

from indicators import load_and_preprocess_data

# -----------------------------------------------------------------------------
# Bybit demo credentials and constants
# -----------------------------------------------------------------------------
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', 'SNX2WlOlLVoFcbVG30')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', 'jMUpiSbuCUnnxn5F3fHdOfoXThNTTOcMxQtM')
BYBIT_BASE_URL = 'https://api-demo.bybit.com'

SYMBOL = 'BTCUSDT'
INTERVAL = '1'  # 1-minute candles; can be '5', '15', '60' etc in bybit API
STARTING_CAPITAL = 100000.0
RISK_PCT_PER_TRADE = 0.02
MAX_POSITION_RISK = 0.02
MODEL_PATH = 'model_eurusd_best.zip'  # ideally the BTC-specific model path if available
WINDOW_SIZE = 30

# Generate feature names from precomputed data
FEATURE_COLS = [
    # Momentum (8 features)
    "rsi_7", "rsi_14", "rsi_21",
    "macd_line", "macd_signal", "macd_diff",
    "roc_12",
    "stoch_k", "stoch_d",
    # Volatility (7 features)
    "atr_14", "atr_pct",
    "bb_width", "bb_pct",
    # Trend (8 features)
    "ema_9_slope", "ema_21_slope",
    "close_ema9_diff", "close_ema21_diff", "close_sma200_diff",
    "vortex_pos", "vortex_neg",
    # Volume (1 feature)
    "obv_trend",
    # Additional (1 feature)
    "cci",
    # Market Sentiment (1 feature)
    "market_sentiment",
]

ACTION_MAP = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
for dir_ in [0, 1]:
    for sl in [5, 10, 15, 25, 30, 60, 90, 120]:
        for tp in [5, 10, 15, 25, 30, 60, 90, 120]:
            ACTION_MAP.append(("OPEN", dir_, float(sl), float(tp)))


def _bybit_signature(timestamp, body=''):
    payload = f"{timestamp}{BYBIT_API_KEY}{5000}{body}"
    return hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()


def _get(endpoint, params=None):
    if params is None:
        params = {}
    timestamp = str(int(time.time() * 1000))
    body = ''
    sign = _bybit_signature(timestamp, body)
    headers = {
        'X-BAPI-API-KEY': BYBIT_API_KEY,
        'X-BAPI-SIGN': sign,
        'X-BAPI-TIMESTAMP': timestamp,
        'X-BAPI-RECV-WINDOW': '5000',
        'Content-Type': 'application/json'
    }
    url = BYBIT_BASE_URL + endpoint
    if params:
        url += '?' + '&'.join([f"{k}={v}" for k, v in params.items()])
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    try:
        return r.json()
    except requests.exceptions.JSONDecodeError:
        print(f"API Error: {r.status_code} - {r.text}")
        return {}


def _post(endpoint, params=None):
    if params is None:
        params = {}
    timestamp = str(int(time.time() * 1000))
    body = json.dumps(params)
    sign = _bybit_signature(timestamp, body)
    headers = {
        'X-BAPI-API-KEY': BYBIT_API_KEY,
        'X-BAPI-SIGN': sign,
        'X-BAPI-TIMESTAMP': timestamp,
        'X-BAPI-RECV-WINDOW': '5000',
        'Content-Type': 'application/json'
    }
    r = requests.post(BYBIT_BASE_URL + endpoint, headers=headers, data=body, timeout=20)
    r.raise_for_status()
    try:
        return r.json()
    except requests.exceptions.JSONDecodeError:
        print(f"API Error: {r.status_code} - {r.text}")
        return {}


def fetch_recent_candles(symbol=SYMBOL, interval=INTERVAL, limit=100):
    endpoint = '/v5/market/kline'
    params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': limit}
    resp = _get(endpoint, params)
    if 'result' not in resp or 'list' not in resp['result']:
        print("No candle data received")
        return pd.DataFrame()
    df = pd.DataFrame(resp['result']['list'])
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # Convert ms → datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')

    df = df.dropna(subset=['datetime'])  # remove bad rows
    df = df.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
    df = df.astype(float)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df


def get_account_balance():
    endpoint = '/v5/account/wallet-balance'
    params = {'accountType': 'UNIFIED'}
    resp = _get(endpoint, params)
    if 'result' not in resp or 'list' not in resp['result']:
        return 0.0
    for item in resp['result']['list']:
        if item.get('accountType') == 'UNIFIED':
            for coin in item.get('coin', []):
                if coin.get('coin') == 'USDT':
                    return float(coin.get('availableToWithdraw', 0.0))
    return 0.0


# def set_leverage(symbol: str, leverage: int):
#     endpoint = '/v5/position/set-leverage'
#     params = {'category': 'linear', 'symbol': symbol, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
#     return _post(endpoint, params)


def infer_position_qty(equity_usd, sl_pips):
    if sl_pips <= 0:
        return 0.001
    risk_amount = equity_usd * RISK_PCT_PER_TRADE
    tick_value = 1  # 1 USD per 1 point for 1 BTCUSDT contract
    qty = risk_amount / (sl_pips * tick_value)
    return max(0.001, qty)


def construct_observation(recent_df):
    source = recent_df.copy()
    source['Price'] = source['Close']
    start = max(0, len(source) - WINDOW_SIZE)
    window = source.iloc[start:]
    # If indicator columns missing, you must compute with indicators.py helper
    if not set(FEATURE_COLS).issubset(window.columns):
        window, _ = load_and_preprocess_data('data/USD_INR Historical Data.csv')
    features = window[FEATURE_COLS].values
    pos = np.zeros(3, dtype=np.float32)
    obs = np.concatenate([features, np.tile(pos, (WINDOW_SIZE, 1))], axis=1)
    return obs.astype(np.float32)


def place_order(symbol, side, qty, sl_price, tp_price):
    endpoint = '/v5/order/create'
    params = {
        'category': 'linear',
        'symbol': symbol,
        'side': side,
        'orderType': 'Market',
        'qty': str(qty),
        'timeInForce': 'GTC',
        'stopLoss': str(sl_price),
        'takeProfit': str(tp_price),
    }
    return _post(endpoint, params)


def apply_demo_funds(amount=100000):
    endpoint = '/v5/account/demo-apply-money'
    params = {'amount': str(amount), 'coin': 'USDT'}
    return _post(endpoint, params)


def live_loop():
    model = PPO.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

    # set_leverage(SYMBOL, LEVERAGE)
    # print(f"Set leverage {LEVERAGE} for {SYMBOL}")

    open_position = False
    equity = STARTING_CAPITAL
    pos_side = None

    while True:
        try:
            df = fetch_recent_candles(limit=WINDOW_SIZE + 20)
            if len(df) < WINDOW_SIZE:
                time.sleep(10)
                continue

            obs = construct_observation(df)
            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            action = int(action[0])
            act_type, direction, sl_pips, tp_pips = ACTION_MAP[action]

            print(f"{datetime.utcnow()} action {action} -> {act_type}, {direction}, sl={sl_pips}, tp={tp_pips}")

            if act_type == 'OPEN' and not open_position:
                qty = infer_position_qty(equity, sl_pips)
                if direction == 1:
                    side = 'Buy'
                    sl_price = df.Close.iloc[-1] - sl_pips
                    tp_price = df.Close.iloc[-1] + tp_pips
                else:
                    side = 'Sell'
                    sl_price = df.Close.iloc[-1] + sl_pips
                    tp_price = df.Close.iloc[-1] - tp_pips

                res = place_order(SYMBOL, side, qty, sl_price, tp_price)
                print('Order placed:', res)
                open_position = True
                pos_side = side

            elif act_type == 'CLOSE' and open_position:
                opp_side = 'Sell' if pos_side == 'Buy' else 'Buy'
                res = close_position(SYMBOL, opp_side)
                print('Close position:', res)
                open_position = False
                pos_side = None

            # update equity from account every cycle
            equity = get_account_balance()
            print(f'Equity: {equity:.2f}')
            time.sleep(60)

        except KeyboardInterrupt:
            print('Stopped by user')
            break
        except Exception as e:
            print('Error:', e)
            time.sleep(15)


if __name__ == '__main__':
    print('Starting Bybit BTC demo trading loop')
    live_loop()
