import time
import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from stable_baselines3 import PPO

from indicators import load_and_preprocess_data

# -----------------------------------------------------------------------------
# Put your Vantage paper-trading credentials here
# -----------------------------------------------------------------------------
VANTAGE_API_KEY = os.getenv('VANTAGE_API_KEY', 'YOUR_VANTAGE_API_KEY')
VANTAGE_ACCOUNT_ID = os.getenv('VANTAGE_ACCOUNT_ID', '')
VANTAGE_API_URL = ''  # replace with actual Vantage endpoint

SYMBOL = 'USDINR'  # confirm with Vantage symbol naming scheme (e.g. USDINR or USD/INR)
TIMEFRAME = 'M1'  # 1-minute bars
MAX_POS_LOTS = 1.0  # maximum position size, adjust for risk

# Risk budgeting
INITIAL_CAPITAL = 10000.0
LEVERAGE = 500
RISK_PCT_PER_TRADE = 0.02  # 2% of capital per trade

# Training artifacts
MODEL_PATH = 'model_eurusd_best.zip'
WINDOW_SIZE = 30

# If you already have prepared feature columns in indicators.py, use same list
_, FEATURE_COLS = load_and_preprocess_data('data/USD_INR Historical Data.csv', start_date='2000-01-01', end_date='2000-02-01')

# Linear action map from trading_env; same mapping required in train agent
ACTION_MAP = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
for dir_ in [0, 1]:
    for sl in [5, 10, 15, 25, 30, 60, 90, 120]:
        for tp in [5, 10, 15, 25, 30, 60, 90, 120]:
            ACTION_MAP.append(("OPEN", dir_, float(sl), float(tp)))


def fetch_recent_candles(symbol, timeframe='M1', count=200):
    """Fetch OHLC candle history from Vantage API (demo)."""
    # placeholder for a real Vantage REST endpoint. Second argument in request may differ.
    endpoint = f'{VANTAGE_API_URL}/v1/market-history'
    payload = {
        'apikey': VANTAGE_API_KEY,
        'symbol': symbol,
        'interval': timeframe,
        'count': count,
    }

    r = requests.get(endpoint, params=payload, timeout=20)
    r.raise_for_status()

    data = r.json()
    
    # Expect data in list of {timestamp, open, high, low, close, volume}
    df = pd.DataFrame(data['candles'])
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].rename(
        columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})

    return df


def send_order(symbol, direction, volume, sl_pips, tp_pips):
    """Place an order in Vantage demo account."""
    endpoint = f'{VANTAGE_API_URL}/v1/orders'
    side = 'BUY' if direction == 1 else 'SELL'

    body = {
        'apikey': VANTAGE_API_KEY,
        'account_id': VANTAGE_ACCOUNT_ID,
        'symbol': symbol,
        'side': side,
        'type': 'market',
        'volume': volume,
        'stop_loss_pips': sl_pips,
        'take_profit_pips': tp_pips,
    }

    r = requests.post(endpoint, json=body, timeout=20)
    r.raise_for_status()
    return r.json()


def close_position(position_id):
    endpoint = f'{VANTAGE_API_URL}/v1/orders/{position_id}/close'
    body = {
        'apikey': VANTAGE_API_KEY,
        'account_id': VANTAGE_ACCOUNT_ID,
    }
    r = requests.post(endpoint, json=body, timeout=20)
    r.raise_for_status()
    return r.json()


def get_account_info():
    endpoint = f'{VANTAGE_API_URL}/v1/accounts/{VANTAGE_ACCOUNT_ID}'
    params = {'apikey': VANTAGE_API_KEY}
    r = requests.get(endpoint, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def infer_position_size(equity_usd, sl_pips):
    if sl_pips <= 0 or np.isclose(sl_pips, 0):
        return MAX_POS_LOTS

    capital_risk = equity_usd * RISK_PCT_PER_TRADE
    pip_value_per_lot = 0.0001 * 100000
    max_risk_lots = (capital_risk / (sl_pips * pip_value_per_lot))

    # enforce at least 0.01 lot and at most max by configuration
    size = max(0.01, min(max_risk_lots, MAX_POS_LOTS))
    return size


def construct_observation(recent_df):
    # compute indicators on fresh history, same as training path
    source = recent_df.copy()
    source['Price'] = source['Close']

    # to reduce overhead, call load_and_preprocess_data semantics for this DF
    # we can pass via CSV in-memory or add helper to compute indicators directly.
    # For simplicity, recompute manually with same logic borrowed from indicators.py
    # (for the demo we can ensure `recent_df` already contains these columns from function)

    start = len(source) - WINDOW_SIZE
    window = source.iloc[start:]
    features = window[FEATURE_COLS].values

    pos = np.zeros(3, dtype=np.float32)
    obs = np.concatenate([features, np.tile(pos, (WINDOW_SIZE, 1))], axis=1)
    return obs.astype(np.float32)


def live_loop():
    model = PPO.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

    current_equity = INITIAL_CAPITAL
    open_position_info = None

    while True:
        try:
            df = fetch_recent_candles(SYMBOL, timeframe=TIMEFRAME, count=WINDOW_SIZE + 40)
            if len(df) < WINDOW_SIZE:
                print("Not enough data—waiting")
                time.sleep(10)
                continue

            df_prepared, _ = load_and_preprocess_data('data/USD_INR Historical Data.csv')
            # TODO: apply live sample indicator update for speed
            obs = construct_observation(df)

            action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
            action = int(action[0])
            act_type, direction, sl_pips, tp_pips = ACTION_MAP[action]

            print(f"[{datetime.utcnow()}] Action {action} -> {act_type}, dir={direction}, sl={sl_pips}, tp={tp_pips}")

            if act_type == 'OPEN' and open_position_info is None:
                if not (sl_pips and tp_pips):
                    print("Invalid SL/TP, skipping")
                else:
                    pos_size = infer_position_size(current_equity, sl_pips)
                    print(f"Placing order: direction={direction}, size={pos_size:.2f} lots")
                    res = send_order(SYMBOL, direction, pos_size, sl_pips, tp_pips)
                    open_position_info = res.get('position_id')
                    print("Order result:", res)

            elif act_type == 'CLOSE' and open_position_info is not None:
                print(f"Closing position {open_position_info}")
                res = close_position(open_position_info)
                print("Close result:", res)
                open_position_info = None

            elif act_type == 'HOLD':
                print("No order, HOLD")

            else:
                print("No trade action taken")

            account_info = get_account_info()
            current_equity = float(account_info['equity'])
            print(f"Equity now {current_equity:.2f} USD")

            time.sleep(60)  # cycle once per bar (maybe M1)

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting.")
            break
        except Exception as exc:
            print("Exception in live loop:", type(exc).__name__, exc)
            time.sleep(15)


if __name__ == '__main__':
    print("Starting Vantage demo live trading loop")
    print(f"Account: {VANTAGE_ACCOUNT_ID}, Leverage={LEVERAGE}, Capital=${INITIAL_CAPITAL}")
    live_loop()
