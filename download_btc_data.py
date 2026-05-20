import requests
import pandas as pd
import os

def download_btc_data():
    # Using Binance API for historical klines (free, no key needed)
    symbol = 'BTCUSDT'
    interval = '1h'  # 1 hour candles
    limit = 1000  # max per request
    start_time = '1500000000000'  # approx 2017-07-14 (BTC launch-ish)

    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&limit={limit}'

    response = requests.get(url)
    data = response.json()

    # Parse to DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Save to data/
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/BTCUSDT_1h.csv')
    print(f"Downloaded {len(df)} BTCUSDT 1h candles to data/BTCUSDT_1h.csv")

if __name__ == '__main__':
    download_btc_data()