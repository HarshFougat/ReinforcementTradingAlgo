import pandas as pd
import pandas_ta as ta
import numpy as np


def load_and_preprocess_data(csv_path: str, start_date: str = None, end_date: str = None):
    """
    Advanced forex/crypto data preprocessing with 25+ technical indicators and market sentiment.
    
    CSV expected columns: [Time (EET), Open, High, Low, Close, Volume] OR [Date, Open, High, Low, Price, ...]
    
    Args:
        csv_path: Path to the CSV file
        start_date: Optional start date (e.g., "2000-01-01")
        end_date: Optional end date (e.g., "2015-12-31")
    
    Returns:
        df: DataFrame with OHLCV + all indicators
        feature_cols: List of 25+ technical indicators for the agent
    """
    # Determine date column and read CSV
    date_col = None
    try:
        df_peek = pd.read_csv(csv_path, nrows=1)
        if "Time (EET)" in df_peek.columns:
            date_col = "Time (EET)"
        elif "Date" in df_peek.columns:
            date_col = "Date"
    except:
        pass
    
    # Read CSV with proper date parsing
    if date_col:
        df = pd.read_csv(csv_path, parse_dates=[date_col], dayfirst=True)
    else:
        df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    
    # Set datetime index
    if "Time (EET)" in df.columns:
        df = df.set_index("Time (EET)")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    else:
        df = df.set_index(df.columns[0])
    
    df.sort_index(inplace=True)
    
    # Remove duplicate index labels (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    
    # Filter by date range
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]

    # Handle price column names
    if "Price" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Price"]
    
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    
    # Ensure numeric
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # ============ MOMENTUM INDICATORS ============
    
    # RSI variations (7, 14, 21)
    df["rsi_7"] = ta.rsi(df["Close"], length=7)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["rsi_21"] = ta.rsi(df["Close"], length=21)
    
    # MACD (trend following)
    macd_result = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd_result is not None:
        df["macd_line"] = macd_result.iloc[:, 0]  # MACD line
        df["macd_signal"] = macd_result.iloc[:, 1]  # Signal line
        df["macd_diff"] = macd_result.iloc[:, 2]  # MACD histogram
    else:
        df["macd_line"] = np.nan
        df["macd_signal"] = np.nan
        df["macd_diff"] = np.nan
    
    # Rate of Change (ROC)
    df["roc_12"] = ta.roc(df["Close"], length=12)
    
    # Stochastic Oscillator (fast)
    stoch_result = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    df["stoch_k"] = stoch_result.iloc[:, 0]  # Fast K
    df["stoch_d"] = stoch_result.iloc[:, 1]  # Fast D
    
    # ============ VOLATILITY INDICATORS ============
    
    # ATR (Average True Range)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_pct"] = (df["atr_14"] / df["Close"]) * 100  # ATR as % of price
    
    # Bollinger Bands (upper, middle, lower)
    bb_result = ta.bbands(df["Close"], length=20, std=2)
    df["bb_upper"] = bb_result.iloc[:, 2]  # Upper band
    df["bb_mid"] = bb_result.iloc[:, 1]    # Middle band (SMA)
    df["bb_lower"] = bb_result.iloc[:, 0]  # Lower band
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])  # Position in bands
    
    # ============ TREND INDICATORS ============
    
    # Moving Averages (EMA and SMA)
    df["ema_9"] = ta.ema(df["Close"], length=9)
    df["ema_21"] = ta.ema(df["Close"], length=21)
    df["ema_50"] = ta.ema(df["Close"], length=50)
    df["sma_200"] = ta.sma(df["Close"], length=200)
    
    # EMA slopes
    df["ema_9_slope"] = df["ema_9"].diff()
    df["ema_21_slope"] = df["ema_21"].diff()
    
    # Price position relative to EMA
    df["close_ema9_diff"] = df["Close"] - df["ema_9"]
    df["close_ema21_diff"] = df["Close"] - df["ema_21"]
    df["close_sma200_diff"] = df["Close"] - df["sma_200"]
    
    # Vortex Indicator (trend direction and strength)
    vortex = ta.vortex(df["High"], df["Low"], df["Close"], length=14)
    df["vortex_pos"] = vortex.iloc[:, 0]  # VI+
    df["vortex_neg"] = vortex.iloc[:, 1]  # VI-
    
    # ============ VOLUME INDICATORS ============
    
    # On-Balance Volume (OBV) trend
    df["obv"] = ta.obv(df["Close"], df["Volume"])
    df["obv_ema"] = ta.ema(df["obv"], length=14)
    df["obv_trend"] = np.sign(df["obv_ema"].diff())  # OBV direction
    
    # ============ ADDITIONAL INDICATORS ============
    
    # Commodity Channel Index (CCI)
    df["cci"] = ta.cci(df["High"], df["Low"], df["Close"], length=20)
    
    # Drop NaN rows
    df.dropna(inplace=True)

    # ============ MARKET SENTIMENT CALCULATION ============
    # Combine multiple signals to create a sentiment score
    
    def calculate_sentiment(row):
        """
        Calculate market sentiment (-1 to +1):
        +1: Strongly bullish
        -1: Strongly bearish
        """
        sentiment = 0.0
        
        # RSI-based sentiment (overbought/oversold)
        if row['rsi_14'] < 30:
            sentiment += 0.3
        elif row['rsi_14'] > 70:
            sentiment -= 0.3
        
        # MACD-based sentiment (direction)
        if row['macd_line'] > row['macd_signal']:
            sentiment += 0.2
        else:
            sentiment -= 0.2
        
        # Price vs MA 200 (long-term trend)
        if row['close_sma200_diff'] > 0:
            sentiment += 0.2
        else:
            sentiment -= 0.2
        
        # EMA 9/21 cross (short-term momentum)
        if row['ema_9'] > row['ema_21']:
            sentiment += 0.15
        else:
            sentiment -= 0.15
        
        # Stochastic (overbought/oversold)
        if row['stoch_k'] < 20:
            sentiment += 0.15
        elif row['stoch_k'] > 80:
            sentiment -= 0.15
        
        return np.clip(sentiment, -1, 1)
    
    df["market_sentiment"] = df.apply(calculate_sentiment, axis=1)

    # ============ FEATURE COLUMNS FOR AGENT ============
    feature_cols = [
        # Momentum (8 features)
        "rsi_7", "rsi_14", "rsi_21",
        "macd_line", "macd_signal", "macd_diff",
        "roc_12",
        
        # Volatility (7 features)
        "stoch_k", "stoch_d",
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

    return df, feature_cols
