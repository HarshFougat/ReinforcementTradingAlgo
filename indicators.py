import pandas as pd
import pandas_ta as ta


def load_and_preprocess_data(csv_path: str, start_date: str = None, end_date: str = None):
    """
    Loads forex/crypto data from CSV and preprocesses it by adding RELATIVE technical features.

    CSV expected columns: [Time (EET), Open, High, Low, Close, Volume] OR [Date, Open, High, Low, Price, ...]
    The returned DataFrame still contains OHLCV for env internals,
    but `feature_cols` lists only the RELATIVE columns to feed the agent.
    
    Args:
        csv_path: Path to the CSV file
        start_date: Optional start date (e.g., "2000-01-01")
        end_date: Optional end date (e.g., "2015-12-31")
    """
    # Try to determine the date column and read CSV accordingly
    date_col = None
    try:
        # First check what columns exist
        df = pd.read_csv(csv_path, nrows=1)
        if "Time (EET)" in df.columns:
            date_col = "Time (EET)"
        elif "Date" in df.columns:
            date_col = "Date"
    except:
        pass
    
    # Read the CSV with appropriate date column
    if date_col:
        df = pd.read_csv(csv_path, parse_dates=[date_col], dayfirst=True)
    else:
        df = pd.read_csv(csv_path)

    # Strip any trailing spaces in headers
    df.columns = df.columns.str.strip()
    
    # Handle different date column names
    if "Time (EET)" in df.columns:
        date_col = "Time (EET)"
    elif "Date" in df.columns:
        date_col = "Date"
    else:
        # If no recognized date column, check if first column is datetime
        date_col = df.columns[0]
    
    # Datetime index
    if date_col in df.columns:
        df = df.set_index(date_col)
    df.sort_index(inplace=True)
    
    # Filter by date range if provided
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]

    # Handle different price column names: "Price" in USD_INR, "Close" in EUR/USD
    if "Price" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Price"]
    
    # Handle missing Volume column (common in some datasets)
    if "Volume" not in df.columns:
        df["Volume"] = 1.0  # Default volume
    
    # Ensure numeric for required columns
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # ---- Technicals ----
    # RSI and ATR (already scale-invariant-ish)
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # Moving averages
    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)

    # Slopes of the MAs
    df["ma_20_slope"] = df["ma_20"].diff()
    df["ma_50_slope"] = df["ma_50"].diff()

    # Distance of price from each MA (relative level)
    df["close_ma20_diff"] = df["Close"] - df["ma_20"]
    df["close_ma50_diff"] = df["Close"] - df["ma_50"]

    # MA divergence: MA20 vs MA50
    df["ma_spread"] = df["ma_20"] - df["ma_50"]
    df["ma_spread_slope"] = df["ma_spread"].diff()

    # Drop initial NaNs from indicators
    df.dropna(inplace=True)

    # Columns the AGENT should see (no raw price levels / raw MAs)
    feature_cols = [
        "rsi_14",
        "atr_14",
        "ma_20_slope",
        "ma_50_slope",
        "close_ma20_diff",
        "close_ma50_diff",
        "ma_spread",
        "ma_spread_slope",
    ]

    return df, feature_cols
