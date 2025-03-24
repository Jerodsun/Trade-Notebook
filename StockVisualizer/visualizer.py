import os
import pandas as pd
import numpy as np
import sqlite3
import json
import datetime as dt
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# Set the plotly template
load_figure_template("bootstrap")

FILE_PATH = os.environ.get("SPX_DATA_FILE", "DAT_ASCII_SPXUSD_M1_202502.csv")


# Database connection functions
def create_database(file_path, db_path="spx_data.db"):
    """Create SQLite database from the Excel file with SPX data"""

    # Try reading as CSV
    try:
        df = pd.read_csv(file_path, header=None)
        if ";" in df.iloc[0, 0]:  # Check if first cell contains semicolons
            split_data = df[0].str.split(";", expand=True)
            df = split_data
            df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    except Exception as csv_error:
        print(f"Error reading as CSV: {csv_error}")
        return

    # Process datetime column
    try:
        # Format: YYYYMMDD HHMMSS
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")
    except Exception as e:
        print(f"Error parsing datetime column: {e}")
        # Try another common format
        try:
            df["datetime"] = pd.to_datetime(df["datetime"])
        except:
            print("Unable to parse datetime column")
            return

    # Convert to EST and filter market hours
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M%S")
    df["datetime"] = df["datetime"].dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="NaT"
    )

    market_open = dt.time(9, 30)
    market_close = dt.time(16, 0)
    df = df[
        (df["datetime"].dt.time >= market_open)
        & (df["datetime"].dt.time <= market_close)
    ]

    # Create date and time columns for easier filtering
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill any missing values
    df = df.fillna(method="ffill")

    # Create the database and save the data
    conn = sqlite3.connect(db_path)
    df.to_sql("price_data", conn, if_exists="replace", index=False)

    # Create indices for faster queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON price_data (date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_datetime ON price_data (datetime)")
    conn.commit()
    conn.close()

    print(f"Database created successfully at {db_path}")
    return


def get_dates(db_path="spx_data.db"):
    """Get list of available dates in the database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT date FROM price_data ORDER BY date"
    dates = pd.read_sql_query(query, conn)
    conn.close()
    return dates["date"].tolist()


def get_data_for_date(date, db_path="spx_data.db"):
    """Get minute data for a specific date"""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM price_data WHERE date = '{date}' ORDER BY datetime"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_data_for_range(start_date, end_date, db_path="spx_data.db"):
    """Get data for a date range"""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM price_data WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY datetime"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_daily_data(db_path="spx_data.db"):
    """Get daily OHLC data by aggregating minute data"""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        date,
        MIN(datetime) as datetime,
        FIRST_VALUE(open) OVER (PARTITION BY date ORDER BY datetime) as open,
        MAX(high) as high,
        MIN(low) as low,
        LAST_VALUE(close) OVER (PARTITION BY date ORDER BY datetime 
            RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as close,
        SUM(volume) as volume
    FROM price_data
    GROUP BY date
    ORDER BY date
    """
    try:
        df = pd.read_sql_query(query, conn)
    except:
        # Alternative query for SQLite versions that don't support window functions
        query_alt = """
        SELECT 
            date,
            MIN(datetime) as datetime,
            (SELECT open FROM price_data p2 WHERE p2.date = p1.date ORDER BY datetime LIMIT 1) as open,
            MAX(high) as high,
            MIN(low) as low,
            (SELECT close FROM price_data p2 WHERE p2.date = p1.date ORDER BY datetime DESC LIMIT 1) as close,
            SUM(volume) as volume
        FROM price_data p1
        GROUP BY date
        ORDER BY date
        """
        df = pd.read_sql_query(query_alt, conn)
    conn.close()
    return df


def calculate_stats(df):
    """Calculate trading statistics for the provided dataframe"""
    if df.empty:
        return {}

    # Basic stats
    stats = {
        "open": df["open"].iloc[0],
        "high": df["high"].max(),
        "low": df["low"].min(),
        "close": df["close"].iloc[-1],
        "range": df["high"].max() - df["low"].min(),
        "range_percent": (df["high"].max() - df["low"].min())
        / df["open"].iloc[0]
        * 100,
        "change": df["close"].iloc[-1] - df["open"].iloc[0],
        "change_percent": (df["close"].iloc[-1] - df["open"].iloc[0])
        / df["open"].iloc[0]
        * 100,
    }

    # Volatility
    if len(df) > 1:
        returns = df["close"].pct_change().dropna()
        stats["volatility"] = (
            returns.std() * np.sqrt(len(df)) * 100
        )  # Intraday volatility
    else:
        stats["volatility"] = 0

    # Calculate VWAP if volume data is available and not all zeros
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["vwap"] = (
            df["volume"] * (df["high"] + df["low"] + df["close"]) / 3
        ).cumsum() / df["volume"].cumsum()
        stats["vwap"] = df["vwap"].iloc[-1]
    else:
        stats["vwap"] = (df["high"] + df["low"] + df["close"]).mean() / 3

    return stats
