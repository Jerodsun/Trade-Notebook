import os
import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# Set the plotly template
load_figure_template("bootstrap")

FILE_PATH = os.environ.get("SPX_DATA_FILE", "DAT_ASCII_SPXUSD_M1_202502.csv")


# Database connection functions
def create_database(file_path, db_path="spx_data.db"):
    """Create SQLite database from the Excel file with SPX data"""

    if os.path.exists(db_path):
        return

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
        "volume": df["volume"].sum(),
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


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.title = "SPX Price History Dashboard"

# Define app layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "SPX Price History Dashboard", className="text-center my-4"
                        ),
                        html.P(
                            "Analyze SPX price data with multiple timeframes",
                            className="text-center",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Data Selection"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Select View:"),
                                                        dcc.RadioItems(
                                                            id="view-selector",
                                                            options=[
                                                                {
                                                                    "label": "Day View",
                                                                    "value": "day",
                                                                },
                                                                {
                                                                    "label": "Week View",
                                                                    "value": "week",
                                                                },
                                                                {
                                                                    "label": "Month View",
                                                                    "value": "month",
                                                                },
                                                            ],
                                                            value="day",
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Select Date:"),
                                                        html.Div(
                                                            id="date-picker-container",
                                                            children=[
                                                                dcc.DatePickerSingle(
                                                                    id="date-picker",
                                                                    placeholder="Select a date",
                                                                    className="mb-3",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Chart Type:"),
                                                        dcc.Dropdown(
                                                            id="chart-type",
                                                            options=[
                                                                {
                                                                    "label": "Candlestick",
                                                                    "value": "candle",
                                                                },
                                                                {
                                                                    "label": "OHLC",
                                                                    "value": "ohlc",
                                                                },
                                                                {
                                                                    "label": "Line (Close)",
                                                                    "value": "line",
                                                                },
                                                            ],
                                                            value="candle",
                                                            clearable=False,
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Indicators:"),
                                                        dcc.Checklist(
                                                            id="indicators",
                                                            options=[
                                                                {
                                                                    "label": "SMA",
                                                                    "value": "sma",
                                                                },
                                                                {
                                                                    "label": "EMA",
                                                                    "value": "ema",
                                                                },
                                                                {
                                                                    "label": "Bollinger Bands",
                                                                    "value": "bb",
                                                                },
                                                                {
                                                                    "label": "VWAP",
                                                                    "value": "vwap",
                                                                },
                                                            ],
                                                            value=["vwap"],
                                                            inline=True,
                                                            className="mb-3",
                                                        ),
                                                    ],
                                                    width=8,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Button(
                                                            "Load Data",
                                                            id="load-button",
                                                            className="btn btn-primary",
                                                        )
                                                    ],
                                                    width=4,
                                                    className="d-flex align-items-end",
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Spinner(
                            dcc.Graph(id="price-chart", style={"height": "60vh"}),
                            color="primary",
                        )
                    ],
                    width=12,
                    className="my-3",
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Statistics"),
                                dbc.CardBody(id="stats-container", children=[]),
                            ]
                        )
                    ],
                    width=12,
                    className="mb-3",
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [html.Div(id="db-status", className="text-center text-muted mb-3")],
                    width=12,
                )
            ]
        ),
        # Store components for holding data without callbacks
        dcc.Store(id="available-dates"),
        dcc.Store(id="current-data"),
        dcc.Store(id="chart-data"),
    ]
)


# Initialize database on startup
@app.callback(
    [
        Output("db-status", "children"),
        Output("available-dates", "data"),
        Output("date-picker", "min_date_allowed"),
        Output("date-picker", "max_date_allowed"),
        Output("date-picker", "date"),
        Output("date-picker", "initial_visible_month"),
    ],
    [Input("load-button", "n_clicks")],
    prevent_initial_call=False,
)
def initialize_data(n_clicks):
    file_path = FILE_PATH
    db_path = "spx_data.db"

    # Create database
    create_database(file_path, db_path)

    # Get available dates
    try:
        dates = get_dates(db_path)
        dates_str = [str(d) for d in dates]

        if dates:
            min_date = dates[0]
            max_date = dates[-1]
            default_date = max_date
            initial_month = max_date
            status = f"Database loaded successfully. Data available from {min_date} to {max_date}."
        else:
            # If no dates found
            today = dt.date.today()
            last_month = today - timedelta(days=30)
            min_date = last_month
            max_date = today
            default_date = today
            initial_month = today
            status = "No data found in database. Please check your file."
    except Exception as e:
        # Handle any errors
        print(f"Error loading dates: {e}")
        today = dt.date.today()
        last_month = today - timedelta(days=30)
        min_date = last_month
        max_date = today
        default_date = today
        initial_month = today
        dates_str = []
        status = f"Error loading database: {str(e)}"

    return status, dates_str, min_date, max_date, default_date, initial_month


@app.callback(
    [Output("current-data", "data"), Output("chart-data", "data")],
    [Input("date-picker", "date"), Input("view-selector", "value")],
    prevent_initial_call=True,
)
def update_data_for_view(selected_date, view_type):
    """Update the data based on the selected date and view type"""
    if selected_date is None:
        return {}, {}

    selected_date = dt.datetime.strptime(selected_date.split("T")[0], "%Y-%m-%d").date()

    if view_type == "day":
        # For day view, get minute data for the selected date
        df = get_data_for_date(selected_date)
        chart_title = f"SPX Minute Data - {selected_date}"
    elif view_type == "week":
        # For week view, get data for 5 trading days (or less) ending on the selected date
        end_date = selected_date
        start_date = end_date - timedelta(
            days=7
        )  # Get 7 calendar days (roughly 5 trading days)
        df = get_data_for_range(start_date, end_date)
        chart_title = f"SPX Week View - {start_date} to {end_date}"
    elif view_type == "month":
        # For month view, get data for ~21 trading days ending on the selected date
        end_date = selected_date
        start_date = end_date - timedelta(days=31)  # Approximately 1 month
        df = get_data_for_range(start_date, end_date)
        chart_title = f"SPX Month View - {start_date} to {end_date}"

    # Convert DataFrame to dictionary for storage
    data_dict = df.to_dict("records") if not df.empty else {}

    # Create chart data dictionary
    chart_data = {"title": chart_title, "view_type": view_type}

    return data_dict, chart_data


@app.callback(
    [Output("price-chart", "figure"), Output("stats-container", "children")],
    [
        Input("current-data", "data"),
        Input("chart-data", "data"),
        Input("chart-type", "value"),
        Input("indicators", "value"),
    ],
    prevent_initial_call=True,
)
def update_chart(data, chart_info, chart_type, indicators):
    """Update the price chart based on the selected options"""
    if not data or not chart_info:
        # Return empty figure if no data
        return go.Figure().update_layout(title="No data available"), html.Div(
            "No data available"
        )

    # Convert dictionary back to DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        return go.Figure().update_layout(
            title="No data available for the selected period"
        ), html.Div("No data available")

    # Make sure datetime is properly formatted
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2],
        subplot_titles=(chart_info.get("title", "SPX Price Data"), "Volume"),
    )

    # Add price data based on chart type
    if chart_type == "candle":
        fig.add_trace(
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
    elif chart_type == "ohlc":
        fig.add_trace(
            go.Ohlc(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
    else:  # Line chart
        fig.add_trace(
            go.Scatter(x=df["datetime"], y=df["close"], mode="lines", name="Close"),
            row=1,
            col=1,
        )

    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=df["datetime"],
            y=df["volume"],
            name="Volume",
            marker=dict(color="rgba(0, 128, 255, 0.5)"),
        ),
        row=2,
        col=1,
    )

    # Add indicators
    if indicators:
        if "sma" in indicators:
            # 20-period SMA
            df["sma20"] = df["close"].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df["sma20"],
                    mode="lines",
                    line=dict(width=1, color="blue"),
                    name="SMA (20)",
                ),
                row=1,
                col=1,
            )

        if "ema" in indicators:
            # 9-period EMA
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df["ema9"],
                    mode="lines",
                    line=dict(width=1, color="orange"),
                    name="EMA (9)",
                ),
                row=1,
                col=1,
            )

        if "bb" in indicators:
            # Bollinger Bands
            window = 20
            df["sma20"] = df["close"].rolling(window=window).mean()
            df["bb_std"] = df["close"].rolling(window=window).std()
            df["bb_upper"] = df["sma20"] + (df["bb_std"] * 2)
            df["bb_lower"] = df["sma20"] - (df["bb_std"] * 2)

            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df["bb_upper"],
                    mode="lines",
                    line=dict(width=1, color="green"),
                    name="BB Upper",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df["bb_lower"],
                    mode="lines",
                    line=dict(width=1, color="green"),
                    name="BB Lower",
                    fill="tonexty",
                    fillcolor="rgba(0, 255, 0, 0.1)",
                ),
                row=1,
                col=1,
            )

        if "vwap" in indicators:
            # VWAP - Volume Weighted Average Price
            if "volume" in df.columns and df["volume"].sum() > 0:
                df["vwap"] = (
                    df["volume"] * (df["high"] + df["low"] + df["close"]) / 3
                ).cumsum() / df["volume"].cumsum()
            else:
                # If volume is zero or missing, use a simple average instead
                df["vwap"] = ((df["high"] + df["low"] + df["close"]) / 3).cumsum() / (
                    df.index + 1
                )

            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df["vwap"],
                    mode="lines",
                    line=dict(width=1, color="purple"),
                    name="VWAP",
                ),
                row=1,
                col=1,
            )

    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white",
        hovermode="x unified",
    )

    # Use different X-axis tick formats based on view type
    if chart_info.get("view_type") == "day":
        # For intraday data, show time
        fig.update_xaxes(tickformat="%H:%M", title_text="Time", gridcolor="lightgray")
    else:
        # For multi-day views, show date
        fig.update_xaxes(tickformat="%m-%d", title_text="Date", gridcolor="lightgray")

    fig.update_yaxes(title_text="Price", gridcolor="lightgray", row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor="lightgray", row=2, col=1)

    # Calculate and display statistics
    stats = calculate_stats(df)

    # Create stats cards
    stats_cards = []
    if stats:
        # Create a row with multiple stat cards
        row1_cards = [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("OHLC", className="text-center"),
                        dbc.CardBody(
                            [
                                html.P(f"O: {stats['open']:.2f}", className="m-0"),
                                html.P(f"H: {stats['high']:.2f}", className="m-0"),
                                html.P(f"L: {stats['low']:.2f}", className="m-0"),
                                html.P(f"C: {stats['close']:.2f}", className="m-0"),
                            ]
                        ),
                    ]
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Change", className="text-center"),
                        dbc.CardBody(
                            [
                                html.P(f"{stats['change']:.2f} pts", className="m-0"),
                                html.P(
                                    [
                                        f"{stats['change_percent']:.2f}%",
                                        html.I(
                                            className=f"ms-2 fas fa-arrow-{'up' if stats['change'] >= 0 else 'down'}",
                                            style={
                                                "color": (
                                                    "green"
                                                    if stats["change"] >= 0
                                                    else "red"
                                                )
                                            },
                                        ),
                                    ],
                                    className="m-0",
                                ),
                            ]
                        ),
                    ]
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Range", className="text-center"),
                        dbc.CardBody(
                            [
                                html.P(f"{stats['range']:.2f} pts", className="m-0"),
                                html.P(
                                    f"{stats['range_percent']:.2f}% of open",
                                    className="m-0",
                                ),
                            ]
                        ),
                    ]
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Volume", className="text-center"),
                        dbc.CardBody(
                            [html.P(f"{stats['volume']:,.0f}", className="m-0")]
                        ),
                    ]
                ),
                width=3,
            ),
        ]

        row2_cards = [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Volatility", className="text-center"),
                        dbc.CardBody(
                            [html.P(f"{stats['volatility']:.2f}%", className="m-0")]
                        ),
                    ]
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("VWAP", className="text-center"),
                        dbc.CardBody(
                            [
                                html.P(f"{stats['vwap']:.2f}", className="m-0"),
                                html.P(
                                    [
                                        f"{(stats['close'] - stats['vwap']):.2f} from close",
                                        html.I(
                                            className=f"ms-2 fas fa-arrow-{'up' if stats['close'] >= stats['vwap'] else 'down'}",
                                            style={
                                                "color": (
                                                    "green"
                                                    if stats["close"] >= stats["vwap"]
                                                    else "red"
                                                )
                                            },
                                        ),
                                    ],
                                    className="m-0 small",
                                ),
                            ]
                        ),
                    ]
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Time Period", className="text-center"),
                        dbc.CardBody(
                            [
                                html.P(
                                    f"From: {df['datetime'].min().strftime('%Y-%m-%d %H:%M')}",
                                    className="m-0 small",
                                ),
                                html.P(
                                    f"To: {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}",
                                    className="m-0 small",
                                ),
                            ]
                        ),
                    ]
                ),
                width=6,
            ),
        ]

        stats_cards = [dbc.Row(row1_cards, className="mb-3"), dbc.Row(row2_cards)]

    return fig, stats_cards


if __name__ == "__main__":
    app.run_server(debug=True)
