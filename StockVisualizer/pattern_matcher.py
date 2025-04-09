import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import pytz
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

# Database setup
DB_PATH = "spx_data_all.db"


# Function to fetch the latest data from Alpaca
def fetch_latest_data():
    """Fetch the latest SPY data from Alpaca API"""
    try:
        # Initialize the Alpaca API
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

        # Get today's date
        today = datetime.now().date()

        # Format dates - use today for both start and end to ensure we get today's data
        start_date = today.strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        # Fetch data (using SPY as a proxy for SPX)
        symbol = "SPY"
        bars = api.get_bars(symbol, TimeFrame.Minute, start=start_date).df

        # If we didn't get any data for today, try getting yesterday's data
        if bars.empty:
            # Calculate the last weekday
            if today.weekday() == 0:  # Monday
                last_weekday = today - timedelta(days=3)
            elif today.weekday() == 6:  # Sunday
                last_weekday = today - timedelta(days=2)
            else:
                last_weekday = today - timedelta(days=1)

            start_date = last_weekday.strftime("%Y-%m-%d")
            bars = api.get_bars(symbol, TimeFrame.Minute, start=start_date).df

            if bars.empty:
                print(f"No data available for {start_date} or {today}")
                return None

        # Convert UTC time to Eastern Time (US market)
        bars = bars.reset_index()
        bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")

        # Reset index to get datetime as a column
        bars = bars.reset_index()

        # Format the dataframe to match our database schema
        formatted_df = pd.DataFrame()
        formatted_df["datetime"] = bars["timestamp"]
        formatted_df["date"] = bars["timestamp"].dt.date.astype(str)
        formatted_df["open"] = bars["open"]
        formatted_df["high"] = bars["high"]
        formatted_df["low"] = bars["low"]
        formatted_df["close"] = bars["close"]
        formatted_df["volume"] = bars["volume"]

        formatted_df = filter_trading_hours(formatted_df)

        return formatted_df

    except Exception as e:
        print(f"Error fetching data from Alpaca: {e}")
        return None


# Function to store latest data in database
def store_latest_data(df):
    """Store the latest data in the SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)

        # Check if the table exists
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'"
        )
        if not cursor.fetchone():
            # Create the table if it doesn't exist
            cursor.execute(
                """
            CREATE TABLE price_data (
                datetime TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
            """
            )

        # Get the latest date in the database
        cursor.execute("SELECT MAX(date) FROM price_data")
        latest_date = cursor.fetchone()[0]

        # Filter data to only include new dates
        if latest_date:
            new_data = df[df["date"] > latest_date]
        else:
            new_data = df

        # Store the data
        if not new_data.empty:
            new_data.to_sql("price_data", conn, if_exists="append", index=False)
            print(f"Added {len(new_data)} new rows to the database")
        else:
            print("No new data to add to the database")

        conn.close()
        return True

    except Exception as e:
        print(f"Error storing data in database: {e}")
        return False


def get_dates(db_path):
    """Get list of available dates in the database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT date FROM price_data ORDER BY date"
    dates = pd.read_sql_query(query, conn)
    conn.close()
    return dates["date"].tolist()


def filter_trading_hours(df):
    """Filter dataframe to only include regular trading hours (9:30 AM to 4:00 PM Eastern)"""
    df = df.copy()
    # Convert datetime to pandas datetime if it's not already
    if not isinstance(df["datetime"].iloc[0], pd.Timestamp):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Extract time part
    df["time"] = df["datetime"].dt.time

    # Filter for regular trading hours (9:30 AM to 4:00 PM Eastern)
    market_open = pd.to_datetime("09:30:00").time()
    market_close = pd.to_datetime("16:00:00").time()

    filtered_df = df[(df["time"] >= market_open) & (df["time"] <= market_close)]

    # Drop the temporary time column
    filtered_df = filtered_df.drop("time", axis=1)

    return filtered_df


def find_similar_patterns_ml(
    db_path, pattern_date, pattern_length=60, top_n=5, live_data=None
):
    """
    Find similar patterns using machine learning techniques

    This function:
    1. Extracts the first hour of trading for the target date
    2. Extracts the first hour for all other dates
    3. Normalizes the price patterns
    4. Uses nearest neighbors to find the most similar patterns

    If live_data is provided, it will be used as a target pattern instead of querying the database
    """
    conn = sqlite3.connect(db_path)

    # Handle pattern data - either from database or live data
    if live_data is not None and not live_data.empty:
        # Use live data as pattern
        pattern_df = live_data.head(pattern_length)
        pattern_date = "Today (Live)"
    else:
        # Get pattern data for the selected date from database
        pattern_query = f"""
        SELECT datetime, open, high, low, close
        FROM price_data 
        WHERE date = '{pattern_date}'
        ORDER BY datetime
        LIMIT {pattern_length}
        """
        pattern_df = pd.read_sql_query(pattern_query, conn)
        pattern_df["datetime"] = pd.to_datetime(pattern_df["datetime"])

    if len(pattern_df) < pattern_length:
        print(f"Warning: Found only {len(pattern_df)} data points for {pattern_date}")
        if len(pattern_df) == 0:
            conn.close()
            return [], None

    # Get data for all available dates
    dates_query = "SELECT DISTINCT date FROM price_data ORDER BY date"
    dates = pd.read_sql_query(dates_query, conn)["date"].tolist()
    # Remove the pattern date from the list of dates
    # dates_query = "SELECT DISTINCT date FROM price_data WHERE date != ? ORDER BY date"
    # dates = pd.read_sql_query(dates_query, conn, params=(pattern_date,))[
    #     "date"
    # ].tolist()

    # Extract features from the pattern
    pattern_features = extract_features(pattern_df)

    # Collect features for all dates
    all_features = []
    date_dfs = []

    for date in dates:
        # Get first hour data
        query = f"""
        SELECT datetime, open, high, low, close
        FROM price_data 
        WHERE date = '{date}'
        ORDER BY datetime
        LIMIT {pattern_length}
        """

        date_df = pd.read_sql_query(query, conn)
        date_df["datetime"] = pd.to_datetime(date_df["datetime"])

        # Skip if not enough data
        if len(date_df) < pattern_length:
            continue

        # Extract features
        features = extract_features(date_df)

        all_features.append(features)
        date_dfs.append((date, date_df))

    # Convert to numpy array for ML
    X = np.array(all_features)

    # Find nearest neighbors
    if len(X) > 0:
        # Use PCA to reduce dimensionality
        pca = PCA(n_components=min(10, X.shape[1]))
        X_pca = pca.fit_transform(X)
        pattern_pca = pca.transform(pattern_features.reshape(1, -1))

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(top_n, len(X)))
        nn.fit(X_pca)

        distances, indices = nn.kneighbors(pattern_pca)

        # Get similar patterns with their similarity scores
        similar_patterns = []
        for i, idx in enumerate(indices[0]):
            date, df = date_dfs[idx]
            similarity = 1 / (
                1 + distances[0][i]
            )  # Convert distance to similarity score
            similar_patterns.append((date, similarity, df))

        conn.close()
        return similar_patterns, pattern_df

    conn.close()
    return [], pattern_df


def extract_features(df):
    """
    Extract features from a price dataframe

    Features include:
    1. Normalized prices
    2. Returns
    3. Volatility measures
    4. Technical indicators
    """
    # Normalize prices
    scaler = MinMaxScaler()
    close_norm = scaler.fit_transform(df["close"].values.reshape(-1, 1)).flatten()

    # Calculate returns
    returns = np.diff(close_norm)
    returns = np.append(0, returns)  # Add 0 for the first point

    # Calculate volatility (rolling standard deviation)
    window = 5
    vol = []
    for i in range(len(close_norm)):
        if i < window:
            vol.append(np.std(close_norm[: i + 1]))
        else:
            vol.append(np.std(close_norm[i - window + 1 : i + 1]))

    # Combine features
    features = np.concatenate([close_norm, returns, vol])

    return features


def get_full_day_data(date, db_path):
    """Get full day data for a specific date"""
    conn = sqlite3.connect(db_path)
    query = f"SELECT datetime, open, high, low, close FROM price_data WHERE date = '{date}' ORDER BY datetime"
    df = pd.read_sql_query(query, conn)
    df["datetime"] = pd.to_datetime(df["datetime"])
    conn.close()
    return df


def create_comparison_charts(
    pattern_date, similar_patterns, db_path, pattern_length=60, pattern_df_full=None
):
    """Create a figure with multiple subplots comparing the pattern day with similar days"""
    # Get full day data for pattern date (if not provided)
    if pattern_df_full is None:
        pattern_df_full = get_full_day_data(pattern_date, db_path)

    # Calculate number of rows needed
    n_rows = len(similar_patterns) + 1  # +1 for the pattern day

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,  # Changed to False to allow individual x-axis zooming
        vertical_spacing=0.05,  # Increased spacing for better separation
        subplot_titles=[f"Pattern Day: {pattern_date}"]
        + [
            f"Match {i+1}: {date} (Similarity: {similarity:.4f})"
            for i, (date, similarity, _) in enumerate(similar_patterns)
        ],
    )

    # Add candlestick chart for pattern day
    fig.add_trace(
        go.Candlestick(
            x=pattern_df_full["datetime"],
            open=pattern_df_full["open"],
            high=pattern_df_full["high"],
            low=pattern_df_full["low"],
            close=pattern_df_full["close"],
            name=f"Pattern Day: {pattern_date}",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Set y-axis range for pattern day based on its own data
    pattern_min = pattern_df_full["low"].min() * 0.999  # Add small buffer
    pattern_max = pattern_df_full["high"].max() * 1.001
    fig.update_yaxes(range=[pattern_min, pattern_max], row=1, col=1)

    # Add vertical line at the pattern length mark
    if len(pattern_df_full) > pattern_length:
        cutoff_time = pattern_df_full.iloc[pattern_length - 1]["datetime"]

        fig.add_vline(
            x=cutoff_time, line=dict(color="red", width=1, dash="dash"), row=1, col=1
        )

        # Add annotation
        fig.add_annotation(
            x=cutoff_time,
            y=pattern_df_full["high"].max(),
            text="Pattern End",
            showarrow=True,
            arrowhead=1,
            row=1,
            col=1,
        )

    # Add charts for similar days
    for i, (date, similarity, _) in enumerate(similar_patterns):
        # Get full day data
        similar_df_full = get_full_day_data(date, db_path)

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=similar_df_full["datetime"],
                open=similar_df_full["open"],
                high=similar_df_full["high"],
                low=similar_df_full["low"],
                close=similar_df_full["close"],
                name=f"Match {i+1}: {date}",
                showlegend=False,
            ),
            row=i + 2,
            col=1,
        )

        # Set y-axis range for this similar day based on its own data
        similar_min = similar_df_full["low"].min() * 0.999  # Add small buffer
        similar_max = similar_df_full["high"].max() * 1.001
        fig.update_yaxes(range=[similar_min, similar_max], row=i + 2, col=1)

        # Add vertical line at the pattern length mark
        if len(similar_df_full) > pattern_length:
            cutoff_time = similar_df_full.iloc[pattern_length - 1]["datetime"]

            fig.add_vline(
                x=cutoff_time,
                line=dict(color="red", width=1, dash="dash"),
                row=i + 2,
                col=1,
            )

            # Add annotation
            fig.add_annotation(
                x=cutoff_time,
                y=similar_df_full["high"].max(),
                text="Pattern End",
                showarrow=True,
                arrowhead=1,
                row=i + 2,
                col=1,
            )

    # Update layout
    fig.update_layout(
        height=300 * n_rows,  # Increased height per row for better visibility
        title=f"Comparison of {pattern_date} with Most Similar Trading Days (First {pattern_length} minutes)",
        template="plotly_white",
    )

    # Update y-axis titles and make each chart independently zoomable
    for i in range(1, n_rows + 1):
        fig.update_yaxes(
            title_text="Price",
            row=i,
            col=1,
            autorange=False,  # Disable autorange to keep our custom scaling
        )

        # Make each x-axis independently zoomable
        fig.update_xaxes(
            title_text="Time" if i == n_rows else "",
            row=i,
            col=1,
            rangeslider_visible=False,  # Disable rangeslider
            autorange=True,  # Allow zooming on x-axis
        )

    return fig


def create_individual_normalized_charts(
    pattern_date, similar_patterns, db_path, pattern_length=60, pattern_df_full=None
):
    """Create individual normalized charts for each pattern day"""
    # Get full day data for pattern date (if not provided)
    if pattern_df_full is None:
        pattern_df_full = get_full_day_data(pattern_date, db_path)

    # Calculate number of rows needed
    n_rows = len(similar_patterns) + 1  # +1 for the pattern day

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,  # Allow individual x-axis zooming
        vertical_spacing=0.05,
        subplot_titles=[f"Pattern Day: {pattern_date} (Normalized)"]
        + [
            f"Match {i+1}: {date} (Normalized, Similarity: {similarity:.4f})"
            for i, (date, similarity, _) in enumerate(similar_patterns)
        ],
    )

    # Normalize and add pattern day
    pattern_close_first = pattern_df_full.iloc[0]["close"]
    pattern_df_full["normalized"] = (
        pattern_df_full["close"] / pattern_close_first
    ) * 100

    # Add pattern day trace
    fig.add_trace(
        go.Scatter(
            x=pattern_df_full["datetime"],
            y=pattern_df_full["normalized"],
            mode="lines",
            name=f"Pattern Day: {pattern_date}",
            line=dict(color="black", width=2),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Set y-axis range with buffer
    y_min = pattern_df_full["normalized"].min() * 0.999
    y_max = pattern_df_full["normalized"].max() * 1.001
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)

    # Add vertical line at pattern length mark
    if len(pattern_df_full) > pattern_length:
        cutoff_time = pattern_df_full.iloc[pattern_length - 1]["datetime"]

        fig.add_vline(
            x=cutoff_time, line=dict(color="red", width=1, dash="dash"), row=1, col=1
        )

        # Add annotation
        fig.add_annotation(
            x=cutoff_time,
            y=pattern_df_full["normalized"].max(),
            text="Pattern End",
            showarrow=True,
            arrowhead=1,
            row=1,
            col=1,
        )

    # Add similar days
    colors = ["blue", "green", "red", "purple", "orange"]

    for i, (date, similarity, _) in enumerate(similar_patterns):
        color = colors[i % len(colors)]

        # Get full day data
        similar_df_full = get_full_day_data(date, db_path)

        # Normalize
        similar_close_first = similar_df_full.iloc[0]["close"]
        similar_df_full["normalized"] = (
            similar_df_full["close"] / similar_close_first
        ) * 100

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=similar_df_full["datetime"],
                y=similar_df_full["normalized"],
                mode="lines",
                name=f"Match {i+1}: {date}",
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=i + 2,
            col=1,
        )

        # Set y-axis range with buffer
        y_min = similar_df_full["normalized"].min() * 0.999
        y_max = similar_df_full["normalized"].max() * 1.001
        fig.update_yaxes(range=[y_min, y_max], row=i + 2, col=1)

        # Add vertical line at pattern length mark
        if len(similar_df_full) > pattern_length:
            cutoff_time = similar_df_full.iloc[pattern_length - 1]["datetime"]

            fig.add_vline(
                x=cutoff_time,
                line=dict(color="red", width=1, dash="dash"),
                row=i + 2,
                col=1,
            )

            # Add annotation
            fig.add_annotation(
                x=cutoff_time,
                y=similar_df_full["normalized"].max(),
                text="Pattern End",
                showarrow=True,
                arrowhead=1,
                row=i + 2,
                col=1,
            )

    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        title="Individual Normalized Price Comparisons (Base 100)",
        template="plotly_white",
    )

    # Update y-axis titles and make each chart independently zoomable
    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Price (Base 100)", row=i, col=1, autorange=False)

        # Make each x-axis independently zoomable
        fig.update_xaxes(
            title_text="Time" if i == n_rows else "",
            row=i,
            col=1,
            rangeslider_visible=False,
            autorange=True,
        )

    return fig


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.name = "SPX Pattern Matcher"
app.title = "SPX Pattern Matcher"


# Get available dates
available_dates = get_dates(DB_PATH)
print(f"Found {len(available_dates)} trading days in database")

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "SPX Historical Pattern Match", className="text-center my-4"
                        ),
                        html.P(
                            "Find similar trading days based on initial intraday price patterns",
                            className="text-center mb-4",
                        ),
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Settings"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                # Mode selector
                                                dbc.Col(
                                                    [
                                                        html.Label("Mode:"),
                                                        dcc.RadioItems(
                                                            id="mode-selector",
                                                            options=[
                                                                {
                                                                    "label": "Historical Date",
                                                                    "value": "historical",
                                                                },
                                                                {
                                                                    "label": "Live Data (Today)",
                                                                    "value": "live",
                                                                },
                                                            ],
                                                            value="historical",
                                                            inline=True,
                                                            className="mb-2",
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                # Date selector (only shown in historical mode)
                                                dbc.Col(
                                                    [
                                                        html.Label("Select Date:"),
                                                        dcc.Dropdown(
                                                            id="date-dropdown",
                                                            options=[
                                                                {
                                                                    "label": date,
                                                                    "value": date,
                                                                }
                                                                for date in available_dates
                                                            ],
                                                            value=available_dates[-1]
                                                            if available_dates
                                                            else None,
                                                            clearable=False,
                                                        ),
                                                    ],
                                                    width=4,
                                                    id="date-selector-col",
                                                ),
                                                # Pattern length
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Pattern Length (minutes):"
                                                        ),
                                                        dcc.Slider(
                                                            id="pattern-length-slider",
                                                            min=30,
                                                            max=120,
                                                            step=5,
                                                            value=60,
                                                            marks={
                                                                30: "30m",
                                                                60: "1h",
                                                                90: "1.5h",
                                                                120: "2h",
                                                            },
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                # Number of matches
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Number of Matches:"
                                                        ),
                                                        dcc.Slider(
                                                            id="num-matches-slider",
                                                            min=3,
                                                            max=10,
                                                            step=1,
                                                            value=5,
                                                            marks={
                                                                i: str(i)
                                                                for i in range(3, 11, 1)
                                                            },
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
                                                        dbc.Button(
                                                            "Refresh Live Data",
                                                            id="refresh-data-button",
                                                            color="success",
                                                            className="mt-4 w-100",
                                                        )
                                                    ],
                                                    width={"size": 3, "offset": 0},
                                                    id="refresh-button-col",
                                                    style={"display": "none"},
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            "Find Similar Patterns",
                                                            id="find-patterns-button",
                                                            color="primary",
                                                            className="mt-4 w-100",
                                                        )
                                                    ],
                                                    width={"size": 6, "offset": 3},
                                                    id="find-button-col",
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        # Status and info message
        dbc.Row(
            [
                dbc.Col(
                    [html.Div(id="status-message", className="text-center")], width=12
                )
            ],
            className="mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4(
                            "Individual Normalized Charts", className="text-center my-3"
                        ),
                        html.P(
                            "Each chart shows normalized prices (Base 100) for better trend comparison.",
                            className="text-center text-muted",
                        ),
                        dbc.Spinner(
                            dcc.Graph(
                                id="individual-normalized-charts",
                                style={"height": "auto"},
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "modeBarButtonsToAdd": [
                                        "drawline",
                                        "drawopenpath",
                                        "eraseshape",
                                    ],
                                },
                            ),
                            color="primary",
                            type="border",
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
        # Individual comparison charts
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4(
                            "Individual Daily Charts", className="text-center my-3"
                        ),
                        html.P(
                            "Each chart is independently zoomable. Use the zoom tools to examine specific areas of interest.",
                            className="text-center text-muted",
                        ),
                        dbc.Spinner(
                            dcc.Graph(
                                id="comparison-charts",
                                style={"height": "auto"},
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "modeBarButtonsToAdd": [
                                        "drawline",
                                        "drawopenpath",
                                        "eraseshape",
                                    ],
                                },
                            ),
                            color="primary",
                            type="border",
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
)


# Callback to show/hide date selector based on mode
@app.callback(
    [
        Output("date-selector-col", "style"),
        Output("refresh-button-col", "style"),
        Output("find-button-col", "width"),
    ],
    [Input("mode-selector", "value")],
)
def toggle_date_selector(mode):
    if mode == "historical":
        return ({"display": "block"}, {"display": "none"}, {"size": 6, "offset": 3})
    else:  # live mode
        return ({"display": "none"}, {"display": "block"}, {"size": 6, "offset": 0})


# Callback to refresh live data
@app.callback(
    Output("status-message", "children", allow_duplicate=True),
    [Input("refresh-data-button", "n_clicks")],
    prevent_initial_call=True,
)
def refresh_live_data(n_clicks):
    if not n_clicks:
        return ""

    try:
        # Fetch latest data from Alpaca
        latest_data = fetch_latest_data()

        if latest_data is not None and not latest_data.empty:
            # Store the data in the database
            store_latest_data(latest_data)

            return dbc.Alert(
                f"Successfully refreshed data. Found {len(latest_data)} data points.",
                color="success",
            )
        else:
            return dbc.Alert(
                "Failed to fetch live data from Alpaca. Check your API credentials and connection.",
                color="danger",
            )

    except Exception as e:
        return dbc.Alert(f"Error refreshing data: {str(e)}", color="danger")


@app.callback(
    [
        Output("comparison-charts", "figure"),
        Output("individual-normalized-charts", "figure"),
        Output("status-message", "children", allow_duplicate=True),
    ],
    [Input("find-patterns-button", "n_clicks")],
    [
        State("mode-selector", "value"),
        State("date-dropdown", "value"),
        State("pattern-length-slider", "value"),
        State("num-matches-slider", "value"),
    ],
    prevent_initial_call=True,
)
def update_charts(n_clicks, mode, selected_date, pattern_length, top_n):
    if not n_clicks:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Click 'Find Similar Patterns' to start")
        return empty_fig, empty_fig, ""

    # Empty figure in case of errors
    empty_fig = go.Figure()
    empty_fig.update_layout(title="No data available")

    try:
        # Handle mode selection
        if mode == "historical":
            if not selected_date:
                return (
                    empty_fig,
                    empty_fig,
                    dbc.Alert("Please select a date", color="warning"),
                )

            # Find similar patterns using historical data
            similar_patterns, pattern_df = find_similar_patterns_ml(
                DB_PATH, selected_date, pattern_length, top_n
            )
            pattern_date_display = selected_date

            # Create charts with historical data
            if not similar_patterns or pattern_df is None:
                return (
                    empty_fig,
                    empty_fig,
                    dbc.Alert(
                        f"No similar patterns found for {pattern_date_display}",
                        color="warning",
                    ),
                )

            # Create individual comparison charts for historical mode
            comparison_fig = create_comparison_charts(
                pattern_date_display,
                similar_patterns,
                DB_PATH,
                pattern_length,
                pattern_df,
            )

            # Create individual normalized charts for historical mode
            normalized_fig = create_individual_normalized_charts(
                pattern_date_display,
                similar_patterns,
                DB_PATH,
                pattern_length,
                pattern_df,
            )

        else:  # Live mode
            # Fetch latest data from Alpaca
            live_data = fetch_latest_data()

            if live_data is None or live_data.empty:
                return (
                    empty_fig,
                    empty_fig,
                    dbc.Alert(
                        "Failed to fetch live data from Alpaca. Check your API credentials or try again later.",
                        color="danger",
                    ),
                )

            # Add debugging info
            print(f"Live data retrieved: {len(live_data)} points")
            print(f"First few rows of live data:\n{live_data.head()}")
            print(
                f"Date range in live data: {live_data['date'].min()} to {live_data['date'].max()}"
            )

            # Filter to trading hours if needed
            live_data = filter_trading_hours(live_data)

            # Make sure we store this data in the database
            store_latest_data(live_data)

            # Find similar patterns using live data
            similar_patterns, pattern_df = find_similar_patterns_ml(
                DB_PATH, None, pattern_length, top_n, live_data=live_data
            )

            # Use today's actual date for display
            today_date = datetime.now().date().strftime("%Y-%m-%d")
            pattern_date_display = f"Today ({today_date})"

            if not similar_patterns or pattern_df is None:
                return (
                    empty_fig,
                    empty_fig,
                    dbc.Alert(
                        f"No similar patterns found for {pattern_date_display}. You may need to wait for more data points today.",
                        color="warning",
                    ),
                )

            # Create charts with live data
            comparison_fig = create_comparison_charts(
                pattern_date_display,
                similar_patterns,
                DB_PATH,
                pattern_length,
                live_data,
            )

            normalized_fig = create_individual_normalized_charts(
                pattern_date_display,
                similar_patterns,
                DB_PATH,
                pattern_length,
                live_data,
            )

        # Create success message with details
        success_message = dbc.Alert(
            f"Found {len(similar_patterns)} similar patterns to {pattern_date_display} using a {pattern_length}-minute pattern window.",
            color="success",
        )

        return comparison_fig, normalized_fig, success_message

    except Exception as e:
        import traceback

        print(f"Error finding pattern matches: {e}")
        print(traceback.format_exc())
        return (
            empty_fig,
            empty_fig,
            dbc.Alert(f"Error: {str(e)}", color="danger"),
        )


# Run app
if __name__ == "__main__":
    app.run(debug=True, port=8055)
