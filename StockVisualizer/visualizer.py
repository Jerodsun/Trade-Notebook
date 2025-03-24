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

from pattern_matching_ui import create_pattern_matching_tab, register_pattern_callbacks

# Set the plotly template
load_figure_template("bootstrap")

FILE_PATH = os.environ.get("SPX_DATA_FILE", "DAT_ASCII_SPXUSD_M1_202502.csv")


# Annotation system functions
def create_annotations_table(db_path="spx_data.db"):
    """Create annotations table if it doesn't exist"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create annotations table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        datetime TEXT,
        price REAL,
        text TEXT,
        arrow_direction TEXT,
        color TEXT,
        created_at TEXT
    )
    """
    )

    conn.commit()
    conn.close()


def save_annotation(annotation, db_path="spx_data.db"):
    """Save a new annotation to the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract annotation data
    date = annotation.get("date")
    datetime_str = annotation.get("datetime")
    price = annotation.get("price")
    text = annotation.get("text")
    arrow_direction = annotation.get("arrow_direction", "up")
    color = annotation.get("color", "red")
    created_at = dt.datetime.now().isoformat()

    # Insert annotation into database
    cursor.execute(
        """
    INSERT INTO annotations (date, datetime, price, text, arrow_direction, color, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (date, datetime_str, price, text, arrow_direction, color, created_at),
    )

    # Get the ID of the inserted annotation
    annotation_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return annotation_id


def get_annotations_for_date(date, db_path="spx_data.db"):
    """Get all annotations for a specific date"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
    SELECT id, date, datetime, price, text, arrow_direction, color, created_at
    FROM annotations
    WHERE date = ?
    ORDER BY datetime
    """,
        (date,),
    )

    # Fetch all annotations
    annotations = []
    for row in cursor.fetchall():
        annotations.append(
            {
                "id": row[0],
                "date": row[1],
                "datetime": row[2],
                "price": row[3],
                "text": row[4],
                "arrow_direction": row[5],
                "color": row[6],
                "created_at": row[7],
            }
        )

    conn.close()
    return annotations


def get_annotations_for_range(start_date, end_date, db_path="spx_data.db"):
    """Get all annotations for a date range"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
    SELECT id, date, datetime, price, text, arrow_direction, color, created_at
    FROM annotations
    WHERE date >= ? AND date <= ?
    ORDER BY datetime
    """,
        (start_date, end_date),
    )

    # Fetch all annotations
    annotations = []
    for row in cursor.fetchall():
        annotations.append(
            {
                "id": row[0],
                "date": row[1],
                "datetime": row[2],
                "price": row[3],
                "text": row[4],
                "arrow_direction": row[5],
                "color": row[6],
                "created_at": row[7],
            }
        )

    conn.close()
    return annotations


def delete_annotation(annotation_id, db_path="spx_data.db"):
    """Delete an annotation from the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))

    conn.commit()
    conn.close()

    return True


def update_annotation(annotation, db_path="spx_data.db"):
    """Update an existing annotation"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract annotation data
    annotation_id = annotation.get("id")
    text = annotation.get("text")
    color = annotation.get("color")
    arrow_direction = annotation.get("arrow_direction")

    # Update annotation in database
    cursor.execute(
        """
    UPDATE annotations
    SET text = ?, color = ?, arrow_direction = ?
    WHERE id = ?
    """,
        (text, color, arrow_direction, annotation_id),
    )

    conn.commit()
    conn.close()

    return True


# Function to add annotations to a figure
def add_annotations_to_figure(fig, annotations, y_domain=[0, 1]):
    """Add annotations to a plotly figure"""
    if not annotations:
        return fig

    for annotation in annotations:
        # Parse datetime and price
        datetime_str = annotation.get("datetime")
        price = annotation.get("price")
        text = annotation.get("text")
        color = annotation.get("color", "red")
        arrow_direction = annotation.get("arrow_direction", "up")

        # Set arrow properties based on direction
        if arrow_direction == "up":
            ay = -40
            arrowhead = 2
        elif arrow_direction == "down":
            ay = 40
            arrowhead = 2
        else:  # None or other
            ay = 0
            arrowhead = 0

        # Add the annotation to the figure
        fig.add_annotation(
            x=datetime_str,
            y=price,
            text=text,
            showarrow=True,
            arrowhead=arrowhead,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            ax=0,
            ay=ay,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=color,
            borderwidth=2,
            borderpad=4,
            font=dict(color=color, size=12),
        )

    return fig


def create_annotation_modal():
    """Create a modal dialog for adding/editing annotations with manual input fields"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Add Chart Annotation")),
            dbc.ModalBody(
                [
                    dbc.Form(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Annotation Text:"),
                                            dbc.Textarea(
                                                id="annotation-text",
                                                placeholder="Enter your annotation...",
                                                rows=3,
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("DateTime:"),
                                            dbc.Input(
                                                id="annotation-datetime",
                                                type="text",
                                                placeholder="Format: YYYY-MM-DD HH:MM:SS",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Price:"),
                                            dbc.Input(
                                                id="annotation-price",
                                                type="number",
                                                placeholder="4500.00",
                                                step="0.01",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ]
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Color:"),
                                            dbc.Select(
                                                id="annotation-color",
                                                options=[
                                                    {"label": "Red", "value": "red"},
                                                    {
                                                        "label": "Green",
                                                        "value": "green",
                                                    },
                                                    {"label": "Blue", "value": "blue"},
                                                    {
                                                        "label": "Orange",
                                                        "value": "orange",
                                                    },
                                                    {
                                                        "label": "Purple",
                                                        "value": "purple",
                                                    },
                                                ],
                                                value="red",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Arrow Direction:"),
                                            dbc.Select(
                                                id="annotation-arrow",
                                                options=[
                                                    {"label": "Up", "value": "up"},
                                                    {"label": "Down", "value": "down"},
                                                    {"label": "None", "value": "none"},
                                                ],
                                                value="up",
                                            ),
                                        ],
                                        width=6,
                                    ),
                                ]
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.P(
                                                "You can also click on the chart to set position automatically.",
                                                className="text-muted small",
                                            ),
                                            html.Div(
                                                id="annotation-coordinates",
                                                className="text-muted",
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button(
                        "Cancel",
                        id="cancel-annotation-button",
                        className="ms-auto",
                        color="secondary",
                    ),
                    dbc.Button("Save", id="save-annotation-button", color="primary"),
                ]
            ),
        ],
        id="annotation-modal",
        centered=True,
    )


def create_annotation_list_item(annotation):
    """Create a list item for an annotation"""
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P(
                                        f"{annotation['text']}",
                                        className="mb-0 fw-bold",
                                    ),
                                    html.Small(
                                        f"At: {annotation['datetime']} - Price: {annotation['price']}",
                                        className="text-muted",
                                    ),
                                ],
                                className="me-auto",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        html.I(className="fas fa-pencil-alt"),
                                        id={
                                            "type": "edit-annotation",
                                            "index": annotation["id"],
                                        },
                                        color="link",
                                        size="sm",
                                        className="p-0 me-2 annotation-action-button",
                                    ),
                                    dbc.Button(
                                        html.I(className="fas fa-trash"),
                                        id={
                                            "type": "delete-annotation",
                                            "index": annotation["id"],
                                        },
                                        color="link",
                                        size="sm",
                                        className="p-0 text-danger annotation-action-button annotation-delete-button",
                                    ),
                                ],
                                className="d-flex",
                            ),
                        ],
                        className="d-flex justify-content-between align-items-center",
                    ),
                ],
                className="p-2",
            )
        ],
        className="mb-2 annotation-card",
        style={"borderLeft": f"4px solid {annotation['color']}"},
    )


# Database connection functions
def create_database(file_path, db_path="spx_data.db"):
    """Create SQLite database from the Excel file with SPX data"""

    if os.path.exists(db_path):
        # Create annotations table even if the database already exists
        create_annotations_table(db_path)
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

    # Create annotations table
    create_annotations_table(db_path)

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


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css",
    ],
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
                                                                {
                                                                    "label": "Pattern Matching",
                                                                    "value": "pattern",
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
                            dcc.Graph(
                                id="price-chart",
                                style={"height": "60vh"},
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToAdd": [
                                        "drawline",
                                        "drawopenpath",
                                        "eraseshape",
                                    ],
                                    "modeBarButtonsToRemove": ["lasso2d"],
                                },
                            ),
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
                    [html.Div(id="tab-content")],
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
                    width=8,
                    className="mb-3",
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Div(
                                            [
                                                html.Span(
                                                    "Annotations", className="h5 me-2"
                                                ),
                                                dbc.Button(
                                                    html.I(className="fas fa-plus"),
                                                    id="add-annotation-button",
                                                    color="primary",
                                                    size="sm",
                                                    className="me-2",
                                                ),
                                                dbc.Button(
                                                    html.I(className="fas fa-sync-alt"),
                                                    id="refresh-annotations-button",
                                                    color="secondary",
                                                    size="sm",
                                                    className="me-2",
                                                ),
                                            ],
                                            className="d-flex align-items-center",
                                        ),
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="annotations-list",
                                            style={
                                                "maxHeight": "300px",
                                                "overflowY": "auto",
                                            },
                                            children=[],
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                    width=4,
                    className="mb-3",
                ),
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
        dcc.Store(id="annotation-data"),
        dcc.Store(id="active-annotation", data=None),
        # Add the annotation modal
        create_annotation_modal(),
    ],
    fluid=True,
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
    [
        Output("current-data", "data"),
        Output("chart-data", "data"),
        Output("annotation-data", "data"),
    ],
    [Input("date-picker", "date"), Input("view-selector", "value")],
    prevent_initial_call=True,
)
def update_data_for_view(selected_date, view_type):
    """Update the data based on the selected date and view type"""
    if selected_date is None:
        return {}, {}, []

    # Initialize df as an empty DataFrame to avoid UnboundLocalError
    df = pd.DataFrame()
    
    selected_date = dt.datetime.strptime(selected_date.split("T")[0], "%Y-%m-%d").date()

    if view_type == "day":
        # For day view, get minute data for the selected date
        df = get_data_for_date(selected_date)
        chart_title = f"SPX Minute Data - {selected_date}"

        # Get annotations for the selected date
        annotations = get_annotations_for_date(str(selected_date))
    elif view_type == "week":
        # For week view, get data for 5 trading days (or less) ending on the selected date
        end_date = selected_date
        start_date = end_date - timedelta(
            days=7
        )  # Get 7 calendar days (roughly 5 trading days)
        df = get_data_for_range(start_date, end_date)
        chart_title = f"SPX Week View - {start_date} to {end_date}"

        # Get annotations for the date range
        annotations = get_annotations_for_range(str(start_date), str(end_date))
    elif view_type == "month":
        # For month view, get data for ~21 trading days ending on the selected date
        end_date = selected_date
        start_date = end_date - timedelta(days=31)  # Approximately 1 month
        df = get_data_for_range(start_date, end_date)
        chart_title = f"SPX Month View - {start_date} to {end_date}"

        # Get annotations for the date range
        annotations = get_annotations_for_range(str(start_date), str(end_date))
    elif view_type == "pattern":
        # For pattern view, we don't need to load data here
        # The pattern matching tab will handle its own data loading
        return {}, {"title": "Pattern Matching", "view_type": "pattern"}, []

    # Convert DataFrame to dictionary for storage
    data_dict = df.to_dict("records") if not df.empty else {}

    # Create chart data dictionary
    chart_data = {"title": chart_title, "view_type": view_type}

    return data_dict, chart_data, annotations
@app.callback(
    [Output("price-chart", "figure"), Output("stats-container", "children")],
    [
        Input("current-data", "data"),
        Input("chart-data", "data"),
        Input("chart-type", "value"),
        Input("indicators", "value"),
        Input("annotation-data", "data"),
    ],
    prevent_initial_call=True,
)
def update_chart(data, chart_info, chart_type, indicators, annotations):
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

    # Create figure
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(chart_info.get("title", "SPX Price Data"),),
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

    # After creating the figure with all indicators, add annotations
    if annotations:
        fig = add_annotations_to_figure(fig, annotations)

    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
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
        # Determine change direction class
        change_class = "text-success" if stats["change"] >= 0 else "text-danger"
        change_icon = "fa fa-arrow-up" if stats["change"] >= 0 else "fa fa-arrow-down"

        # Determine VWAP comparison class
        vwap_class = (
            "text-success" if stats["close"] >= stats["vwap"] else "text-danger"
        )
        vwap_icon = (
            "fa fa-arrow-up" if stats["close"] >= stats["vwap"] else "fa fa-arrow-down"
        )

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
                    ],
                    className="h-100",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Change", className="text-center"),
                        dbc.CardBody(
                            [
                                html.P(
                                    f"{stats['change']:.2f} pts",
                                    className=f"m-0 {change_class}",
                                ),
                                html.P(
                                    [
                                        f"{stats['change_percent']:.2f}%",
                                        html.I(
                                            className=f"ms-2 {change_icon}",
                                            style={"color": "inherit"},
                                        ),
                                    ],
                                    className=f"m-0 {change_class}",
                                ),
                            ]
                        ),
                    ],
                    className="h-100",
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
                                    className="m-0 small",
                                ),
                            ]
                        ),
                    ],
                    className="h-100",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Volume", className="text-center"),
                        dbc.CardBody(
                            [
                                html.P(
                                    f"{stats['volume']:,.0f}"
                                    if "volume" in stats
                                    else "N/A",
                                    className="m-0",
                                )
                            ]
                        ),
                    ],
                    className="h-100",
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
                    ],
                    className="h-100",
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
                                            className=f"ms-2 {vwap_icon}",
                                            style={"color": "inherit"},
                                        ),
                                    ],
                                    className=f"m-0 small {vwap_class}",
                                ),
                            ]
                        ),
                    ],
                    className="h-100",
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
                    ],
                    className="h-100",
                ),
                width=6,
            ),
        ]

        stats_cards = [dbc.Row(row1_cards, className="mb-3"), dbc.Row(row2_cards)]

    return fig, stats_cards


# Open the annotation modal when the Add button is clicked
@app.callback(
    [Output("annotation-modal", "is_open"), Output("active-annotation", "data")],
    [
        Input("add-annotation-button", "n_clicks"),
        Input("cancel-annotation-button", "n_clicks"),
        Input("save-annotation-button", "n_clicks"),
        Input({"type": "edit-annotation", "index": dash.ALL}, "n_clicks"),
    ],
    [
        State("annotation-modal", "is_open"),
        State("annotation-data", "data"),
        State("active-annotation", "data"),
    ],
    prevent_initial_call=True,
)
def toggle_annotation_modal(
    add_clicks,
    cancel_clicks,
    save_clicks,
    edit_clicks,
    is_open,
    annotations,
    active_annotation,
):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, active_annotation

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Handle "Add Annotation" button
    if trigger_id == "add-annotation-button":
        return True, {}  # Use empty object instead of None to trigger pre-fill

    # Handle "Cancel" button
    elif trigger_id == "cancel-annotation-button":
        return False, None

    # Handle "Save" button
    elif trigger_id == "save-annotation-button":
        return False, active_annotation

    # Handle "Edit" button
    elif "edit-annotation" in trigger_id:
        # Parse the annotation ID from the trigger
        annotation_id = json.loads(trigger_id)["index"]

        # Find the annotation with the matching ID
        annotation = next((a for a in annotations if a["id"] == annotation_id), None)

        if annotation:
            return True, annotation

    return is_open, active_annotation


# Update the active annotation when a point is clicked on the chart
@app.callback(
    [
        Output("annotation-coordinates", "children"),
        Output("active-annotation", "data", allow_duplicate=True),
    ],
    [Input("price-chart", "clickData")],
    [
        State("active-annotation", "data"),
        State("annotation-modal", "is_open"),
        State("current-data", "data"),
    ],
    prevent_initial_call=True,
)
def update_click_data(click_data, active_annotation, modal_open, data_records):
    if not modal_open or not click_data:
        # Only process clicks when the modal is open
        raise dash.exceptions.PreventUpdate

    # Extract the point data from the click
    point_data = click_data["points"][0]
    x_value = point_data.get("x")
    y_value = point_data.get("y")

    # Convert data records back to DataFrame for reference
    df = pd.DataFrame(data_records) if data_records else pd.DataFrame()

    # Prepare the annotation data
    if df.empty:
        date_str = x_value.split("T")[0] if "T" in x_value else x_value
    else:
        # If we have actual data, get the date in the proper format
        date_str = pd.to_datetime(x_value).date().isoformat()

    # Update or create the annotation object
    if active_annotation is None:
        active_annotation = {}

    active_annotation.update({"date": date_str, "datetime": x_value, "price": y_value})

    # Display the coordinates to the user
    coords_text = f"DateTime: {x_value}, Price: {y_value:.2f}"

    return coords_text, active_annotation


# Save the annotation when the Save button is clicked
@app.callback(
    Output("annotation-data", "data", allow_duplicate=True),
    [Input("save-annotation-button", "n_clicks")],
    [
        State("active-annotation", "data"),
        State("annotation-text", "value"),
        State("annotation-color", "value"),
        State("annotation-arrow", "value"),
        State("annotation-datetime", "value"),
        State("annotation-price", "value"),
        State("date-picker", "date"),
        State("annotation-data", "data"),
    ],
    prevent_initial_call=True,
)
def save_annotation_callback(
    n_clicks,
    active_annotation,
    text,
    color,
    arrow,
    datetime_str,
    price,
    current_date,
    existing_annotations,
):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # Initialize active_annotation if it doesn't exist
    if active_annotation is None:
        active_annotation = {}

    # Check if we need to use manually entered values
    if datetime_str and price is not None:
        try:
            # Try to parse the datetime string
            dt_obj = pd.to_datetime(datetime_str)

            # Extract date part for SQLite filtering
            date_str = dt_obj.date().isoformat()

            # Update the annotation with manual values
            active_annotation.update(
                {
                    "date": date_str,
                    "datetime": dt_obj.isoformat(),
                    "price": float(price),
                }
            )
        except Exception as e:
            print(f"Error parsing manual datetime: {e}")
            # If parsing fails and we don't have click data, use the current date
            if "date" not in active_annotation:
                selected_date = pd.to_datetime(current_date.split("T")[0]).date()
                active_annotation.update(
                    {
                        "date": selected_date.isoformat(),
                        "datetime": f"{selected_date.isoformat()}T12:00:00",  # Use noon as default time
                        "price": float(price) if price is not None else 0.0,
                    }
                )
    elif (
        "date" not in active_annotation
        or "datetime" not in active_annotation
        or "price" not in active_annotation
    ):
        # If we don't have complete data, we can't save
        print("Incomplete annotation data - cannot save")
        raise dash.exceptions.PreventUpdate

    # Update the annotation with form values
    active_annotation.update({"text": text, "color": color, "arrow_direction": arrow})

    # Check if this is an edit or a new annotation
    if "id" in active_annotation:
        # This is an edit - update the existing annotation
        update_annotation(active_annotation)

        # Update the annotation in the list
        for i, ann in enumerate(existing_annotations):
            if ann["id"] == active_annotation["id"]:
                existing_annotations[i] = active_annotation
                break
    else:
        # This is a new annotation - save it to the database
        annotation_id = save_annotation(active_annotation)

        # Add the ID to the annotation and add it to the list
        active_annotation["id"] = annotation_id
        if existing_annotations is None:
            existing_annotations = []
        existing_annotations.append(active_annotation)

    return existing_annotations


# Delete an annotation when the delete button is clicked
@app.callback(
    Output("annotation-data", "data", allow_duplicate=True),
    [Input({"type": "delete-annotation", "index": dash.ALL}, "n_clicks")],
    [State("annotation-data", "data")],
    prevent_initial_call=True,
)
def delete_annotation_callback(delete_clicks, annotations):
    ctx = callback_context
    if not ctx.triggered or not annotations:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if "delete-annotation" in trigger_id:
        # Parse the annotation ID from the trigger
        annotation_id = json.loads(trigger_id)["index"]

        # Delete from database
        delete_annotation(annotation_id)

        # Remove from the list
        annotations = [a for a in annotations if a["id"] != annotation_id]

        return annotations

    raise dash.exceptions.PreventUpdate


# Update the annotations list display
@app.callback(
    Output("annotations-list", "children"),
    [Input("annotation-data", "data"), Input("refresh-annotations-button", "n_clicks")],
    prevent_initial_call=True,
)
def update_annotations_list(annotations, n_clicks):
    if not annotations:
        return [html.P("No annotations available", className="text-muted")]

    # Sort annotations by datetime
    sorted_annotations = sorted(annotations, key=lambda x: x["datetime"])

    # Create list items for each annotation
    annotation_items = [create_annotation_list_item(ann) for ann in sorted_annotations]

    return annotation_items


# Update form fields when editing an annotation
@app.callback(
    [
        Output("annotation-text", "value"),
        Output("annotation-color", "value"),
        Output("annotation-arrow", "value"),
        Output("annotation-datetime", "value"),
        Output("annotation-price", "value"),
    ],
    [Input("active-annotation", "data")],
    prevent_initial_call=True,
)
def populate_annotation_form(active_annotation):
    if not active_annotation:
        # For new annotations, set default values
        return "", "red", "up", "", None

    # For editing or when chart is clicked, populate with existing values
    datetime_val = ""
    price_val = None

    if "datetime" in active_annotation:
        # Convert ISO format to human-readable format for the input field
        try:
            dt_obj = pd.to_datetime(active_annotation["datetime"])
            datetime_val = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        except:
            datetime_val = active_annotation["datetime"]

    if "price" in active_annotation:
        price_val = active_annotation["price"]

    return (
        active_annotation.get("text", ""),
        active_annotation.get("color", "red"),
        active_annotation.get("arrow_direction", "up"),
        datetime_val,
        price_val,
    )


@app.callback(
    Output("tab-content", "children"),
    [Input("view-selector", "value")],
)
def update_tab_content(view_type):
    if view_type == "pattern":
        return create_pattern_matching_tab()
    return html.Div()  # Return empty div for other views


register_pattern_callbacks(app, db_path="spx_data.db")

if __name__ == "__main__":
    app.run_server(debug=True)
