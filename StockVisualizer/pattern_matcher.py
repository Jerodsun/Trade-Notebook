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
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Database setup
DB_PATH = "spx_data.db"


def get_dates(db_path):
    """Get list of available dates in the database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT date FROM price_data ORDER BY date"
    dates = pd.read_sql_query(query, conn)
    conn.close()
    return dates["date"].tolist()


def find_similar_patterns_ml(db_path, pattern_date, pattern_length=60, top_n=5):
    """
    Find similar patterns using machine learning techniques

    This function:
    1. Extracts the first hour of trading for the target date
    2. Extracts the first hour for all other dates
    3. Normalizes the price patterns
    4. Uses nearest neighbors to find the most similar patterns
    """
    conn = sqlite3.connect(db_path)

    # Get pattern data for the selected date
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
    dates_query = "SELECT DISTINCT date FROM price_data WHERE date != ? ORDER BY date"
    dates = pd.read_sql_query(dates_query, conn, params=(pattern_date,))[
        "date"
    ].tolist()

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
    pattern_date, similar_patterns, db_path, pattern_length=60
):
    """Create a figure with multiple subplots comparing the pattern day with similar days"""
    # Get full day data for pattern date
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
    pattern_date, similar_patterns, db_path, pattern_length=60
):
    """Create individual normalized charts for each pattern day"""
    # Get full day data for pattern date
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
                        html.H1("SPX Pattern Finder", className="text-center my-4"),
                        html.P(
                            "Find similar trading days based on the first hour of trading",
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
                                                # Date selector
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
                                                            max=5,
                                                            step=1,
                                                            value=3,
                                                            marks={
                                                                i: str(i)
                                                                for i in range(3, 6, 1)
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
                                                            "Find Similar Patterns",
                                                            id="find-patterns-button",
                                                            color="primary",
                                                            className="mt-4 w-100",
                                                        )
                                                    ],
                                                    width={"size": 6, "offset": 3},
                                                )
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


@app.callback(
    [
        Output("comparison-charts", "figure"),
        Output("individual-normalized-charts", "figure"),  # Add this line
        Output("status-message", "children"),
    ],
    [Input("find-patterns-button", "n_clicks")],
    [
        State("date-dropdown", "value"),
        State("pattern-length-slider", "value"),
        State("num-matches-slider", "value"),
    ],
    prevent_initial_call=True,
)
def update_charts(n_clicks, selected_date, pattern_length, top_n):
    if not selected_date:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No date selected")
        return (
            empty_fig,
            empty_fig,
            empty_fig,
            dbc.Alert("Please select a date", color="warning"),
        )  # Add empty_fig

    # Status message
    status = dbc.Spinner(
        spinner_style={"width": "1rem", "height": "1rem"},
        children=[html.Span("Finding similar patterns...")],
    )

    try:
        # Find similar patterns using ML approach
        similar_patterns, pattern_df = find_similar_patterns_ml(
            DB_PATH, selected_date, pattern_length, top_n
        )

        # Empty figure in case of errors
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")

        if not similar_patterns or pattern_df is None:
            return (
                empty_fig,
                empty_fig,
                empty_fig,  # Add empty_fig
                dbc.Alert(
                    f"No similar patterns found for {selected_date}", color="warning"
                ),
            )

        # Create individual comparison charts
        comparison_fig = create_comparison_charts(
            selected_date, similar_patterns, DB_PATH, pattern_length
        )

        # Create individual normalized charts
        normalized_fig = create_individual_normalized_charts(
            selected_date, similar_patterns, DB_PATH, pattern_length
        )

        # Create success message with details
        success_message = dbc.Alert(
            f"Found {len(similar_patterns)} similar patterns to {selected_date} using a {pattern_length}-minute pattern window.",
            color="success",
        )

        return comparison_fig, normalized_fig, success_message

    except Exception as e:
        print(f"Error finding pattern matches: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return (
            empty_fig,
            empty_fig,
            empty_fig,  # Add empty_fig
            dbc.Alert(f"Error: {str(e)}", color="danger"),
        )


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
