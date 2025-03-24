import os
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# Import functions from your existing modules
from pattern_matcher import (
    find_similar_patterns,
    create_pattern_comparison_figure,
    analyze_pattern_outcomes,
    create_outcome_figure,
)

from visualizer import create_database, get_dates, get_data_for_date, calculate_stats

# Initialize the Dash app with Bootstrap
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

# Define the database path
DB_PATH = "spx_data.db"

# Check if database exists, if not create it
if not os.path.exists(DB_PATH):
    FILE_PATH = os.environ.get("SPX_DATA_FILE", "output.txt")
    if os.path.exists(FILE_PATH):
        create_database(FILE_PATH, DB_PATH)
    else:
        print(f"Warning: Data file {FILE_PATH} not found!")

# Get available dates
try:
    available_dates = get_dates(DB_PATH)
    print(f"Found {len(available_dates)} trading days in database")
except Exception as e:
    available_dates = []
    print(f"Error loading dates: {e}")

# App layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("SPX Pattern Matcher", className="text-center my-4"),
                        html.P(
                            "Find similar trading patterns based on the first hour of trading",
                            className="text-center mb-4",
                        ),
                    ]
                ),
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
                                                        html.Label(
                                                            "Similarity Method:"
                                                        ),
                                                        dbc.RadioItems(
                                                            id="similarity-method",
                                                            options=[
                                                                {
                                                                    "label": "DTW (Dynamic Time Warping)",
                                                                    "value": "dtw",
                                                                },
                                                                {
                                                                    "label": "Euclidean Distance",
                                                                    "value": "euclidean",
                                                                },
                                                                {
                                                                    "label": "Correlation",
                                                                    "value": "correlation",
                                                                },
                                                            ],
                                                            value="dtw",
                                                            inline=True,
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Projection Hours:"),
                                                        dcc.Slider(
                                                            id="projection-hours-slider",
                                                            min=1,
                                                            max=24,
                                                            step=1,
                                                            value=6,
                                                            marks={
                                                                1: "1h",
                                                                6: "6h",
                                                                12: "12h",
                                                                24: "Full Day",
                                                            },
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="mt-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            "Find Pattern Matches",
                                                            id="find-patterns-button",
                                                            color="primary",
                                                            className="mt-3 w-100",
                                                        ),
                                                    ],
                                                    width={"size": 6, "offset": 3},
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Spinner(
                            dcc.Graph(
                                id="pattern-comparison-graph", style={"height": "600px"}
                            ),
                            color="primary",
                            type="border",
                            fullscreen=False,
                        ),
                    ],
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Spinner(
                            dcc.Graph(
                                id="outcome-projection-graph", style={"height": "500px"}
                            ),
                            color="primary",
                            type="border",
                            fullscreen=False,
                        ),
                    ],
                    width=12,
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4(
                            "Pattern Match Statistics", className="text-center mb-3"
                        ),
                        html.Div(id="outcome-stats-table"),
                    ],
                    width=12,
                ),
            ]
        ),
        # Store for intermediate data
        dcc.Store(id="similar-patterns-store"),
        dcc.Store(id="pattern-data-store"),
    ],
    fluid=True,
)


# Callback to find pattern matches
@app.callback(
    [
        Output("similar-patterns-store", "data"),
        Output("pattern-data-store", "data"),
        Output("pattern-comparison-graph", "figure"),
        Output("outcome-projection-graph", "figure"),
        Output("outcome-stats-table", "children"),
    ],
    [Input("find-patterns-button", "n_clicks")],
    [
        State("date-dropdown", "value"),
        State("pattern-length-slider", "value"),
        State("num-matches-slider", "value"),
        State("similarity-method", "value"),
        State("projection-hours-slider", "value"),
    ],
    prevent_initial_call=True,
)
def update_pattern_matches(
    n_clicks, selected_date, pattern_length, top_n, method, hours_forward
):
    if not selected_date:
        return None, None, {}, {}, "Please select a date"

    # Ensure selected_date is a string in the right format (YYYY-MM-DD)
    if isinstance(selected_date, datetime):
        selected_date = selected_date.strftime("%Y-%m-%d")

    # Find similar patterns
    try:
        similar_patterns, pattern_df = find_similar_patterns(
            DB_PATH, selected_date, pattern_length, top_n, method
        )

        # Create empty figure objects in case of errors
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")

        # Store pattern data
        pattern_data = pattern_df.to_dict("records") if pattern_df is not None else []

        # Store similar patterns data (we can't store the DataFrame objects directly)
        similar_patterns_data = []
        for date, similarity, df in similar_patterns:
            similar_patterns_data.append(
                {"date": date, "similarity": similarity, "data": df.to_dict("records")}
            )

        if not similar_patterns or pattern_df is None:
            return (
                [],
                pattern_data,
                empty_fig,
                empty_fig,
                "No pattern matches found for the selected date",
            )

        # Create pattern comparison figure
        comparison_fig = create_pattern_comparison_figure(
            pattern_df,
            similar_patterns,
            title=f"Pattern Comparison for {selected_date} (First {pattern_length} Minutes)",
        )

        # Create outcome projection figure
        outcome_fig = create_outcome_figure(
            pattern_df, similar_patterns, DB_PATH, hours_forward
        )

        if outcome_fig is None:
            outcome_fig = empty_fig.update_layout(title="No outcome data available")

        # Analyze pattern outcomes
        outcomes_df = analyze_pattern_outcomes(DB_PATH, similar_patterns, hours_forward)

        # Create outcomes table
        if len(outcomes_df) > 0:
            outcomes_table = dbc.Table.from_dataframe(
                outcomes_df[
                    [
                        "date",
                        "similarity",
                        "pct_change_high",
                        "pct_change_low",
                        "pct_change_close",
                    ]
                ].round(4),
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
            )
        else:
            outcomes_table = "No outcome data available"

        return (
            similar_patterns_data,
            pattern_data,
            comparison_fig,
            outcome_fig,
            outcomes_table,
        )

    except Exception as e:
        print(f"Error finding pattern matches: {e}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return None, None, empty_fig, empty_fig, f"Error: {str(e)}"


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
