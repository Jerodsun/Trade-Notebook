import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go

from pattern_matcher import (
    find_similar_patterns,
    create_pattern_comparison_figure,
    analyze_pattern_outcomes,
    create_outcome_figure,
)


def create_pattern_matching_tab():
    """Create a pattern matching tab for the SPX Visualizer application"""
    return dbc.Card(
        [
            dbc.CardHeader("Pattern Matching"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Target Date (First Hour Pattern):"),
                                    dcc.DatePickerSingle(
                                        id="pattern-date-picker",
                                        placeholder="Select a date",
                                        className="mb-3",
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Matching Method:"),
                                    dcc.Dropdown(
                                        id="pattern-method",
                                        options=[
                                            {
                                                "label": "Dynamic Time Warping",
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
                                        clearable=False,
                                        className="mb-3",
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Number of Matches:"),
                                    dcc.Slider(
                                        id="pattern-top-n",
                                        min=1,
                                        max=10,
                                        step=1,
                                        value=5,
                                        marks={i: str(i) for i in range(1, 11)},
                                        className="mb-3",
                                    ),
                                ],
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Analysis Period (hours forward):"),
                                    dcc.Slider(
                                        id="pattern-hours-forward",
                                        min=1,
                                        max=24,
                                        step=1,
                                        value=6,
                                        marks={i: str(i) for i in [1, 3, 6, 12, 24]},
                                        className="mb-3",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Button(
                                        "Find Similar Patterns",
                                        id="find-patterns-button",
                                        className="btn btn-primary mt-4",
                                    ),
                                ],
                                width=6,
                                className="d-flex align-items-end justify-content-end",
                            ),
                        ],
                    ),
                    dbc.Spinner(
                        dcc.Graph(
                            id="pattern-chart",
                            style={"height": "60vh"},
                        ),
                        color="primary",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Outcome Analysis"),
                                            dbc.CardBody(
                                                [
                                                    dbc.Spinner(
                                                        dcc.Graph(
                                                            id="outcome-chart",
                                                            style={"height": "40vh"},
                                                        ),
                                                        color="primary",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                width=8,
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Pattern Statistics"),
                                            dbc.CardBody(
                                                [
                                                    html.Div(id="pattern-stats-table"),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                width=4,
                            ),
                        ],
                        className="mt-3",
                    ),
                    # Store for pattern matching results
                    dcc.Store(id="pattern-results"),
                ]
            ),
        ]
    )


def register_pattern_callbacks(app, db_path="spx_data.db"):
    """Register callbacks for the pattern matching functionality"""

    # Initialize date picker when the app loads
    @app.callback(
        [
            Output("pattern-date-picker", "min_date_allowed"),
            Output("pattern-date-picker", "max_date_allowed"),
            Output("pattern-date-picker", "date"),
            Output("pattern-date-picker", "initial_visible_month"),
        ],
        [Input("available-dates", "data")],
        prevent_initial_call=True,
    )
    def initialize_pattern_date_picker(available_dates):
        if not available_dates:
            today = datetime.today()
            last_month = today - timedelta(days=30)
            return last_month, today, today, today

        dates = available_dates
        if dates:
            min_date = dates[0]
            max_date = dates[-1]
            default_date = max_date
            initial_month = max_date
        else:
            today = datetime.today()
            last_month = today - timedelta(days=30)
            min_date = last_month
            max_date = today
            default_date = today
            initial_month = today

        return min_date, max_date, default_date, initial_month

    # Find similar patterns when the button is clicked
    @app.callback(
        [
            Output("pattern-results", "data"),
            Output("pattern-chart", "figure"),
            Output("pattern-stats-table", "children"),
            Output("outcome-chart", "figure"),
        ],
        [Input("find-patterns-button", "n_clicks")],
        [
            State("pattern-date-picker", "date"),
            State("pattern-method", "value"),
            State("pattern-top-n", "value"),
            State("pattern-hours-forward", "value"),
        ],
        prevent_initial_call=True,
    )
    def find_patterns(n_clicks, pattern_date, method, top_n, hours_forward):
        if not pattern_date:
            return {}, go.Figure(), None, go.Figure()

        # Convert date string to date object
        pattern_date = pattern_date.split("T")[0]

        # Find similar patterns
        try:
            similar_patterns, pattern_df = find_similar_patterns(
                db_path, pattern_date, pattern_length=60, top_n=top_n, method=method
            )
        except Exception as e:
            print(f"Error finding patterns: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"Error: {str(e)}")
            return {}, empty_fig, f"Error: {str(e)}", empty_fig

        if not similar_patterns or pattern_df is None or pattern_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No pattern data found for the selected date")
            return {}, empty_fig, "No data available", empty_fig

        # Create comparison figure
        comparison_fig = create_pattern_comparison_figure(
            pattern_df,
            similar_patterns,
            title=f"First Hour Trading Pattern Comparison - {pattern_date}",
        )

        # Analyze outcomes
        outcomes_df = analyze_pattern_outcomes(db_path, similar_patterns, hours_forward)

        # Create outcomes table
        if outcomes_df is not None and not outcomes_df.empty:
            # Format the DataFrame for display
            display_df = outcomes_df.copy()

            # Format percentage columns
            pct_cols = ["pct_change_high", "pct_change_low", "pct_change_close"]
            for col in pct_cols:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")

            # Format price columns
            price_cols = ["subsequent_high", "subsequent_low", "subsequent_close"]
            for col in price_cols:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

            # Format similarity score
            display_df["similarity"] = display_df["similarity"].apply(
                lambda x: f"{x:.4f}"
            )

            # Create the table
            outcomes_table = dbc.Table.from_dataframe(
                display_df[
                    [
                        "date",
                        "similarity",
                        "pct_change_high",
                        "pct_change_low",
                        "pct_change_close",
                    ]
                ],
                striped=True,
                bordered=True,
                hover=True,
                className="table-sm",
            )

            # Style the table rows based on the outcome
            styled_table = html.Div(
                [
                    html.H5("Pattern Outcome Projections"),
                    html.P(
                        f"Based on {len(similar_patterns)} similar patterns, {hours_forward} hours forward",
                        className="text-muted",
                    ),
                    outcomes_table,
                ]
            )
        else:
            styled_table = html.Div("No outcome data available")

        # Create outcome projection chart
        outcome_fig = create_outcome_figure(
            pattern_df, similar_patterns, db_path, hours_forward
        )
        if outcome_fig is None:
            outcome_fig = go.Figure()
            outcome_fig.update_layout(title="No outcome data available")

        # Store results for later use
        results_data = {
            "pattern_date": pattern_date,
            "method": method,
            "top_n": top_n,
            "similar_dates": [date for date, _, _ in similar_patterns],
            "similarities": [similarity for _, similarity, _ in similar_patterns],
        }

        return results_data, comparison_fig, styled_table, outcome_fig

    # Add the pattern matching tab when view selector changes to "pattern" (we'll add this option later)
    @app.callback(
        Output("tab-content", "children", allow_duplicate=True),
        [Input("view-selector", "value")],
        prevent_initial_call=True,
    )
    def update_tab_content(view_type):
        if view_type == "pattern":
            return create_pattern_matching_tab()
        # For other view types, we'll let the existing callbacks handle it
        raise dash.exceptions.PreventUpdate
