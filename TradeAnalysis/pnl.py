import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# Create the Dash application
app = dash.Dash(__name__, title="Trading Activity Visualization")

# Define the layout
app.layout = html.Div(
    [
        html.H1(
            "Trading Activity Analysis", style={"textAlign": "center", "margin": "20px"}
        ),
        html.Div(
            [
                html.Label("Select Timeframe:"),
                dcc.RadioItems(
                    id="timeframe-selector",
                    options=[
                        {"label": "All Time", "value": "all"},
                        {"label": "6 Months", "value": "6m"},
                        {"label": "3 Months", "value": "3m"},
                        {"label": "1 Month", "value": "1m"},
                    ],
                    value="all",
                    labelStyle={"display": "inline-block", "margin-right": "20px"},
                ),
            ],
            style={"margin": "20px"},
        ),
        html.Div(
            [
                html.Label("Show Trade Types:"),
                dcc.Checklist(
                    id="trade-type-filter",
                    options=[
                        {"label": "Buys", "value": "buy"},
                        {"label": "Sells", "value": "sell"},
                        {"label": "Shorts", "value": "short"},
                        {"label": "Covers", "value": "cover"},
                        {"label": "Options", "value": "option"},
                    ],
                    value=["buy", "sell", "short", "cover", "option"],
                    labelStyle={"display": "inline-block", "margin-right": "20px"},
                ),
            ],
            style={"margin": "20px"},
        ),
        dcc.Graph(id="trading-graph", style={"height": "600px"}),
        html.Div(
            [
                html.H2("Trading Statistics", style={"textAlign": "center"}),
                html.Div(
                    id="trading-stats",
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "justifyContent": "space-around",
                    },
                ),
            ],
            style={"margin": "30px"},
        ),
    ]
)


def load_and_process_data(file_path):
    """Load and process the trading data from CSV"""
    df = pd.read_csv(file_path, skipinitialspace=True)

    # Clean up column names
    df.columns = [col.strip() for col in df.columns]

    # Convert dates to datetime
    df["Run Date"] = pd.to_datetime(df["Run Date"], errors="coerce")

    # Sort by date
    df = df.sort_values("Run Date")

    # Determine trade types
    def get_trade_type(action):
        if not isinstance(action, str):
            return "unknown"

        action = action.upper()
        if "BOUGHT" in action and "SHORT COVER" not in action:
            return "buy"
        elif "SOLD" in action and "SHORT SALE" not in action:
            return "sell"
        elif "SHORT SALE" in action:
            return "short"
        elif "SHORT COVER" in action:
            return "cover"
        else:
            return "unknown"

    df["trade_type"] = df["Action"].apply(get_trade_type)

    # Determine if it's an option trade
    def is_option(row):
        if not isinstance(row["Symbol"], str) and not isinstance(row["Action"], str):
            return False

        symbol = str(row["Symbol"]).upper() if isinstance(row["Symbol"], str) else ""
        action = str(row["Action"]).upper() if isinstance(row["Action"], str) else ""

        return (
            "CALL" in symbol or "PUT" in symbol or "CALL" in action or "PUT" in action
        )

    df["is_option"] = df.apply(is_option, axis=1)

    # Filter out non-trade transactions
    trade_types = ["buy", "sell", "short", "cover"]
    df = df[df["trade_type"].isin(trade_types)]

    # Calculate cumulative P&L
    # First, clean the Amount column to ensure it's numeric
    df["Amount ($)"] = pd.to_numeric(df["Amount ($)"], errors="coerce")

    # Calculate running sum of amounts (P&L)
    df["Cumulative P&L"] = df["Amount ($)"].cumsum()

    return df


def generate_trade_markers(df, selected_trade_types, include_options):
    """Generate scatter plot data for trade markers"""
    markers = []

    # Define colors and symbols for trade types
    trade_colors = {
        "buy": "green",
        "sell": "red",
        "short": "purple",
        "cover": "blue",
        "unknown": "gray",
    }

    trade_symbols = {
        "buy": "circle",
        "sell": "circle-open",
        "short": "triangle-down",
        "cover": "triangle-up",
        "unknown": "x",
    }

    # Filter trades based on selected types and options flag
    for trade_type in selected_trade_types:
        filtered_df = df[df["trade_type"] == trade_type]

        if not include_options and "option" not in selected_trade_types:
            filtered_df = filtered_df[~filtered_df["is_option"]]
        elif "option" in selected_trade_types and not include_options:
            filtered_df = filtered_df[filtered_df["is_option"]]

        if not filtered_df.empty:
            marker = go.Scatter(
                x=filtered_df["Run Date"],
                y=filtered_df["Cumulative P&L"],
                mode="markers",
                name=f"{trade_type.title()} Trades",
                marker=dict(
                    size=10,
                    color=trade_colors.get(trade_type, "gray"),
                    symbol=trade_symbols.get(trade_type, "circle"),
                ),
                hoverinfo="text",
                hovertext=[
                    f"Date: {row['Run Date'].strftime('%Y-%m-%d')}<br>"
                    f"Symbol: {row['Symbol']}<br>"
                    f"Action: {row['Action']}<br>"
                    f"Quantity: {row['Quantity']}<br>"
                    f"Amount: ${row['Amount ($)']:,.2f}<br>"
                    f"Net Value: ${row['Quantity'] * row['Price ($)']:,.2f}<br>"
                    f"P&L: ${row['Cumulative P&L']:,.2f}"
                    for _, row in filtered_df.iterrows()
                ],
            )
            markers.append(marker)

    return markers


@app.callback(
    [Output("trading-graph", "figure"), Output("trading-stats", "children")],
    [Input("timeframe-selector", "value"), Input("trade-type-filter", "value")],
)
def update_graph(timeframe, selected_trade_types):
    # Load the data (in a real app, you'd probably load this once and store it)
    file_path = "History_for_Account_Z09915588.csv"
    df = load_and_process_data(file_path)

    # Filter by timeframe
    if timeframe != "all":
        end_date = df["Run Date"].max()
        if timeframe == "1m":
            start_date = end_date - pd.DateOffset(months=1)
        elif timeframe == "3m":
            start_date = end_date - pd.DateOffset(months=3)
        elif timeframe == "6m":
            start_date = end_date - pd.DateOffset(months=6)
        df = df[df["Run Date"] >= start_date]

    # Create the main figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the cumulative P&L line
    fig.add_trace(
        go.Scatter(
            x=df["Run Date"],
            y=df["Cumulative P&L"],
            mode="lines",
            name="Cumulative P&L",
            line=dict(color="blue", width=2),
        )
    )

    # Determine if options are included in the selection
    include_options = "option" in selected_trade_types
    # Remove 'option' from trade types since it's a flag, not a trade type
    trade_types = [t for t in selected_trade_types if t != "option"]

    # Add trade markers
    for trace in generate_trade_markers(df, trade_types, include_options):
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        title="Trading Activity and Cumulative P&L",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        template="plotly_white",
    )

    # Add a zero reference line
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")

    # Calculate stats for the trading summary
    total_trades = len(df)
    buy_trades = len(df[df["trade_type"] == "buy"])
    sell_trades = len(df[df["trade_type"] == "sell"])
    short_trades = len(df[df["trade_type"] == "short"])
    cover_trades = len(df[df["trade_type"] == "cover"])
    option_trades = len(df[df["is_option"]])
    trading_days = len(df["Run Date"].dt.date.unique())
    avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0

    # Net P&L for the period
    net_pl = df["Amount ($)"].sum()

    # Create the stats boxes
    stats_boxes = [
        html.Div(
            [html.H3("Total Trades"), html.P(f"{total_trades}")],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [html.H3("Buy Trades"), html.P(f"{buy_trades}", style={"color": "green"})],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [html.H3("Sell Trades"), html.P(f"{sell_trades}", style={"color": "red"})],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [
                html.H3("Options Trades"),
                html.P(f"{option_trades}", style={"color": "orange"}),
            ],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [
                html.H3("Short Sales"),
                html.P(f"{short_trades}", style={"color": "purple"}),
            ],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [
                html.H3("Short Covers"),
                html.P(f"{cover_trades}", style={"color": "blue"}),
            ],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [html.H3("Trading Days"), html.P(f"{trading_days}")],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [
                html.H3("Net P&L"),
                html.P(
                    f"${net_pl:,.2f}", style={"color": "green" if net_pl > 0 else "red"}
                ),
            ],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
                "fontWeight": "bold",
            },
        ),
    ]

    return fig, stats_boxes


if __name__ == "__main__":
    app.run(debug=True)
