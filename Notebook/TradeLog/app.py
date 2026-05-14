import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, date
import db_manager
import price_fetcher

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Paper Trade Log"
)


# Layout Components
def get_ticker_header():
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        id="live-tickers",
                        className="d-flex justify-content-center gap-4",
                    )
                ],
                width=12,
            )
        ],
        className="mb-4",
    )


def get_new_position_form():
    return dbc.Card(
        [
            dbc.CardHeader("Log New Trade"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Symbol"),
                                    dbc.Input(
                                        id="pos-symbol",
                                        placeholder="e.g. SPX",
                                        type="text",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Strategy"),
                                    dbc.Select(
                                        id="pos-strategy",
                                        options=[
                                            {
                                                "label": "0DTE Scalp",
                                                "value": "0DTE Scalp",
                                            },
                                            {
                                                "label": "0DTE Trend",
                                                "value": "0DTE Trend",
                                            },
                                            {
                                                "label": "Day Trade",
                                                "value": "Day Trade",
                                            },
                                            {"label": "Swing", "value": "Swing"},
                                            {"label": "Lotto", "value": "Lotto"},
                                        ],
                                        value="0DTE Scalp",
                                    ),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Option Type"),
                                    dbc.Select(
                                        id="pos-type",
                                        options=[
                                            {"label": "Call", "value": "Call"},
                                            {"label": "Put", "value": "Put"},
                                        ],
                                        value="Call",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Strike Price"),
                                    dbc.Input(id="pos-strike", type="number", step=5),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Option Expiry"),
                                    dcc.DatePickerSingle(
                                        id="pos-expiry",
                                        date=date.today(),
                                        className="w-100",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Action"),
                                    dbc.Select(
                                        id="exec-action",
                                        options=[
                                            {"label": "Buy", "value": "Buy"},
                                            {"label": "Sell", "value": "Sell"},
                                        ],
                                        value="Buy",
                                    ),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Trade Date"),
                                    dcc.DatePickerSingle(
                                        id="exec-date",
                                        date=date.today(),
                                        className="w-100",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Execution Time"),
                                    dbc.Input(
                                        id="exec-time",
                                        type="text",
                                        placeholder="HH:MM:SS",
                                        value=datetime.now().strftime("%H:%M:%S"),
                                    ),
                                ],
                                width=6,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Quantity"),
                                    dbc.Input(
                                        id="exec-qty", type="number", min=1, value=1
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Fill Price"),
                                    dbc.Input(
                                        id="exec-opt-price", type="number", step=0.05
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Underlying"),
                                    dbc.Input(
                                        id="exec-und-price", type="number", step=0.01
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
                                    dbc.Label("Journal Notes"),
                                    dbc.Textarea(
                                        id="pos-notes",
                                        placeholder="Setup/Confidence/Plan...",
                                        className="journal-entry-area",
                                        style={"height": "150px"},
                                    ),
                                ]
                            )
                        ],
                        className="mb-3",
                    ),
                    dbc.Button(
                        "Log Position",
                        id="btn-open-pos",
                        color="primary",
                        className="w-100",
                    ),
                    html.Div(id="open-pos-msg", className="mt-2"),
                ]
            ),
        ],
        className="mb-4",
    )


def get_open_positions_view():
    return html.Div(
        [
            html.Div(
                [
                    html.H4("Active Positions", className="d-inline"),
                    html.Div(id="today-pl-summary", className="float-end"),
                ],
                className="mb-4",
            ),
            html.Div(id="open-positions-list"),
        ]
    )


# Main Layout
app.layout = dbc.Container(
    [
        dcc.Interval(id="ticker-interval", interval=60 * 1000, n_intervals=0),
        dcc.ConfirmDialog(
            id="confirm-delete",
            message="Are you sure you want to delete this trade? This action cannot be undone.",
        ),
        dcc.Store(id="delete-id-store"),
        html.Div(
            [
                html.H1("Q2 2026 Trading Log", className="text-center my-4"),
                html.P(
                    "jsun",
                    className="text-center text-muted mb-3",
                ),
                get_ticker_header(),
            ]
        ),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Journal & Entry",
                    tab_id="tab-journal",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col([get_new_position_form()], width=12, lg=4),
                                dbc.Col([get_open_positions_view()], width=12, lg=8),
                            ],
                            className="mt-4",
                        )
                    ],
                ),
                dbc.Tab(
                    label="P&L Dashboard",
                    tab_id="tab-dashboard",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    [
                                                        html.Span(
                                                            "Market Pulse: SPX & NDX",
                                                            className="fw-bold",
                                                        ),
                                                        dbc.RadioItems(
                                                            id="market-timeframe",
                                                            options=[
                                                                {
                                                                    "label": "Today",
                                                                    "value": "1d",
                                                                },
                                                                {
                                                                    "label": "This Week",
                                                                    "value": "5d",
                                                                },
                                                            ],
                                                            value="1d",
                                                            inline=True,
                                                            className="float-end",
                                                            inputStyle={
                                                                "margin-left": "15px",
                                                                "margin-right": "5px",
                                                            },
                                                            labelStyle={
                                                                "font-size": "0.85rem",
                                                                "color": "#64748b",
                                                            },
                                                        ),
                                                    ]
                                                ),
                                                dbc.CardBody(
                                                    [
                                                        dcc.Graph(
                                                            id="market-pulse-chart",
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="mt-4 shadow-sm",
                                        ),
                                    ],
                                    width=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader(
                                                    "Performance Overview",
                                                    className="fw-bold",
                                                ),
                                                dbc.CardBody(
                                                    [
                                                        dcc.Graph(
                                                            id="pl-chart",
                                                            config={
                                                                "displayModeBar": False
                                                            },
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="mt-4 shadow-sm",
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        )
                    ],
                ),
                dbc.Tab(
                    label="Trade History",
                    tab_id="tab-history",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            id="closed-positions-list", className="mt-4"
                                        )
                                    ],
                                    width=12,
                                )
                            ]
                        )
                    ],
                ),
            ],
            id="tabs",
            active_tab="tab-journal",
        ),
    ],
    fluid=True,
)


# Callbacks
@app.callback(
    Output("open-pos-msg", "children"),
    Input("btn-open-pos", "n_clicks"),
    State("pos-symbol", "value"),
    State("pos-type", "value"),
    State("pos-strike", "value"),
    State("pos-expiry", "date"),
    State("exec-action", "value"),
    State("exec-qty", "value"),
    State("exec-opt-price", "value"),
    State("exec-und-price", "value"),
    State("pos-notes", "value"),
    State("pos-strategy", "value"),
    State("exec-date", "date"),
    State("exec-time", "value"),
    prevent_initial_call=True,
)
def handle_open_position(
    n_clicks,
    symbol,
    opt_type,
    strike,
    expiry,
    action,
    qty,
    opt_price,
    und_price,
    notes,
    strategy,
    ex_date,
    ex_time,
):
    if not all(
        [
            symbol,
            opt_type,
            strike,
            expiry,
            action,
            qty,
            opt_price,
            und_price,
            ex_date,
            ex_time,
        ]
    ):
        return dbc.Alert(
            "Please fill all required fields.", color="danger", className="mt-3"
        )

    try:
        # Combine date and time
        timestamp = f"{ex_date} {ex_time}"
        pos_id = db_manager.create_position(
            symbol, strike, expiry, opt_type, notes, strategy
        )
        db_manager.add_execution(
            pos_id, action, qty, opt_price, und_price, timestamp=timestamp
        )
        return dbc.Alert(
            f"Position opened successfully (ID: {pos_id})",
            color="success",
            className="mt-3",
        )
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger", className="mt-3")


@app.callback(
    [Output("confirm-delete", "displayed"), Output("delete-id-store", "data")],
    Input({"type": "btn-delete-pos", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def trigger_confirm(n_clicks):
    if not any(n_clicks):
        return False, None
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"]
    import json

    trigger_json = trigger.split(".n_clicks")[0]
    pos_id = json.loads(trigger_json)["index"]
    return True, pos_id


@app.callback(
    [
        Output("open-positions-list", "children"),
        Output("closed-positions-list", "children"),
        Output("pl-chart", "figure"),
        Output("today-pl-summary", "children"),
    ],
    [
        Input("btn-open-pos", "n_clicks"),
        Input("open-pos-msg", "children"),
        Input({"type": "btn-close-exec", "index": dash.ALL}, "n_clicks"),
        Input("confirm-delete", "submit_n_clicks"),
        Input({"type": "btn-save-notes", "index": dash.ALL}, "n_clicks"),
        Input({"type": "btn-delete-exec", "index": dash.ALL}, "n_clicks"),
    ],
    [
        State("delete-id-store", "data"),
        State({"type": "journal-notes", "index": dash.ALL}, "value"),
        State({"type": "journal-notes", "index": dash.ALL}, "id"),
        State({"type": "close-action", "index": dash.ALL}, "value"),
        State({"type": "close-qty", "index": dash.ALL}, "value"),
        State({"type": "close-opt-price", "index": dash.ALL}, "value"),
        State({"type": "close-und-price", "index": dash.ALL}, "value"),
        State({"type": "close-time", "index": dash.ALL}, "value"),
        State({"type": "close-action", "index": dash.ALL}, "id"),
    ],
    prevent_initial_call=False,
)
def update_views(
    n1,
    n2,
    n3,
    n_confirm,
    n5,
    n6,
    delete_id,
    notes_values,
    notes_ids,
    close_actions,
    close_qtys,
    close_opt_prices,
    close_und_prices,
    close_times,
    close_ids,
):
    ctx = callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]["prop_id"]
        import json

        if "confirm-delete" in trigger and n_confirm:
            db_manager.delete_position(delete_id)

        elif "btn-save-notes" in trigger:
            trigger_json = trigger.split(".n_clicks")[0]
            pos_id = json.loads(trigger_json)["index"]
            for val, nid in zip(notes_values, notes_ids):
                if nid["index"] == pos_id:
                    db_manager.update_position_notes(pos_id, val)
                    break

        elif "btn-delete-exec" in trigger:
            trigger_json = trigger.split(".n_clicks")[0]
            exec_id = json.loads(trigger_json)["index"]
            db_manager.delete_execution(exec_id)

        elif "btn-close-exec" in trigger:
            trigger_json = trigger.split(".n_clicks")[0]
            pos_id = json.loads(trigger_json)["index"]
            # Find values for this pos_id
            for i, cid in enumerate(close_ids):
                if cid["index"] == pos_id:
                    action = close_actions[i]
                    qty = close_qtys[i]
                    opt_p = close_opt_prices[i]
                    und_p = close_und_prices[i]
                    ex_time = close_times[i]
                    if all([action, qty, opt_p, und_p, ex_time]):
                        timestamp = f"{date.today().strftime('%Y-%m-%d')} {ex_time}"
                        db_manager.add_execution(
                            pos_id, action, qty, opt_p, und_p, timestamp=timestamp
                        )
                    break

    # Fetch Open Positions
    open_pos = db_manager.get_open_positions()
    open_list = []
    today_pl = 0
    today_str = date.today().strftime("%Y-%m-%d")

    for pos in open_pos:
        execs = db_manager.get_executions(pos["id"])
        if not execs:
            continue

        exec_rows = [
            html.Tr(
                [
                    html.Td(pd.to_datetime(e["timestamp"]).strftime("%H:%M:%S")),
                    html.Td(e["action"]),
                    html.Td(e["quantity"]),
                    html.Td(f"${e['option_price']:.2f}"),
                    html.Td(f"${e['underlying_price']:.2f}"),
                    html.Td(f"{e['delta']:.3f}" if e["delta"] else "-"),
                    html.Td(f"{e['theta']:.3f}" if e["theta"] else "-"),
                    html.Td(
                        dbc.Button(
                            "✕",
                            id={"type": "btn-delete-exec", "index": e["id"]},
                            color="danger",
                            size="sm",
                            outline=True,
                            style={"padding": "0 5px", "line-height": "1"},
                        )
                    ),
                ]
            )
            for e in execs
        ]

        start_time = pd.to_datetime(execs[0]["timestamp"])
        held_delta = datetime.now() - start_time
        held_str = (
            f"{held_delta.seconds // 60}m {held_delta.seconds % 60}s"
            if held_delta.days == 0
            else f"{held_delta.days}d"
        )

        card = dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    (
                                        html.Span(
                                            pos["strategy"],
                                            className="strategy-badge me-2",
                                        )
                                        if pos["strategy"]
                                        else None
                                    ),
                                    html.Span(
                                        f"{pos['symbol']} {pos['strike']} {pos['option_type']} | Held: {held_str}"
                                    ),
                                ]
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Delete Position",
                                    id={"type": "btn-delete-pos", "index": pos["id"]},
                                    color="danger",
                                    size="sm",
                                ),
                                width="auto",
                            ),
                        ],
                        justify="between",
                        align="center",
                    )
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Journal & Reflections",
                                            className="form-label",
                                        ),
                                        dbc.Textarea(
                                            id={
                                                "type": "journal-notes",
                                                "index": pos["id"],
                                            },
                                            value=pos["notes"],
                                            className="journal-entry-area mb-2",
                                            placeholder="How are you feeling? What's the plan? Log your thoughts here...",
                                        ),
                                        dbc.Button(
                                            "Save Journal Entry",
                                            id={
                                                "type": "btn-save-notes",
                                                "index": pos["id"],
                                            },
                                            color="secondary",
                                            size="sm",
                                            className="mb-3",
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                        dbc.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Time"),
                                            html.Th("Action"),
                                            html.Th("Qty"),
                                            html.Th("Fill"),
                                            html.Th("Und"),
                                            html.Th("Δ"),
                                            html.Th("Θ"),
                                            html.Th(""),
                                        ]
                                    )
                                ),
                                html.Tbody(exec_rows),
                            ],
                            bordered=True,
                            size="sm",
                            responsive=True,
                            className="mt-2",
                        ),
                        html.Div(
                            [
                                html.H6(
                                    "Add Execution",
                                    className="mb-2",
                                    style={
                                        "font-size": "0.85rem",
                                        "text-transform": "uppercase",
                                        "color": "#64748b",
                                    },
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Select(
                                                id={
                                                    "type": "close-action",
                                                    "index": pos["id"],
                                                },
                                                options=[
                                                    {"label": "Buy", "value": "Buy"},
                                                    {"label": "Sell", "value": "Sell"},
                                                ],
                                                value=(
                                                    "Sell"
                                                    if execs[0]["action"] == "Buy"
                                                    else "Buy"
                                                ),
                                            ),
                                            width=12,
                                            sm=2,
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id={
                                                    "type": "close-qty",
                                                    "index": pos["id"],
                                                },
                                                type="number",
                                                value=execs[0]["quantity"],
                                            ),
                                            width=12,
                                            sm=2,
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id={
                                                    "type": "close-opt-price",
                                                    "index": pos["id"],
                                                },
                                                placeholder="Price",
                                                type="number",
                                                step=0.01,
                                            ),
                                            width=12,
                                            sm=2,
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id={
                                                    "type": "close-und-price",
                                                    "index": pos["id"],
                                                },
                                                placeholder="Und",
                                                type="number",
                                                step=0.01,
                                            ),
                                            width=12,
                                            sm=2,
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id={
                                                    "type": "close-time",
                                                    "index": pos["id"],
                                                },
                                                value=datetime.now().strftime(
                                                    "%H:%M:%S"
                                                ),
                                                type="text",
                                            ),
                                            width=12,
                                            sm=2,
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                "Log",
                                                id={
                                                    "type": "btn-close-exec",
                                                    "index": pos["id"],
                                                },
                                                color="primary",
                                                size="sm",
                                                className="w-100",
                                            ),
                                            width=12,
                                            sm=2,
                                        ),
                                    ],
                                    className="g-2",
                                ),
                            ],
                            className="p-3 rounded-3 mt-3",
                            style={
                                "background-color": "#f8fafc",
                                "border": "1px dashed #cbd5e1",
                            },
                        ),
                    ]
                ),
            ],
            className="mb-4",
        )
        open_list.append(card)

    # Fetch Closed Positions
    closed_pos = db_manager.get_closed_positions()
    closed_list = []
    total_realized_pl = 0
    pl_data = []

    for pos in closed_pos:
        execs = db_manager.get_executions(pos["id"])
        if not execs:
            continue

        # Calculate P&L for this position
        pos_pl = 0
        exec_rows = []
        for e in execs:
            if e["action"] == "Buy":
                pos_pl -= e["quantity"] * e["option_price"] * 100
            else:
                pos_pl += e["quantity"] * e["option_price"] * 100

            exec_rows.append(
                html.Tr(
                    [
                        html.Td(
                            pd.to_datetime(e["timestamp"]).strftime("%Y-%m-%d %H:%M")
                        ),
                        html.Td(e["action"]),
                        html.Td(e["quantity"]),
                        html.Td(f"${e['option_price']:.2f}"),
                        html.Td(f"${e['underlying_price']:.2f}"),
                        html.Td(f"{e['delta']:.3f}" if e["delta"] else "-"),
                        html.Td(f"{e['theta']:.3f}" if e["theta"] else "-"),
                    ]
                )
            )

        if pos["created_at"].startswith(today_str):
            today_pl += pos_pl

        total_realized_pl += pos_pl
        pl_data.append({"date": pos["created_at"], "pl": pos_pl})

        card = dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    (
                                        html.Span(
                                            pos["strategy"],
                                            className="strategy-badge me-2",
                                        )
                                        if pos["strategy"]
                                        else None
                                    ),
                                    html.Span(
                                        f"{pos['symbol']} {pos['strike']} {pos['option_type']} | Exp: {pos['expiration']}"
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    html.Span(
                                        f"P&L: ${pos_pl:,.2f}",
                                        className=f"me-3 {'pl-positive' if pos_pl >= 0 else 'pl-negative'}",
                                    ),
                                    dbc.Button(
                                        "Delete",
                                        id={
                                            "type": "btn-delete-pos",
                                            "index": pos["id"],
                                        },
                                        color="danger",
                                        size="sm",
                                        outline=True,
                                    ),
                                ],
                                width="auto",
                            ),
                        ],
                        justify="between",
                        align="center",
                    )
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Final Reflections & Analysis",
                                            className="form-label",
                                        ),
                                        dbc.Textarea(
                                            id={
                                                "type": "journal-notes",
                                                "index": pos["id"],
                                            },
                                            value=pos["notes"],
                                            className="journal-entry-area mb-2",
                                            placeholder="Add your post-trade analysis here...",
                                        ),
                                        dbc.Button(
                                            "Update Reflections",
                                            id={
                                                "type": "btn-save-notes",
                                                "index": pos["id"],
                                            },
                                            color="secondary",
                                            size="sm",
                                            className="mb-3",
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                        dbc.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Time"),
                                            html.Th("Action"),
                                            html.Th("Qty"),
                                            html.Th("Fill"),
                                            html.Th("Und"),
                                            html.Th("Δ"),
                                            html.Th("Θ"),
                                        ]
                                    )
                                ),
                                html.Tbody(exec_rows),
                            ],
                            bordered=True,
                            size="sm",
                            responsive=True,
                        ),
                    ]
                ),
            ],
            className="mb-4",
        )
        closed_list.append(card)

    today_pl_view = html.Div(
        [
            html.H5(
                [
                    "Today's Realized: ",
                    html.Span(
                        f"${today_pl:,.2f}",
                        className="pl-positive" if today_pl >= 0 else "pl-negative",
                    ),
                ],
                className="mb-0 d-inline me-4",
            ),
            html.H5(
                [
                    "Account Value: ",
                    html.Span(
                        f"${db_manager.get_starting_balance() + total_realized_pl:,.2f}",
                        className="fw-bold",
                    ),
                ],
                className="mb-0 d-inline",
            ),
        ]
    )

    # Generate P&L Chart
    fig = go.Figure()
    if pl_data:
        df_pl = pd.DataFrame(pl_data).sort_values("date")
        df_pl["cumulative_pl"] = df_pl["pl"].cumsum()

        fig.add_trace(
            go.Scatter(
                x=df_pl["date"],
                y=df_pl["cumulative_pl"],
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="#0f766e", width=3),
                marker=dict(size=6, color="#6366f1"),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.05)",
            )
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        template="plotly_white",
        hovermode="x unified",
    )

    return open_list, closed_list, fig, today_pl_view


@app.callback(
    Output("live-tickers", "children"), Input("ticker-interval", "n_intervals")
)
def update_tickers(n):
    prices = price_fetcher.get_index_prices()
    ticker_elements = []

    for symbol, data in prices.items():
        if data:
            name = "SPX" if symbol == "^SPX" else "NDX"
            color = "pl-positive" if data["change"] >= 0 else "pl-negative"
            ticker_elements.append(
                html.Div(
                    [
                        html.Span(f"{name}: ", className="text-muted fw-bold"),
                        html.Span(f"{data['price']:,.2f} ", className="fw-bold"),
                        html.Span(f"({data['percent']:+.2f}%)", className=color),
                    ]
                )
            )
        else:
            ticker_elements.append(
                html.Div(f"{symbol} Unavailable", className="text-muted")
            )

    return ticker_elements


@app.callback(
    Output("market-pulse-chart", "figure"),
    [Input("market-timeframe", "value"), Input("ticker-interval", "n_intervals")],
)
def update_market_pulse(timeframe, n):
    fig = go.Figure()
    symbols = ["^SPX", "^NDX"]
    colors = {"^SPX": "#0f766e", "^NDX": "#6366f1"}

    # Store history for day markers
    main_history = None

    for symbol in symbols:
        history = price_fetcher.get_index_history(symbol, timeframe)
        if history:
            if not main_history:
                main_history = history
            # Normalize to % change from start of period
            start_price = history["prices"][0]
            pct_changes = [
                (p - start_price) / start_price * 100 for p in history["prices"]
            ]

            name = "SPX" if symbol == "^SPX" else "NDX"
            fig.add_trace(
                go.Scatter(
                    x=history["times"],
                    y=pct_changes,
                    mode="lines",
                    name=name,
                    line=dict(color=colors[symbol], width=2),
                )
            )

    # Add day markers for 5d view
    if timeframe == "5d" and main_history:
        # Find the points where the date changes
        dates = [t.date() for t in main_history["times"]]
        for i in range(1, len(dates)):
            if dates[i] != dates[i - 1]:
                # Draw a vertical line at the start of the new day
                fig.add_vline(
                    x=main_history["times"][i],
                    line_width=1,
                    line_dash="dot",
                    line_color="#cbd5e1",
                )

    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(
                    bounds=[16, 9.5], pattern="hour"
                ),  # hide non-trading hours (4pm to 9:30am)
            ],
        ),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="% Change"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


if __name__ == "__main__":
    app.run_server() # (debug=True)
