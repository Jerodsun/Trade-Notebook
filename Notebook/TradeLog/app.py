import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, date
import db_manager

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Paper Trade Log")

# Layout Components
def get_new_position_form():
    return dbc.Card([
        dbc.CardHeader("Log New Position"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Symbol"),
                    dbc.Input(id="pos-symbol", placeholder="e.g. SPX", type="text"),
                ], width=3),
                dbc.Col([
                    dbc.Label("Option Type"),
                    dbc.Select(id="pos-type", options=[
                        {"label": "Call", "value": "Call"},
                        {"label": "Put", "value": "Put"},
                    ], value="Call"),
                ], width=3),
                dbc.Col([
                    dbc.Label("Strike"),
                    dbc.Input(id="pos-strike", type="number", step=1),
                ], width=3),
                dbc.Col([
                    dbc.Label("Expiration"),
                    dcc.DatePickerSingle(id="pos-expiry", date=date.today()),
                ], width=3),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Action"),
                    dbc.Select(id="exec-action", options=[
                        {"label": "Buy", "value": "Buy"},
                        {"label": "Sell", "value": "Sell"},
                    ], value="Buy"),
                ], width=3),
                dbc.Col([
                    dbc.Label("Quantity"),
                    dbc.Input(id="exec-qty", type="number", min=1, value=1),
                ], width=3),
                dbc.Col([
                    dbc.Label("Option Price"),
                    dbc.Input(id="exec-opt-price", type="number", step=0.01),
                ], width=3),
                dbc.Col([
                    dbc.Label("Underlying Price"),
                    dbc.Input(id="exec-und-price", type="number", step=0.01),
                ], width=3),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Initial Notes"),
                    dbc.Textarea(id="pos-notes", placeholder="Why are you taking this trade?"),
                ])
            ], className="mb-3"),
            dbc.Button("Open Position", id="btn-open-pos", color="primary"),
            html.Div(id="open-pos-msg", className="mt-2")
        ])
    ], className="mb-4")

def get_open_positions_view():
    return html.Div([
        html.H4("Open Positions"),
        html.Div(id="open-positions-list")
    ])

# Main Layout
app.layout = dbc.Container([
    html.Div([
        html.H1("Option Paper Trading Log", className="text-center my-4"),
        html.P("Index Options Strategy Journal", className="text-center text-muted mb-5"),
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="Journal & Entry", tab_id="tab-journal", children=[
            dbc.Row([
                dbc.Col([get_new_position_form()], width=12, lg=4),
                dbc.Col([get_open_positions_view()], width=12, lg=8)
            ], className="mt-4")
        ]),
        dbc.Tab(label="P&L Dashboard & History", tab_id="tab-history", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Overview"),
                        dbc.CardBody([
                            dcc.Graph(id="pl-chart", config={'displayModeBar': False}),
                        ])
                    ], className="mt-4"),
                ], width=12),
                dbc.Col([
                    html.H4("Trade History", className="mt-5 mb-4"),
                    html.Div(id="closed-positions-list")
                ], width=12)
            ])
        ]),
    ], id="tabs", active_tab="tab-journal")
], fluid=True)

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
    prevent_initial_call=True
)
def handle_open_position(n_clicks, symbol, opt_type, strike, expiry, action, qty, opt_price, und_price, notes):
    if not all([symbol, opt_type, strike, expiry, action, qty, opt_price, und_price]):
        return dbc.Alert("Please fill all required fields.", color="danger", className="mt-3")
    
    try:
        pos_id = db_manager.create_position(symbol, strike, expiry, opt_type, notes)
        db_manager.add_execution(pos_id, action, qty, opt_price, und_price)
        return dbc.Alert(f"Position opened successfully (ID: {pos_id})", color="success", className="mt-3")
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger", className="mt-3")

@app.callback(
    [Output("open-positions-list", "children"),
     Output("closed-positions-list", "children"),
     Output("pl-chart", "figure")],
    [Input("btn-open-pos", "n_clicks"),
     Input("open-pos-msg", "children"),
     Input({"type": "btn-close-exec", "index": dash.ALL}, "n_clicks"),
     Input({"type": "btn-delete-pos", "index": dash.ALL}, "n_clicks")],
    prevent_initial_call=False
)
def update_views(n1, n2, n3, n4):
    ctx = callback_context
    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id']
        if 'btn-delete-pos' in trigger:
            import json
            # Extract position ID from trigger string like '{"index":1,"type":"btn-delete-pos"}.n_clicks'
            trigger_json = trigger.split('.n_clicks')[0]
            pos_id = json.loads(trigger_json)['index']
            db_manager.delete_position(pos_id)

    # Fetch Open Positions
    open_pos = db_manager.get_open_positions()
    open_list = []
    for pos in open_pos:
        execs = db_manager.get_executions(pos['id'])
        exec_rows = [html.Tr([
            html.Td(e['timestamp']),
            html.Td(e['action']),
            html.Td(e['quantity']),
            html.Td(f"${e['option_price']:.2f}"),
            html.Td(f"${e['underlying_price']:.2f}"),
            html.Td(f"{e['delta']:.3f}" if e['delta'] else "-"),
            html.Td(f"{e['theta']:.3f}" if e['theta'] else "-"),
        ]) for e in execs]
        
        card = dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(f"{pos['symbol']} {pos['strike']} {pos['option_type']} | Exp: {pos['expiration']}"),
                    dbc.Col(dbc.Button("Delete", id={"type": "btn-delete-pos", "index": pos['id']}, color="danger", size="sm"), width="auto")
                ], justify="between", align="center")
            ),
            dbc.CardBody([
                html.P([html.Strong("Journal: "), pos['notes']], className="mb-3"),
                dbc.Table([
                    html.Thead(html.Tr([html.Th("Time"), html.Th("Action"), html.Th("Qty"), html.Th("Opt Price"), html.Th("Und Price"), html.Th("Delta"), html.Th("Theta")])),
                    html.Tbody(exec_rows)
                ], bordered=True, size="sm", responsive=True),
                
                html.Div([
                    html.H6("Log Closing/Scaling Execution", className="mt-4 mb-3"),
                    dbc.Row([
                        dbc.Col(dbc.Select(id={"type": "close-action", "index": pos['id']}, options=[{"label": "Buy", "value": "Buy"}, {"label": "Sell", "value": "Sell"}], value="Sell" if execs[0]['action'] == 'Buy' else "Buy"), width=12, sm=2),
                        dbc.Col(dbc.Input(id={"type": "close-qty", "index": pos['id']}, type="number", value=execs[0]['quantity']), width=12, sm=2),
                        dbc.Col(dbc.Input(id={"type": "close-opt-price", "index": pos['id']}, placeholder="Opt Price", type="number", step=0.01), width=12, sm=3),
                        dbc.Col(dbc.Input(id={"type": "close-und-price", "index": pos['id']}, placeholder="Und Price", type="number", step=0.01), width=12, sm=3),
                        dbc.Col(dbc.Button("Log", id={"type": "btn-close-exec", "index": pos['id']}, color="primary", size="sm", className="w-100"), width=12, sm=2)
                    ], className="g-2")
                ], className="bg-dark p-3 rounded-3 mt-3")
            ])
        ], className="mb-4 shadow-sm")
        open_list.append(card)

    # Fetch Closed Positions
    closed_pos = db_manager.get_closed_positions()
    closed_rows = []
    total_pl = 0
    pl_data = []
    
    for pos in closed_pos:
        execs = db_manager.get_executions(pos['id'])
        # Calculate P&L for this position
        pos_pl = 0
        for e in execs:
            if e['action'] == 'Buy':
                pos_pl -= e['quantity'] * e['option_price'] * 100
            else:
                pos_pl += e['quantity'] * e['option_price'] * 100
        
        total_pl += pos_pl
        pl_data.append({'date': pos['created_at'], 'pl': pos_pl})
        
        closed_rows.append(html.Tr([
            html.Td(pos['created_at']),
            html.Td(f"{pos['symbol']} {pos['strike']} {pos['option_type']}"),
            html.Td(pos['expiration']),
            html.Td(f"${pos_pl:,.2f}", className="pl-positive" if pos_pl >= 0 else "pl-negative"),
            html.Td(pos['notes']),
            html.Td(dbc.Button("Delete", id={"type": "btn-delete-pos", "index": pos['id']}, color="danger", size="sm"))
        ]))
    
    closed_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Opened"), html.Th("Option"), html.Th("Expiry"), html.Th("P&L"), html.Th("Notes"), html.Th("Action")])),
        html.Tbody(closed_rows)
    ], bordered=True, hover=True, responsive=True)

    # Generate P&L Chart
    fig = go.Figure()
    if pl_data:
        df_pl = pd.DataFrame(pl_data).sort_values('date')
        df_pl['cumulative_pl'] = df_pl['pl'].cumsum()
        
        # Determine color based on P&L trend
        line_color = '#22c55e' if df_pl['cumulative_pl'].iloc[-1] >= 0 else '#ef4444'
        
        fig.add_trace(go.Scatter(
            x=df_pl['date'], 
            y=df_pl['cumulative_pl'], 
            mode='lines+markers', 
            name='Cumulative P&L',
            line=dict(color=line_color, width=3),
            marker=dict(size=8, color='#38bdf8'),
            fill='tozeroy',
            fillcolor=f'rgba({56 if line_color == "#22c55e" else 239}, {189 if line_color == "#22c55e" else 68}, {248 if line_color == "#22c55e" else 68}, 0.1)'
        ))
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        xaxis=dict(showgrid=True, gridcolor='#334155'),
        yaxis=dict(showgrid=True, gridcolor='#334155'),
        template="plotly_dark",
        hovermode="x unified"
    )

    return open_list, closed_table, fig

@app.callback(
    Output({"type": "btn-close-exec", "index": dash.MATCH}, "disabled"),
    Input({"type": "btn-close-exec", "index": dash.MATCH}, "n_clicks"),
    State({"type": "close-action", "index": dash.MATCH}, "value"),
    State({"type": "close-qty", "index": dash.MATCH}, "value"),
    State({"type": "close-opt-price", "index": dash.MATCH}, "value"),
    State({"type": "close-und-price", "index": dash.MATCH}, "value"),
    State({"type": "btn-close-exec", "index": dash.MATCH}, "id"),
    prevent_initial_call=True
)
def handle_close_execution(n_clicks, action, qty, opt_price, und_price, btn_id):
    if n_clicks:
        pos_id = btn_id['index']
        if all([action, qty, opt_price, und_price]):
            db_manager.add_execution(pos_id, action, qty, opt_price, und_price)
    return False

if __name__ == "__main__":
    app.run_server(debug=True)
