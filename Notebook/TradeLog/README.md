# Paper Trade Log

A position-centric trading log for paper trading index options, built with Python, Dash, and SQLite.

## Features
- **Position-Centric Tracking**: Link multiple executions (buys/sells) to a single trade idea.
- **Journaling**: Add notes to every position to track your thought process.
- **Automated Greeks**: Estimates Delta, Gamma, Theta, and Vega at the time of execution using the Black-Scholes model.
- **P&L Dashboard**: Real-time cumulative P&L chart and trade history.

## Setup
Ensure you have the required dependencies installed:
```bash
pip install dash dash-bootstrap-components pandas scipy numpy plotly
```

## Running the App
Navigate to the `PaperTradeLog` directory and run:
```bash
python app.py
```
The app will be available at `http://127.0.0.1:8050/`.

## Usage
1.  **Journal & Entry Tab**:
    *   Fill out the "Log New Position" form to start a trade.
    *   Include your initial notes and the execution details (Option Price, Underlying Price).
    *   The app will automatically calculate the IV and Greeks.
    *   Open positions appear below. Use the "Add Execution" form on an open position to scale out or close the trade.
2.  **P&L Dashboard & History Tab**:
    *   View your cumulative P&L chart.
    *   Review closed trades, their net P&L, and your journal notes.
