import sqlite3
import os
from greeks import calculate_iv, calculate_greeks, get_days_to_expiration

DB_PATH = os.path.join(os.path.dirname(__file__), 'paper_trades.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn

def create_position(symbol, strike, expiration, option_type, notes=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO positions (symbol, strike, expiration, option_type, notes, status)
        VALUES (?, ?, ?, ?, ?, 'Open')
    ''', (symbol, strike, expiration, option_type, notes))
    pos_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pos_id

def add_execution(position_id, action, quantity, option_price, underlying_price, timestamp=None):
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get position details for Greeks calculation
    cursor.execute('SELECT strike, expiration, option_type FROM positions WHERE id = ?', (position_id,))
    pos = cursor.fetchone()
    
    iv, delta, gamma, theta, vega = None, None, None, None, None
    
    if pos:
        T = get_days_to_expiration(pos['expiration'])
        r = 0.045 # Assuming a 4.5% risk-free rate, can be made dynamic
        
        iv = calculate_iv(underlying_price, pos['strike'], T, r, option_price, pos['option_type'])
        greeks = calculate_greeks(underlying_price, pos['strike'], T, r, iv, pos['option_type'])
        
        delta = greeks['delta']
        gamma = greeks['gamma']
        theta = greeks['theta']
        vega = greeks['vega']

    if timestamp:
        cursor.execute('''
            INSERT INTO executions (position_id, action, quantity, option_price, underlying_price, timestamp, 
                                   implied_volatility, delta, gamma, theta, vega)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (position_id, action, quantity, option_price, underlying_price, timestamp, iv, delta, gamma, theta, vega))
    else:
        cursor.execute('''
            INSERT INTO executions (position_id, action, quantity, option_price, underlying_price, 
                                   implied_volatility, delta, gamma, theta, vega)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (position_id, action, quantity, option_price, underlying_price, iv, delta, gamma, theta, vega))
    
    # Check if position should be closed (simplistic: if net quantity is 0)
    cursor.execute('SELECT action, quantity FROM executions WHERE position_id = ?', (position_id,))
    execs = cursor.fetchall()
    net_qty = 0
    for e in execs:
        if e['action'] == 'Buy':
            net_qty += e['quantity']
        else:
            net_qty -= e['quantity']
            
    if net_qty == 0:
        cursor.execute('UPDATE positions SET status = "Closed" WHERE id = ?', (position_id,))
    else:
        cursor.execute('UPDATE positions SET status = "Open" WHERE id = ?', (position_id,))

    conn.commit()
    conn.close()

def get_open_positions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM positions WHERE status = "Open" ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_closed_positions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM positions WHERE status = "Closed" ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_executions(position_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM executions WHERE position_id = ? ORDER BY timestamp ASC', (position_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_position_notes(position_id, notes):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE positions SET notes = ? WHERE id = ?', (notes, position_id))
    conn.commit()
    conn.close()

def delete_position(position_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM positions WHERE id = ?', (position_id,))
    conn.commit()
    conn.close()
