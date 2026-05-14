import sqlite3
import os

def init_db():
    db_path = 'paper_trades.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute('PRAGMA foreign_keys = ON')

    # Create positions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strike REAL NOT NULL,
            expiration DATE NOT NULL,
            option_type TEXT NOT NULL, -- 'Call' or 'Put'
            status TEXT DEFAULT 'Open', -- 'Open' or 'Closed'
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create executions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            action TEXT NOT NULL, -- 'Buy' or 'Sell'
            quantity INTEGER NOT NULL,
            option_price REAL NOT NULL,
            underlying_price REAL NOT NULL,
            implied_volatility REAL,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            FOREIGN KEY (position_id) REFERENCES positions (id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database initialized at {os.path.abspath(db_path)}")

if __name__ == "__main__":
    init_db()
