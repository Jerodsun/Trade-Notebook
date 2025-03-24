
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

