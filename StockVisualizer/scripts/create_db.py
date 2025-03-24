import os
import pandas as pd
import sqlite3
import datetime as dt


# Database connection functions
def create_database(directory_path, db_path="spx_data_all.db"):
    """Create SQLite database from all CSV files in the directory with SPX data"""

    all_data = []

    # Iterate through all files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, header=None)
                if ";" in df.iloc[0, 0]:  # Check if first cell contains semicolons
                    split_data = df[0].str.split(";", expand=True)
                    df = split_data
                    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
                all_data.append(df)
            except Exception as csv_error:
                print(f"Error reading {file_path} as CSV: {csv_error}")
                continue

    if not all_data:
        print("No CSV files found or all files failed to read.")
        return

    # Concatenate all dataframes
    df = pd.concat(all_data, ignore_index=True)

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

    print(f"Database created successfully at {db_path}")
    return


if __name__ == "__main__":
    current_directory = os.getcwd()
    create_database("data/", "spx_data_all_temp.db")
