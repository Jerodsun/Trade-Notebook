import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler


def find_similar_patterns(
    db_path, pattern_date, pattern_length=60, top_n=5, method="dtw"
):
    """
    Find the most similar patterns to a given pattern in SPX time series data.

    Parameters:
    -----------
    db_path : str
        Path to the SQLite database containing SPX data
    pattern_date : str or datetime
        The date of the pattern to match (will use first hour of trading)
    pattern_length : int
        Length of the pattern in minutes (default: 60 for first hour)
    top_n : int
        Number of similar patterns to return
    method : str
        Similarity method: 'euclidean', 'correlation', or 'dtw'

    Returns:
    --------
    similar_patterns : list of tuples
        List of (date, similarity_score, pattern_df) for the most similar patterns
    pattern_df : DataFrame
        The pattern that was matched against
    """
    conn = sqlite3.connect(db_path)

    # Convert pattern_date to string format if it's a datetime
    if isinstance(pattern_date, datetime):
        pattern_date = pattern_date.strftime("%Y-%m-%d")

    # Get pattern data (first hour of trading for the specified date)
    pattern_query = f"""
    SELECT datetime, open, high, low, close
    FROM price_data 
    WHERE date = '{pattern_date}'
    ORDER BY datetime
    LIMIT {pattern_length}
    """

    pattern_df = pd.read_sql_query(pattern_query, conn)
    pattern_df["datetime"] = pd.to_datetime(pattern_df["datetime"])

    if len(pattern_df) < pattern_length:
        print(
            f"Warning: Found only {len(pattern_df)} data points for {pattern_date}, expected {pattern_length}"
        )
        if len(pattern_df) == 0:
            conn.close()
            return [], None

    # Get all available dates except the pattern date
    dates_query = "SELECT DISTINCT date FROM price_data WHERE date != ? ORDER BY date"
    dates = pd.read_sql_query(dates_query, conn, params=(pattern_date,))[
        "date"
    ].tolist()

    # Extract pattern values (we'll use close price for comparison)
    pattern_values = pattern_df["close"].values

    # Normalize the pattern
    scaler = MinMaxScaler()
    pattern_normalized = scaler.fit_transform(pattern_values.reshape(-1, 1)).flatten()

    # Store similarity scores
    similarity_scores = []

    # For each date, compare its first hour pattern with our target pattern
    for date in dates:
        # Get first hour data for this date
        query = f"""
        SELECT datetime, open, high, low, close
        FROM price_data 
        WHERE date = '{date}'
        ORDER BY datetime
        LIMIT {pattern_length}
        """

        date_df = pd.read_sql_query(query, conn)

        # Skip if we don't have enough data points
        if len(date_df) < pattern_length:
            continue

        date_df["datetime"] = pd.to_datetime(date_df["datetime"])

        # Extract and normalize the comparison window
        comp_values = date_df["close"].values
        comp_normalized = scaler.fit_transform(comp_values.reshape(-1, 1)).flatten()

        # Calculate similarity based on chosen method
        if method == "dtw":
            # For DTW we'll use a simplified version without the full DTW algorithm
            # since it's computationally expensive
            try:
                from fastdtw import fastdtw

                pattern_normalized = pattern_normalized.flatten()
                comp_normalized = comp_normalized.flatten()
                distance, _ = fastdtw(pattern_normalized, comp_normalized)
                similarity = 1 / (1 + distance)
            except ImportError:
                # Fall back to Euclidean if fastdtw is not installed
                distance = euclidean(pattern_normalized, comp_normalized)
                similarity = 1 / (1 + distance)
        if method == "euclidean":
            distance = euclidean(pattern_normalized, comp_normalized)
            similarity = 1 / (1 + distance)
        elif method == "correlation":
            if np.std(pattern_normalized) == 0 or np.std(comp_normalized) == 0:
                similarity = 0  # Avoid division by zero for flat patterns
            else:
                similarity = np.corrcoef(pattern_normalized, comp_normalized)[0, 1]
                # Handle NaN values
                if np.isnan(similarity):
                    similarity = 0

        # Store the date, similarity score, and dataframe
        similarity_scores.append((date, similarity, date_df))

    conn.close()

    # Sort by similarity (higher is better)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Return the top N similar patterns
    return similarity_scores[:top_n], pattern_df


def create_pattern_comparison_figure(
    pattern_df, similar_patterns, title="Pattern Comparison"
):
    """
    Create a plotly figure comparing the pattern with similar patterns.

    Parameters:
    -----------
    pattern_df : pandas DataFrame
        DataFrame containing the pattern data
    similar_patterns : list of tuples
        List of (date, similarity_score, pattern_df) from find_similar_patterns
    title : str
        Title for the figure

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The comparison figure
    """
    # Create figure
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(title, "Normalized Comparison"),
        row_heights=[0.7, 0.3],
    )

    # Add original pattern
    fig.add_trace(
        go.Candlestick(
            x=pattern_df["datetime"],
            open=pattern_df["open"],
            high=pattern_df["high"],
            low=pattern_df["low"],
            close=pattern_df["close"],
            name=f"Original Pattern",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Add normalized comparison traces
    # Normalize original pattern for comparison
    scaler = MinMaxScaler()
    pattern_close = pattern_df["close"].values
    pattern_normalized = scaler.fit_transform(pattern_close.reshape(-1, 1)).flatten()

    # Add normalized original pattern
    fig.add_trace(
        go.Scatter(
            x=pattern_df["datetime"],
            y=pattern_normalized,
            mode="lines",
            name=f"Original (Norm)",
            line=dict(color="black", width=2),
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    # Add similar patterns
    colors = ["blue", "green", "red", "purple", "orange"]

    for i, (date, similarity, similar_df) in enumerate(similar_patterns):
        color = colors[i % len(colors)]

        # Add OHLC for similar pattern
        fig.add_trace(
            go.Candlestick(
                x=similar_df["datetime"],
                open=similar_df["open"],
                high=similar_df["high"],
                low=similar_df["low"],
                close=similar_df["close"],
                name=f"Match {i+1}: {date} ({similarity:.4f})",
                showlegend=True,
                increasing=dict(line=dict(color=color)),
                decreasing=dict(line=dict(color=color)),
                visible="legendonly",  # Hide by default, toggle from legend
            ),
            row=1,
            col=1,
        )

        # Add normalized comparison
        similar_close = similar_df["close"].values
        similar_normalized = scaler.fit_transform(
            similar_close.reshape(-1, 1)
        ).flatten()

        fig.add_trace(
            go.Scatter(
                x=similar_df["datetime"],
                y=similar_normalized,
                mode="lines",
                name=f"Match {i+1} (Norm)",
                line=dict(color=color),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Normalized", row=2, col=1)

    return fig


def analyze_pattern_outcomes(db_path, similar_patterns, hours_forward=6):
    """
    Analyze what typically happens after the matched patterns.

    Parameters:
    -----------
    db_path : str
        Path to the SQLite database
    similar_patterns : list of tuples
        List of (date, similarity_score, pattern_df) from find_similar_patterns
    hours_forward : int
        Number of hours after the pattern to analyze

    Returns:
    --------
    outcomes_df : pandas DataFrame
        DataFrame with outcome statistics
    """
    conn = sqlite3.connect(db_path)

    outcomes = []

    # For each similar pattern, get the subsequent price action
    for date, similarity, pattern_df in similar_patterns:
        # Get the last datetime in the pattern
        last_datetime = pattern_df["datetime"].max()

        # Calculate end datetime (pattern end + hours_forward)
        end_datetime = pd.to_datetime(last_datetime) + pd.Timedelta(hours=hours_forward)

        # Format datetimes for SQL query
        last_datetime_str = last_datetime.strftime("%Y-%m-%d %H:%M:%S")
        end_datetime_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Query for subsequent price action
        query = f"""
        SELECT datetime, open, high, low, close
        FROM price_data 
        WHERE datetime > '{last_datetime_str}' AND datetime <= '{end_datetime_str}'
        ORDER BY datetime
        """

        subsequent_df = pd.read_sql_query(query, conn)

        if len(subsequent_df) == 0:
            continue
        subsequent_df["datetime"] = pd.to_datetime(subsequent_df["datetime"])

        # Get the pattern's last close price
        pattern_close = pattern_df["close"].iloc[-1]

        # Calculate metrics
        subsequent_high = subsequent_df["high"].max()
        subsequent_low = subsequent_df["low"].min()
        subsequent_close = subsequent_df["close"].iloc[-1]

        # Calculate percentage changes
        pct_change_high = (subsequent_high - pattern_close) / pattern_close * 100
        pct_change_low = (subsequent_low - pattern_close) / pattern_close * 100
        pct_change_close = (subsequent_close - pattern_close) / pattern_close * 100

        # Store outcome data
        outcomes.append(
            {
                "date": date,
                "similarity": similarity,
                "subsequent_high": subsequent_high,
                "subsequent_low": subsequent_low,
                "subsequent_close": subsequent_close,
                "pct_change_high": pct_change_high,
                "pct_change_low": pct_change_low,
                "pct_change_close": pct_change_close,
            }
        )

    conn.close()

    # Convert to DataFrame
    outcomes_df = pd.DataFrame(outcomes)

    # Calculate summary statistics
    if len(outcomes_df) > 0:
        outcomes_df = outcomes_df.sort_values("similarity", ascending=False)

        # Add weighted average row
        weights = outcomes_df["similarity"] / outcomes_df["similarity"].sum()
        weighted_averages = {
            "date": "Weighted Avg",
            "similarity": outcomes_df["similarity"].mean(),
            "subsequent_high": (outcomes_df["subsequent_high"] * weights).sum(),
            "subsequent_low": (outcomes_df["subsequent_low"] * weights).sum(),
            "subsequent_close": (outcomes_df["subsequent_close"] * weights).sum(),
            "pct_change_high": (outcomes_df["pct_change_high"] * weights).sum(),
            "pct_change_low": (outcomes_df["pct_change_low"] * weights).sum(),
            "pct_change_close": (outcomes_df["pct_change_close"] * weights).sum(),
        }

        # Add simple average row
        simple_averages = {
            "date": "Simple Avg",
            "similarity": outcomes_df["similarity"].mean(),
            "subsequent_high": outcomes_df["subsequent_high"].mean(),
            "subsequent_low": outcomes_df["subsequent_low"].mean(),
            "subsequent_close": outcomes_df["subsequent_close"].mean(),
            "pct_change_high": outcomes_df["pct_change_high"].mean(),
            "pct_change_low": outcomes_df["pct_change_low"].mean(),
            "pct_change_close": outcomes_df["pct_change_close"].mean(),
        }

        # Append summary rows
        outcomes_df = pd.concat(
            [outcomes_df, pd.DataFrame([weighted_averages, simple_averages])],
            ignore_index=True,
        )

    return outcomes_df


def create_outcome_figure(pattern_df, similar_patterns, db_path, hours_forward=6):
    """
    Create a figure showing the average outcome after the matched patterns.

    Parameters:
    -----------
    pattern_df : pandas DataFrame
        DataFrame containing the pattern data
    similar_patterns : list of tuples
        List of (date, similarity_score, pattern_df) from find_similar_patterns
    db_path : str
        Path to the SQLite database
    hours_forward : int
        Number of hours after the pattern to analyze

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The outcome projection figure
    """
    conn = sqlite3.connect(db_path)

    # Get the pattern's end datetime and close price
    pattern_end_datetime = pattern_df["datetime"].max()
    pattern_close = pattern_df["close"].iloc[-1]

    # List to store all subsequent price data
    all_subsequent_data = []

    # For each similar pattern, get the subsequent price action
    for date, similarity, similar_df in similar_patterns:
        # Get the last datetime in the pattern
        last_datetime = similar_df["datetime"].max()

        # Calculate end datetime
        end_datetime = pd.to_datetime(last_datetime) + pd.Timedelta(hours=hours_forward)

        # Format datetimes for SQL query
        last_datetime_str = last_datetime.strftime("%Y-%m-%d %H:%M:%S")
        end_datetime_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Query for subsequent price action
        query = f"""
        SELECT datetime, open, high, low, close
        FROM price_data 
        WHERE datetime > '{last_datetime_str}' AND datetime <= '{end_datetime_str}'
        ORDER BY datetime
        """

        subsequent_df = pd.read_sql_query(query, conn)

        if len(subsequent_df) == 0:
            continue
        subsequent_df["datetime"] = pd.to_datetime(subsequent_df["datetime"])

        # Get the pattern's last close price
        similar_close = similar_df["close"].iloc[-1]

        # Normalize the subsequent data relative to the pattern's close price
        subsequent_df["minutes_from_end"] = (
            subsequent_df["datetime"] - pd.to_datetime(last_datetime)
        ).dt.total_seconds() / 60
        subsequent_df["normalized_open"] = (
            subsequent_df["open"] / similar_close * pattern_close
        )
        subsequent_df["normalized_high"] = (
            subsequent_df["high"] / similar_close * pattern_close
        )
        subsequent_df["normalized_low"] = (
            subsequent_df["low"] / similar_close * pattern_close
        )
        subsequent_df["normalized_close"] = (
            subsequent_df["close"] / similar_close * pattern_close
        )

        # Add date and similarity information
        subsequent_df["date"] = date
        subsequent_df["similarity"] = similarity

        all_subsequent_data.append(subsequent_df)

    conn.close()

    if not all_subsequent_data:
        # No subsequent data available
        return None

    # Combine all subsequent data
    combined_df = pd.concat(all_subsequent_data)

    # Create time bins for aggregating data (e.g., every 15 minutes)
    bin_size = 15  # minutes
    combined_df["time_bin"] = (combined_df["minutes_from_end"] / bin_size).astype(
        int
    ) * bin_size

    # Calculate weighted averages for each time bin
    weights = combined_df["similarity"] / combined_df["similarity"].sum()
    weighted_df = combined_df.copy()
    weighted_df["weighted_normalized_close"] = (
        weighted_df["normalized_close"] * weighted_df["similarity"]
    )

    # Group by time bin and calculate statistics
    aggregated = weighted_df.groupby("time_bin").agg(
        {
            "weighted_normalized_close": "sum",
            "similarity": "sum",
            "normalized_high": "max",
            "normalized_low": "min",
            "normalized_close": ["mean", "std"],
        }
    )

    # Flatten multi-level columns
    aggregated.columns = ["_".join(col).strip() for col in aggregated.columns.values]

    # Calculate weighted average
    aggregated["weighted_avg_close"] = (
        aggregated["weighted_normalized_close"] / aggregated["similarity_sum"]
    )

    # Calculate upper and lower bands (1 standard deviation)
    aggregated["upper_band"] = (
        aggregated["normalized_close_mean"] + aggregated["normalized_close_std"]
    )
    aggregated["lower_band"] = (
        aggregated["normalized_close_mean"] - aggregated["normalized_close_std"]
    )

    # Create time values for x-axis
    time_values = [
        pattern_end_datetime + pd.Timedelta(minutes=int(t)) for t in aggregated.index
    ]

    # Create the figure
    fig = go.Figure()

    # Add the original pattern's close price
    fig.add_trace(
        go.Scatter(
            x=[pattern_end_datetime],
            y=[pattern_close],
            mode="markers",
            marker=dict(size=10, color="black"),
            name="Pattern End",
        )
    )

    # Add the weighted average projection
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=aggregated["weighted_avg_close"],
            mode="lines",
            line=dict(color="blue", width=2),
            name="Weighted Avg Projection",
        )
    )

    # Add the simple average projection
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=aggregated["normalized_close_mean"],
            mode="lines",
            line=dict(color="green", width=2),
            name="Average Projection",
        )
    )

    # Add the high/low range
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=aggregated["normalized_high_max"],
            mode="lines",
            line=dict(color="red", width=1, dash="dash"),
            name="High Range",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=aggregated["normalized_low_min"],
            mode="lines",
            line=dict(color="red", width=1, dash="dash"),
            name="Low Range",
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.1)",
        )
    )

    # Add uncertainty bands (1 standard deviation)
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=aggregated["upper_band"],
            mode="lines",
            line=dict(color="rgba(0, 128, 0, 0.3)", width=1),
            name="Upper Band (1σ)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=aggregated["lower_band"],
            mode="lines",
            line=dict(color="rgba(0, 128, 0, 0.3)", width=1),
            name="Lower Band (1σ)",
            fill="tonexty",
            fillcolor="rgba(0, 128, 0, 0.1)",
        )
    )

    # Update layout
    hours_text = "hour" if hours_forward == 1 else "hours"
    fig.update_layout(
        title=f"Projected Price Movement ({hours_forward} {hours_text} after pattern)",
        xaxis_title="Time",
        yaxis_title="Projected Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig
