import robin_stocks.robinhood as rh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import getpass
import time
from scipy.interpolate import griddata


def login_to_robinhood():
    """
    Log in to Robinhood with user credentials
    """
    username = input("Enter your Robinhood username (email): ")
    password = getpass.getpass("Enter your Robinhood password: ")

    # Log in to Robinhood
    login = rh.login(username, password)
    print("Login successful")
    return login


def get_option_chain_data(ticker="SPY"):
    """
    Fetch all available option chain data for the given ticker
    """
    # Get all available expiration dates
    print(f"Fetching option expiration dates for {ticker}...")
    exp_dates = rh.options.get_chains(ticker)

    # Initialize empty list to store all option data
    all_options = []

    # Process each expiration date
    for date in exp_dates["expiration_dates"]:
        print(f"Fetching options for expiration: {date}")

        # Get option chain for both calls and puts
        for option_type in ["call", "put"]:
            options = rh.options.find_options_by_expiration_and_type(
                ticker, date, option_type
            )

            # Add options to our list
            if options:
                all_options.extend(options)

        # Brief pause to avoid hitting rate limits
        time.sleep(0.5)

    # Convert to DataFrame for easier processing
    options_df = pd.DataFrame(all_options)
    return options_df


def calculate_days_to_expiry(expiry_date):
    """
    Calculate days to expiry from today
    """
    today = datetime.now().date()
    expiry = datetime.strptime(expiry_date, "%Y-%m-%d").date()
    return (expiry - today).days


def process_option_data(options_df, ticker="SPY"):
    """
    Process the raw option data to extract metrics needed for volatility surface
    """
    # Get the current stock price
    stock_price = float(rh.stocks.get_latest_price(ticker)[0])
    print(f"Current {ticker} price: ${stock_price}")

    # Add important calculated fields
    options_df["days_to_expiry"] = options_df["expiration_date"].apply(
        calculate_days_to_expiry
    )
    options_df["strike_price"] = options_df["strike_price"].astype(float)
    options_df["implied_volatility"] = options_df["implied_volatility"].astype(float)
    options_df["moneyness"] = options_df["strike_price"] / stock_price

    # Filter out options with no implied volatility or expired options
    options_df = options_df[options_df["implied_volatility"] > 0]
    options_df = options_df[options_df["days_to_expiry"] > 0]

    # Keep only relevant columns
    cols_to_keep = [
        "strike_price",
        "expiration_date",
        "days_to_expiry",
        "type",
        "implied_volatility",
        "moneyness",
    ]
    cleaned_df = options_df[cols_to_keep]

    return cleaned_df, stock_price


def create_volatility_surface(options_df):
    """
    Create and plot a volatility surface
    """
    # Extract data points for the surface
    x = options_df["moneyness"].values
    y = options_df["days_to_expiry"].values
    z = options_df["implied_volatility"].values

    # Create a grid for the surface
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)

    # Interpolate the surface
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method="cubic")

    # Create the mesh grid
    X, Y = np.meshgrid(xi, yi)

    # Create 3D plot
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(X, Y, zi, cmap=cm.coolwarm, linewidth=0, antialiased=True)

    # Add color bar and labels
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("Moneyness (Strike/Spot)")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Volatility")
    plt.title("SPY Options Implied Volatility Surface")

    # Show the plot
    plt.savefig("volatility_surface.png")
    print("Volatility surface plot saved as 'volatility_surface.png'")
    plt.show()

    # Also return the raw surface data
    return X, Y, zi


def calculate_local_volatility(X, Y, Z, current_price):
    """
    Calculate local volatility from the implied volatility surface using Dupire's formula

    This is a simplified implementation and assumes:
    - Z is the implied volatility surface
    - X is the moneyness grid (K/S)
    - Y is the time to expiry grid in days
    """
    # Convert days to years for financial calculations
    Y_years = Y / 365.0

    # Get the step sizes
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # Initialize local volatility surface
    local_vol = np.zeros_like(Z)

    # Simple differentiation for internal points (not the edges)
    for i in range(1, Z.shape[0] - 1):
        for j in range(1, Z.shape[1] - 1):
            if Y_years[i, j] > 0 and X[i, j] > 0:
                # Get implied volatility at this point
                sigma = Z[i, j]

                # Calculate derivatives
                d_sigma_dt = (Z[i + 1, j] - Z[i - 1, j]) / (2 * dy)
                d_sigma_dk = (Z[i, j + 1] - Z[i, j - 1]) / (2 * dx)
                d2_sigma_dk2 = (Z[i, j + 1] - 2 * Z[i, j] + Z[i, j - 1]) / (dx**2)

                K = X[i, j] * current_price
                t = Y_years[i, j]

                # Terms in Dupire's formula
                term1 = sigma**2
                term2 = 2 * t * d_sigma_dt
                term3 = 2 * sigma * t * d_sigma_dk * K
                term4 = t * sigma * K**2 * d2_sigma_dk2

                # Calculate local volatility (with a safety check)
                denominator = (
                    1 + K * t * d_sigma_dk
                ) ** 2 + t * K**2 * sigma * d2_sigma_dk2

                if denominator > 0:
                    local_vol[i, j] = np.sqrt((term1 + term2) / denominator)
                else:
                    local_vol[
                        i, j
                    ] = sigma  # Fallback to implied vol if calculation fails

    # Handle the edges by simple extension
    local_vol[0, :] = local_vol[1, :]
    local_vol[-1, :] = local_vol[-2, :]
    local_vol[:, 0] = local_vol[:, 1]
    local_vol[:, -1] = local_vol[:, -2]

    return local_vol


def plot_local_volatility(X, Y, local_vol):
    """
    Plot the local volatility surface
    """
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(
        X, Y, local_vol, cmap=cm.viridis, linewidth=0, antialiased=True
    )

    # Add color bar and labels
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("Moneyness (Strike/Spot)")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Local Volatility")
    plt.title("SPY Options Local Volatility Surface")

    # Show the plot
    plt.savefig("local_volatility_surface.png")
    print("Local volatility surface plot saved as 'local_volatility_surface.png'")
    plt.show()

    return local_vol


def main():
    # Login to Robinhood
    login_to_robinhood()

    # Get option chain data
    options_df = get_option_chain_data("SPY")

    # Process the data
    processed_data, current_price = process_option_data(options_df)

    print(f"Processed {len(processed_data)} option contracts")

    # Create implied volatility surface
    X, Y, Z = create_volatility_surface(processed_data)

    # Calculate and plot local volatility surface
    local_vol = calculate_local_volatility(X, Y, Z, current_price)
    plot_local_volatility(X, Y, local_vol)


if __name__ == "__main__":
    main()
